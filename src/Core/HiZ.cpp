#include "Core/HiZ.h"

#include "Core/HiZBuild.h"
#include "Settings.h"
#include "Util/D3D.h"

#include <algorithm>
#include <cstring>
#include <d3dcompiler.h>
#include <wrl/client.h>

namespace CBRO::Core::HiZ
{
	using Microsoft::WRL::ComPtr;

	namespace
	{
		// The GPU build (shaders, constant layout, pyramid layout) is in HiZBuild.h; this file owns the D3D objects,
		// the capture/readback ring and the published snapshots. The readback is one flat buffer (far pyramid, near
		// pyramid, block-change map) mapped without any CPU processing: the snapshot points into the mapping.
		constexpr std::uint32_t kUAVs = Build::kUAVs;
		constexpr std::uint32_t kRingSize = Build::kRingSize;
		// Published snapshots point into their readback's mapping, so a readback slot stays mapped until its snapshot
		// has been superseded kSnapshots times (a frame's readers, the cull stage's threads and the lamp culls, are done
		// long before: they finish inside their own frame, and Poll runs at the next frame's cull start).
		constexpr std::uint32_t kSnapshots = 4;
		constexpr std::uint32_t kSlots = kSnapshots + 4;  // ... plus up to four captures in flight on the GPU

		struct Slot
		{
			enum class State : std::uint8_t
			{
				kFree,
				kPending,  // copied into on the GPU, not yet mapped
				kMapped,   // held by a published snapshot
			};
			ComPtr<ID3D11Buffer> staging;
			std::uint64_t        frame{ 0 };
			Camera               camera{};
			std::uint32_t        captureIndex{ 0 };
			std::uint32_t        mergedFrames{ 1 };
			float                mergeMove{ 0.0f };
			State                state{ State::kFree };
			const std::uint8_t*  mapped{ nullptr };
		};

		struct Gpu
		{
			ComPtr<ID3D11ComputeShader>       build;
			ComPtr<ID3D11ComputeShader>       tail;
			ComPtr<ID3D11Buffer>              params;
			// Private copy of the main depth: the compute pass reads it, so the engine's depth target
			// never has to be unbound (ENB and Upscaling track the output-merger bindings).
			ComPtr<ID3D11Texture2D>           depthCopy;
			ComPtr<ID3D11ShaderResourceView>  depthCopySRV;
			ComPtr<ID3D11Texture2D>           ringNear;
			ComPtr<ID3D11Texture2D>           ringFar;
			ComPtr<ID3D11Texture2D>           refFar;
			ComPtr<ID3D11Texture2D>           refNear;
			ComPtr<ID3D11Texture2D>           ringDrawn;
			ComPtr<ID3D11Buffer>              out;
			ComPtr<ID3D11UnorderedAccessView> uavs[kUAVs];  // ring near, ring far, ref far, ref near, out, ring drawn
			std::array<Slot, kSlots>          slots;
			Build::Layout                     layout;
			Build::OutputLayout               output;
			std::uint32_t                     srcWidth{ 0 };
			std::uint32_t                     srcHeight{ 0 };
			std::uint32_t                     dstWidth{ 0 };
			std::uint32_t                     dstHeight{ 0 };
			bool                              initialized{ false };
			bool                              failed{ false };
		};
		Gpu g_gpu;

		std::array<Snapshot, kSnapshots> g_snapshots;
		std::atomic<int>                 g_published{ -1 };

		// The captures in the GPU ring (their cameras decide which slices the shader merges). A max over frames can only
		// make more visible, whatever the cameras were, so the gate is only about usefulness; v1.17's 2 units / 0.03
		// degrees broke the chain on idle sway, and the published depth then alternated between a 4-frame max and a
		// single frame, flipping the edge objects in bulk.
		struct RingEntry
		{
			std::uint64_t frame{ 0 };
			float         eye[3]{};
			float         rotate[3][3]{};
			bool          valid{ false };
		};
		std::array<RingEntry, kRingSize> g_ring{};
		std::uint32_t                    g_ringCur{ 0 };
		constexpr float                  kMergeMove = 16.0f;                     // units of camera movement per frame
		constexpr float                  kMergeTurn = 2.0f * 0.0087f * 0.0087f;  // ||dR||^2 for ~0.5 degrees per frame

		// The block-change map as last read back (the GPU keeps the authoritative one and the references).
		std::vector<std::uint32_t> g_blockChangedAt;
		std::uint32_t              g_blocksW{ 0 };
		std::uint32_t              g_blocksH{ 0 };
		std::uint32_t              g_captureIndex{ 0 };
		std::uint32_t              g_blocksChanged{ 0 };  // since the last log
		BlockChangeKinds           g_blockChangeKinds{};  // ... by what the newest capture holds there (v1.52 diagnostic)
		bool                       g_refValid{ false };   // the GPU references hold linear depths of the current projection
		float                      g_refDepthA{ 0.0f };
		float                      g_refDepthB{ 0.0f };

		struct Counters
		{
			std::uint64_t captures{ 0 };
			std::uint64_t noFreeSlot{ 0 };
			std::uint64_t readbacks{ 0 };
			std::uint64_t merges{ 0 };        // published depths merged with at least one earlier capture
			std::uint64_t mergedFrames{ 0 };  // captures merged in, summed
			std::array<std::uint64_t, 9> spans{};  // published depths by the number of frames they span (index 8 = 8 or more)
			std::int32_t  depthIndex{ -1 };
		};
		Counters g_counters;

		// How many of the previous captures (consecutive, newest first) this capture's level 0 merges: those whose camera
		// is nearly a_camera (the chain stops at the first that moved too far). a_move receives the largest distance.
		std::uint32_t MergeChain(std::uint64_t a_frame, const Camera& a_camera, std::uint32_t a_maxFrames, float& a_move)
		{
			a_move = 0.0f;
			std::uint32_t merged = 0;
			for (std::uint32_t k = 1; k < a_maxFrames && k < kRingSize; ++k) {
				const auto& older = g_ring[(g_ringCur + kRingSize - k) % kRingSize];
				if (!older.valid || older.frame + k + 2 < a_frame) {
					break;
				}
				const float dx = older.eye[0] - a_camera.eye[0];
				const float dy = older.eye[1] - a_camera.eye[1];
				const float dz = older.eye[2] - a_camera.eye[2];
				float       squared = 0.0f;
				for (int r = 0; r < 3; ++r) {
					for (int c = 0; c < 3; ++c) {
						const float d = older.rotate[r][c] - a_camera.rotate[r][c];
						squared += d * d;
					}
				}
				const float distance = std::sqrt(dx * dx + dy * dy + dz * dz);
				if (distance > kMergeMove * static_cast<float>(k) || squared > kMergeTurn * static_cast<float>(k)) {
					break;
				}
				a_move = std::max(a_move, distance);
				++merged;
			}
			return merged;
		}

		void Fail(std::string_view a_what, HRESULT a_hr)
		{
			logger::error("hi-z: {} failed (hr 0x{:08X}); occlusion culling disabled for this session", a_what, static_cast<std::uint32_t>(a_hr));
			g_gpu.failed = true;
		}

		bool Compile(ID3D11Device* a_device, const char* a_entry, ComPtr<ID3D11ComputeShader>& a_out)
		{
			const auto scale = std::format("{:.8f}", 1.0f + kCacheDepthTolerance * 0.5f);
			const auto slack = std::format("{:.8f}", kCacheDepthSlack * 0.5f);
			const D3D_SHADER_MACRO macros[]{
				{ "CBRO_SCALE", scale.c_str() },
				{ "CBRO_SLACK", slack.c_str() },
				{ nullptr, nullptr }
			};
			ComPtr<ID3DBlob> code;
			ComPtr<ID3DBlob> errors;
			auto             hr = D3DCompile(
                Build::kSource, sizeof(Build::kSource) - 1, "CBRO_HiZ", macros, nullptr, a_entry, "cs_5_0",
                D3DCOMPILE_OPTIMIZATION_LEVEL3, 0, code.GetAddressOf(), errors.GetAddressOf());
			if (FAILED(hr)) {
				logger::error("hi-z: shader '{}' compile errors: {}", a_entry, errors ? static_cast<const char*>(errors->GetBufferPointer()) : "?");
				Fail("D3DCompile", hr);
				return false;
			}
			hr = a_device->CreateComputeShader(code->GetBufferPointer(), code->GetBufferSize(), nullptr, a_out.GetAddressOf());
			if (FAILED(hr)) {
				Fail("CreateComputeShader", hr);
				return false;
			}
			return true;
		}

		bool Initialize(ID3D11Device* a_device)
		{
			if (g_gpu.initialized) {
				return !g_gpu.failed;
			}
			g_gpu.initialized = true;

			if (!Compile(a_device, "build", g_gpu.build) || !Compile(a_device, "tail", g_gpu.tail)) {
				return false;
			}
			D3D11_BUFFER_DESC desc{};
			desc.ByteWidth = sizeof(Build::Params);
			desc.Usage = D3D11_USAGE_DEFAULT;
			desc.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
			const auto hr = a_device->CreateBuffer(&desc, nullptr, g_gpu.params.GetAddressOf());
			if (FAILED(hr)) {
				Fail("CreateBuffer(params)", hr);
				return false;
			}
			logger::info("hi-z: compute shaders ready (build + tail)");
			return true;
		}

		// Unmaps the readback a snapshot points into, if any (nothing reads the snapshot any more).
		void ReleaseSlot(ID3D11DeviceContext* a_context, Snapshot& a_snapshot)
		{
			if (a_snapshot.slot >= 0) {
				auto& slot = g_gpu.slots[static_cast<std::size_t>(a_snapshot.slot)];
				if (slot.state == Slot::State::kMapped) {
					if (a_context && slot.staging) {
						a_context->Unmap(slot.staging.Get(), 0);
					}
					slot.state = Slot::State::kFree;
					slot.mapped = nullptr;
				}
			}
			a_snapshot.slot = -1;
			a_snapshot.texels = nullptr;
			a_snapshot.nearest = nullptr;
			a_snapshot.drawn = nullptr;
		}

		bool CreateTexture(ID3D11Device* a_device, std::uint32_t a_slices, ComPtr<ID3D11Texture2D>& a_texture, ComPtr<ID3D11UnorderedAccessView>& a_uav, std::string_view a_what)
		{
			D3D11_TEXTURE2D_DESC desc{};
			desc.Width = g_gpu.dstWidth;
			desc.Height = g_gpu.dstHeight;
			desc.MipLevels = 1;
			desc.ArraySize = a_slices;
			desc.Format = DXGI_FORMAT_R32_FLOAT;
			desc.SampleDesc.Count = 1;
			desc.Usage = D3D11_USAGE_DEFAULT;
			desc.BindFlags = D3D11_BIND_UNORDERED_ACCESS;
			auto hr = a_device->CreateTexture2D(&desc, nullptr, a_texture.GetAddressOf());
			if (FAILED(hr)) {
				Fail(std::format("CreateTexture2D({})", a_what), hr);
				return false;
			}
			hr = a_device->CreateUnorderedAccessView(a_texture.Get(), nullptr, a_uav.GetAddressOf());
			if (FAILED(hr)) {
				Fail(std::format("CreateUnorderedAccessView({})", a_what), hr);
				return false;
			}
			return true;
		}

		bool EnsureTargets(ID3D11Device* a_device, ID3D11DeviceContext* a_context, const D3D11_TEXTURE2D_DESC& a_depthDesc, ID3D11ShaderResourceView* a_depthSRV)
		{
			const auto srcWidth = a_depthDesc.Width;
			const auto srcHeight = a_depthDesc.Height;
			if (srcWidth == g_gpu.srcWidth && srcHeight == g_gpu.srcHeight && g_gpu.out && g_gpu.depthCopySRV) {
				return true;
			}

			Reset();  // (unmaps whatever the old slots held)
			const auto factor = std::max(1u, Settings::Get().hiZDownsample);
			g_gpu.srcWidth = srcWidth;
			g_gpu.srcHeight = srcHeight;
			g_gpu.dstWidth = (srcWidth + factor - 1) / factor;
			g_gpu.dstHeight = (srcHeight + factor - 1) / factor;
			g_blocksW = (g_gpu.dstWidth + 7) / 8;
			g_blocksH = (g_gpu.dstHeight + 7) / 8;
			g_blockChangedAt.assign(static_cast<std::size_t>(g_blocksW) * g_blocksH, 0);
			Build::ComputeLayout(g_gpu.layout, g_gpu.dstWidth, g_gpu.dstHeight);
			g_gpu.output = Build::ComputeOutput(g_gpu.layout, g_blocksW * g_blocksH);
			g_gpu.depthCopy.Reset();
			g_gpu.depthCopySRV.Reset();
			g_gpu.ringNear.Reset();
			g_gpu.ringFar.Reset();
			g_gpu.refFar.Reset();
			g_gpu.refNear.Reset();
			g_gpu.ringDrawn.Reset();
			g_gpu.out.Reset();
			for (auto& uav : g_gpu.uavs) {
				uav.Reset();
			}
			for (auto& slot : g_gpu.slots) {
				slot = {};
			}

			// Same layout and typeless format as the engine's depth (CopyResource needs both), readable only.
			D3D11_TEXTURE2D_DESC copyDesc = a_depthDesc;
			copyDesc.Usage = D3D11_USAGE_DEFAULT;
			copyDesc.BindFlags = D3D11_BIND_SHADER_RESOURCE;
			copyDesc.CPUAccessFlags = 0;
			copyDesc.MiscFlags = 0;
			auto hr = a_device->CreateTexture2D(&copyDesc, nullptr, g_gpu.depthCopy.GetAddressOf());
			if (FAILED(hr)) {
				Fail("CreateTexture2D(depth copy)", hr);
				return false;
			}
			D3D11_SHADER_RESOURCE_VIEW_DESC srvDesc{};
			a_depthSRV->GetDesc(&srvDesc);
			hr = a_device->CreateShaderResourceView(g_gpu.depthCopy.Get(), &srvDesc, g_gpu.depthCopySRV.GetAddressOf());
			if (FAILED(hr)) {
				Fail("CreateShaderResourceView(depth copy)", hr);
				return false;
			}

			if (!CreateTexture(a_device, kRingSize, g_gpu.ringNear, g_gpu.uavs[0], "ring near") ||
				!CreateTexture(a_device, kRingSize, g_gpu.ringFar, g_gpu.uavs[1], "ring far") ||
				!CreateTexture(a_device, 1, g_gpu.refFar, g_gpu.uavs[2], "reference far") ||
				!CreateTexture(a_device, 1, g_gpu.refNear, g_gpu.uavs[3], "reference near") ||
				!CreateTexture(a_device, kRingSize, g_gpu.ringDrawn, g_gpu.uavs[5], "ring drawn")) {
				return false;
			}

			D3D11_BUFFER_DESC bufferDesc{};
			bufferDesc.ByteWidth = g_gpu.output.bytes;
			bufferDesc.Usage = D3D11_USAGE_DEFAULT;
			bufferDesc.BindFlags = D3D11_BIND_UNORDERED_ACCESS;
			bufferDesc.MiscFlags = D3D11_RESOURCE_MISC_BUFFER_ALLOW_RAW_VIEWS;
			hr = a_device->CreateBuffer(&bufferDesc, nullptr, g_gpu.out.GetAddressOf());
			if (FAILED(hr)) {
				Fail("CreateBuffer(hi-z output)", hr);
				return false;
			}
			D3D11_UNORDERED_ACCESS_VIEW_DESC uavDesc{};
			uavDesc.Format = DXGI_FORMAT_R32_TYPELESS;
			uavDesc.ViewDimension = D3D11_UAV_DIMENSION_BUFFER;
			uavDesc.Buffer.FirstElement = 0;
			uavDesc.Buffer.NumElements = g_gpu.output.bytes / 4;
			uavDesc.Buffer.Flags = D3D11_BUFFER_UAV_FLAG_RAW;
			hr = a_device->CreateUnorderedAccessView(g_gpu.out.Get(), &uavDesc, g_gpu.uavs[4].GetAddressOf());
			if (FAILED(hr)) {
				Fail("CreateUnorderedAccessView(hi-z output)", hr);
				return false;
			}
			// The change map persists in the output buffer across captures (only marked blocks are written): start it at 0.
			const UINT zeros[4]{};
			a_context->ClearUnorderedAccessViewUint(g_gpu.uavs[4].Get(), zeros);

			bufferDesc.Usage = D3D11_USAGE_STAGING;
			bufferDesc.BindFlags = 0;
			bufferDesc.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
			bufferDesc.MiscFlags = 0;
			for (auto& slot : g_gpu.slots) {
				hr = a_device->CreateBuffer(&bufferDesc, nullptr, slot.staging.GetAddressOf());
				if (FAILED(hr)) {
					Fail("CreateBuffer(staging)", hr);
					return false;
				}
			}

			logger::info(
				"hi-z: depth {}x{} -> hi-z {}x{} (factor {}; nearest and farthest per texel, and level 0's farthest drawn), {} levels and the {}x{} block map built on the GPU, read from a private copy; readback {} KB x {} slots",
				srcWidth, srcHeight, g_gpu.dstWidth, g_gpu.dstHeight, factor, g_gpu.layout.levels, g_blocksW, g_blocksH, g_gpu.output.bytes / 1024, kSlots);
			return true;
		}

		// The DSV bound after the pre-pass is the main depth; find its RendererData entry
		// (Phase 0: depthStencilTargets[2] with ENB + Upscaling, but never assume).
		std::int32_t FindDepthTarget(ID3D11DepthStencilView* a_dsv)
		{
			const auto data = RE::BSGraphics::RendererData::GetSingleton();
			if (!data) {
				return -1;
			}
			if (a_dsv) {
				ID3D11Resource* resource = nullptr;
				a_dsv->GetResource(&resource);
				std::int32_t found = -1;
				for (std::size_t i = 0; i < std::size(data->depthStencilTargets) && resource; ++i) {
					if (reinterpret_cast<std::uintptr_t>(data->depthStencilTargets[i].texture) == reinterpret_cast<std::uintptr_t>(resource)) {
						found = static_cast<std::int32_t>(i);
						break;
					}
				}
				if (resource) {
					resource->Release();
				}
				if (found >= 0) {
					return found;
				}
			}
			return data->depthStencilTargets[2].srViewDepth ? 2 : -1;
		}
	}

	float Snapshot::MaxDepth(float a_x0, float a_y0, float a_x1, float a_y1) const noexcept
	{
		// Pick the level where the rectangle spans at most 4 texels per axis (<= 5x5 reads).
		auto          span = std::max(a_x1 - a_x0, a_y1 - a_y0);
		std::uint32_t level = 0;
		while (span > 4.0f && level + 1 < levels) {
			span *= 0.5f;
			++level;
		}

		const float scale = 1.0f / static_cast<float>(1u << level);
		const auto  w = width[level];
		const auto  h = height[level];
		const auto  clampX = [&](float a_v) { return static_cast<std::uint32_t>(std::clamp(a_v * scale, 0.0f, static_cast<float>(w - 1))); };
		const auto  clampY = [&](float a_v) { return static_cast<std::uint32_t>(std::clamp(a_v * scale, 0.0f, static_cast<float>(h - 1))); };
		const auto  x0 = clampX(a_x0);
		const auto  x1 = clampX(a_x1);
		const auto  y0 = clampY(a_y0);
		const auto  y1 = clampY(a_y1);

		const auto* base = texels + offset[level];
		float       farthest = 0.0f;
		for (auto y = y0; y <= y1; ++y) {
			const auto* row = base + y * w;
			for (auto x = x0; x <= x1; ++x) {
				farthest = std::max(farthest, row[x]);
			}
		}
		return farthest;
	}

	bool SphereRays::Misses(float a_x0, float a_y0, float a_x1, float a_y1) const noexcept
	{
		// The ray in the rectangle closest to the center's direction (per-axis scaling keeps the
		// clamp exact), and the steepest ray in it for the cosine bound.
		const float du = (std::clamp(centerX, a_x0, a_x1) - centerX) / texelsPerTanX;
		const float dv = (std::clamp(centerY, a_y0, a_y1) - centerY) / texelsPerTanY;
		const float lateral = depth * std::sqrt(du * du + dv * dv);
		const float tu = std::max(std::abs(a_x0 - axisX), std::abs(a_x1 - axisX)) / texelsPerTanX;
		const float tv = std::max(std::abs(a_y0 - axisY), std::abs(a_y1 - axisY)) / texelsPerTanY;
		return lateral > radius * std::sqrt(1.0f + tu * tu + tv * tv);
	}

	bool Snapshot::AllNearer(float a_x0, float a_y0, float a_x1, float a_y1, float a_threshold, std::uint32_t a_refine, const SphereRays* a_rays) const noexcept
	{
		auto          span = std::max(a_x1 - a_x0, a_y1 - a_y0);
		std::uint32_t level = 0;
		while (span > 4.0f && level + 1 < levels) {
			span *= 0.5f;
			++level;
		}
		return AllNearerAt(level, a_x0, a_y0, a_x1, a_y1, a_threshold, a_refine, a_rays);
	}

	bool Snapshot::AllNearerAt(std::uint32_t a_level, float a_x0, float a_y0, float a_x1, float a_y1, float a_threshold, std::uint32_t a_refine, const SphereRays* a_rays) const noexcept
	{
		const float size = static_cast<float>(1u << a_level);  // level-0 texels per texel at this level
		const float scale = 1.0f / size;
		const auto  w = width[a_level];
		const auto  h = height[a_level];
		const auto  x0 = static_cast<std::uint32_t>(std::clamp(a_x0 * scale, 0.0f, static_cast<float>(w - 1)));
		const auto  x1 = static_cast<std::uint32_t>(std::clamp(a_x1 * scale, 0.0f, static_cast<float>(w - 1)));
		const auto  y0 = static_cast<std::uint32_t>(std::clamp(a_y0 * scale, 0.0f, static_cast<float>(h - 1)));
		const auto  y1 = static_cast<std::uint32_t>(std::clamp(a_y1 * scale, 0.0f, static_cast<float>(h - 1)));

		const auto* base = texels + offset[a_level];
		for (auto y = y0; y <= y1; ++y) {
			const auto* row = base + y * w;
			for (auto x = x0; x <= x1; ++x) {
				if (row[x] < a_threshold) {
					continue;
				}
				// Only the part of the rectangle inside this texel matters.
				const float tx0 = static_cast<float>(x) * size;
				const float ty0 = static_cast<float>(y) * size;
				const float sx0 = std::max(a_x0, tx0);
				const float sy0 = std::max(a_y0, ty0);
				const float sx1 = std::max(sx0, std::min(a_x1, tx0 + size - 0.001f));
				const float sy1 = std::max(sy0, std::min(a_y1, ty0 + size - 0.001f));
				// (a quarter texel of slack covers the render's sub-pixel jitter)
				if (a_rays && a_rays->Misses(sx0 - 0.25f, sy0 - 0.25f, sx1 + 0.25f, sy1 + 0.25f)) {
					continue;  // the object can't be seen through this texel at all
				}
				if (a_refine == 0 || a_level == 0) {
					return false;
				}
				if (!AllNearerAt(a_level - 1, sx0, sy0, sx1, sy1, a_threshold, a_refine - 1, a_rays)) {
					return false;
				}
			}
		}
		return true;
	}

	Snapshot::FartherScan Snapshot::ScanFarther(float a_x0, float a_y0, float a_x1, float a_y1, float a_threshold, const SphereRays* a_rays) const noexcept
	{
		FartherScan scan{};
		const auto  w = width[0];
		const auto  h = height[0];
		const auto  x0 = static_cast<std::uint32_t>(std::clamp(a_x0, 0.0f, static_cast<float>(w - 1)));
		const auto  x1 = static_cast<std::uint32_t>(std::clamp(a_x1, 0.0f, static_cast<float>(w - 1)));
		const auto  y0 = static_cast<std::uint32_t>(std::clamp(a_y0, 0.0f, static_cast<float>(h - 1)));
		const auto  y1 = static_cast<std::uint32_t>(std::clamp(a_y1, 0.0f, static_cast<float>(h - 1)));
		for (auto y = y0; y <= y1; ++y) {
			const auto* row = texels + offset[0] + y * w;
			for (auto x = x0; x <= x1; ++x) {
				if (row[x] < a_threshold) {
					continue;
				}
				const float fx = static_cast<float>(x), fy = static_cast<float>(y);
				if (a_rays && a_rays->Misses(fx - 0.25f, fy - 0.25f, fx + 1.249f, fy + 1.249f)) {
					continue;
				}
				if (scan.texels++ == 0) {
					scan.x = fx;
					scan.y = fy;
				}
				const float nearDepth = nearest[offset[0] + y * w + x];
				scan.farPlane += row[x] >= 0.99999f;
				scan.firstPerson += row[x] >= 0.99999f && nearDepth <= 0.0f;
				scan.farthest = std::max(scan.farthest, row[x]);
				if (row[x] >= 0.99999f) {
					scan.farPlaneNearest = std::min(scan.farPlaneNearest, nearDepth);  // (< 1: something is drawn in that texel too)
					if (drawn) {
						scan.farPlaneDrawn = std::max(scan.farPlaneDrawn, drawn[y * w + x]);
					}
				}
			}
		}
		return scan;
	}

	bool Snapshot::AllFarther(float a_x0, float a_y0, float a_x1, float a_y1, float a_threshold, std::uint32_t a_refine) const noexcept
	{
		auto          span = std::max(a_x1 - a_x0, a_y1 - a_y0);
		std::uint32_t level = 0;
		while (span > 4.0f && level + 1 < levels) {
			span *= 0.5f;
			++level;
		}
		return AllFartherAt(level, a_x0, a_y0, a_x1, a_y1, a_threshold, a_refine);
	}

	bool Snapshot::AllFartherAt(std::uint32_t a_level, float a_x0, float a_y0, float a_x1, float a_y1, float a_threshold, std::uint32_t a_refine) const noexcept
	{
		const float size = static_cast<float>(1u << a_level);
		const float scale = 1.0f / size;
		const auto  w = width[a_level];
		const auto  h = height[a_level];
		const auto  x0 = static_cast<std::uint32_t>(std::clamp(a_x0 * scale, 0.0f, static_cast<float>(w - 1)));
		const auto  x1 = static_cast<std::uint32_t>(std::clamp(a_x1 * scale, 0.0f, static_cast<float>(w - 1)));
		const auto  y0 = static_cast<std::uint32_t>(std::clamp(a_y0 * scale, 0.0f, static_cast<float>(h - 1)));
		const auto  y1 = static_cast<std::uint32_t>(std::clamp(a_y1 * scale, 0.0f, static_cast<float>(h - 1)));

		const auto* base = nearest + offset[a_level];
		for (auto y = y0; y <= y1; ++y) {
			const auto* row = base + y * w;
			for (auto x = x0; x <= x1; ++x) {
				if (row[x] >= a_threshold) {
					continue;  // nothing in this texel is nearer than the threshold
				}
				const float tx0 = static_cast<float>(x) * size;
				const float ty0 = static_cast<float>(y) * size;
				const float sx0 = std::max(a_x0, tx0);
				const float sy0 = std::max(a_y0, ty0);
				const float sx1 = std::max(sx0, std::min(a_x1, tx0 + size - 0.001f));
				const float sy1 = std::max(sy0, std::min(a_y1, ty0 + size - 0.001f));
				if (a_refine == 0 || a_level == 0) {
					return false;
				}
				if (!AllFartherAt(a_level - 1, sx0, sy0, sx1, sy1, a_threshold, a_refine - 1)) {
					return false;
				}
			}
		}
		return true;
	}

	bool Snapshot::NoSurfaceBetween(float a_x0, float a_y0, float a_x1, float a_y1, float a_near, float a_far, std::uint32_t a_refine, const SphereRays* a_rays, bool a_drawnOnly) const noexcept
	{
		auto          span = std::max(a_x1 - a_x0, a_y1 - a_y0);
		std::uint32_t level = 0;
		while (span > 4.0f && level + 1 < levels) {
			span *= 0.5f;
			++level;
		}
		return NoSurfaceBetweenAt(level, a_x0, a_y0, a_x1, a_y1, a_near, a_far, a_refine, a_rays, a_drawnOnly && drawn);
	}

	bool Snapshot::NoSurfaceBetweenAt(std::uint32_t a_level, float a_x0, float a_y0, float a_x1, float a_y1, float a_near, float a_far, std::uint32_t a_refine, const SphereRays* a_rays, bool a_drawnOnly) const noexcept
	{
		const float size = static_cast<float>(1u << a_level);
		const float scale = 1.0f / size;
		const auto  w = width[a_level];
		const auto  h = height[a_level];
		const auto  x0 = static_cast<std::uint32_t>(std::clamp(a_x0 * scale, 0.0f, static_cast<float>(w - 1)));
		const auto  x1 = static_cast<std::uint32_t>(std::clamp(a_x1 * scale, 0.0f, static_cast<float>(w - 1)));
		const auto  y0 = static_cast<std::uint32_t>(std::clamp(a_y0 * scale, 0.0f, static_cast<float>(h - 1)));
		const auto  y1 = static_cast<std::uint32_t>(std::clamp(a_y1 * scale, 0.0f, static_cast<float>(h - 1)));

		const auto* farBase = texels + offset[a_level];
		const auto* nearBase = nearest + offset[a_level];
		const auto* frontBase = a_drawnOnly && a_level == 0 ? drawn : farBase;  // (the coarse levels keep the farthest depth)
		for (auto y = y0; y <= y1; ++y) {
			for (auto x = x0; x <= x1; ++x) {
				const auto i = y * w + x;
				if (frontBase[i] < a_near || nearBase[i] > a_far) {
					continue;  // every surface here is in front of the range, or behind it
				}
				const float tx0 = static_cast<float>(x) * size;
				const float ty0 = static_cast<float>(y) * size;
				const float sx0 = std::max(a_x0, tx0);
				const float sy0 = std::max(a_y0, ty0);
				const float sx1 = std::max(sx0, std::min(a_x1, tx0 + size - 0.001f));
				const float sy1 = std::max(sy0, std::min(a_y1, ty0 + size - 0.001f));
				// (a quarter texel of slack covers the render's sub-pixel jitter, as in AllNearerAt)
				if (a_rays && a_rays->Misses(sx0 - 0.25f, sy0 - 0.25f, sx1 + 0.25f, sy1 + 0.25f)) {
					continue;  // no ray through this texel meets the sphere: what is seen here lies outside it
				}
				if (a_refine == 0 || a_level == 0) {
					return false;
				}
				if (!NoSurfaceBetweenAt(a_level - 1, sx0, sy0, sx1, sy1, a_near, a_far, a_refine - 1, a_rays, a_drawnOnly)) {
					return false;
				}
			}
		}
		return true;
	}

	bool Capture(std::uint64_t a_frame, const Camera& a_camera)
	{
		if (g_gpu.failed) {
			return false;
		}
		const auto device = Util::GetDevice();
		const auto context = Util::GetContext();
		const auto data = RE::BSGraphics::RendererData::GetSingleton();
		if (!device || !context || !data || !Initialize(device)) {
			return false;
		}

		// Save the compute state we touch. The output merger is only read (which depth is bound), never changed.
		ID3D11DepthStencilView*    dsv = nullptr;
		ID3D11ComputeShader*       oldShader = nullptr;
		ID3D11ClassInstance*       oldInstances[256]{};
		UINT                       oldInstanceCount = 256;
		ID3D11ShaderResourceView*  oldSRV = nullptr;
		ID3D11UnorderedAccessView* oldUAVs[kUAVs]{};
		ID3D11Buffer*              oldCB = nullptr;
		context->OMGetRenderTargets(0, nullptr, &dsv);

		bool       ok = false;
		const auto index = FindDepthTarget(dsv);
		if (index != g_counters.depthIndex) {
			logger::info("hi-z: main depth is depthStencilTargets[{}]", index);
			g_counters.depthIndex = index;
		}

		const auto depthSRV = index >= 0 ? reinterpret_cast<ID3D11ShaderResourceView*>(data->depthStencilTargets[index].srViewDepth) : nullptr;
		const auto depthTex = index >= 0 ? reinterpret_cast<ID3D11Texture2D*>(data->depthStencilTargets[index].texture) : nullptr;

		if (depthSRV && depthTex) {
			D3D11_TEXTURE2D_DESC desc{};
			depthTex->GetDesc(&desc);
			if (EnsureTargets(device, context, desc, depthSRV)) {
				Slot* slot = nullptr;
				for (auto& candidate : g_gpu.slots) {
					if (candidate.state == Slot::State::kFree) {
						slot = &candidate;
						break;
					}
				}
				if (!slot) {
					++g_counters.noFreeSlot;
				} else {
					context->CSGetShader(&oldShader, oldInstances, &oldInstanceCount);
					context->CSGetShaderResources(0, 1, &oldSRV);
					context->CSGetUnorderedAccessViews(0, kUAVs, oldUAVs);
					context->CSGetConstantBuffers(0, 1, &oldCB);

					// This capture's ring slice, and the older captures of a still camera merged into its level 0.
					const auto& settings = Settings::Get();
					const auto  maxFrames = std::min<std::uint32_t>(settings.hiZTemporalFrames, kRingSize);
					float       mergeMove = 0.0f;
					const auto  mergeCount = maxFrames > 1 ? MergeChain(a_frame, a_camera, maxFrames, mergeMove) : 0u;
					auto&       entry = g_ring[g_ringCur];
					entry.frame = a_frame;
					std::copy_n(a_camera.eye, 3, entry.eye);
					std::memcpy(entry.rotate, a_camera.rotate, sizeof(entry.rotate));
					entry.valid = true;
					const auto captureIndex = ++g_captureIndex;
					// The block-change map compares linear depths against references of the same projection; without
					// one (or after a change) every block counts as changed and takes this depth as its reference.
					const bool linear = a_camera.viewSpace && a_camera.depthB < 0.0f;
					const bool reset = !g_refValid || g_refDepthA != a_camera.depthA || g_refDepthB != a_camera.depthB;

					Build::Params params{};
					params.srcSize[0] = g_gpu.srcWidth;
					params.srcSize[1] = g_gpu.srcHeight;
					params.dstSize[0] = g_gpu.dstWidth;
					params.dstSize[1] = g_gpu.dstHeight;
					params.factor = std::max(1u, settings.hiZDownsample);
					params.nearReject = settings.worldDepthMin;
					params.depthMin = settings.worldDepthMin;
					params.invRange = 1.0f / std::max(settings.worldDepthMax - settings.worldDepthMin, 1.0e-6f);
					params.depthA = a_camera.depthA;
					params.depthB = a_camera.depthB;
					params.captureIndex = captureIndex;
					params.mergeCount = mergeCount;
					params.ringCur = g_ringCur;
					params.ringSize = kRingSize;
					params.flags = (reset ? Build::kFlagReset : 0u) | (linear ? Build::kFlagLinear : 0u);
					params.levels = g_gpu.layout.levels;
					params.blocksW = g_blocksW;
					params.nearOffset = g_gpu.output.nearOffset;
					params.changedOffset = g_gpu.output.changedOffset;
					params.drawnOffset = g_gpu.output.drawnOffset;
					for (std::uint32_t level = 0; level < g_gpu.layout.levels; ++level) {
						params.dims[level][0] = g_gpu.layout.width[level];
						params.dims[level][1] = g_gpu.layout.height[level];
						params.dims[level][2] = g_gpu.layout.offset[level];
					}
					context->UpdateSubresource(g_gpu.params.Get(), 0, nullptr, &params, 0, 0);

					// A bound depth target can't be read by a shader, but it can be copied.
					context->CopyResource(g_gpu.depthCopy.Get(), depthTex);

					ID3D11UnorderedAccessView* uavs[kUAVs]{};
					for (std::uint32_t i = 0; i < kUAVs; ++i) {
						uavs[i] = g_gpu.uavs[i].Get();
					}
					const auto cb = g_gpu.params.Get();
					const auto srv = g_gpu.depthCopySRV.Get();
					context->CSSetShader(g_gpu.build.Get(), nullptr, 0);
					context->CSSetShaderResources(0, 1, &srv);
					context->CSSetUnorderedAccessViews(0, kUAVs, uavs, nullptr);
					context->CSSetConstantBuffers(0, 1, &cb);
					context->Dispatch(g_blocksW, g_blocksH, 1);
					if (g_gpu.layout.levels > Build::kGroupLevels) {
						context->CSSetShader(g_gpu.tail.Get(), nullptr, 0);
						context->Dispatch(1, 1, 1);
					}

					ID3D11ShaderResourceView*  nullSRV = nullptr;
					ID3D11UnorderedAccessView* nullUAVs[kUAVs]{};
					context->CSSetShaderResources(0, 1, &nullSRV);
					context->CSSetUnorderedAccessViews(0, kUAVs, nullUAVs, nullptr);
					context->CopyResource(slot->staging.Get(), g_gpu.out.Get());

					UINT keepCounts[kUAVs];
					std::fill_n(keepCounts, kUAVs, static_cast<UINT>(-1));
					context->CSSetShader(oldShader, oldInstances, oldInstanceCount);
					context->CSSetShaderResources(0, 1, &oldSRV);
					context->CSSetUnorderedAccessViews(0, kUAVs, oldUAVs, keepCounts);
					context->CSSetConstantBuffers(0, 1, &oldCB);

					g_ringCur = (g_ringCur + 1) % kRingSize;
					g_refValid = linear;
					if (linear) {
						g_refDepthA = a_camera.depthA;
						g_refDepthB = a_camera.depthB;
					}
					slot->frame = a_frame;
					slot->camera = a_camera;
					slot->captureIndex = captureIndex;
					slot->mergedFrames = 1 + mergeCount;
					slot->mergeMove = mergeMove;
					slot->state = Slot::State::kPending;
					++g_counters.captures;
					ok = true;
				}
			}
		}

		for (UINT i = 0; i < oldInstanceCount && i < std::size(oldInstances); ++i) {
			if (oldInstances[i]) {
				oldInstances[i]->Release();
			}
		}
		for (auto* uav : oldUAVs) {
			if (uav) {
				uav->Release();
			}
		}
		for (IUnknown* object : std::initializer_list<IUnknown*>{ dsv, oldShader, oldSRV, oldCB }) {
			if (object) {
				object->Release();
			}
		}
		return ok;
	}

	void Poll()
	{
		const auto context = Util::GetContext();
		if (!context || g_gpu.failed) {
			return;
		}

		// Oldest first; once one is still in flight, newer ones are too. Only the newest finished one is
		// published (after a hitch several finish at once; older ones would be superseded unused).
		std::array<Slot*, kSlots> order{};
		std::size_t               count = 0;
		for (auto& slot : g_gpu.slots) {
			if (slot.state == Slot::State::kPending) {
				order[count++] = &slot;
			}
		}
		std::sort(order.begin(), order.begin() + count, [](const Slot* a, const Slot* b) { return a->frame < b->frame; });

		Slot* newest = nullptr;
		for (std::size_t i = 0; i < count; ++i) {
			D3D11_MAPPED_SUBRESOURCE mapped{};
			const auto               hr = context->Map(order[i]->staging.Get(), 0, D3D11_MAP_READ, D3D11_MAP_FLAG_DO_NOT_WAIT, &mapped);
			if (hr == DXGI_ERROR_WAS_STILL_DRAWING) {
				break;
			}
			if (FAILED(hr)) {
				order[i]->state = Slot::State::kFree;
				continue;
			}
			if (newest) {  // superseded before use
				context->Unmap(newest->staging.Get(), 0);
				newest->state = Slot::State::kFree;
				newest->mapped = nullptr;
			}
			newest = order[i];
			newest->mapped = static_cast<const std::uint8_t*>(mapped.pData);
			newest->state = Slot::State::kMapped;
		}
		if (!newest) {
			return;
		}

		const auto current = g_published.load();
		const auto target = (current + 1 + static_cast<int>(kSnapshots)) % static_cast<int>(kSnapshots);
		auto&      snapshot = g_snapshots[static_cast<std::size_t>(target)];
		ReleaseSlot(context, snapshot);  // published kSnapshots readbacks ago: nothing reads it any more
		snapshot.frame = newest->frame;
		snapshot.camera = newest->camera;
		snapshot.mergedFrames = newest->mergedFrames;
		snapshot.mergeMove = newest->mergeMove;
		snapshot.readbackIndex = newest->captureIndex;
		snapshot.levels = g_gpu.layout.levels;
		snapshot.width = g_gpu.layout.width;
		snapshot.height = g_gpu.layout.height;
		snapshot.offset = g_gpu.layout.offset;
		snapshot.texels = reinterpret_cast<const float*>(newest->mapped);
		snapshot.nearest = reinterpret_cast<const float*>(newest->mapped + g_gpu.output.nearOffset);
		snapshot.drawn = reinterpret_cast<const float*>(newest->mapped + g_gpu.output.drawnOffset);
		snapshot.slot = static_cast<std::int32_t>(newest - g_gpu.slots.data());

		// The change map: the GPU's per-block capture index of the last change, including changes captures that were
		// never published made (the map only ever moves forward).
		const auto* changed = reinterpret_cast<const std::uint32_t*>(newest->mapped + g_gpu.output.changedOffset);
		// (what the newest capture holds at each changed block: level 3 has one texel per block)
		const bool classify = snapshot.levels > 3 && snapshot.width[3] == g_blocksW && snapshot.height[3] == g_blocksH;
		for (std::size_t i = 0; i < g_blockChangedAt.size(); ++i) {
			if (changed[i] != g_blockChangedAt[i]) {
				g_blockChangedAt[i] = changed[i];
				++g_blocksChanged;
				if (classify) {
					const auto  x = static_cast<std::uint32_t>(i % g_blocksW);
					const auto  y = static_cast<std::uint32_t>(i / g_blocksW);
					const auto  at = snapshot.offset[3] + y * snapshot.width[3] + x;
					const float farthest = snapshot.texels[at];
					const float nearest = snapshot.nearest[at];
					if (nearest <= 0.0f) {
						++g_blockChangeKinds.firstPerson;  // (first-person pixels count as nearest 0, farthest 1)
					} else if (farthest >= 0.99999f) {
						++g_blockChangeKinds.farPlane;  // (sky: cleared to the far plane)
					}
					if (y * 3 >= g_blocksH * 2) {
						++g_blockChangeKinds.lowerThird;
					}
				}
			}
		}

		if (snapshot.mergedFrames > 1) {
			++g_counters.merges;
			g_counters.mergedFrames += snapshot.mergedFrames - 1;
		}
		++g_counters.spans[std::min<std::uint32_t>(snapshot.mergedFrames, 8)];
		g_published.store(target, std::memory_order_release);
		++g_counters.readbacks;
	}

	BlockChanges Blocks() noexcept
	{
		return { g_blockChangedAt.empty() ? nullptr : g_blockChangedAt.data(), g_blocksW, g_blocksH };
	}

	std::uint32_t TakeBlocksChanged() noexcept
	{
		return std::exchange(g_blocksChanged, 0u);
	}

	BlockChangeKinds TakeBlockChangeKinds() noexcept
	{
		return std::exchange(g_blockChangeKinds, BlockChangeKinds{});
	}

	const Snapshot* Latest() noexcept
	{
		const auto index = g_published.load(std::memory_order_acquire);
		return index >= 0 ? &g_snapshots[static_cast<std::size_t>(index)] : nullptr;
	}

	void Reset()
	{
		g_published.store(-1);
		g_refValid = false;
		const auto context = Util::GetContext();
		for (auto& snapshot : g_snapshots) {
			ReleaseSlot(context, snapshot);
		}
		for (auto& slot : g_gpu.slots) {
			if (slot.state == Slot::State::kMapped && context && slot.staging) {
				context->Unmap(slot.staging.Get(), 0);
			}
			slot.state = Slot::State::kFree;
			slot.mapped = nullptr;
		}
		for (auto& entry : g_ring) {
			entry.valid = false;
		}
	}

	bool Failed() noexcept
	{
		return g_gpu.failed;
	}

	std::string Describe()
	{
		const auto  snapshot = Latest();
		std::string spans;
		for (std::size_t i = 1; i < g_counters.spans.size(); ++i) {
			if (g_counters.spans[i]) {
				spans += std::format(" {}{}:{}", i, i == 8 ? "+" : "", g_counters.spans[i]);
			}
		}
		g_counters.spans = {};
		return std::format(
			"hi-z {}x{} ({} levels, GPU-built) captures={} readbacks={} noFreeSlot={} latest=frame {} | temporal: {} readbacks merged with {} earlier captures in all; this interval's depths by frames spanned:{}",
			g_gpu.dstWidth, g_gpu.dstHeight, g_gpu.layout.levels, g_counters.captures, g_counters.readbacks, g_counters.noFreeSlot,
			snapshot ? snapshot->frame : 0, g_counters.merges, g_counters.mergedFrames, spans.empty() ? " none" : spans);
	}
}
