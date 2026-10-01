// Offline brute-force check of the GPU Hi-Z build (src/Core/HiZBuild.h, compiled as is): the two compute shaders run
// on a D3D11 device (hardware, else WARP) over synthetic depth frames, and their output buffer (far pyramid, near
// pyramid, block-change map) is compared with a CPU model of what HiZ.cpp computed at readback time until v1.24:
//  1. level 0 = per-texel min/max over factor x factor depth texels, first person (d < 0.01) as near 0 / far 1;
//  2. the temporal merge over the previous captures the CPU says to merge (min of nearest, max of farthest), through
//     a ring that wraps;
//  3. every pyramid level = min/max over 2x2 of the level below, odd sizes clamped; bit-exact;
//  4. the block-change map: a block is marked with the capture's index when a texel's linear farthest depth got farther,
//     or its nearest nearer, than its reference allows (reset and no-linear captures mark every block); a marked block's
//     reference is its new depth, an unmarked block's follows the nearest farthest / farthest nearest depth seen since
//     its mark (v1.27), and marks persist across captures. Blocks within float noise of the threshold may differ
//     between the GPU's and the CPU's arithmetic and are reported separately;
//  5. (v1.57) level 0's farthest drawn depth: the max over the texel's depth pixels where something was drawn (the far
//     plane and first person left out as 0), merged over the same captures by max; bit-exact.
// Sizes include odd ones (161x102: partial 8x8 blocks, odd mip sizes), one where the tail pass builds a single level,
// and one with fewer than 4 levels (no tail pass; the group's level-3 store must stay in bounds).
// Build: cl /std:c++latest /O2 /EHsc /I src tools\tests\hiz_gpu_test.cpp d3d11.lib d3dcompiler.lib   (VS x64 prompt)
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <random>
#include <string>
#include <vector>

#define NOMINMAX
#define WIN32_LEAN_AND_MEAN
#include <d3d11.h>
#include <d3dcompiler.h>
#include <wrl/client.h>

#include "Core/HiZ.h"  // kCacheDepthSlack / kCacheDepthTolerance (declarations only; nothing of HiZ.cpp is linked)
#include "Core/HiZBuild.h"

using namespace CBRO::Core::HiZ::Build;
using Microsoft::WRL::ComPtr;

static int g_failures = 0;
#define CHECK(cond, ...)                                     \
	do {                                                     \
		if (!(cond)) {                                       \
			if (g_failures++ < 30) std::printf(__VA_ARGS__); \
		}                                                    \
	} while (0)

// A standard-Z camera like FO4's (near 15, far 70000): ndc = depthA + depthB / z, buffer depth = 0.01 + 0.99 * ndc.
constexpr float kDepthA = 1.000214f;
constexpr float kDepthB = -15.0032f;
constexpr float kDepthMin = 0.01f;
constexpr float kInvRange = 1.0f / 0.99f;
constexpr float kScale = 1.0f + CBRO::Core::HiZ::kCacheDepthTolerance * 0.5f;
constexpr float kSlack = CBRO::Core::HiZ::kCacheDepthSlack * 0.5f;
constexpr float kInf = std::numeric_limits<float>::infinity();

static float BufferDepth(float a_z) { return std::clamp(kDepthMin + 0.99f * (kDepthA + kDepthB / a_z), kDepthMin, 0.99999f); }
static float LinearDepth(float a_d)
{
	const float ndc = (a_d - kDepthMin) * kInvRange;
	const bool  sky = a_d >= 0.999999f || ndc >= kDepthA;
	return sky ? kInf : kDepthB / (ndc - kDepthA);
}

// ---- synthetic depth frames --------------------------------------------------------------------------

struct Wall
{
	float x0, y0, x1, y1;  // fractions of the frame
	float z;
};

struct Scene
{
	std::uint32_t      w{ 0 };
	std::uint32_t      h{ 0 };
	std::vector<float> depth;
};

static std::mt19937 rng(20260929);
static float        U(float a_lo, float a_hi) { return std::uniform_real_distribution<float>(a_lo, a_hi)(rng); }

// Ground receding upward, sky in the top fifth, walls, a first-person block bottom right, per-texel noise on z.
static Scene MakeScene(std::uint32_t a_w, std::uint32_t a_h, const std::vector<Wall>& a_walls, float a_noise)
{
	Scene scene{ a_w, a_h, std::vector<float>(static_cast<std::size_t>(a_w) * a_h) };
	for (std::uint32_t y = 0; y < a_h; ++y) {
		for (std::uint32_t x = 0; x < a_w; ++x) {
			const float fx = (x + 0.5f) / a_w;
			const float fy = (y + 0.5f) / a_h;
			float       z = 40.0f + 4000.0f * (1.0f - fy) * (1.0f - fy);  // ground: near at the bottom, far at the top
			bool        sky = fy < 0.2f;
			for (const auto& wall : a_walls) {
				if (fx >= wall.x0 && fx < wall.x1 && fy >= wall.y0 && fy < wall.y1) {
					z = std::min(z, wall.z);
					sky = false;
				}
			}
			float d;
			if (fx > 0.8f && fy > 0.85f) {
				d = 0.005f;  // first-person weapon
			} else if (sky) {
				d = 1.0f;
			} else {
				d = BufferDepth(z * (1.0f + U(-a_noise, a_noise)));
			}
			scene.depth[static_cast<std::size_t>(y) * a_w + x] = d;
		}
	}
	return scene;
}

// ---- the CPU model (HiZ.cpp up to v1.24, at readback time) --------------------------------------------

struct Model
{
	std::uint32_t                                srcW, srcH, factor, dstW, dstH, blocksW, blocksH;
	Layout                                       layout;
	std::array<std::vector<float>, kRingSize>    ringNear, ringFar, ringDrawn;
	std::vector<float>                           refFar, refNear;
	std::vector<std::uint32_t>                   changedAt;
	std::vector<float>                           farPyramid, nearPyramid;
	std::vector<float>                           drawn;  // level 0's farthest drawn depth (v1.57), merged
	std::vector<float>                           margin;  // per block: how close the change decision was (for tolerance)
	std::vector<bool>                            marked;  // per block: marked this capture
	std::vector<float>                           refFarBefore, refNearBefore;  // the references before this capture (to adopt a borderline GPU decision)
	std::uint32_t                                lastCapture{ 0 };

	Model(std::uint32_t a_srcW, std::uint32_t a_srcH, std::uint32_t a_factor) :
		srcW(a_srcW), srcH(a_srcH), factor(a_factor)
	{
		dstW = (srcW + factor - 1) / factor;
		dstH = (srcH + factor - 1) / factor;
		blocksW = (dstW + 7) / 8;
		blocksH = (dstH + 7) / 8;
		ComputeLayout(layout, dstW, dstH);
		const std::size_t level0 = static_cast<std::size_t>(dstW) * dstH;
		for (auto& r : ringNear) r.assign(level0, 1.0f);
		for (auto& r : ringFar) r.assign(level0, 0.0f);
		for (auto& r : ringDrawn) r.assign(level0, 0.0f);
		drawn.assign(level0, 0.0f);
		refFar.assign(level0, 0.0f);
		refNear.assign(level0, 0.0f);
		changedAt.assign(static_cast<std::size_t>(blocksW) * blocksH, 0);
		farPyramid.assign(layout.total, 0.0f);
		nearPyramid.assign(layout.total, 0.0f);
		margin.assign(changedAt.size(), kInf);
		marked.assign(changedAt.size(), false);
	}

	void Capture(const Scene& a_scene, std::uint32_t a_captureIndex, std::uint32_t a_mergeCount, std::uint32_t a_ringCur, std::uint32_t a_flags)
	{
		// 1. reduce
		auto& near0 = ringNear[a_ringCur];
		auto& far0 = ringFar[a_ringCur];
		auto& drawn0 = ringDrawn[a_ringCur];
		for (std::uint32_t y = 0; y < dstH; ++y) {
			for (std::uint32_t x = 0; x < dstW; ++x) {
				float farthest = 0.0f, nearest = 1.0f, drawnDepth = 0.0f;
				for (std::uint32_t dy = 0; dy < factor; ++dy) {
					for (std::uint32_t dx = 0; dx < factor; ++dx) {
						const auto px = std::min(x * factor + dx, srcW - 1);
						const auto py = std::min(y * factor + dy, srcH - 1);
						const float d = a_scene.depth[static_cast<std::size_t>(py) * srcW + px];
						const bool  firstPerson = d < kDepthMin;
						farthest = std::max(farthest, firstPerson ? 1.0f : d);
						nearest = std::min(nearest, firstPerson ? 0.0f : d);
						drawnDepth = std::max(drawnDepth, (firstPerson || d >= 0.999999f) ? 0.0f : d);  // (nothing drawn and the sky left out)
					}
				}
				near0[static_cast<std::size_t>(y) * dstW + x] = nearest;
				far0[static_cast<std::size_t>(y) * dstW + x] = farthest;
				drawn0[static_cast<std::size_t>(y) * dstW + x] = drawnDepth;
			}
		}
		// 2. merge
		const std::size_t level0 = static_cast<std::size_t>(dstW) * dstH;
		std::copy_n(far0.begin(), level0, farPyramid.begin());
		std::copy_n(near0.begin(), level0, nearPyramid.begin());
		std::copy_n(drawn0.begin(), level0, drawn.begin());
		for (std::uint32_t k = 1; k <= a_mergeCount; ++k) {
			const auto slice = (a_ringCur + kRingSize - k) % kRingSize;
			for (std::size_t i = 0; i < level0; ++i) {
				farPyramid[i] = std::max(farPyramid[i], ringFar[slice][i]);
				nearPyramid[i] = std::min(nearPyramid[i], ringNear[slice][i]);
				drawn[i] = std::max(drawn[i], ringDrawn[slice][i]);
			}
		}
		// 3. mips (v1.24 BuildMips)
		for (std::uint32_t level = 1; level < layout.levels; ++level) {
			const auto srcWl = layout.width[level - 1];
			const auto srcHl = layout.height[level - 1];
			const auto dstWl = layout.width[level];
			const auto dstHl = layout.height[level];
			for (int which = 0; which < 2; ++which) {
				auto&        pyramid = which == 0 ? farPyramid : nearPyramid;
				const float* src = pyramid.data() + layout.offset[level - 1];
				float*       dst = pyramid.data() + layout.offset[level];
				for (std::uint32_t y = 0; y < dstHl; ++y) {
					const float* rowA = src + static_cast<std::size_t>(std::min(y * 2, srcHl - 1)) * srcWl;
					const float* rowB = src + static_cast<std::size_t>(std::min(y * 2 + 1, srcHl - 1)) * srcWl;
					for (std::uint32_t x = 0; x < dstWl; ++x) {
						const auto xa = std::min(x * 2, srcWl - 1);
						const auto xb = std::min(x * 2 + 1, srcWl - 1);
						dst[static_cast<std::size_t>(y) * dstWl + x] = which == 0 ?
						                                                   std::max(std::max(rowA[xa], rowA[xb]), std::max(rowB[xa], rowB[xb])) :
						                                                   std::min(std::min(rowA[xa], rowA[xb]), std::min(rowB[xa], rowB[xb]));
					}
				}
			}
		}
		// 4. block changes (v1.24 MarkChangedBlocks, with the reference on the GPU's side of the fence)
		refFarBefore = refFar;
		refNearBefore = refNear;
		lastCapture = a_captureIndex;
		const bool reset = (a_flags & kFlagReset) != 0;
		const bool linear = (a_flags & kFlagLinear) != 0;
		for (std::uint32_t by = 0; by < blocksH; ++by) {
			for (std::uint32_t bx = 0; bx < blocksW; ++bx) {
				const auto b = static_cast<std::size_t>(by) * blocksW + bx;
				bool       changed = reset || !linear;
				float      closest = kInf;
				if (!changed) {
					for (std::uint32_t y = by * 8; y < std::min(by * 8 + 8, dstH); ++y) {
						for (std::uint32_t x = bx * 8; x < std::min(bx * 8 + 8, dstW); ++x) {
							const auto  i = static_cast<std::size_t>(y) * dstW + x;
							const float farNew = LinearDepth(farPyramid[i]);
							const float nearNew = LinearDepth(nearPyramid[i]);
							const float farLimit = refFar[i] * kScale + kSlack;
							const float nearLimit = nearNew * kScale + kSlack;
							changed = changed || farNew > farLimit || nearLimit < refNear[i];
							const float m1 = std::abs(farNew - farLimit);
							const float m2 = std::abs(nearLimit - refNear[i]);
							if (!std::isnan(m1)) closest = std::min(closest, m1 / std::max(1.0f, std::abs(farLimit)));
							if (!std::isnan(m2)) closest = std::min(closest, m2 / std::max(1.0f, std::abs(refNear[i])));
						}
					}
				}
				margin[b] = closest;
				marked[b] = changed;
				if (changed) {
					changedAt[b] = a_captureIndex;
				}
				if (linear) {
					// A marked block's reference is its new depth; an unmarked block's follows the nearest farthest depth
					// (and the farthest nearest depth) seen since its mark (v1.27).
					for (std::uint32_t y = by * 8; y < std::min(by * 8 + 8, dstH); ++y) {
						for (std::uint32_t x = bx * 8; x < std::min(bx * 8 + 8, dstW); ++x) {
							const auto  i = static_cast<std::size_t>(y) * dstW + x;
							const float farNew = LinearDepth(farPyramid[i]);
							const float nearNew = LinearDepth(nearPyramid[i]);
							refFar[i] = changed ? farNew : std::min(refFar[i], farNew);
							refNear[i] = changed ? nearNew : std::max(refNear[i], nearNew);
						}
					}
				}
			}
		}
	}
};

// ---- the GPU under test --------------------------------------------------------------------------------

struct Gpu
{
	ComPtr<ID3D11Device>              device;
	ComPtr<ID3D11DeviceContext>       context;
	ComPtr<ID3D11ComputeShader>       build, tail;
	ComPtr<ID3D11Buffer>              params, out, staging;
	ComPtr<ID3D11Texture2D>           depth, ringNear, ringFar, refFar, refNear, ringDrawn;
	ComPtr<ID3D11ShaderResourceView>  depthSRV;
	ComPtr<ID3D11UnorderedAccessView> uavs[kUAVs];
	OutputLayout                      output;
	std::string                       driver;

	bool CreateDevice()
	{
		const D3D_FEATURE_LEVEL levels[]{ D3D_FEATURE_LEVEL_11_0 };
		for (const auto type : { D3D_DRIVER_TYPE_HARDWARE, D3D_DRIVER_TYPE_WARP }) {
			D3D_FEATURE_LEVEL got{};
			const auto        hr = D3D11CreateDevice(nullptr, type, nullptr, 0, levels, 1, D3D11_SDK_VERSION, device.GetAddressOf(), &got, context.GetAddressOf());
			if (SUCCEEDED(hr)) {
				driver = type == D3D_DRIVER_TYPE_HARDWARE ? "hardware" : "WARP";
				return true;
			}
		}
		std::printf("no D3D11 device\n");
		return false;
	}

	bool Compile(const char* a_entry, ComPtr<ID3D11ComputeShader>& a_out)
	{
		const auto             scale = std::to_string(kScale);
		const auto             slack = std::to_string(kSlack);
		const D3D_SHADER_MACRO macros[]{ { "CBRO_SCALE", scale.c_str() }, { "CBRO_SLACK", slack.c_str() }, { nullptr, nullptr } };
		ComPtr<ID3DBlob>       code, errors;
		auto                   hr = D3DCompile(kSource, sizeof(kSource) - 1, "CBRO_HiZ", macros, nullptr, a_entry, "cs_5_0", D3DCOMPILE_OPTIMIZATION_LEVEL3, 0, code.GetAddressOf(), errors.GetAddressOf());
		if (FAILED(hr)) {
			std::printf("shader %s: %s\n", a_entry, errors ? static_cast<const char*>(errors->GetBufferPointer()) : "?");
			return false;
		}
		hr = device->CreateComputeShader(code->GetBufferPointer(), code->GetBufferSize(), nullptr, a_out.GetAddressOf());
		return SUCCEEDED(hr);
	}

	bool Init()
	{
		if (!CreateDevice() || !Compile("build", build) || !Compile("tail", tail)) {
			return false;
		}
		// The macro values must be what HiZ.cpp compiles with, or the test compares against the wrong tolerance.
		if (std::stof(std::to_string(kScale)) != kScale) {
			std::printf("warning: CBRO_SCALE rounds in the macro (%s)\n", std::to_string(kScale).c_str());
		}
		D3D11_BUFFER_DESC desc{};
		desc.ByteWidth = sizeof(Params);
		desc.Usage = D3D11_USAGE_DEFAULT;
		desc.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
		return SUCCEEDED(device->CreateBuffer(&desc, nullptr, params.GetAddressOf()));
	}

	bool Texture(std::uint32_t a_w, std::uint32_t a_h, std::uint32_t a_slices, ComPtr<ID3D11Texture2D>& a_tex, ComPtr<ID3D11UnorderedAccessView>* a_uav)
	{
		D3D11_TEXTURE2D_DESC desc{};
		desc.Width = a_w;
		desc.Height = a_h;
		desc.MipLevels = 1;
		desc.ArraySize = a_slices;
		desc.Format = DXGI_FORMAT_R32_FLOAT;
		desc.SampleDesc.Count = 1;
		desc.Usage = D3D11_USAGE_DEFAULT;
		desc.BindFlags = a_uav ? D3D11_BIND_UNORDERED_ACCESS : D3D11_BIND_SHADER_RESOURCE;
		if (FAILED(device->CreateTexture2D(&desc, nullptr, a_tex.ReleaseAndGetAddressOf()))) {
			return false;
		}
		return !a_uav || SUCCEEDED(device->CreateUnorderedAccessView(a_tex.Get(), nullptr, a_uav->ReleaseAndGetAddressOf()));
	}

	bool Resize(const Model& a_model)
	{
		if (!Texture(a_model.srcW, a_model.srcH, 1, depth, nullptr) ||
			FAILED(device->CreateShaderResourceView(depth.Get(), nullptr, depthSRV.ReleaseAndGetAddressOf())) ||
			!Texture(a_model.dstW, a_model.dstH, kRingSize, ringNear, &uavs[0]) ||
			!Texture(a_model.dstW, a_model.dstH, kRingSize, ringFar, &uavs[1]) ||
			!Texture(a_model.dstW, a_model.dstH, 1, refFar, &uavs[2]) ||
			!Texture(a_model.dstW, a_model.dstH, 1, refNear, &uavs[3]) ||
			!Texture(a_model.dstW, a_model.dstH, kRingSize, ringDrawn, &uavs[5])) {
			return false;
		}
		output = ComputeOutput(a_model.layout, a_model.blocksW * a_model.blocksH);
		D3D11_BUFFER_DESC desc{};
		desc.ByteWidth = output.bytes;
		desc.Usage = D3D11_USAGE_DEFAULT;
		desc.BindFlags = D3D11_BIND_UNORDERED_ACCESS;
		desc.MiscFlags = D3D11_RESOURCE_MISC_BUFFER_ALLOW_RAW_VIEWS;
		if (FAILED(device->CreateBuffer(&desc, nullptr, out.ReleaseAndGetAddressOf()))) {
			return false;
		}
		D3D11_UNORDERED_ACCESS_VIEW_DESC uavDesc{};
		uavDesc.Format = DXGI_FORMAT_R32_TYPELESS;
		uavDesc.ViewDimension = D3D11_UAV_DIMENSION_BUFFER;
		uavDesc.Buffer.NumElements = output.bytes / 4;
		uavDesc.Buffer.Flags = D3D11_BUFFER_UAV_FLAG_RAW;
		if (FAILED(device->CreateUnorderedAccessView(out.Get(), &uavDesc, uavs[4].ReleaseAndGetAddressOf()))) {
			return false;
		}
		const UINT zeros[4]{};
		context->ClearUnorderedAccessViewUint(uavs[4].Get(), zeros);
		desc.Usage = D3D11_USAGE_STAGING;
		desc.BindFlags = 0;
		desc.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
		desc.MiscFlags = 0;
		return SUCCEEDED(device->CreateBuffer(&desc, nullptr, staging.ReleaseAndGetAddressOf()));
	}

	// Runs both dispatches like HiZ::Capture and reads the whole output buffer back (blocking: this is a test).
	std::vector<std::uint8_t> Run(const Model& a_model, const Scene& a_scene, std::uint32_t a_captureIndex, std::uint32_t a_mergeCount, std::uint32_t a_ringCur, std::uint32_t a_flags)
	{
		context->UpdateSubresource(depth.Get(), 0, nullptr, a_scene.depth.data(), a_scene.w * 4, 0);
		Params p{};
		p.srcSize[0] = a_model.srcW;
		p.srcSize[1] = a_model.srcH;
		p.dstSize[0] = a_model.dstW;
		p.dstSize[1] = a_model.dstH;
		p.factor = a_model.factor;
		p.nearReject = kDepthMin;
		p.depthMin = kDepthMin;
		p.invRange = kInvRange;
		p.depthA = kDepthA;
		p.depthB = kDepthB;
		p.captureIndex = a_captureIndex;
		p.mergeCount = a_mergeCount;
		p.ringCur = a_ringCur;
		p.ringSize = kRingSize;
		p.flags = a_flags;
		p.levels = a_model.layout.levels;
		p.blocksW = a_model.blocksW;
		p.nearOffset = output.nearOffset;
		p.changedOffset = output.changedOffset;
		p.drawnOffset = output.drawnOffset;
		for (std::uint32_t level = 0; level < a_model.layout.levels; ++level) {
			p.dims[level][0] = a_model.layout.width[level];
			p.dims[level][1] = a_model.layout.height[level];
			p.dims[level][2] = a_model.layout.offset[level];
		}
		context->UpdateSubresource(params.Get(), 0, nullptr, &p, 0, 0);

		ID3D11UnorderedAccessView* raw[kUAVs]{};
		for (std::uint32_t i = 0; i < kUAVs; ++i) raw[i] = uavs[i].Get();
		auto* srv = depthSRV.Get();
		auto* cb = params.Get();
		context->CSSetShader(build.Get(), nullptr, 0);
		context->CSSetShaderResources(0, 1, &srv);
		context->CSSetUnorderedAccessViews(0, kUAVs, raw, nullptr);
		context->CSSetConstantBuffers(0, 1, &cb);
		context->Dispatch(a_model.blocksW, a_model.blocksH, 1);
		if (a_model.layout.levels > kGroupLevels) {
			context->CSSetShader(tail.Get(), nullptr, 0);
			context->Dispatch(1, 1, 1);
		}
		ID3D11UnorderedAccessView* nulls[kUAVs]{};
		context->CSSetUnorderedAccessViews(0, kUAVs, nulls, nullptr);
		context->CopyResource(staging.Get(), out.Get());

		std::vector<std::uint8_t>  bytes(output.bytes);
		D3D11_MAPPED_SUBRESOURCE   mapped{};
		if (SUCCEEDED(context->Map(staging.Get(), 0, D3D11_MAP_READ, 0, &mapped))) {
			std::memcpy(bytes.data(), mapped.pData, output.bytes);
			context->Unmap(staging.Get(), 0);
		} else {
			std::printf("Map failed\n");
			++g_failures;
		}
		return bytes;
	}
};

// ---- the run -----------------------------------------------------------------------------------------------

struct Totals
{
	std::uint64_t texels{ 0 }, texelMismatches{ 0 }, blocks{ 0 }, blocksMarked{ 0 }, blockMismatches{ 0 }, borderline{ 0 };
};

// A block decided within float noise of the threshold may go either way; the model then adopts the GPU's decision (mark
// and reference), so the tolerated split does not make every later decision for that block differ.
static void AdoptBlock(Model& a_model, std::size_t a_b, bool a_gpuMarked)
{
	const auto bx = static_cast<std::uint32_t>(a_b % a_model.blocksW);
	const auto by = static_cast<std::uint32_t>(a_b / a_model.blocksW);
	for (std::uint32_t y = by * 8; y < std::min(by * 8 + 8, a_model.dstH); ++y) {
		for (std::uint32_t x = bx * 8; x < std::min(bx * 8 + 8, a_model.dstW); ++x) {
			const auto  i = static_cast<std::size_t>(y) * a_model.dstW + x;
			const float farNew = LinearDepth(a_model.farPyramid[i]);
			const float nearNew = LinearDepth(a_model.nearPyramid[i]);
			a_model.refFar[i] = a_gpuMarked ? farNew : std::min(a_model.refFarBefore[i], farNew);
			a_model.refNear[i] = a_gpuMarked ? nearNew : std::max(a_model.refNearBefore[i], nearNew);
		}
	}
}

static void Compare(Model& a_model, const std::vector<std::uint8_t>& a_bytes, const OutputLayout& a_out, const char* a_what, Totals& a_totals)
{
	// (`near` and `far` are Windows macros)
	const auto* farOut = reinterpret_cast<const float*>(a_bytes.data());
	const auto* nearOut = reinterpret_cast<const float*>(a_bytes.data() + a_out.nearOffset);
	const auto* changed = reinterpret_cast<const std::uint32_t*>(a_bytes.data() + a_out.changedOffset);
	const auto* drawnOut = reinterpret_cast<const float*>(a_bytes.data() + a_out.drawnOffset);
	for (std::uint32_t y = 0; y < a_model.dstH; ++y) {
		for (std::uint32_t x = 0; x < a_model.dstW; ++x) {
			const auto i = static_cast<std::size_t>(y) * a_model.dstW + x;
			++a_totals.texels;
			const bool ok = drawnOut[i] == a_model.drawn[i];
			if (!ok) ++a_totals.texelMismatches;
			CHECK(ok, "%s: level-0 drawn texel (%u,%u): gpu %.7f, cpu %.7f\n", a_what, x, y, drawnOut[i], a_model.drawn[i]);
		}
	}
	for (std::uint32_t level = 0; level < a_model.layout.levels; ++level) {
		const auto w = a_model.layout.width[level];
		const auto h = a_model.layout.height[level];
		const auto off = a_model.layout.offset[level];
		for (std::uint32_t y = 0; y < h; ++y) {
			for (std::uint32_t x = 0; x < w; ++x) {
				const auto i = off + static_cast<std::size_t>(y) * w + x;
				++a_totals.texels;
				const bool ok = farOut[i] == a_model.farPyramid[i] && nearOut[i] == a_model.nearPyramid[i];
				if (!ok) ++a_totals.texelMismatches;
				CHECK(ok, "%s: level %u texel (%u,%u): gpu far %.7f near %.7f, cpu far %.7f near %.7f\n", a_what, level, x, y, farOut[i], nearOut[i], a_model.farPyramid[i], a_model.nearPyramid[i]);
			}
		}
	}
	for (std::size_t b = 0; b < a_model.changedAt.size(); ++b) {
		++a_totals.blocks;
		if (a_model.marked[b]) ++a_totals.blocksMarked;
		if (changed[b] == a_model.changedAt[b]) {
			continue;
		}
		if (a_model.margin[b] < 1.0e-4f) {  // decided within float noise of the threshold: arithmetic may differ
			++a_totals.borderline;
			AdoptBlock(a_model, b, changed[b] == a_model.lastCapture);
			a_model.changedAt[b] = changed[b];
			continue;
		}
		++a_totals.blockMismatches;
		CHECK(false, "%s: block %zu (%zu,%zu): gpu changedAt %u, cpu %u (margin %.3g)\n", a_what, b, b % a_model.blocksW, b / a_model.blocksW, changed[b], a_model.changedAt[b], a_model.margin[b]);
	}
}

int main()
{
	Gpu gpu;
	if (!gpu.Init()) {
		return 1;
	}
	std::printf("device: %s\n", gpu.driver.c_str());

	struct Config
	{
		std::uint32_t w, h, factor;
	};
	const Config configs[]{ { 1280, 800, 4 }, { 321, 203, 2 }, { 100, 60, 8 }, { 25, 20, 8 }, { 1920, 1080, 4 }, { 640, 360, 1 } };
	const Wall   wallA{ 0.10f, 0.30f, 0.35f, 0.90f, 120.0f };
	const Wall   wallB{ 0.55f, 0.25f, 0.75f, 0.70f, 60.0f };
	const Wall   wallC{ 0.40f, 0.10f, 0.50f, 0.60f, 900.0f };  // in the sky band too

	Totals totals;
	for (const auto& config : configs) {
		Model model(config.w, config.h, config.factor);
		if (!gpu.Resize(model)) {
			std::printf("resize %ux%u failed\n", config.w, config.h);
			return 1;
		}
		std::printf("depth %ux%u / %u -> hi-z %ux%u, %u levels, blocks %ux%u, output %u bytes%s\n",
			config.w, config.h, config.factor, model.dstW, model.dstH, model.layout.levels, model.blocksW, model.blocksH, gpu.output.bytes,
			model.layout.levels > kGroupLevels ? "" : " (no tail pass)");

		// A capture sequence the way Runtime + HiZ::Capture would drive it (ring wraps after 8).
		struct Step
		{
			const char*   what;
			Scene         scene;
			std::uint32_t mergeCount;
			std::uint32_t flags;
		};
		std::vector<Step> steps;
		steps.push_back({ "first (reset)", MakeScene(config.w, config.h, { wallA, wallB }, 0.002f), 0, kFlagReset | kFlagLinear });
		steps.push_back({ "jitter, merge 1", MakeScene(config.w, config.h, { wallA, wallB }, 0.002f), 1, kFlagLinear });
		steps.push_back({ "jitter, merge 2", MakeScene(config.w, config.h, { wallA, wallB }, 0.002f), 2, kFlagLinear });
		steps.push_back({ "wall B gone (farther), merge 3", MakeScene(config.w, config.h, { wallA }, 0.002f), 3, kFlagLinear });
		steps.push_back({ "wall C appears (nearer in the sky band), merge 0", MakeScene(config.w, config.h, { wallA, wallC }, 0.002f), 0, kFlagLinear });
		steps.push_back({ "same, merge 1", MakeScene(config.w, config.h, { wallA, wallC }, 0.002f), 1, kFlagLinear });
		steps.push_back({ "no linear form (all changed, references kept)", MakeScene(config.w, config.h, { wallA, wallC }, 0.002f), 0, 0 });
		steps.push_back({ "reset after no-linear", MakeScene(config.w, config.h, { wallA, wallC }, 0.002f), 0, kFlagReset | kFlagLinear });
		steps.push_back({ "jitter, merge 4 (ring wraps)", MakeScene(config.w, config.h, { wallA, wallC }, 0.002f), 4, kFlagLinear });
		steps.push_back({ "big jitter (some blocks move), merge 7", MakeScene(config.w, config.h, { wallA, wallC }, 0.02f), 7, kFlagLinear });
		steps.push_back({ "walls A+B back, merge 2", MakeScene(config.w, config.h, { wallA, wallB }, 0.002f), 2, kFlagLinear });

		std::uint32_t captureIndex = 100u * static_cast<std::uint32_t>(&config - configs);  // any monotonic sequence
		std::uint32_t ringCur = 0;
		for (const auto& step : steps) {
			++captureIndex;
			model.Capture(step.scene, captureIndex, step.mergeCount, ringCur, step.flags);
			const auto bytes = gpu.Run(model, step.scene, captureIndex, step.mergeCount, ringCur, step.flags);
			Totals     before = totals;
			Compare(model, bytes, gpu.output, step.what, totals);
			std::printf("  %-52s marked %5llu of %5llu blocks | mismatches: texels %llu, blocks %llu, borderline %llu\n", step.what,
				totals.blocksMarked - before.blocksMarked, totals.blocks - before.blocks,
				totals.texelMismatches - before.texelMismatches, totals.blockMismatches - before.blockMismatches, totals.borderline - before.borderline);
			ringCur = (ringCur + 1) % kRingSize;
		}
	}

	std::printf("\n%llu pyramid texels compared, %llu differ | %llu block decisions compared (%llu marked), %llu differ, %llu borderline tolerated\n",
		totals.texels, totals.texelMismatches, totals.blocks, totals.blocksMarked, totals.blockMismatches, totals.borderline);
	std::printf(g_failures ? "FAILED (%d)\n" : "OK\n", g_failures);
	return g_failures ? 1 : 0;
}
