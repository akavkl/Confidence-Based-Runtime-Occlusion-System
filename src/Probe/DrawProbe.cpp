#include "Probe/ProbeInternal.h"

#include "Util/D3D.h"
#include "Util/Hooking.h"

// Counts draw calls submitted by the game on the immediate context (per render stage) and
// records which depth-stencil views are bound while they happen. The context is whatever
// RendererData holds, i.e. ENB's proxy when ENB is active, so this counts game submissions.

namespace CBRO::Probe::Draw
{
	namespace
	{
		// ID3D11DeviceContext vtable indices (IUnknown 0-2, ID3D11DeviceChild 3-6, then d3d11.h order).
		constexpr std::size_t kDrawIndexed = 12;
		constexpr std::size_t kDraw = 13;
		constexpr std::size_t kDrawIndexedInstanced = 20;
		constexpr std::size_t kDrawInstanced = 21;
		constexpr std::size_t kOMSetRenderTargets = 33;
		constexpr std::size_t kOMSetRenderTargetsAndUAVs = 34;
		constexpr std::size_t kDrawAuto = 38;
		constexpr std::size_t kDrawIndexedInstancedIndirect = 39;
		constexpr std::size_t kDrawInstancedIndirect = 40;
		constexpr std::size_t kRSSetViewports = 44;

		// FO4 draws first-person geometry with viewport depth [0, 0.01] and the world with [0.01, 1].
		constexpr float kFirstPersonMaxDepth = 0.0101f;

		constexpr std::size_t kStages = static_cast<std::size_t>(Stage::kCount);

		std::atomic<ID3D11DeviceContext*> g_immediate{ nullptr };

		struct FrameCounters
		{
			std::array<std::atomic<std::uint64_t>, kStages> immediate{};
			std::array<std::atomic<std::uint64_t>, kStages> firstPerson{};  // immediate draws in the first-person depth range
			std::atomic<std::uint64_t>                      otherContext{ 0 };
			std::atomic<std::uint64_t>                      offRenderThread{ 0 };
		};
		FrameCounters              g_frame;
		std::atomic<std::uint64_t> g_immediateThisFrame{ 0 };

		// Render-thread-only aggregates, touched in EndFrame.
		struct Interval
		{
			std::uint64_t                      frames{ 0 };
			std::array<std::uint64_t, kStages> immediate{};
			std::array<std::uint64_t, kStages> firstPerson{};
			std::uint64_t                      maxFrameTotal{ 0 };
			std::uint64_t                      otherContext{ 0 };
		};
		Interval g_interval;

		struct DsvRecord
		{
			ID3D11DepthStencilView*                         dsv{ nullptr };
			std::array<std::atomic<std::uint64_t>, kStages> draws{};
			std::uint64_t                                   binds{ 0 };
			std::string                                     texture;
			std::string                                     view;
			std::string                                     match;
		};

		constexpr std::size_t          kMaxDsvs = 32;
		constexpr std::size_t          kNoDsv = ~std::size_t{ 0 };
		std::mutex                     g_dsvLock;
		std::array<DsvRecord, kMaxDsvs> g_dsvs{};
		std::size_t                    g_dsvCount{ 0 };
		std::atomic<std::size_t>       g_currentDsv{ kNoDsv };

		std::atomic<bool> g_firstPersonViewport{ false };
		std::atomic<bool> g_viewportSampled{ false };
		D3D11_VIEWPORT    g_prepassViewport{};
		UINT              g_prepassViewportCount{ 0 };

		std::string MatchDepthTarget(ID3D11DepthStencilView* a_dsv, ID3D11Resource* a_resource)
		{
			const auto data = RE::BSGraphics::RendererData::GetSingleton();
			if (!data) {
				return "?";
			}
			std::string result;
			for (std::size_t i = 0; i < std::size(data->depthStencilTargets); ++i) {
				const auto& target = data->depthStencilTargets[i];
				for (std::size_t k = 0; k < 4; ++k) {
					const auto view = reinterpret_cast<std::uintptr_t>(a_dsv);
					if (view == reinterpret_cast<std::uintptr_t>(target.dsView[k])) {
						result += std::format("depth[{}].dsView[{}] ", i, k);
					} else if (view == reinterpret_cast<std::uintptr_t>(target.dsViewReadOnlyDepth[k])) {
						result += std::format("depth[{}].dsViewReadOnlyDepth[{}] ", i, k);
					} else if (view == reinterpret_cast<std::uintptr_t>(target.dsViewReadOnlyStencil[k])) {
						result += std::format("depth[{}].dsViewReadOnlyStencil[{}] ", i, k);
					} else if (view == reinterpret_cast<std::uintptr_t>(target.dsViewReadOnlyDepthStencil[k])) {
						result += std::format("depth[{}].dsViewReadOnlyDepthStencil[{}] ", i, k);
					}
				}
				if (a_resource && reinterpret_cast<std::uintptr_t>(a_resource) == reinterpret_cast<std::uintptr_t>(target.texture)) {
					result += std::format("(texture of depth[{}]) ", i);
				}
			}
			return result.empty() ? "no RendererData match" : result;
		}

		void DescribeDsv(DsvRecord& a_record)
		{
			ID3D11Resource* resource = nullptr;
			a_record.dsv->GetResource(&resource);
			if (resource) {
				ID3D11Texture2D* texture = nullptr;
				if (SUCCEEDED(resource->QueryInterface(IID_PPV_ARGS(&texture))) && texture) {
					a_record.texture = Util::DescribeTexture(texture);
					texture->Release();
				}
			}

			D3D11_DEPTH_STENCIL_VIEW_DESC desc{};
			a_record.dsv->GetDesc(&desc);
			a_record.view = std::format(
				"fmt={} dim={} mip={} flags={}{}",
				Util::FormatName(desc.Format), static_cast<int>(desc.ViewDimension),
				desc.ViewDimension == D3D11_DSV_DIMENSION_TEXTURE2D ? desc.Texture2D.MipSlice : 0,
				(desc.Flags & D3D11_DSV_READ_ONLY_DEPTH) ? "RO-depth " : "",
				(desc.Flags & D3D11_DSV_READ_ONLY_STENCIL) ? "RO-stencil" : "");

			a_record.match = MatchDepthTarget(a_record.dsv, resource);
			if (resource) {
				resource->Release();
			}
		}

		void SetCurrentDsv(ID3D11DepthStencilView* a_dsv)
		{
			if (!a_dsv) {
				g_currentDsv.store(kNoDsv, std::memory_order_relaxed);
				return;
			}

			std::scoped_lock lock(g_dsvLock);
			for (std::size_t i = 0; i < g_dsvCount; ++i) {
				if (g_dsvs[i].dsv == a_dsv) {
					++g_dsvs[i].binds;
					g_currentDsv.store(i, std::memory_order_relaxed);
					return;
				}
			}
			if (g_dsvCount == kMaxDsvs) {
				g_currentDsv.store(kNoDsv, std::memory_order_relaxed);
				return;
			}

			auto& record = g_dsvs[g_dsvCount];
			record.dsv = a_dsv;
			record.binds = 1;
			DescribeDsv(record);
			g_currentDsv.store(g_dsvCount, std::memory_order_relaxed);
			++g_dsvCount;
		}

		void CountDraw(ID3D11DeviceContext* a_context)
		{
			if (a_context != g_immediate.load(std::memory_order_relaxed)) {
				g_frame.otherContext.fetch_add(1, std::memory_order_relaxed);
				return;
			}

			const auto stage = static_cast<std::size_t>(CurrentStage());
			g_frame.immediate[stage].fetch_add(1, std::memory_order_relaxed);
			g_immediateThisFrame.fetch_add(1, std::memory_order_relaxed);
			if (g_firstPersonViewport.load(std::memory_order_relaxed)) {
				g_frame.firstPerson[stage].fetch_add(1, std::memory_order_relaxed);
			}
			if (GetCurrentThreadId() != State().renderThreadId.load(std::memory_order_relaxed)) {
				g_frame.offRenderThread.fetch_add(1, std::memory_order_relaxed);
			}

			if (const auto dsv = g_currentDsv.load(std::memory_order_relaxed); dsv != kNoDsv) {
				g_dsvs[dsv].draws[stage].fetch_add(1, std::memory_order_relaxed);
			}

			if (stage == static_cast<std::size_t>(Stage::kPrePass) && !g_viewportSampled.exchange(true)) {
				g_prepassViewportCount = 1;
				a_context->RSGetViewports(&g_prepassViewportCount, &g_prepassViewport);
			}
		}

		// ---- vtable thunks ----------------------------------------------------------------

		using DrawIndexed_t = void(STDMETHODCALLTYPE*)(ID3D11DeviceContext*, UINT, UINT, INT);
		using Draw_t = void(STDMETHODCALLTYPE*)(ID3D11DeviceContext*, UINT, UINT);
		using DrawIndexedInstanced_t = void(STDMETHODCALLTYPE*)(ID3D11DeviceContext*, UINT, UINT, UINT, INT, UINT);
		using DrawInstanced_t = void(STDMETHODCALLTYPE*)(ID3D11DeviceContext*, UINT, UINT, UINT, UINT);
		using OMSetRenderTargets_t = void(STDMETHODCALLTYPE*)(ID3D11DeviceContext*, UINT, ID3D11RenderTargetView* const*, ID3D11DepthStencilView*);
		using OMSetRenderTargetsAndUAVs_t = void(STDMETHODCALLTYPE*)(ID3D11DeviceContext*, UINT, ID3D11RenderTargetView* const*, ID3D11DepthStencilView*, UINT, UINT, ID3D11UnorderedAccessView* const*, const UINT*);
		using DrawAuto_t = void(STDMETHODCALLTYPE*)(ID3D11DeviceContext*);
		using DrawIndirect_t = void(STDMETHODCALLTYPE*)(ID3D11DeviceContext*, ID3D11Buffer*, UINT);
		using RSSetViewports_t = void(STDMETHODCALLTYPE*)(ID3D11DeviceContext*, UINT, const D3D11_VIEWPORT*);

		DrawIndexed_t               g_drawIndexed{ nullptr };
		Draw_t                      g_draw{ nullptr };
		DrawIndexedInstanced_t      g_drawIndexedInstanced{ nullptr };
		DrawInstanced_t             g_drawInstanced{ nullptr };
		OMSetRenderTargets_t        g_omSetRenderTargets{ nullptr };
		OMSetRenderTargetsAndUAVs_t g_omSetRenderTargetsAndUAVs{ nullptr };
		DrawAuto_t                  g_drawAuto{ nullptr };
		DrawIndirect_t              g_drawIndexedInstancedIndirect{ nullptr };
		DrawIndirect_t              g_drawInstancedIndirect{ nullptr };
		RSSetViewports_t            g_rsSetViewports{ nullptr };

		void STDMETHODCALLTYPE DrawIndexed(ID3D11DeviceContext* a_self, UINT a_indexCount, UINT a_startIndex, INT a_baseVertex)
		{
			CountDraw(a_self);
			g_drawIndexed(a_self, a_indexCount, a_startIndex, a_baseVertex);
		}

		void STDMETHODCALLTYPE DrawNonIndexed(ID3D11DeviceContext* a_self, UINT a_vertexCount, UINT a_startVertex)
		{
			CountDraw(a_self);
			g_draw(a_self, a_vertexCount, a_startVertex);
		}

		void STDMETHODCALLTYPE DrawIndexedInstanced(ID3D11DeviceContext* a_self, UINT a_indexCount, UINT a_instances, UINT a_startIndex, INT a_baseVertex, UINT a_startInstance)
		{
			CountDraw(a_self);
			g_drawIndexedInstanced(a_self, a_indexCount, a_instances, a_startIndex, a_baseVertex, a_startInstance);
		}

		void STDMETHODCALLTYPE DrawInstanced(ID3D11DeviceContext* a_self, UINT a_vertexCount, UINT a_instances, UINT a_startVertex, UINT a_startInstance)
		{
			CountDraw(a_self);
			g_drawInstanced(a_self, a_vertexCount, a_instances, a_startVertex, a_startInstance);
		}

		void STDMETHODCALLTYPE OMSetRenderTargets(ID3D11DeviceContext* a_self, UINT a_numViews, ID3D11RenderTargetView* const* a_rtvs, ID3D11DepthStencilView* a_dsv)
		{
			if (a_self == g_immediate.load(std::memory_order_relaxed)) {
				SetCurrentDsv(a_dsv);
			}
			g_omSetRenderTargets(a_self, a_numViews, a_rtvs, a_dsv);
		}

		void STDMETHODCALLTYPE OMSetRenderTargetsAndUAVs(
			ID3D11DeviceContext*              a_self,
			UINT                              a_numRTVs,
			ID3D11RenderTargetView* const*    a_rtvs,
			ID3D11DepthStencilView*           a_dsv,
			UINT                              a_uavStart,
			UINT                              a_numUAVs,
			ID3D11UnorderedAccessView* const* a_uavs,
			const UINT*                       a_initialCounts)
		{
			if (a_self == g_immediate.load(std::memory_order_relaxed) && a_numRTVs != D3D11_KEEP_RENDER_TARGETS_AND_DEPTH_STENCIL) {
				SetCurrentDsv(a_dsv);
			}
			g_omSetRenderTargetsAndUAVs(a_self, a_numRTVs, a_rtvs, a_dsv, a_uavStart, a_numUAVs, a_uavs, a_initialCounts);
		}

		void STDMETHODCALLTYPE DrawAuto(ID3D11DeviceContext* a_self)
		{
			CountDraw(a_self);
			g_drawAuto(a_self);
		}

		void STDMETHODCALLTYPE DrawIndexedInstancedIndirect(ID3D11DeviceContext* a_self, ID3D11Buffer* a_args, UINT a_offset)
		{
			CountDraw(a_self);
			g_drawIndexedInstancedIndirect(a_self, a_args, a_offset);
		}

		void STDMETHODCALLTYPE DrawInstancedIndirect(ID3D11DeviceContext* a_self, ID3D11Buffer* a_args, UINT a_offset)
		{
			CountDraw(a_self);
			g_drawInstancedIndirect(a_self, a_args, a_offset);
		}

		void STDMETHODCALLTYPE RSSetViewports(ID3D11DeviceContext* a_self, UINT a_count, const D3D11_VIEWPORT* a_viewports)
		{
			if (a_self == g_immediate.load(std::memory_order_relaxed) && a_count > 0 && a_viewports) {
				g_firstPersonViewport.store(a_viewports[0].MaxDepth <= kFirstPersonMaxDepth, std::memory_order_relaxed);
			}
			g_rsSetViewports(a_self, a_count, a_viewports);
		}

		template <class F>
		void Hook(std::uintptr_t a_vtable, std::size_t a_index, F a_thunk, F& a_original, std::string_view a_name)
		{
			a_original = reinterpret_cast<F>(Util::WriteVFunc(a_vtable, a_index, Util::FnAddr(a_thunk), a_name));
		}

		std::string ModulePath(std::uintptr_t a_address)
		{
			HMODULE module = nullptr;
			if (!GetModuleHandleExW(
					GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
					reinterpret_cast<LPCWSTR>(a_address), &module)) {
				return "?";
			}
			wchar_t    buf[MAX_PATH]{};
			const auto len = GetModuleFileNameW(module, buf, MAX_PATH);
			return std::filesystem::path(std::wstring_view(buf, len)).string();
		}

		std::string StageCounts(const std::array<std::uint64_t, kStages>& a_counts, double a_divisor = 1.0)
		{
			std::string result;
			for (std::size_t i = 0; i < kStages; ++i) {
				result += std::format(" {}={:.0f}", StageName(static_cast<Stage>(i)), static_cast<double>(a_counts[i]) / a_divisor);
			}
			return result;
		}
	}

	void Install()
	{
		const auto context = Util::GetContext();
		if (!context) {
			logger::warn("draw counter: no immediate context yet; disabled");
			return;
		}

		const auto vtable = *reinterpret_cast<const std::uintptr_t*>(context);
		logger::info("draw counter: immediate context {} vtable in {}", fmt_ptr(context), ModulePath(vtable));

		Hook(vtable, kDrawIndexed, &DrawIndexed, g_drawIndexed, "d3d:DrawIndexed");
		Hook(vtable, kDraw, &DrawNonIndexed, g_draw, "d3d:Draw");
		Hook(vtable, kDrawIndexedInstanced, &DrawIndexedInstanced, g_drawIndexedInstanced, "d3d:DrawIndexedInstanced");
		Hook(vtable, kDrawInstanced, &DrawInstanced, g_drawInstanced, "d3d:DrawInstanced");
		Hook(vtable, kOMSetRenderTargets, &OMSetRenderTargets, g_omSetRenderTargets, "d3d:OMSetRenderTargets");
		Hook(vtable, kOMSetRenderTargetsAndUAVs, &OMSetRenderTargetsAndUAVs, g_omSetRenderTargetsAndUAVs, "d3d:OMSetRenderTargetsAndUnorderedAccessViews");
		Hook(vtable, kDrawAuto, &DrawAuto, g_drawAuto, "d3d:DrawAuto");
		Hook(vtable, kDrawIndexedInstancedIndirect, &DrawIndexedInstancedIndirect, g_drawIndexedInstancedIndirect, "d3d:DrawIndexedInstancedIndirect");
		Hook(vtable, kDrawInstancedIndirect, &DrawInstancedIndirect, g_drawInstancedIndirect, "d3d:DrawInstancedIndirect");
		Hook(vtable, kRSSetViewports, &RSSetViewports, g_rsSetViewports, "d3d:RSSetViewports");

		// Seed the first-person flag from the viewport that is bound right now.
		UINT           count = 1;
		D3D11_VIEWPORT viewport{};
		context->RSGetViewports(&count, &viewport);
		g_firstPersonViewport.store(count > 0 && viewport.MaxDepth <= kFirstPersonMaxDepth);

		g_immediate.store(context);
	}

	std::uint64_t ImmediateDrawsThisFrame() noexcept
	{
		return g_immediateThisFrame.load(std::memory_order_relaxed);
	}

	void EndFrame(bool a_log)
	{
		if (!g_immediate.load()) {
			return;
		}

		std::array<std::uint64_t, kStages> last{};
		std::array<std::uint64_t, kStages> lastFirstPerson{};
		std::uint64_t                      total = 0;
		for (std::size_t i = 0; i < kStages; ++i) {
			last[i] = g_frame.immediate[i].exchange(0, std::memory_order_relaxed);
			lastFirstPerson[i] = g_frame.firstPerson[i].exchange(0, std::memory_order_relaxed);
			total += last[i];
			g_interval.immediate[i] += last[i];
			g_interval.firstPerson[i] += lastFirstPerson[i];
		}
		const auto otherContext = g_frame.otherContext.exchange(0, std::memory_order_relaxed);
		const auto offRenderThread = g_frame.offRenderThread.exchange(0, std::memory_order_relaxed);
		g_immediateThisFrame.store(0, std::memory_order_relaxed);
		g_interval.otherContext += otherContext;
		g_interval.maxFrameTotal = std::max(g_interval.maxFrameTotal, total);
		++g_interval.frames;

		if (!a_log) {
			return;
		}

		logger::info(
			"draws last frame: immediate={} by stage:{} | other contexts={} | immediate draws off the pre-pass thread={}",
			total, StageCounts(last), otherContext, offRenderThread);

		std::uint64_t intervalTotal = 0;
		for (const auto count : g_interval.immediate) {
			intervalTotal += count;
		}
		const auto frames = static_cast<double>(std::max<std::uint64_t>(1, g_interval.frames));
		logger::info(
			"draws avg over {} frames: immediate={:.0f} (max {}) by stage:{} | other contexts={:.0f}",
			g_interval.frames, static_cast<double>(intervalTotal) / frames, g_interval.maxFrameTotal,
			StageCounts(g_interval.immediate, frames), static_cast<double>(g_interval.otherContext) / frames);
		logger::info(
			"first-person depth-range draws: last frame by stage:{} | avg by stage:{}",
			StageCounts(lastFirstPerson), StageCounts(g_interval.firstPerson, frames));

		if (g_viewportSampled.load()) {
			logger::info(
				"prepass viewport (first draw): x={} y={} w={} h={} depth=[{},{}]",
				g_prepassViewport.TopLeftX, g_prepassViewport.TopLeftY, g_prepassViewport.Width, g_prepassViewport.Height,
				g_prepassViewport.MinDepth, g_prepassViewport.MaxDepth);
		}

		{
			std::scoped_lock lock(g_dsvLock);
			std::vector<std::pair<std::uint64_t, std::size_t>> order;
			for (std::size_t i = 0; i < g_dsvCount; ++i) {
				std::uint64_t draws = 0;
				for (auto& count : g_dsvs[i].draws) {
					draws += count.load(std::memory_order_relaxed);
				}
				order.emplace_back(draws, i);
			}
			std::ranges::sort(order, std::greater{});
			for (const auto& [draws, i] : order) {
				auto&                              record = g_dsvs[i];
				std::array<std::uint64_t, kStages> perStage{};
				for (std::size_t s = 0; s < kStages; ++s) {
					perStage[s] = record.draws[s].exchange(0, std::memory_order_relaxed);
				}
				logger::info(
					"  dsv {} draws={} binds={} [{}] {} | {} | by stage:{}",
					fmt_ptr(record.dsv), draws, record.binds, record.match, record.view, record.texture, StageCounts(perStage));
				record.dsv = nullptr;
				record.binds = 0;
				record.texture.clear();
				record.view.clear();
				record.match.clear();
			}
			g_dsvCount = 0;
			g_currentDsv.store(kNoDsv, std::memory_order_relaxed);
		}

		g_interval = {};
		g_viewportSampled.store(false);
	}
}
