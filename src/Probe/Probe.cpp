#include "Probe/Probe.h"
#include "Probe/ProbeInternal.h"

#include "Settings.h"
#include "Util/D3D.h"
#include "Util/Hooking.h"

namespace CBRO::Probe
{
	namespace
	{
		SharedState g_state;

		struct TraceBuffer
		{
			std::mutex               lock;
			std::vector<std::string> events;
			std::int64_t             start{ 0 };
			std::int64_t             frequency{ 1 };
			std::uint64_t            firstFrame{ 0 };
		};
		TraceBuffer g_trace;

		// Frame-boundary bookkeeping. Written by F4SE messages on the main thread and
		// consumed at the deferred pre-pass, so everything crossing threads is atomic.
		struct Control
		{
			std::atomic<std::int64_t> traceCountdown{ -1 };   // frames until an automatic trace opens
			std::atomic<std::int64_t> surveyCountdown{ -1 };  // frames until an automatic survey
			std::atomic<bool>         dumpPending{ true };

			std::uint32_t traceFramesLeft{ 0 };
			bool          traceKeyDown{ false };
			bool          surveyKeyDown{ false };
			bool          drawHooksInstalled{ false };
			std::uint32_t lastScreenWidth{ 0 };
			std::uint32_t lastScreenHeight{ 0 };

			std::int64_t  lastFrameQPC{ 0 };
			double        frameMsSum{ 0.0 };
			double        frameMsMax{ 0.0 };
			std::uint32_t frameMsCount{ 0 };

			std::array<std::atomic<std::uint64_t>, static_cast<std::size_t>(Stage::kCount)> stageCalls{};
			std::array<std::atomic<std::uint32_t>, static_cast<std::size_t>(Stage::kCount)> stageThread{};
		};
		Control g_control;

		std::int64_t QPC() noexcept
		{
			LARGE_INTEGER value{};
			QueryPerformanceCounter(&value);
			return value.QuadPart;
		}

		// ---- trace window ----------------------------------------------------------------

		void FlushTrace()
		{
			std::vector<std::string> events;
			std::uint64_t            firstFrame = 0;
			{
				std::scoped_lock lock(g_trace.lock);
				events.swap(g_trace.events);
				firstFrame = g_trace.firstFrame;
			}
			if (events.empty()) {
				return;
			}
			logger::info("---- trace (window opened at frame {}), {} events ----", firstFrame, events.size());
			for (const auto& event : events) {
				logger::info("  {}", event);
			}
		}

		void OpenTraceWindow(std::string_view a_reason, std::uint64_t a_frame)
		{
			const auto frames = Settings::Get().traceFrames;
			if (frames == 0) {
				return;
			}
			logger::info("trace: opening {}-frame window at frame {} ({})", frames, a_frame, a_reason);
			{
				std::scoped_lock lock(g_trace.lock);
				g_trace.events.clear();
				g_trace.start = QPC();
				g_trace.firstFrame = a_frame;
			}
			g_control.traceFramesLeft = frames;
			g_state.tracing.store(true);
		}

		bool KeyPressed(std::uint32_t a_vk, bool& a_wasDown)
		{
			if (a_vk == 0) {
				return false;
			}
			const bool down = (GetAsyncKeyState(static_cast<int>(a_vk)) & 0x8000) != 0;
			const bool pressed = down && !a_wasDown;
			a_wasDown = down;
			return pressed;
		}

		bool GameWindowFocused()
		{
			const auto main = RE::Main::GetSingleton();
			return main && GetForegroundWindow() == reinterpret_cast<HWND>(main->hwnd);
		}

		// ---- one-shot dumps ----------------------------------------------------------------

		void DumpRenderTargets()
		{
			const auto data = RE::BSGraphics::RendererData::GetSingleton();
			const auto state = CBRO::Engine::BSGraphics::State::GetSingleton();
			if (!data || !state) {
				logger::warn("dump: renderer not ready (data={} state={})", fmt_ptr(data), fmt_ptr(state));
				return;
			}

			logger::info("==== render target dump (frame {}) ====", g_state.frame.load());
			logger::info(
				"state: screen={}x{} backBuffer={}x{} fbViewport=[{:.3f},{:.3f},{:.3f},{:.3f}] frameCount={}",
				state->screenWidth, state->screenHeight, state->backBufferWidth, state->backBufferHeight,
				state->frameBufferViewport.left, state->frameBufferViewport.right,
				state->frameBufferViewport.top, state->frameBufferViewport.bottom, state->frameCount);

			for (std::size_t i = 0; i < std::size(data->depthStencilTargets); ++i) {
				const auto& target = data->depthStencilTargets[i];
				const auto  texture = reinterpret_cast<ID3D11Texture2D*>(target.texture);
				if (!texture) {
					continue;
				}
				logger::info(
					"depth[{:2}] tex={} {} | {} | dsv0={} roDepth0={}",
					i, fmt_ptr(texture), Util::DescribeTexture(texture),
					Util::DescribeSRV(reinterpret_cast<ID3D11ShaderResourceView*>(target.srViewDepth)),
					fmt_ptr(target.dsView[0]), fmt_ptr(target.dsViewReadOnlyDepth[0]));
			}

			// (The render-target manager's logical -> platform table isn't in the RD headers; its OG offsets were never
			// verified anyway. The platform indices above are what the engine binds.)
			constexpr std::size_t kMainDepthMips = 39;
			const auto            depthMips = reinterpret_cast<ID3D11Texture2D*>(data->renderTargets[kMainDepthMips].texture);
			logger::info(
				"renderTargets[39] (kMainDepthMips, platform index) {} | {}",
				Util::DescribeTexture(depthMips),
				Util::DescribeSRV(reinterpret_cast<ID3D11ShaderResourceView*>(data->renderTargets[kMainDepthMips].srView)));
		}

		void LogFrameSummary(std::uint64_t a_frame)
		{
			const auto main = g_state.mainThreadId.load();
			const auto render = g_state.renderThreadId.load();

			std::string stages;
			for (std::size_t i = 1; i < static_cast<std::size_t>(Stage::kCount); ++i) {
				const auto calls = g_control.stageCalls[i].exchange(0);
				const auto tid = g_control.stageThread[i].load();
				stages += std::format(" {}={}@{}", StageName(static_cast<Stage>(i)), calls, tid);
			}

			const auto ui = RE::UI::GetSingleton();
			const bool loading = ui && ui->GetMenuOpen("LoadingMenu");
			const auto rootCamera = g_state.worldCamera.load();
			char       cameraName[64]{};
			TryGetObjectName(reinterpret_cast<const void*>(rootCamera), cameraName, sizeof(cameraName));

			// Previs (BSPreCulledObjects) state as the engine sees it: active = enabled && INI && !suspended.
			// Toggle in game with the console command `tpc` (TogglePreCulling).
			const auto readByte = [](std::uint64_t a_id) { return *reinterpret_cast<const std::uint8_t*>(CBRO::Engine::OG(a_id).address()) != 0; };
			const bool previsEnabled = readByte(493183);
			const bool previsSuspended = readByte(718924);
			const bool previsIni = readByte(1472203);  // bUsePreCulledObjects:Display

			const auto frames = std::max<std::uint32_t>(1, g_control.frameMsCount);
			logger::info("==== summary frame {} ====", a_frame);
			logger::info(
				"frame time (base, pre-pass to pre-pass): avg {:.2f} ms ({:.0f} fps), max {:.2f} ms over {} frames",
				g_control.frameMsSum / frames, 1000.0 / std::max(0.001, g_control.frameMsSum / frames), g_control.frameMsMax, g_control.frameMsCount);
			g_control.frameMsSum = 0.0;
			g_control.frameMsMax = 0.0;
			g_control.frameMsCount = 0;
			logger::info(
				"previs: {} (enabled={} bUsePreCulledObjects={} suspended={})",
				previsEnabled && previsIni && !previsSuspended ? "ACTIVE" : "OFF", previsEnabled, previsIni, previsSuspended);
			logger::info(
				"threads: main={} prepass={} prepassOnMainThread={} | stage calls since last summary (count@tid):{}",
				main, render, main == render, stages);
			logger::info(
				"world camera: {} '{}' | menuMode={} loadingMenu={}",
				fmt_ptr(rootCamera), cameraName, ui ? ui->menuMode : 0u, loading);
		}

		void OnFrameBoundary()
		{
			const auto& settings = Settings::Get();

			g_state.renderThreadId.store(GetCurrentThreadId());
			if (const auto main = RE::Main::GetSingleton()) {
				g_state.mainThreadId.store(main->threadID);
			}
			g_state.worldCamera.store(reinterpret_cast<std::uintptr_t>(RE::Main::WorldRootCamera()));

			const auto frame = g_state.frame.fetch_add(1) + 1;
			const bool summary = frame % settings.summaryIntervalFrames == 0;

			// Base (rendered) frame time: pre-pass to pre-pass, unaffected by frame generation.
			const auto now = QPC();
			if (g_control.lastFrameQPC) {
				const double ms = static_cast<double>(now - g_control.lastFrameQPC) * 1000.0 / static_cast<double>(g_trace.frequency);
				g_control.frameMsSum += ms;
				g_control.frameMsMax = std::max(g_control.frameMsMax, ms);
				++g_control.frameMsCount;
			}
			g_control.lastFrameQPC = now;

			// close out the previous frame
			if (summary) {
				LogFrameSummary(frame - 1);
			}
			Cull::EndFrame(summary);
			Groups::EndFrame(summary);
			Draw::EndFrame(summary);

			if (g_state.tracing.load()) {
				if (g_control.traceFramesLeft > 0) {
					--g_control.traceFramesLeft;
				}
				if (g_control.traceFramesLeft == 0) {
					g_state.tracing.store(false);
					FlushTrace();
				}
			}

			// open the next windows
			const bool focused = GameWindowFocused();
			const bool traceKey = focused && KeyPressed(settings.traceHotkey, g_control.traceKeyDown);
			const bool surveyKey = focused && KeyPressed(settings.surveyHotkey, g_control.surveyKeyDown);

			if (!g_state.tracing.load()) {
				if (traceKey) {
					OpenTraceWindow("hotkey", frame);
				} else if (const auto countdown = g_control.traceCountdown.load(); countdown >= 0) {
					if (countdown == 0) {
						OpenTraceWindow("after load", frame);
					}
					g_control.traceCountdown.store(countdown - 1);
				}
			}

			bool survey = surveyKey;
			if (const auto countdown = g_control.surveyCountdown.load(); countdown >= 0) {
				survey |= countdown == 0;
				g_control.surveyCountdown.store(countdown - 1);
			}
			if (survey) {
				g_control.dumpPending.store(true);
				if (settings.sceneSurvey) {
					Scene::RequestSurvey();
				}
			}

			if (const auto state = CBRO::Engine::BSGraphics::State::GetSingleton()) {
				if (state->screenWidth != g_control.lastScreenWidth || state->screenHeight != g_control.lastScreenHeight) {
					g_control.lastScreenWidth = state->screenWidth;
					g_control.lastScreenHeight = state->screenHeight;
					g_control.dumpPending.store(true);
				}
			}
			if (g_control.dumpPending.exchange(false)) {
				DumpRenderTargets();
			}

			if (!g_control.drawHooksInstalled && settings.drawCallCounter) {
				g_control.drawHooksInstalled = true;
				Draw::Install();
			}
		}

		class StageListener final :
			public Hooks::RenderStages::Listener
		{
		public:
			void OnStageBegin(Stage a_stage) override
			{
				if (a_stage == Stage::kPrePass) {
					OnFrameBoundary();
				}

				const auto index = static_cast<std::size_t>(a_stage);
				g_control.stageCalls[index].fetch_add(1, std::memory_order_relaxed);
				g_control.stageThread[index].store(GetCurrentThreadId(), std::memory_order_relaxed);

				if (Tracing()) {
					TraceEvent(std::format("{} begin (immediate draws so far {})", StageName(a_stage), Draw::ImmediateDrawsThisFrame()));
				}
			}

			void OnStageEnd(Stage a_stage) override
			{
				if (Tracing()) {
					TraceEvent(std::format("{} end (immediate draws so far {})", StageName(a_stage), Draw::ImmediateDrawsThisFrame()));
				}
			}
		};
		StageListener g_stageListener;
	}

	SharedState& State() noexcept
	{
		return g_state;
	}

	std::string_view ThreadKindName(ThreadKind a_kind) noexcept
	{
		switch (a_kind) {
		case ThreadKind::kMain:
			return "main"sv;
		case ThreadKind::kRender:
			return "render"sv;
		case ThreadKind::kOther:
			return "other"sv;
		default:
			return "?"sv;
		}
	}

	void TraceEvent(std::string a_text)
	{
		if (!Tracing()) {
			return;
		}
		const auto tid = GetCurrentThreadId();
		std::scoped_lock lock(g_trace.lock);
		if (g_trace.events.size() >= 8192) {
			return;
		}
		const double ms = static_cast<double>(QPC() - g_trace.start) * 1000.0 / static_cast<double>(g_trace.frequency);
		g_trace.events.push_back(std::format("+{:9.3f}ms tid={:5} {:6} {}", ms, tid, ThreadKindName(ClassifyThread(tid)), a_text));
	}

	void Install()
	{
		const auto& settings = Settings::Get();

		LARGE_INTEGER frequency{};
		QueryPerformanceFrequency(&frequency);
		g_trace.frequency = frequency.QuadPart;

		if (!REL::Module::get().is_og()) {
			logger::warn("probe: hook offsets and vtable IDs are OG-only; runtime is not 1.10.163, installing nothing");
			return;
		}

		if (settings.renderStageHooks) {
			Hooks::RenderStages::AddListener(&g_stageListener);
		}
		if (settings.cullingHooks) {
			Cull::Install();
		}
		if (settings.cullingGroupHooks) {
			Groups::Install();
		}
	}

	void OnPostPostLoad()
	{
		if (!REL::Module::get().is_og()) {
			return;
		}

		logger::info("---- hook chain after all plugins loaded ----");
		Hooks::RenderStages::LogChain();
		if (Settings::Get().cullingHooks) {
			Cull::VerifyChain();
		}
	}

	void OnGameDataReady()
	{
		Scene::LogGameSettings();
	}

	void OnGameLoaded()
	{
		const auto delay = static_cast<std::int64_t>(Settings::Get().traceDelayFramesAfterLoad);
		g_control.traceCountdown.store(delay);
		g_control.surveyCountdown.store(delay);
		g_control.dumpPending.store(true);
		logger::info("game loaded: trace, render-target dump and scene survey scheduled in {} frames", delay);
	}
}
