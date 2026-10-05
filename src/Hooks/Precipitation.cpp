#include "Hooks/Precipitation.h"

#include "Engine/Compat.h"
#include "Hooks/PrevisFeed.h"
#include "Util/Hooking.h"

namespace CBRO::Hooks::Precipitation
{
	namespace
	{
		// 856638 (0x140649BE0): mov rax, rsp; push rbp; push r14 (6 bytes, position-independent), then lea rbp, [rax-0x1B8].
		constexpr std::uint64_t                kRainPassID = 856638;
		constexpr std::array<std::uint8_t, 6> kRainPassPrologue{ 0x48, 0x8B, 0xC4, 0x55, 0x41, 0x56 };

		using PassFn = void (*)(std::uintptr_t, std::uintptr_t);
		Util::SwitchableHook g_hook;
		std::uintptr_t       g_original{ 0 };
		double               g_qpcPerMs{ 0.0 };

		std::atomic<std::uint64_t> g_runs{ 0 };
		std::atomic<std::uint64_t> g_previsActive{ 0 };
		std::atomic<std::int64_t>  g_ticksActive{ 0 };
		std::atomic<std::int64_t>  g_ticksInactive{ 0 };

		void PassThunk(std::uintptr_t a_precipitation, std::uintptr_t a_extra)
		{
			const bool    window = PrevisFeed::BeginRainWindow();  // (this frame's previs query skipped: its list is empty)
			const bool    active = PrevisFeed::ActiveNow();
			LARGE_INTEGER start{}, end{};
			QueryPerformanceCounter(&start);
			reinterpret_cast<PassFn>(g_original)(a_precipitation, a_extra);
			QueryPerformanceCounter(&end);
			PrevisFeed::EndRainWindow(window);
			g_runs.fetch_add(1, std::memory_order_relaxed);
			if (active) {
				g_previsActive.fetch_add(1, std::memory_order_relaxed);
			}
			(active ? g_ticksActive : g_ticksInactive).fetch_add(end.QuadPart - start.QuadPart, std::memory_order_relaxed);
		}
	}

	void Install()
	{
		LARGE_INTEGER frequency{};
		QueryPerformanceFrequency(&frequency);
		g_qpcPerMs = static_cast<double>(frequency.QuadPart) / 1000.0;
		g_original = Util::DetourSwitchable(g_hook, CBRO::Engine::OG(kRainPassID).address(), Util::FnAddr(&PassThunk), kRainPassPrologue, "precipitation:rain occlusion pass");
		logger::info("precipitation: rain occlusion pass {} (diagnostic: runs, previs state, CPU time)", g_original ? "hooked" : "NOT hooked");
	}

	bool Hooked() noexcept
	{
		return g_original != 0;
	}

	Stats Take() noexcept
	{
		const auto ms = [](std::int64_t a_ticks) { return g_qpcPerMs > 0.0 ? static_cast<double>(a_ticks) / g_qpcPerMs : 0.0; };
		return {
			g_runs.exchange(0, std::memory_order_relaxed), g_previsActive.exchange(0, std::memory_order_relaxed),
			ms(g_ticksActive.exchange(0, std::memory_order_relaxed)), ms(g_ticksInactive.exchange(0, std::memory_order_relaxed))
		};
	}
}
