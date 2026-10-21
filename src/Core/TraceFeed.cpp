#include "Core/TraceFeed.h"

#include "Hooks/CullGroups.h"
#include "Settings.h"

#include <algorithm>
#include <bit>

namespace CBRO::Core::TraceFeed
{
	namespace
	{
		using Refused = void (*)(std::uintptr_t, std::uintptr_t);

		constexpr std::uint32_t kMinRefresh = 4;
		constexpr std::uint32_t kMaxRefresh = 1024;

		bool g_allowed{ false };  // bTraceFeed, and the accumulator hook installed

		// The consumer's request (CBRO_SetTraceFeed, main thread), taken at the next cull begin.
		std::atomic<Refused>       g_requested{ nullptr };
		std::atomic<std::uint32_t> g_requestedMask{ 0 };
		std::atomic<float>         g_requestedRadius{ 0.0f };

		// This frame's (BeginFrame; read by the walk on any thread, written only before it starts)
		std::atomic<Refused> g_refused{ nullptr };
		bool                 g_on{ false };
		std::uint32_t        g_mask{ 0 };
		float                g_radius{ 0.0f };

		void Forward(const void* a_accumulator, RE::NiAVObject* a_object)
		{
			if (const auto refused = g_refused.load(std::memory_order_relaxed)) {
				refused(reinterpret_cast<std::uintptr_t>(a_accumulator), reinterpret_cast<std::uintptr_t>(a_object));
			}
		}
	}

	void Install()
	{
		g_allowed = Settings::Get().traceFeed;
		logger::info("trace feed: {}", g_allowed ? "on request (a path tracer's scene: Renderer Replacer)" : "off (bTraceFeed=0)");
	}

	void BeginFrame() noexcept
	{
		const auto refused = g_allowed ? g_requested.load(std::memory_order_acquire) : nullptr;
		const bool on = refused != nullptr;
		const auto mask = g_requestedMask.load(std::memory_order_relaxed);
		const auto radius = g_requestedRadius.load(std::memory_order_relaxed);
		if (on != g_on || (on && (mask != g_mask || radius != g_radius))) {
			if (on) {
				logger::info("trace feed: on (refused registrations handed over; what CBRO leaves out filed once every {} frames, and when it moves, within {:.0f} units)", mask + 1, radius);
			} else {
				logger::info("trace feed: off");
			}
		}
		g_on = on;
		g_mask = mask;
		g_radius = radius;
		g_refused.store(refused, std::memory_order_relaxed);
		Hooks::CullGroups::SetRefusedObserver(on ? &Forward : nullptr);
	}

	bool On() noexcept
	{
		return g_on;
	}

	bool Turn(const void* a_object, std::uint32_t a_clock) noexcept
	{
		const auto hash = static_cast<std::uint32_t>(((reinterpret_cast<std::uintptr_t>(a_object) >> 4) * 0x9E3779B97F4A7C15ull) >> 32);
		return ((hash + a_clock) & g_mask) == 0;
	}

	bool InReach(const RE::NiBound& a_bound, const float* a_eye) noexcept
	{
		const float dx = a_bound.center.x - a_eye[0], dy = a_bound.center.y - a_eye[1], dz = a_bound.center.z - a_eye[2];
		const float reach = g_radius + std::max(a_bound.fRadius, 0.0f);
		return dx * dx + dy * dy + dz * dz <= reach * reach;
	}
}

bool CBRO_SetTraceFeed(const CBRO_TraceFeed* a_feed)
{
	using namespace CBRO::Core::TraceFeed;
	if (!a_feed || !a_feed->refused) {
		g_requested.store(nullptr, std::memory_order_release);
		return true;
	}
	if (a_feed->size < sizeof(CBRO_TraceFeed) || !g_allowed) {
		return false;
	}
	const auto refresh = std::bit_floor(std::clamp(a_feed->refreshFrames, kMinRefresh, kMaxRefresh));
	g_requestedMask.store(refresh - 1, std::memory_order_relaxed);
	g_requestedRadius.store(std::max(a_feed->radius, 0.0f), std::memory_order_relaxed);
	g_requested.store(a_feed->refused, std::memory_order_release);
	return true;
}
