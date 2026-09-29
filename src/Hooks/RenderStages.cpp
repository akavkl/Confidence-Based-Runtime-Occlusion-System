#include "Hooks/RenderStages.h"

#include "Util/Hooking.h"

namespace CBRO::Hooks::RenderStages
{
	namespace
	{
		std::atomic<std::uint32_t> g_stage{ static_cast<std::uint32_t>(Stage::kNone) };
		std::array<Listener*, 4>   g_listeners{};
		std::size_t                g_listenerCount{ 0 };
		bool                       g_installed{ false };

		Stage Enter(Stage a_stage)
		{
			for (std::size_t i = 0; i < g_listenerCount; ++i) {
				g_listeners[i]->OnStageBegin(a_stage);
			}
			return static_cast<Stage>(g_stage.exchange(static_cast<std::uint32_t>(a_stage)));
		}

		void Leave(Stage a_stage, Stage a_previous)
		{
			g_stage.store(static_cast<std::uint32_t>(a_previous));
			for (std::size_t i = g_listenerCount; i-- > 0;) {
				g_listeners[i]->OnStageEnd(a_stage);
			}
		}

		using StageFn = std::uintptr_t (*)(std::uintptr_t, std::uintptr_t, std::uintptr_t, std::uintptr_t);

		// Forwards all four integer argument registers so the wrapped call keeps whatever
		// arguments it had, and passes RAX back in case the callee returns something.
		template <Stage S>
		struct StageHook
		{
			static inline std::uintptr_t original{ 0 };

			static std::uintptr_t thunk(std::uintptr_t a1, std::uintptr_t a2, std::uintptr_t a3, std::uintptr_t a4)
			{
				const auto previous = Enter(S);
				const auto result = reinterpret_cast<StageFn>(original)(a1, a2, a3, a4);
				Leave(S, previous);
				return result;
			}
		};

		struct StageSite
		{
			Stage           stage;
			std::uint64_t   functionID;
			std::ptrdiff_t  ogOffset;
			std::uintptr_t* original;
			std::uintptr_t  thunk;
			std::uintptr_t  site{ 0 };
		};

		template <Stage S>
		StageSite MakeSite(std::uint64_t a_functionID, std::ptrdiff_t a_ogOffset)
		{
			return { S, a_functionID, a_ogOffset, &StageHook<S>::original, Util::FnAddr(&StageHook<S>::thunk) };
		}

		std::array<StageSite, 7> g_sites{
			MakeSite<Stage::kCull>(984743, 0x16A),              // DrawWorld::Render_PreUI
			MakeSite<Stage::kPrePass>(984743, 0x17F),
			MakeSite<Stage::kHBAO>(984743, 0x1BA),
			MakeSite<Stage::kSunCascades>(984743, 0x1BF),
			MakeSite<Stage::kForward>(984743, 0x1C9),
			MakeSite<Stage::kPostResolveDepth>(338205, 0x1DC),  // ForwardAlphaImpl
			MakeSite<Stage::kFirstPersonAlpha>(338205, 0x253),
		};

		// The stages CBRO always hooks: the two it acts in, and the two more its per-frame timing splits the frame by.
		bool AlwaysHooked(Stage a_stage) noexcept
		{
			return a_stage == Stage::kCull || a_stage == Stage::kPrePass || a_stage == Stage::kSunCascades || a_stage == Stage::kForward;
		}

		const StageSite* FindSite(Stage a_stage) noexcept
		{
			for (const auto& site : g_sites) {
				if (site.stage == a_stage) {
					return &site;
				}
			}
			return nullptr;
		}
	}

	std::string_view StageName(Stage a_stage) noexcept
	{
		switch (a_stage) {
		case Stage::kNone:
			return "none"sv;
		case Stage::kCull:
			return "cull"sv;
		case Stage::kPrePass:
			return "prepass"sv;
		case Stage::kHBAO:
			return "hbao"sv;
		case Stage::kSunCascades:
			return "sunCascades"sv;
		case Stage::kForward:
			return "forward"sv;
		case Stage::kPostResolveDepth:
			return "postResolveDepth"sv;
		case Stage::kFirstPersonAlpha:
			return "firstPersonAlpha"sv;
		default:
			return "?"sv;
		}
	}

	Stage Current() noexcept
	{
		return static_cast<Stage>(g_stage.load(std::memory_order_relaxed));
	}

	void AddListener(Listener* a_listener)
	{
		if (g_listenerCount < g_listeners.size()) {
			g_listeners[g_listenerCount++] = a_listener;
		}
	}

	void Install(bool a_allStages)
	{
		if (g_installed) {
			return;
		}
		g_installed = true;

		for (auto& site : g_sites) {
			if (!a_allStages && !AlwaysHooked(site.stage)) {
				continue;
			}
			site.site = CBRO::Engine::OG(site.functionID).address() + site.ogOffset;
			const auto name = std::format("stage:{}", StageName(site.stage));
			*site.original = Util::WriteCall5(site.site, site.thunk, name);
			if (*site.original) {
				logger::info("hook {}: installed", name);
			}
		}
	}

	std::uintptr_t ThunkOf(Stage a_stage) noexcept
	{
		const auto site = FindSite(a_stage);
		return site && *site->original ? site->thunk : 0;
	}

	std::uintptr_t OriginalOf(Stage a_stage) noexcept
	{
		const auto site = FindSite(a_stage);
		return site ? *site->original : 0;
	}

	void LogChain()
	{
		for (const auto& site : g_sites) {
			if (!*site.original) {
				continue;
			}
			const auto chain = Util::DescribeCodeAddress(Util::ReadCall5Target(site.site));
			const bool outermost = chain.find("CBRO.dll") != std::string::npos;
			logger::info(
				"stage:{} site now calls {} ({})",
				StageName(site.stage), chain, outermost ? "ours is outermost" : "another plugin wrapped ours; still chained");
		}
	}
}
