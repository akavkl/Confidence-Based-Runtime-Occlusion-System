#pragma once

// Call-site wrappers around DrawWorld render stages (OG offsets from jarari/fo4test; Upscaling
// write_calls the same sites and write_call chaining keeps both hooks alive in either load order).
// Everything here runs on the main thread (Phase 0: render stages share Main::threadID).

namespace CBRO::Hooks::RenderStages
{
	enum class Stage : std::uint32_t
	{
		kNone,
		kCull,              // DrawWorld::Render_PreUI -> DrawWorld cull (718911): feeds the main-camera culling groups
		kPrePass,           // DrawWorld::Render_PreUI -> deferred opaque pre-pass (fills depthStencilTargets[2])
		kHBAO,              // DrawWorld::Render_PreUI -> NVHBAO
		kSunCascades,       // DrawWorld::Render_PreUI -> unbatched sun-cascade setup (1108521): the light's update, the
		                    //   cascade cull over group 0 and (previs off) the shadow-map draws of every caster in range
		kForward,           // DrawWorld::Render_PreUI -> forward pass
		kPostResolveDepth,  // ForwardAlphaImpl -> FinishAccumulating (after depth resolve)
		kFirstPersonAlpha,  // ForwardAlphaImpl -> first-person RenderAlphaGeometry
		kCount
	};

	[[nodiscard]] std::string_view StageName(Stage a_stage) noexcept;

	// The stage currently executing on the render thread (kNone outside wrapped stages).
	[[nodiscard]] Stage Current() noexcept;

	class Listener
	{
	public:
		virtual ~Listener() = default;
		// Called before the stage starts (Current() still returns the enclosing stage).
		virtual void OnStageBegin(Stage) {}
		// Called after the stage returned (Current() already restored).
		virtual void OnStageEnd(Stage) {}
	};

	// Register before Install(). Begin callbacks run in registration order, end callbacks in reverse.
	void AddListener(Listener* a_listener);

	// Hooks the cull, pre-pass, sun-cascades and forward sites (what CBRO's frame timing needs); with a_allStages
	// also the probe-only ones (hbao, post-resolve-depth, first-person alpha).
	void Install(bool a_allStages);
	void LogChain();  // after all plugins loaded: report who else wraps our sites

	// CBRO's thunk on a stage's call site, and what it chains to (0 if the stage isn't hooked): for code that
	// checks whether a site still calls the engine's function (Hooks/CullGroups: the sun's cascade path).
	[[nodiscard]] std::uintptr_t ThunkOf(Stage a_stage) noexcept;
	[[nodiscard]] std::uintptr_t OriginalOf(Stage a_stage) noexcept;
}
