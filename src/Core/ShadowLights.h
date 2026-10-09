#pragma once

// Point-light shadow maps. With previs off, every shadow-casting point light in the loaded cells
// renders its paraboloid shadow maps (8-14k draws per frame in dense night areas). A shadow map can
// only matter if the light can reach a visible pixel, so when the light's whole reach is hidden
// behind the main view's depth, its caster cull (BSShadowParabolicLight -> BSParabolicCullingProcess
// ::Process(camera, ...)) runs in kAllFail mode: the engine restarts the caster list for that
// camera and adds nothing, so the shadow pass draws nothing. Lights reaching the camera, partly
// off-screen, or with unknown reach are never touched.
//
// Spot lights (v1.48; bSpotShadowCulling): their casters come from a culling-group pass, not a traversal, so they
// are dropped at their shadow map's accumulator instead: when the lit volume (a sphere around the part of the light's
// reach inside its shadow frustum, the pyramid the engine itself tests the light's visibility by; the whole reach when
// the frustum's corners lie over 60 degrees off its axis) is hidden behind the main view's depth for confirmFrames
// frames, no caster is filed and the shadow map draws nothing.
// Previs does the same job by marking such lights occluded (FO4-ENGINE-NOTES 6.2a/6.7). Since v1.53 an emptied spot
// light's group pass is skipped outright (its output was dropped anyway), and a kept spot light's casters get the
// per-caster test below as they register into its shadow map (bSpotCasterTrim).
//
// For every other light, its traversal's per-object test (Process(object)) also drops casters that can't
// shadow a visible pixel: the light and every visible point lie in the view cone pushed out until it holds
// the light, so each light-to-pixel segment does too, and a caster entirely outside that region (by more
// than the shadow filter's reach) meets none of them (ShadowGeometry). The subtree goes with the object.

//
// The lamps themselves (v1.37): every shadow-casting lamp the shadow stage processes in a frame is recorded (point
// lights from their culler's sphere, spot lights from BSShadowFrustumLight's cull: the NiLight's position and radius)
// and published at the next cull begin. With the sun off, group 0's only remaining readers are the spot lights' group
// passes (FO4-ENGINE-NOTES 6.2a), so Occlusion leaves a group-0 entry the main view doesn't need out of the group when
// no recorded lamp's shadow of it can reach a visible surface (the same shadow-volume test the point-light casters get).
namespace CBRO::Core::ShadowLights
{
	void Install();
	void LogStats(std::uint32_t a_frames);

	// Takes the vtable hooks out (CBRO off: the engine's own culler runs) or puts them back. Main thread.
	bool SetHooksIn(bool a_in);

	// The main frame's deferred-lights stage (its lamp loop) begins (true, CBRO frames only) or ends (false): spot-light
	// shadow maps are emptied only inside it. Main thread.
	void SetLampStage(bool a_open) noexcept;

	struct Lamp
	{
		RE::NiPoint3 position{};
		float        reach{ 0.0f };
		bool         spot{ false };
		// Spot lamps (v1.81): the lit volume the shadow-map test read, whether the map was emptied, and whether the frame
		// before recorded the same lamp with the same volume.
		RE::NiBound  volume{};
		bool         emptied{ false };
		bool         steady{ false };
	};
	struct LampList
	{
		static constexpr std::size_t kMax = 48;
		std::array<Lamp, kMax>       items{};
		std::uint32_t                count{ 0 };
		std::uint32_t                overflow{ 0 };
		bool                         complete{ false };  // both light kinds recorded (the spot-light hook is in)
	};
	// Previs-driven light occlusion (FO4-ENGINE-NOTES 6.7): with previs active the engine marks lights previs deems
	// unseen as occluded (`BSLight+0x17C`) and unmarks them through the same previs-gated callback (356257:
	// `if (IsActive() && ...)`). With previs inactive for the walk (CBRO's windows, or v1.28's switch) nothing unmarks
	// them, so a lamp marked while previs was active stays dark: DeferredLightsImpl's lamp loop skips every marked
	// shadow light (no shadow map, no light pass). Vanilla with previs off never marks any. This clears the mark on every
	// light of the ShadowSceneNode's lists; to run once per CBRO frame before the deferred-lights stage. Returns how
	// many it cleared. Main thread. a_countFrame: this call counts as the frame's (the stats are per frame; the cull
	// begin's call counts, the deferred-lights stage's does not).
	std::uint32_t UnoccludeLights(bool a_countFrame) noexcept;

	// Diagnostic (bLampDiagnostic): what the engine's shadow-light loop (DeferredLightsImpl, FO4-ENGINE-NOTES 6.4/6.8)
	// decides for each shadow light of a CBRO frame. Begin runs right before the loop (the deferred-lights stage begin):
	// it replicates the loop's static tests per light (hidden, occluded, culler state, room set against the main camera);
	// the light's Update(camera) is hooked to record its result; End runs after the loop and reads the queue slot and the
	// shadow-map slice. LogStats reports the per-frame fates and a few of the lights that did not render. Main thread.
	void LampLoopBegin() noexcept;
	void LampLoopEnd() noexcept;

	// Main thread at the cull begin: last frame's lamps become the readable list; this frame's recording starts over.
	void PublishLamps() noexcept;
	// The published list (read-only until the next PublishLamps; any thread).
	[[nodiscard]] const LampList& Lamps() noexcept;
}
