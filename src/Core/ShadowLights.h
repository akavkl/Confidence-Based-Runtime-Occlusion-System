#pragma once

// Point-light shadow maps. With previs off, every shadow-casting point light in the loaded cells
// renders its paraboloid shadow maps (8-14k draws per frame in dense night areas). A shadow map can
// only matter if the light can reach a visible pixel, so when the light's whole reach is hidden
// behind the main view's depth, its caster cull (BSShadowParabolicLight -> BSParabolicCullingProcess
// ::Process(camera, ...)) runs in kAllFail mode: the engine restarts the caster list for that
// camera and adds nothing, so the shadow pass draws nothing. Lights reaching the camera, partly
// off-screen, or with unknown reach are never touched.
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

	struct Lamp
	{
		RE::NiPoint3 position{};
		float        reach{ 0.0f };
		bool         spot{ false };
	};
	struct LampList
	{
		static constexpr std::size_t kMax = 48;
		std::array<Lamp, kMax>       items{};
		std::uint32_t                count{ 0 };
		std::uint32_t                overflow{ 0 };
	};
	// Main thread at the cull begin: last frame's lamps become the readable list; this frame's recording starts over.
	void PublishLamps() noexcept;
	// The published list (read-only until the next PublishLamps; any thread).
	[[nodiscard]] const LampList& Lamps() noexcept;
}
