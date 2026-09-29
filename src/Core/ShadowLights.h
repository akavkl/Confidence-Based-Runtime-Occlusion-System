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

namespace CBRO::Core::ShadowLights
{
	void Install();
	void LogStats(std::uint32_t a_frames);

	// Takes the two vtable hooks out (CBRO off: the engine's own culler runs) or puts them back. Main thread.
	bool SetHooksIn(bool a_in);
}
