#pragma once

// Diagnostic (bSetDiff): at one spot, which main-view objects CBRO mode draws that previs mode never draws, and the
// reverse. The v1.33/v1.34 runs showed CBRO classic registering 2,200 main-view objects a frame at the load spot
// against real previs' 1,050 with the pre-pass GPU span 2.6 ms against 0.7: the gap to previs is mostly what CBRO
// keeps. Every main-accumulator registration (both modes; Hooks/CullGroups' observer) goes into a per-frame ring;
// at the cull stage's end the settled frames' objects are added to the mode's set for the current A/B location, and
// an object registered by one mode while the other mode's set (already populated at this spot) lacks it is counted
// once, by type, projected size (bound radius over distance), distance, the always-draw flag and previs' own
// not-visible mark. Objects are only dereferenced in the frame that registered them (main thread, the cull's end),
// under SEH. Sets reset at each new A/B location. Cost: one lock-free append per registration, a hash insert per
// settled-frame registration on the main thread.
namespace CBRO::Core::SetDiff
{
	void Install(bool a_enabled);
	// Any thread: a main-accumulator registration (Hooks/CullGroups' main registration observer).
	void Observe(RE::NiAVObject* a_object) noexcept;
	// Main thread, at the cull stage's end: files this frame's registrations under the frame's mode (0 previs, else
	// CBRO) when the frame is settled (past the load and the switch settle, no menu, no hitch).
	void EndCull(int a_mode, bool a_settled, const RE::NiPoint3& a_eye);
	void Reset();  // a new A/B location: both sets start over
	void LogStats();
}
