#pragma once

// Renderer Replacer's path tracer (D:/Projects/Renderer Replacer, src/Render/PathTracer/Scene.cpp) builds its scene from
// BSShaderAccumulator::RegisterObject (1447420): the main accumulator's registrations and the shadow maps' (the sun's
// cascades, the lamps). Its detour sits at that function's entry, beneath CBRO's slot-45 hook (Hooks/CullGroups), so it
// never sees a registration CBRO refuses, and an entry CBRO rejects or never files is registered nowhere: its
// reflections, bounces and traced shadows lost whatever CBRO culled from the screen. Its scene keeps an instance it was
// handed while the object still stands in the world, for up to 300 frames without another registration (its
// kKeepFrames); an instance's transform, skin pose and mesh-LOD level are read only when it is registered.
//
// The tracer cannot say which objects its reflections need before CBRO decides: CBRO decides in DrawWorld's cull
// (Render_PreUI+0x16A, FO4-ENGINE-NOTES 4.1), the trace runs at the deferred composite (728427) later in the same frame,
// and its rays meet only what its scene already holds. So while it asks (CBRO_SetTraceFeed below):
// - every registration CBRO refuses is handed to it (Hooks::CullGroups::SetRefusedObserver). No draw comes back.
// - an object CBRO would leave out of every view (a confirmed-hidden main-only entry; a group-0 entry neither the main
//   view nor a shadow needs) within the tracer's radius is filed instead, on its turn (one frame in refreshFrames, by
//   address) and in every frame its bound moved: as a main-view drop and, with the sun up, out of the sun's cascades.
//   The engine walks it as it walks any kept entry (BSFadeNode::OnVisible with its LOD level, 928617; its children,
//   558046: notes 5.2, 7.8, 7.11) and registers its geometry (962984). CBRO refuses each registration and hands it
//   over. A cell node pruned whole is walked on its own turn (sun up), its entries with it.
// - (v1.91) an object out of the view that no view files (the engine's frustum test leaves it out: a main-only entry,
//   or a group-0 entry with the sun down, or one nothing needs) is filed the same way on its turn, force-visible (the
//   block's force byte, as previs marks its entries), its children with the mark. Behind the camera the tracer otherwise
//   kept it only as long as it had seen it last (300 frames), and never got one it had not seen. With the sun up a
//   group-0 entry whose shadow may be needed is left to the cascades, which file it within their range.
// The rest of the time such an object stays out of the walk, as without the feed.

extern "C"
{
	// The tracer's request. Renderer Replacer keeps a copy of this layout (src/Core/Registrations.cpp): grow it by
	// appending fields, gated on size.
	struct CBRO_TraceFeed
	{
		std::uint32_t size;           // sizeof(CBRO_TraceFeed)
		std::uint32_t refreshFrames;  // an object CBRO leaves out is filed for the tracer once in this many frames (rounded down to a power of two, 4..1024)
		float         radius;         // world units: an object wholly farther from the camera is never filed for it
		void (*refused)(std::uintptr_t a_accumulator, std::uintptr_t a_object);  // each refused registration, on the registering thread
	};

	// Exported. Null (or a null `refused`) turns the feed off. Main thread; takes effect at the next cull begin. False when
	// not taken (bTraceFeed=0, an older layout, CBRO not running).
	__declspec(dllexport) bool CBRO_SetTraceFeed(const CBRO_TraceFeed* a_feed);
}

namespace CBRO::Core::TraceFeed
{
	void Install();  // after Hooks::CullGroups

	// Main thread at each cull begin, before the sun's cascades are read (Occlusion::SetSunCascades asks On()): this
	// frame's request.
	void BeginFrame() noexcept;
	[[nodiscard]] bool On() noexcept;  // this frame

	// An object's turn this frame (one frame in refreshFrames, spread by address).
	[[nodiscard]] bool Turn(const void* a_object, std::uint32_t a_clock) noexcept;
	// Whether any of a bound lies within the tracer's radius of a_eye.
	[[nodiscard]] bool InReach(const RE::NiBound& a_bound, const float* a_eye) noexcept;
}
