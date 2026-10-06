#pragma once

#include "Core/Occlusion.h"
#include "Core/Verdict.h"
#include "Hooks/CullGroups.h"

// Asynchronous verdicts (bAsync). The v1.35 run showed CBRO's frame is paced by the main thread: 1.0-2.4 ms of
// per-entry work inside the engine's scene walk (Block::Add), plus the walk itself. Here the walk's hooks do no
// evaluation: for every entry the engine offers they look the object's record up in a map filled by a worker and
// append the entry to a candidate ring; at the cull stage's end the ring and a copy of the frame's context go to the
// worker, which judges every candidate with the same Judge logic (Occlusion::JudgeAsync: cache reuse, sphere and
// mesh-shape tests, lights, the sun) while the rest of the frame renders, into the other map. At the next cull begin
// the maps swap. So a verdict is used one frame after the walk that offered its object, with the depth frame the
// worker had then.
//
// Safety: the worker's context is the frame's own, with its dilation grown by the camera movement the next frame may
// bring (moveMargin) and its view box widened toward the turn the next frame may bring (Runtime, v1.70). A map's
// verdicts are used only in a frame whose camera stayed within the margins of the camera the worker judged for and
// whose view lies inside the box the worker judged with (BeginFrame); otherwise, and for
// any object the map has no record of with its current bound, the walk judges the object itself (Occlusion), last
// frame's record only continuing its streaks. Objects are dereferenced by the worker only between the cull's end and
// the forward stage's end of the same frame (Wait): the engine frees scene objects in its update, never while it
// renders them. Kept records reused across frames are always safe; hidden ones carry the streaks and evidence the
// synchronous path has.
namespace CBRO::Core::Async
{
	struct Pose
	{
		float eye[3]{};
		float rotate[3][3]{};
		float zoom[2]{};  // the NiCamera frustum's half-extents: a map holds only at the zoom it was judged at
	};

	void Install(bool a_enabled);
	[[nodiscard]] bool Enabled() noexcept;

	// Any thread, during the cull: an entry the walk offered (judged by the worker after the cull).
	void Record(RE::NiAVObject* a_object, const RE::NiBound& a_bound, Hooks::CullGroups::GroupKind a_kind) noexcept;

	// Main thread, at the cull begin: publishes the worker's map when it finished; true when that map holds for a
	// camera at a_pose (within the margins the worker judged with, and a_viewHeld: the frame's view lies inside the
	// worker's), so this frame may hide by it.
	bool BeginFrame(const Pose& a_pose, bool a_viewHeld) noexcept;

	// Last frame's record for an object (any thread during the cull; the map is read-only then), or null, whether or
	// not the map holds for this frame's camera (BeginFrame's answer).
	[[nodiscard]] const Occlusion::Record* Find(const void* a_object) noexcept;

	// Main thread, at the cull stage's end: this frame's candidates and the worker's context (the frame's, dilated
	// by the move margin) go to the worker. a_pose is the camera the context was built for; a_viewWiden (NDC) how far
	// its view's most widened side lies past the frame's (for the log).
	void Submit(const Occlusion::FrameContext& a_context, const Pose& a_pose, float a_moveMargin, float a_viewWiden);

	// Main thread, before the frame's render ends: blocks until the worker is done with the frame's objects.
	void Wait() noexcept;

	// Forget both maps (load, mode switch). Waits for the worker first.
	void Reset() noexcept;

	void LogStats(std::uint32_t a_frames);
}
