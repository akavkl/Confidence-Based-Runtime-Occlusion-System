#pragma once

#include "Core/Async.h"
#include "Core/Occlusion.h"
#include "Core/RecorderFormat.h"
#include "Core/ShadowLights.h"
#include "Core/Verdict.h"

// Diagnostic (bRecorder, v1.95): a recording of CBRO's decisions, to be read offline (tools/recording.py) and,
// later, replayed. From the hotkey press to the next, every frame's camera, view box, sun, lamps and one level of the
// CPU-side Hi-Z go into the recording (up to iRecorderFrames frames: at the limit it saves itself), with every
// candidate CBRO decided on in the frame (the entry walk's
// cheap skips, the top-level decisions, the Block::Add children, node pruning, the main view's registrations in both
// modes and the ones CBRO refused), each with its verdict record, its left-out path and the flags that explain a
// decision made without a verdict (forced, inherited, the player, a sky mesh, a pending marker). Nothing CBRO decides
// changes. The hotkey (iRecorderHotkey, unbound by default) starts a recording and, pressed again, stops it and saves
// it; the save runs on its own thread (Data/CBRO_Recordings/CBRO-<date>-<time>.cbrorec) and reports through the HUD and
// the log.
//
// Cost: off, one relaxed load per decision. Recording, a wait-free append per candidate (~20k a frame outdoors), the
// per-frame hi-z copy, and at the cull's end the new objects' metadata (name, type, reference, bound: read on the main
// thread in the frame that saw them, so no stale pointer is ever read) behind a pointer table: about 1 ms a frame on
// the main thread, most of it in the first frames. Memory: ~0.8 MB a frame outdoors at hi-z level 2 (1800 frames: ~1.4 GB),
// held only while recording. The frames' candidate buffers are allocated and page-touched ahead on a helper thread
// (v1.96.1), so the main thread's appends land in faulted-in pages.
namespace CBRO::Core::Recorder
{
	using Origin = RecorderFormat::Origin;
	using RecorderFormat::Flags;

	void Install(bool a_enabled, std::uint32_t a_frames, std::int32_t a_hiZLevel, std::string a_version);
	[[nodiscard]] bool Enabled() noexcept;

	extern std::atomic<bool> g_on;
	// Any thread: a recording is running (the hooks note their decisions).
	[[nodiscard]] inline bool On() noexcept
	{
		return g_on.load(std::memory_order_relaxed);
	}

	// Main thread, at the cull begin (between two frames' decisions).
	void Toggle();  // the hotkey: start, or stop and save
	void Start();
	void Stop();    // ... and save on a worker thread
	[[nodiscard]] std::string Status();

	// ---- the walk's threads, inside DrawWorld's cull: one decision at a time per thread ---------------------------
	void BeginDecision() noexcept;                                          // a decision starts (its judge and path follow)
	void NoteJudged(const Occlusion::Record& a_record, std::uint8_t a_source) noexcept;  // JudgeOrLookup's record and source
	void NotePath(std::uint8_t a_path) noexcept;                            // RecordLeftOut's path
	void SetWalkSite(std::uint16_t a_site, std::uint16_t a_flags) noexcept;  // the entry walk, before the top-level decision: the loop its entry stands in (0: not the walk's) and its flags (moved, trace turn)
	void NoteFlags(std::uint16_t a_flags) noexcept;                         // flags of the decision in progress (the player, inherited)
	// The decision's end: the candidate, with the record and path noted since BeginDecision.
	void NoteEntry(Origin a_origin, const RE::NiAVObject* a_object, const RE::NiBound& a_bound, std::uint8_t a_kind, std::uint16_t a_flags, std::uint16_t a_site) noexcept;
	void NoteNode(const RE::NiAVObject* a_node, std::uint32_t a_index, bool a_skipped, std::uint32_t a_entries, std::uint16_t a_flags, const RE::NiBound& a_bound) noexcept;
	void NoteRegistration(const RE::NiAVObject* a_object, bool a_refused) noexcept;

	// ---- main thread, at the cull stage's end (both modes) --------------------------------------------------------
	struct FrameInfo
	{
		const Occlusion::FrameContext* context{ nullptr };
		const Async::Pose*             pose{ nullptr };  // the current camera (null: unknown)
		const ShadowLights::LampList*  lamps{ nullptr };
		std::uint64_t                  renderFrame{ 0 };
		std::uint32_t                  location{ 0 };
		int                            mode{ 0 };
		bool                           settled{ false };
		bool                           observeOnly{ false };
	};
	void EndFrame(const FrameInfo& a_info);
}
