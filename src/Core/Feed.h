#pragma once

// Policy for the previs fast path (PREVIS-FEED-PLAN.md §4; the engine facts are in Hooks/PrevisFeed). Each frame, at
// the cull begin, decides which path the frame takes:
//   - previs:  CBRO is off; the engine's previs lists are fed and CBRO's hooks are out (as before).
//   - classic: CBRO is on and judges the engine's full walk. With bPrevisFeed=1, previs is SUSPENDED (the engine's
//              own flush-free byte, the one its cascade cull flips around its sun feed) only inside two windows of a
//              CBRO frame: from Render_PreUI+0x157's return to the cull stage's end (so +0x15C runs the engine's
//              pre-cull helper and the walk, DrawWorld's cull and its jobs take their previs-off flow), and around
//              the cascade cull (its group-0 path, the reader of CBRO's sun verdicts). The rest of the frame (the
//              previs query at +0x80, the lamp shadow maps, the third view, cell loads, the game's update) sees
//              previs active, as vanilla: nothing is flushed, no state is held across frames, and F8 back to previs
//              mode is the engine untouched. (v1.31-v1.32 held the byte across the whole frame instead, fought the
//              engine's per-update release every frame, and left the lamp pass and cell loads with a state vanilla
//              never has.) With bPrevisFeed=0 this is v1.28's behaviour (Runtime switches previs off, which flushes).
//   - feed:    (not built yet: PREVIS-FEED-PLAN.md phase 3) previs stays active and CBRO fills the engine's lists.
// Also the completeness audit (plan §4.5): every iFeedAuditInterval frames on the classic path, CBRO's replica of the
// engine's scene walk (PrevisFeed::EnumerateCandidates) is compared with what the engine actually offered to
// Group::Add that frame. A "missing" object is one the engine filed that the replica would not have: the replica must
// have none before it may ever feed.

namespace CBRO::Core::Feed
{
	enum class Path : std::uint8_t
	{
		kPrevis,
		kClassic,
		kFeed,
	};

	void Install();       // after CullGroups and PrevisFeed are installed
	void OnGameLoaded();

	// Main thread at each cull begin, once the frame's mode is settled (after the hotkey and previs-request handling,
	// before the hooks are synced). Returns the frame's path.
	Path BeginFrame(bool a_cbroMode, std::uint32_t a_clock);
	// Main thread at the cull stage's end (DrawWorld's cull and its jobs done): closes an audit frame.
	void EndCull();

	[[nodiscard]] Path             Current() noexcept;
	[[nodiscard]] std::string_view PathName(Path a_path) noexcept;
	// For the frame-time line: what the engine's previs state is right now ("ACTIVE", "suspended by CBRO", "OFF").
	[[nodiscard]] std::string_view PrevisState() noexcept;
	// Whether Feed manages previs (bPrevisFeed=1): Runtime then never switches previs off.
	[[nodiscard]] bool ManagesPrevis() noexcept;

	void LogStats(std::uint32_t a_frames);
}
