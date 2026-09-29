#pragma once

// Policy for the previs fast path (PREVIS-FEED-PLAN.md §4; the engine facts are in Hooks/PrevisFeed). Each frame, at
// the cull begin, decides which path the frame takes:
//   - previs:  CBRO is off; the engine's previs lists are fed and CBRO's hooks are out (as before).
//   - classic: CBRO is on and judges the engine's full walk. With bPrevisFeed=1, previs is SUSPENDED for the whole
//              time CBRO is on (the engine's own flush-free switch, the one its cascade cull uses every frame) and
//              never switched off, so nothing is flushed and F8 back to previs mode gets full-strength previs at once.
//              Kept suspended across the whole frame rather than around the cull alone, so every reader of
//              IsActive() (the previs query at Render_PreUI+0x80, the helper at +0x15C, the third previs view, the
//              cascades) sees one consistent "previs off" frame, exactly as when previs is disabled, minus the flush.
//              With bPrevisFeed=0 this is v1.28's behaviour (Runtime switches previs off, which flushes it).
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
