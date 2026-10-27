#pragma once

// An object's per-frame verdict record (Core/Occlusion writes it, Core/Async carries it between frames). One record
// per object per judged frame: the view outcome with the evidence a later frame re-checks it by, the sun outcome, the
// confirmation streaks, and the bound it was judged with.
namespace CBRO::Core::Occlusion
{
	enum class Outcome : std::uint8_t
	{
		kNone,
		kOutside,  // entirely outside the current view (geometry only)
		kHidden,   // behind the depth (rejected once its streak reaches confirmFrames)
		kKept,     // drawn: visible, near, at the edge, exempt, a light that reaches in, or a bad bound
	};

	enum class SunOutcome : std::uint8_t
	{
		kNone,      // not evaluated (the main view needed the object anyway)
		kUnneeded,  // sun shadows off, or the shadow can't reach the view (geometry)
		kBehind,    // the shadow falls only behind visible surfaces (needs its streak)
		kNeeded,
		kUnknown,   // the sun's state couldn't be verified: needed
		kLampUnneeded,  // sun off: no shadow-casting lamp's shadow of it can reach a visible surface (never reused: re-judged each frame)
	};

	enum RecordFlags : std::uint8_t
	{
		kRecordDepth = 1 << 0,     // a hidden view outcome with its evidence recorded (rect, threshold, blocks)
		kRecordLight = 1 << 2,     // the object is a light (never skipped as out of view)
		kRecordNoCache = 1 << 3,   // never reuse (a light's reach test, a table-full case)
		kRecordSunEdge = 1 << 4,   // the sun outcome is "needed" only as the shadow may land where the depth frame has nothing (streak carried)
		kRecordSunFar = 1 << 5,    // the sun outcome is "needed", but not by the last cascade: nothing visible in its slab (v1.80)
		// (v1.92.5, diagnostic: the set-diff reasons, set only while Occlusion::SetKeptReasons is on)
		kRecordShaped = 1 << 1,        // the view outcome came through a mesh-shape test
		kRecordShapeNone = 1 << 6,     // a shape was wanted (big enough, a shape type) and none was used: the sphere decided
		kRecordParallaxKept = 1 << 7,  // kept only by the parallax margin: the same tests without it say hidden
	};

	// A cache line each (v1.92.1): the arrays came from new[] and vectors at 16-byte alignment, so every record lay
	// across two lines, and the walk's lookup and copy of the worker's record (written on another core) fetched both.
	struct alignas(64) Record
	{
		const void*   object{ nullptr };
		float         bound[4]{};         // the bound tested (before dilation)
		std::uint32_t viewEpoch{ 0 };
		std::uint32_t readback{ 0 };      // the depth a hidden view outcome holds against (its evaluation's, or its last re-check's)
		std::uint32_t sunReadback{ 0 };   // the depth the sun outcome read
		std::uint16_t rect[4]{};          // hidden evidence: level-0 texels x0, y0, x1, y1 that were all nearer than `threshold`
		float         threshold{ 0.0f };  // ... in buffer depth
		std::uint8_t  blocks[4]{ 255, 0, 0, 0 };     // Hi-Z blocks the evidence covers, widened (255 = none)
		std::uint8_t  sunBlocks[4]{ 255, 0, 0, 0 };  // Hi-Z blocks the sun's shadow test read, widened (255 = none)
		Outcome       outcome{ Outcome::kNone };
		std::uint8_t  verdict{ 0 };       // the Verdict behind the outcome (for the stats)
		SunOutcome    sun{ SunOutcome::kNone };
		std::uint8_t  sunEpoch{ 0 };
		std::uint8_t  streak{ 0 };        // consecutive frames hidden
		std::uint8_t  sunStreak{ 0 };     // consecutive frames with the shadow behind surfaces
		std::uint8_t  flags{ 0 };
		std::uint8_t  cellHint{ 0 };      // mesh shapes: the cell last seen visible (TestShape starts there)
	};
	static_assert(sizeof(Record) == 64);
}
