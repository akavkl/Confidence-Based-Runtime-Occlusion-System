#include "Core/Occlusion.h"

#include "Core/Recorder.h"

#include "Core/Async.h"
#include "Core/DropSet.h"
#include "Core/MeshProxy.h"
#include "Core/ShadowLights.h"
#include "Core/TraceFeed.h"
#include "Hooks/CullGroups.h"
#include "Hooks/PrevisFeed.h"
#include "Settings.h"
#include "Util/Gamebryo.h"

#include <immintrin.h>
#include <intrin.h>
#include <unordered_map>
#include <unordered_set>

namespace CBRO::Core::Occlusion
{
	namespace
	{
		// ---- tunables (copied from Settings once; read on the hot path) --------------------

		struct Tunables
		{
			std::uint32_t confirmFrames{ 2 };
			float         nearDistance{ 48.0f };
			float         depthTolerance{ 0.002f };  // relative, on linear view depth
			float         depthSlack{ 4.0f };        // absolute, game units
			float         depthMin{ 0.01f };
			float         depthRange{ 0.99f };
			bool          observeOnly{ false };
			bool          cullActors{ false };
			bool          cullPlayer{ false };  // the player's own 3D may be culled too (v1.89, bCullPlayer)
			bool          meshShapes{ true };
			bool          lampGroupTrim{ true };  // sun off: a group-0 entry the main view doesn't need is left out when no lamp's shadow of it can reach a visible surface
			bool          emptiedSpotTrim{ true };  // sun up: spot lamps whose shadow map stays empty keep no group-0 entry (v1.81)
			bool          spotOnlyCascadeTrim{ true };  // sun up: entries kept for spot lamps alone stay out of the sun's cascades (v1.82)
			bool          scolRootBounds{ true };  // static collection pieces judged and filed with the collection's bound (v1.83)
			bool          keepFadeState{ true };   // fade nodes left out of the main view keep their fade, as previs's would (v1.88.2)
			bool          entryWalk{ true };       // a node pruning can't leave out whole is walked by CBRO itself (v1.94, bEntryWalk)
			bool          cheapChildren{ true };   // a static mesh child out of the view that no view needs is left out by the cheap tests alone (v1.96, bChildCheapTests)
			bool          childInherit{ true };    // a child with its kept top entry's bound takes its decision, no judgement of its own (v1.98, bChildInherit)
			bool          stableIndex{ true };     // decision parents' records in filing order, found by a cursor (v1.99.2, bStableIndex)
			bool          traceDirect{ true };     // an entry filed for the trace alone is handed to the tracer, not filed (v1.99.3, bTraceDirect)
			bool          deferSunOnly{ true };    // a sun-only drop is filed into group 0 after the main view (v1.99.3, bDeferSunOnly)
			bool          walkLeases{ true };      // clusters of walked entries out of the view are left untouched between walks (v1.98, bWalkLeases)
			std::uint32_t leaseFrames{ 4 };        // ... a cluster is walked again at most this many frames after its last walk (iWalkLeaseFrames)
			float         leaseMargin{ 64.0f };    // ... units a cluster's sphere must clear the view by on top of the verdict's margins, for what moves meanwhile (fWalkLeaseMargin)
			float         viewOverhang{ 0.03f };  // objects and sun shadows: how far (NDC) the view may overhang the depth frame and still be judged by its edge
			float         viewOverhangTurn{ 0.0f };  // ... up to this much where a turned view reaches that far past it (FrameContext::overhang)
		};
		Tunables g_tunables;

		// A Hi-Z texel that isn't clearly nearer is re-checked at up to this many finer levels.
		constexpr std::uint32_t kRefineLevels = 3;
		// ... and when a cached hidden verdict's evidence is re-checked against a newer depth (ViewValid): fewer, since a
		// failure only means the full test runs.
		constexpr std::uint32_t kRecheckRefine = 2;
		// Sun shadows (TestSun): a sweep whose screen box is wider than kSunPieceExtent (NDC) and needed as a whole is
		// tested again in up to kSunPieces equal stretches.
		constexpr int   kSunPieces = 8;
		constexpr float kSunPieceExtent = 0.25f;
		// Light tests (TestSphere): how far (NDC) the current view may overhang the depth frame and still be judged,
		// the overhang taken to hold what the frame's adjacent edge holds. A camera swaying by a fraction of a degree
		// (first-person idle motion) leaves such a strip every few frames; judged unknown, every lamp crossing it
		// dropped out of its emptied state and re-drew its whole shadow map for two frames. 0.03 is ~38 px at 2560.
		// Objects and sun shadows take fViewOverhang (v1.61, default the same): in the v1.60 wobble runs every object
		// and shadow crossing the strip stayed drawn (edge ~700-1,100 a frame, sun "needed" 2,600-4,100 against ~400),
		// then re-confirmed, so the draw count swung by thousands with each small turn. (kLightOverhang: Occlusion.h)
		// A cached kept (drawn) verdict is re-evaluated every this many frames (staggered by object), never sooner: no
		// depth change can make drawing an object unsafe, so this only bounds how long a newly hidden one stays drawn.
		constexpr std::uint32_t kKeptRecheckFrames = 16;
		// Meshes at least this big (bound radius) get a shape test when their sphere can't be settled.
		constexpr float kShapeMinRadius = 200.0f;
		// Merged meshes with more instances than this are only judged by their whole bound.
		constexpr std::uint32_t kMaxInstanceTests = 1024;

		std::atomic<bool>           g_active{ false };
		std::atomic<bool>           g_observeOverride{ false };
		std::array<FrameContext, 2> g_contexts{};
		std::atomic<int>            g_context{ -1 };
		std::atomic<bool>           g_resetRequested{ false };

		// Classification is redone every epoch so lights attached after the first look are noticed.
		constexpr std::uint32_t kEpochFrames = 600;

		// ---- stats: one block per thread, single writer, summed at log time -------------------
		// The tests run on the main thread and every BSJobs worker; shared atomic counters would
		// bounce one cache line between all of them for every object.

		enum Counter : std::size_t
		{
			kTested,
			kRejected,
			kWouldReject,
			kConfirming,
			kVisible,
			kEdge,
			kOutside,
			kBehind,
			kNear,
			kInvalid,
			kExemptType,
			kExemptLight,
			kExemptActor,
			kExemptPlayer,  // offers of the player's 3D left untested (bCullPlayer=0)
			kLightsTested,
			kLightsRejected,
			kNoContext,
			kTableFull,
			kMerged,
			kMergedRejected,
			kMergedConfirming,
			kInstanceTests,
			kInstanceEntries,
			kInstanceEntriesRejected,
			kFramesCulling,
			kFramesBlocked,
			kSkippedHidden,
			kSkippedOutside,
			kShapeTests,
			kShapeHidden,
			kShapeOutside,
			kShapeCells,
			kShapeMoving,
			kShapeAnimated,
			kShapeNoTransform,
			kShared,             // entries of an unknown group or group 0 offered to the filter
			kDropInherited,      // shared entries dropped with their parent (no test)
			kDropFull,           // drops the drop table had no room for (the object stays drawn)
			kRegistered,         // main-view registrations while DrawWorld culled
			kRegistrationsDropped,
			kSunTests,           // group-0 objects the main view doesn't need, whose sun shadow was tested
			kSunOutside,         // ... shadow can't reach the view (or the cascade range)
			kSunHidden,          // ... shadow only falls behind visible surfaces (confirmed)
			kSunConfirming,
			kSunNeeded,
			kSunOff,             // ... sun shadows off this frame, and no lamp known (or trimming off): kept in group 0
			kSunLampUnneeded,    // ... sun off: no recorded lamp's shadow of it can reach a visible surface (left out of group 0)
			kSunLampNeeded,      // ... sun off: some lamp's shadow of it may (kept)
			kSunLampTests,       // ... lamp shadow-volume tests run for the above
			kCasterRejected,     // group-0 entries rejected in every view (neither the main camera nor the sun needs them)
			kCasterSkipped,      // group-0 top-level objects never filed, for the same reason
			kSpotLampKept,       // group-0 entries the sun doesn't need, kept for a spot lamp within reach (sun up, v1.54)
			kSpotLampNodesKept,  // ... cell nodes walked though outside the view and the sun's reach, for the same reason
			kSpotLampsUnknown,   // ... entries kept because the spot lamps weren't all known (hook missing, list overflow)
			kSpotLampsEmptied,   // spot lamps left out of that list: shadow map emptied with a steady lit volume, still hidden (v1.81)
			kSpotLampsLeftLit,   // ... of those, the lamp's map was drawn after all (its casters may have been missing)
			kSpotOnlyLeftOut,    // kSpotLampKept entries left out of the sun's cascades (v1.82)
			kSpotOnlyRegistrations, // ... the cascades' registrations left out for them (their geometry)
			kTraceFiled,          // objects CBRO would leave out of every view, filed for the trace on their turn (Core/TraceFeed, v1.90)
			kTraceMoved,          // ... in a frame their bound moved
			kTraceInherited,      // main-only children of an entry filed so, dropped with it
			kTraceCellNodes,      // cell nodes walked on their turn though pruning would have left them out whole
			kTraceForced,         // out-of-view entries no view files, filed force-visible for the trace on their turn (v1.91)
			kTraceForcedChildren, // ... their children, given the mark with them
			kScolWidened,         // static collection pieces judged and filed with their collection's bound (v1.83)
			kScolRootHits,        // ... their collection root taken from the cache, its parent chain checked (v1.98)
			kScolSlotHits,        // ... parents found in the cache with a root (the chain then checked; v1.98.1)
			kScolChainFails,      // ... of those, the chain check failed: the root found again
			kCellNodesSeen,       // cells' child node 3 (precombined chunks + static refs) offered by the scene walk
			kCellNodesSkipped,    // ... left out whole: outside the view, sun shadow can't reach it
			kCellNodesInView,     // ... walked: the node's bound meets the view
			kCellNodesSunNeeded,  // ... walked: outside the view, but its sun shadow may reach the view
			kCellNodesOther,      // cells' child node 9 offered (never pruned: its role is unknown)
			kRootsSeen,           // roots registered with DrawWorld's cull, offered by its root loop (pruned like node 3)
			kContainersSeen,      // exact-NiNode containers under a cell's node 3, offered by the walk's container loops
			kCellNodesHeld,       // ... holding an always-draw entry or an actor (or too big to scan): walked
			kCellNodeEntriesSkipped,  // entries never filed because their node was skipped whole
			kCellNodesCached,         // ... of the skipped nodes, answered from their clusters' cached spheres, no entry touched (v1.98)
			kCellNodeEntriesCached,   // ... their entries
			kCellNodeClustersRescanned, // ... clusters of such nodes read again on the way (lease due, held, or the array changed; v1.98.1)
			kCellNodesAlwaysDraw, // ... the node's own flags carry always-draw (counted only)
			kWalkNodes,           // nodes walked by CBRO itself (bEntryWalk, v1.94): their entries filed by CBRO, the engine's loop skipped
			kWalkEntries,         // ... their entries seen (null and AppCulled children apart)
			kWalkSkipped,         // ... left out by the cheap tests alone (unmoved, out of view, shadow beyond reach, no lamp, not the trace's turn)
			kWalkOffered,         // ... offered to the top-level decision (SkipTopLevel), as the engine's Group::Add would have
			kWalkFiled,           // ... of them, filed with the engine
			kWalkMarkerFiled,     // ... filed unjudged: a group marker was pending on the entry's block
			kWalkClusters,        // clusters of walked nodes' slots seen (v1.98, bWalkLeases)
			kWalkClustersDormant, // ... left out whole on their cached sphere: within their lease, nothing held, outside the view, shadow unneeded
			kWalkLeased,          // ... their entries, untouched (counted in kWalkEntries too)
			kWalkLeaseFadeVisits, // ... of those, visited after all for a fade mid-way (KeepFadeSeen)
			kWalkLeaseTraceVisits, // ... or for the trace's turn within reach (walked as an entry)
			kWalkLeaseMarkerHeld, // clusters walked after all: a group marker was pending (the next entry filed must take it)
			kWalkClusterNotWalked,    // clusters walked, not leased, because (v1.98.1): the walk never finished them (the pruning scan did)
			kWalkClusterHeldBad,      // ... a member's bound was unusable, or moved, at the last walk
			kWalkClusterHeldKind,     // ... a member is always-draw, a light, a sky mesh, an actor root or the player's 3D
			kWalkClusterHeldMarker,   // ... a member was filed unjudged for a pending marker
			kWalkClusterDue,          // ... the lease ran out (the scheduled re-read)
			kWalkClusterSlotsChanged, // ... the child array changed under the slots
			kWalkClusterInView,       // ... the cached sphere meets the view (with the margins)
			kWalkClusterShadow,       // ... the sun's shadow of the sphere may reach the view
			kWalkClusterLamp,         // ... a spot lamp may need the sphere
			kWalkCulled,              // AppCulled children the walk read the flags of (never entries; v1.98.1)
			kWalkClusterNoMembers,    // clusters walked, not leased: no entry of their own (containers only; v1.99)
			kStableRecorded,          // top-level node decisions recorded in filing order (v1.99.2, bStableIndex)
			kStableParentHits,        // children's decision parents found by the cursor (once per parent)
			kStableParentNotRecorded, // ... not recorded this frame (a nested node, a group filed otherwise; v1.99.3: no scan)
			kStableParentOutOfOrder,  // ... recorded, not found within the look-ahead (the tables answered)
			kTraceHanded,             // top-level entries handed to the tracer directly, not filed (v1.99.3, bTraceDirect)
			kTraceHandedPieces,       // ... the geometry handed with them
			kSunOnlyDeferred,         // sun-only drops held back from the main view, filed into group 0 later (v1.99.3, bDeferSunOnly)
			kSunOnlyHandedPieces,     // ... their geometry handed to the tracer
			kSunOnlyFiledLate,        // ... filed in the cascade cull's window
			kSunOnlyDropped,          // ... never filed: no cascade window opened in their frame
			kSunOnlyKnown,            // ... their pieces not handed again: the tracer keeps them (v2.0, the link)
			kTraceKnownSkipped,       // objects left out on their turn: the tracer keeps them from an earlier handover (v2.0)
			kTraceMoverHeld,          // movers past the tracer's mover radius, not their frame (v2.0)
			kTraceChildHanded,        // pieces filed for the trace alone, handed through the link instead (v2.0)
			kTraceGone,               // entries the walk had handed that left their slot: GONE records (v2.0)
			kWalkFadeRoots,       // roots that are fade nodes: left to the engine, which files them whole into the group array
			kWalkCachesRebuilt,   // slot caches rebuilt (a node's child count changed)
			kLampVolumeOutside,   // lamp casters whose shadow volume lies outside the view (or beyond the lamp's reach)
			kLampVolumeMisses,    // ... lies entirely behind, or entirely in front of, every visible surface (confirmed)
			kLampVolumeConfirming,
			kLampVolumeNeeded,
			kLampVolumeUnknown,
			kCacheHits,           // verdict cache: outcomes reused from last frame's record
			kCacheNew,            // ... objects with no record in last frame's stream (new, or the order changed)
			kCacheEpoch,          // ... re-evaluated: the camera left the epoch
			kCacheBound,          // ... re-evaluated: the object's bound changed
			kCacheDepth,          // ... re-evaluated: a Hi-Z block it read changed
			kCacheMoved,          // ... re-evaluated: a hidden outcome judged from a camera position this one has moved from (v1.88.1)
			kCacheNoCache,        // ... re-evaluated: the record was marked never-reuse (a light's reach, table full)
			kCacheRecheck,        // hidden verdicts reused after their evidence was re-checked against the current depth (their blocks had changed)
			kCacheKeptRecheck,    // kept (drawn) verdicts re-evaluated on their periodic turn (never because the depth changed)
			kCacheSunDepth,       // sun outcomes re-evaluated under a reused view outcome (the blocks their shadow test read changed)
			kStreakFromBackup,    // hidden verdicts without last frame's record whose streak the table's backup continued
			kOverhangHidden,      // objects hidden with part of them in a thin overhang of the view, judged by the depth frame's edge (fViewOverhang)
			kSunOverhang,         // sun shadows found behind surfaces the same way
			kParallaxWidened,     // tests whose screen box widened for a near occluder's parallax since the depth frame (v1.88)
			kFadeKeptSeen,        // fade nodes left out (hidden or out of view) whose frame counter was stamped (v1.88.2; on their turn since v1.92)
			kSunPiecewise,        // sun shadows found behind surfaces stretch by stretch where the whole sweep's box said needed (v1.79)
			kSunFarTests,         // needed sun shadows tested again for the last cascade alone (v1.80)
			kSunFarUnneeded,      // ... that find nothing visible in its slab
			kSunFarLeftOut,       // group-0 entries the main view doesn't need left out of the last cascade
			kSunFarRegistrations, // the last cascade's registrations left out for them (their geometry)
			kWalkJudged,          // async frames: entries the worker's map couldn't answer, judged in the walk (counted twice: the worker judges them too)
			kWalkUnsettled,       // ... of the map's answers, edge and confirming verdicts judged again in the walk (v1.71)
			kWalkSunEdge,         // ... and sun outcomes needed only for the map's no-depth strip, judged again in the walk (v1.73)
			kFinalEdge,           // the walk's decisions (v1.74): kept as edge
			kFinalConfirming,     // ... hidden, still confirming
			kFadeTopOffered,      // survey (v1.92.4; 1 node in kFadeStampFrames, the log scales it): top-level fade nodes offered to the group filter
			kFadeTopKept,         // ... let through (filed)
			kFadeTopKeptEntries,  // ... their geometry records (+0x178): the ChildPush adds they bring
			kFadeTopKeptOne,      // ... let through with exactly one record
			kFadeTopFadingOut,    // ... past the far fade range by distance, currentFade still above 0 (fading out: drawn)
			kFadeTopPastFarBit15, // ... past the far range with flag bit 15 set (filed and drawn whatever the fade)
			kFadeTopFaded,        // ... past the far range at currentFade 0, bit 15 clear: the engine files no mesh of them
			kFadeTopFadedKept,    // ... of those, let through by CBRO (a Group::Add, a verdict, a frustum test and OnVisible for nothing drawn)
			kFadeTopFadedKeptEntries,  // ... their geometry records
			kFadeTopLightingBit33,     // let-through fade nodes with flag bit 33 set: OnVisible re-queues their lighting (v1.94 survey)
			kFadeTopMidFade,           // ... with 0 < currentFade < 1: the same
			kFadeTopNoLighting,        // ... with flag bit 51 set: never re-queued
			kFrameLightingByte,        // frames with the scene node's byte +0x228 set: every visible fade node's lighting re-queued
			kFadeTopLightingBit33Forced,  // ... of the bit-33 ones, filed force-visible (for the trace, or the player's) (v1.94.1)
			kFadeTopMidFadeForced,        // ... of the mid-fade ones, the same
			kFadeTopMidFadeIn,            // ... of the mid-fade ones, fading in (flag bit 37 "complete" clear) (v1.96.1)
			kFadeTopMidFadeInBig,         // ... of those, bound radius +0x1A8 at or above 600: the engine's unseen rule never snaps them to 1
			kFadeAdvanced,                // fade updates run by CBRO on left-out fade nodes whose fade is not complete (v1.97.0)
			kFadeStarted,                 // ... of those, fades that began (currentFade left 0)
			kFadeCompleted,               // ... of those, fades that completed (bit 37 set by the update)
			kFadeStartUnseen,             // let-through fade nodes at currentFade 0, bit 37 clear, frameCounter over 20 frames old: the unseen rule decides at this OnVisible (v1.96.2)
			kFadeStartUnseenNear,         // ... of those, distance fade above fFadeInThreshold: snapped to 1, no fade-in
			kFadeStartSeenLast,           // let-through fade nodes at currentFade 0, bit 37 clear, stamped last frame: OnVisible itself kept them at 0 (far) until now
			kFadeStartSeenRecent,         // ... stamped 2-20 frames ago: a stamp while left out of the main view (CBRO's, or a culling process's OnVisible)
			kFadeStartSeenNear,           // ... of the two above, distance fade above fFadeInThreshold: a fade-in from 0 runs on screen from this frame
			kFadeTopMidFadeInContinuous,  // mid-fade fading-in nodes stamped last frame (the fade ran on screen last frame too)
			kFadeTopMidFadeInResumed,     // ... stamped 2 or more frames ago: left out mid-fade, now back
			kFadeTopMidFadeInActor,       // ... of the fading-in nodes, type 3 (actors: currentFade follows fadeAmount directly)
			kChildOfFade,         // Block::Add entries not decided at the top level (the engine's child pushes, unjudged filings): under a fade node (v1.94.1)
			kChildOfNode,         // ... under an exact NiNode (a chunk container, a cell's node 9)
			kChildOther,          // ... under any other node, or none
			kChildCheapRejected,  // static geometry children out of the view that no view needs, left out by the cheap tests alone, no verdict (v1.96)
			kChildCheapOutside,   // ... the same in a main-only group: left to the engine's frustum test
			kChildInherited,      // children with their kept top entry's bound, decided with it, no judgement (v1.98, bChildInherit)
			kChildInheritedFade,  // ... of them, reached through the fade node filing them (their parent one of its sub-nodes; v1.98.1)
			kChildFadeDecided,    // children dropped or forced with the fade node filing them, their parent one of its sub-nodes (v1.99)
			kCycles,             // TSC cycles in the per-object tests (sampled frames, scaled)
			kCyclesEvaluate,      // ... of which: full view evaluations (sphere tests, mesh shapes, lights)
			kCyclesShape,         // ... of which: the mesh-shape tests inside those evaluations
			kCyclesSun,           // ... of which: sun-shadow evaluations
			kCyclesLookup,        // ... of which: last frame's record looked up (the thread's stream, or the worker's map)
			kCyclesReuse,         // ... of which: whether a record still holds (epoch, bound, depth blocks, re-check)
			kCyclesRecheck,       // ... of kCyclesReuse: hidden verdicts' evidence re-checked against the current depth
			kCyclesRecord,        // ... of which: this frame's record written (the stream, the table's streak backup)
			kCyclesNodeScan,      // ... of which: cell-node pruning's scans of a node's entries
			kCyclesJudge,         // ... of which: the verdict path as a whole (JudgeOrLookup: lookup, checks, evaluations, copy; v1.92)
			kCyclesEntryWalk,     // the entry walk in all (not part of kCycles; its entries' decisions and filings count there too; v1.94)
			kCounterCount
		};

		struct alignas(64) ThreadStats
		{
			std::array<std::atomic<std::uint64_t>, kCounterCount> values{};
		};
		constexpr std::size_t                  kStatSlots = 128;
		std::array<ThreadStats, kStatSlots>    g_threadStats{};
		std::atomic<std::uint32_t>             g_nextStatSlot{ 0 };
		std::array<std::uint64_t, kCounterCount> g_reported{};  // totals at the last LogStats (main thread)
		std::uint64_t                          g_cyclesReported{ 0 };
		ThreadStats*                           g_mainStats{ nullptr };  // the main thread's block (set at BeginFrame)
		std::uint64_t                          g_mainCyclesReported{ 0 };
		std::array<std::uint64_t, kCounterCount> g_mainReported{};  // the main thread's block at the last LogStats

		ThreadStats& LocalStats() noexcept
		{
			// Constant-initialized, so no per-access thread-local init guard (a dynamic initializer would cost one
			// on every Bump; there are several per object tested).
			thread_local ThreadStats* stats = nullptr;
			if (!stats) [[unlikely]] {
				stats = &g_threadStats[g_nextStatSlot.fetch_add(1, std::memory_order_relaxed) % kStatSlots];
			}
			return *stats;
		}

		void Bump(Counter a_counter, std::uint64_t a_amount = 1) noexcept
		{
			auto& value = LocalStats().values[a_counter];
			value.store(value.load(std::memory_order_relaxed) + a_amount, std::memory_order_relaxed);
		}

		std::uint64_t Total(Counter a_counter) noexcept
		{
			std::uint64_t total = 0;
			for (const auto& slot : g_threadStats) {
				total += slot.values[a_counter].load(std::memory_order_relaxed);
			}
			return total;
		}

		// Cycles -> milliseconds, calibrated against QPC between log calls.
		struct TscCalibration
		{
			std::uint64_t tsc{ 0 };
			std::int64_t  qpc{ 0 };
			double        cyclesPerMs{ 0.0 };
		};
		TscCalibration g_calibration;

		void Calibrate() noexcept
		{
			LARGE_INTEGER now{}, frequency{};
			QueryPerformanceCounter(&now);
			QueryPerformanceFrequency(&frequency);
			const auto tsc = __rdtsc();
			if (g_calibration.qpc != 0 && now.QuadPart > g_calibration.qpc) {
				const double ms = static_cast<double>(now.QuadPart - g_calibration.qpc) * 1000.0 / static_cast<double>(frequency.QuadPart);
				if (ms > 100.0) {
					g_calibration.cyclesPerMs = static_cast<double>(tsc - g_calibration.tsc) / ms;
				}
			}
			g_calibration.tsc = tsc;
			g_calibration.qpc = now.QuadPart;
		}

		// ---- per-object history: lock-free open addressing keyed by NiAVObject* ------------
		// value bits: [0,32) clock of the last verdict, [32,40) hidden streak, 40 classified,
		//             41 exempt, [42,44) exemption reason, 44 light, 45 merge-instanced,
		//             46 merged mesh rejected at `clock`, [48,56) classification epoch,
		//             [56,64) merged mesh: index of the instance last seen visible (search hint)

		constexpr std::size_t   kTableBits = 18;  // (v1.75: 17 filled to 70% twice in six exterior minutes of the v1.74 run)
		constexpr std::size_t   kTableSize = std::size_t{ 1 } << kTableBits;
		constexpr std::size_t   kMaxProbe = 64;
		constexpr std::uint64_t kLowBits = 0xFFFFFFFFFFull;  // clock + streak
		constexpr std::uint64_t kClassified = std::uint64_t{ 1 } << 40;
		constexpr std::uint64_t kExempt = std::uint64_t{ 1 } << 41;
		constexpr std::uint64_t kIsLight = std::uint64_t{ 1 } << 44;
		constexpr std::uint64_t kIsMerged = std::uint64_t{ 1 } << 45;
		constexpr std::uint64_t kMergedReject = std::uint64_t{ 1 } << 46;
		constexpr int           kEpochShift = 48;
		constexpr int           kHintShift = 56;
		// Entry::lamp bits [40,60) (v1.92.5, diagnostic, SetKeptReasons on only): bit 40 recorded, [41,44) the Verdict, [44,46) the
		// Outcome, 46 kRecordShaped, 47 kRecordShapeNone, 48 kRecordParallaxKept, [50,60) the clock's low ten bits (read back
		// only in the frame that wrote them). The lamp streak keeps the word's low 40 bits.
		constexpr int           kReasonShift = 40;
		constexpr std::uint64_t kReasonMask = ((std::uint64_t{ 1 } << 20) - 1) << kReasonShift;
		std::atomic<bool>       g_keptReasons{ false };

		struct alignas(32) Entry
		{
			std::atomic<std::uintptr_t> key{ 0 };
			std::atomic<std::uint64_t>  value{ 0 };
			std::atomic<std::uint64_t>  sun{ 0 };   // sun-shadow streak: [0,32) clock of the last hidden verdict, [32,40) streak
			std::atomic<std::uint64_t>  lamp{ 0 };  // lamp-shadow streak (any lamp): same layout
		};
		static_assert(sizeof(Entry) == 32);
		std::unique_ptr<Entry[]>   g_table;
		std::atomic<std::uint32_t> g_used{ 0 };

		Entry* FindEntry(std::uintptr_t a_key) noexcept
		{
			const auto hash = static_cast<std::size_t>(((a_key >> 4) * 0x9E3779B97F4A7C15ull) >> (64 - kTableBits));
			for (std::size_t probe = 0; probe < kMaxProbe; ++probe) {
				auto& entry = g_table[(hash + probe) & (kTableSize - 1)];
				auto  current = entry.key.load(std::memory_order_acquire);
				if (current == a_key) {
					return &entry;
				}
				if (current == 0) {
					if (entry.key.compare_exchange_strong(current, a_key, std::memory_order_acq_rel)) {
						g_used.fetch_add(1, std::memory_order_relaxed);
						return &entry;
					}
					if (current == a_key) {
						return &entry;
					}
				}
			}
			return nullptr;
		}

		// The line an object's entry starts on, fetched ahead (v1.98): the table is 8 MB, a miss on nearly every first probe
		// (0.12 ms a frame listed by the sampler on the main thread, 2026-10-27), and the walk knows the object well before
		// the verdict path's streak backups and classification look it up.
		inline void PrefetchEntry(const void* a_object) noexcept
		{
			if (g_table) {
				const auto key = reinterpret_cast<std::uintptr_t>(a_object);
				const auto hash = static_cast<std::size_t>(((key >> 4) * 0x9E3779B97F4A7C15ull) >> (64 - kTableBits));
				_mm_prefetch(reinterpret_cast<const char*>(&g_table[hash]), _MM_HINT_T1);
			}
		}

		// FindEntry without adding one (the sun streak's backup: only objects the view already judged hidden have it).
		Entry* PeekEntry(std::uintptr_t a_key) noexcept
		{
			const auto hash = static_cast<std::size_t>(((a_key >> 4) * 0x9E3779B97F4A7C15ull) >> (64 - kTableBits));
			for (std::size_t probe = 0; probe < kMaxProbe; ++probe) {
				auto&      entry = g_table[(hash + probe) & (kTableSize - 1)];
				const auto current = entry.key.load(std::memory_order_acquire);
				if (current == a_key) {
					return &entry;
				}
				if (current == 0) {
					return nullptr;
				}
			}
			return nullptr;
		}

		void ClearTable()
		{
			for (std::size_t i = 0; i < kTableSize; ++i) {
				g_table[i].key.store(0, std::memory_order_relaxed);
				g_table[i].value.store(0, std::memory_order_relaxed);
				g_table[i].sun.store(0, std::memory_order_relaxed);
				g_table[i].lamp.store(0, std::memory_order_relaxed);
			}
			g_used.store(0);
		}

		// The table filling up (v1.75): objects not judged for a while are dropped and the rest re-inserted, so the streaks
		// in play survive. Up to v1.74 a full table was cleared, and every hidden object confirmed again at once (the
		// v1.74 run: main view up to 7,092 registrations, the other views 30,022, twice in six minutes). An entry is
		// recent when any of its clocks (the view streak's, the sun's, the lamp's: each stored with its every use) is
		// within kTableKeepFrames. Main thread at the frame begin, as the clear: nothing else reads the table then.
		constexpr std::uint32_t kTableKeepFrames = 600;

		std::size_t CompactTable(std::uint32_t a_clock)
		{
			struct Kept
			{
				std::uintptr_t key;
				std::uint64_t  value, sun, lamp;
			};
			std::vector<Kept> kept;
			kept.reserve(g_used.load());
			const auto recent = [&](std::uint64_t a_field) {
				const auto clock = static_cast<std::uint32_t>(a_field);
				return clock != 0 && a_clock - clock <= kTableKeepFrames;
			};
			for (std::size_t i = 0; i < kTableSize; ++i) {
				const auto& entry = g_table[i];
				const auto  key = entry.key.load(std::memory_order_relaxed);
				if (!key) {
					continue;
				}
				const auto value = entry.value.load(std::memory_order_relaxed);
				const auto sun = entry.sun.load(std::memory_order_relaxed);
				const auto lamp = entry.lamp.load(std::memory_order_relaxed);
				if (recent(value) || recent(sun) || recent(lamp)) {
					kept.push_back({ key, value, sun, lamp });
				}
			}
			ClearTable();
			for (const auto& item : kept) {
				if (const auto entry = FindEntry(item.key)) {
					entry->value.store(item.value, std::memory_order_relaxed);
					entry->sun.store(item.sun, std::memory_order_relaxed);
					entry->lamp.store(item.lamp, std::memory_order_relaxed);
				}
			}
			return kept.size();
		}

		// ---- main-view drops (shared groups) -------------------------------------------------------
		// Entries of a shared group (group 0: the sun's shadow cascades read it too) stay in the group; the
		// ones the main view needn't draw this frame are recorded here, and the main accumulator's
		// registration of them is skipped (MainView).
		DropSet       g_drops;
		DropSet       g_farDrops;  // group-0 entries the sun's last cascade doesn't need this frame (v1.80)
		DropSet       g_sunDrops;  // group-0 entries kept for spot lamps alone: no cascade needs them this frame (v1.82)
		DropSet       g_traceForced;  // entries filed force-visible for the trace this frame, out of view, and their children (v1.91)
		std::uint32_t g_dropCycle{ 0 };  // clock / kTagCycle at the last frame start (main thread)

		// ---- object types -----------------------------------------------------------------------
		// Static scenery (precombined chunks, reference roots, their nodes and meshes) and point/spot
		// lights may be rejected. A light is only rejected when its whole influence sphere is behind
		// visible surfaces, i.e. it can't reach a single visible pixel. Directional/ambient lights,
		// particles, instanced meshes and everything under an Actor stay with the engine.
		// BSSubIndexTriShape (segmented meshes) and BSDynamicTriShape are mostly actor parts, which the
		// actor check keeps; the rest are static scenery (v1.5 run: ~200-400 hidden per frame were kept).

		constexpr std::array kCullableTypes{
			"NiNode"sv,
			"BSFadeNode"sv,
			"BSLeafAnimNode"sv,
			"BSMultiBoundNode"sv,
			"BSTriShape"sv,
			"BSMeshLODTriShape"sv,
			"BSMergeInstancedTriShape"sv,
			"BSCombinedTriShape"sv,
			"BSLODTriShape"sv,
			"BSSubIndexTriShape"sv,
			"BSDynamicTriShape"sv,
			"NiPointLight"sv,
			"NiSpotLight"sv,
		};

		enum TypeFlags : std::uint8_t
		{
			kTypeKnown = 1 << 0,
			kTypeCullable = 1 << 1,
			kTypeLight = 1 << 2,
			kTypeLeafAnim = 1 << 3,  // its meshes sway: never judged by their shape
		};

		constexpr std::size_t                               kTypeSlots = 512;
		std::array<std::atomic<std::uintptr_t>, kTypeSlots> g_typeKeys{};
		std::array<std::atomic<std::uint8_t>, kTypeSlots>   g_typeFlags{};

		std::uint8_t ClassifyTypeName(std::string_view a_name) noexcept
		{
			std::uint8_t flags = kTypeKnown;
			if (std::ranges::find(kCullableTypes, a_name) != kCullableTypes.end()) {
				flags |= kTypeCullable;
			}
			if (a_name.find("Light"sv) != std::string_view::npos) {
				flags |= kTypeLight;
			}
			if (a_name == "BSLeafAnimNode"sv) {
				flags |= kTypeLeafAnim;
			}
			return flags;
		}

		std::uint8_t TypeOf(const RE::NiAVObject* a_object) noexcept
		{
			const auto vtable = Util::TryReadVtable(a_object);
			if (!vtable) {
				return kTypeKnown;
			}
			const auto hash = static_cast<std::size_t>(((vtable >> 3) * 0x9E3779B97F4A7C15ull) >> 55) % kTypeSlots;
			for (std::size_t probe = 0; probe < kTypeSlots; ++probe) {
				const auto i = (hash + probe) % kTypeSlots;
				auto       current = g_typeKeys[i].load(std::memory_order_acquire);
				if (current == vtable) {
					if (const auto flags = g_typeFlags[i].load(std::memory_order_acquire)) {
						return flags;
					}
					break;  // another thread is classifying it: decide locally
				}
				if (current == 0) {
					char name[64]{};
					Util::TryGetRTTIName(a_object, name, sizeof(name));
					const auto flags = ClassifyTypeName(name);
					if (g_typeKeys[i].compare_exchange_strong(current, vtable, std::memory_order_acq_rel)) {
						g_typeFlags[i].store(flags, std::memory_order_release);
						logger::info(
							"occlusion: type {} -> {}{}", name[0] ? name : "(unknown)",
							(flags & kTypeCullable) ? "may be culled" : "never culled", (flags & kTypeLight) ? " (light)" : "");
						return flags;
					}
					if (current == vtable) {
						return flags;
					}
				}
			}
			char name[64]{};
			Util::TryGetRTTIName(a_object, name, sizeof(name));
			return ClassifyTypeName(name);
		}

		// userData of a reference's 3D root points at the TESObjectREFR.
		int FormTypeOf(std::uintptr_t a_form) noexcept
		{
			__try {
				return static_cast<int>(reinterpret_cast<const RE::TESForm*>(a_form)->GetFormType());
			} __except (EXCEPTION_EXECUTE_HANDLER) {
				return -1;
			}
		}

		bool IsActorRoot(const RE::NiAVObject* a_object) noexcept
		{
			return a_object->userData && FormTypeOf(a_object->userData) == static_cast<int>(RE::ENUM_FORM_ID::kACHR);
		}

		bool InsideActor(const RE::NiAVObject* a_object) noexcept
		{
			auto node = a_object;
			for (int depth = 0; node && depth < 32; ++depth) {
				if (IsActorRoot(node)) {
					return true;
				}
				node = node->parent;
			}
			return false;
		}

		// The player (bCullPlayer=0, v1.89). Its third-person 3D (PlayerCharacter::Get3D(false), notes 7.17) and everything
		// under it are never left out of any view or shadow map. A full-body first-person mod shows this same root in first
		// person (True First Person also hangs its shadow clone under it, notes 2.3), and a path tracer that builds its scene
		// from each frame's registrations loses a piece in the frames nothing files it: Renderer Replacer removes the
		// player's pieces after two such frames, and the player's traced shadows lost a piece whenever CBRO judged it hidden
		// or out of view with no sun shadow needed (group 0's rejection from every view), or trimmed it from a lamp's map.
		// The root comes with the frame context (read on the main thread by BeginFrame, compared by address only, its bound
		// copied). A node's world bound holds its children's, so the parent walk runs only for objects whose bound meets it.
		// v1.89.1: not judging them was not enough. Previs files the player (one of its dynamic objects) force-visible, so
		// the main view registers every piece whatever the frustum says; in a CBRO frame the frustum test ran, and a piece
		// out of view (or the head, its sphere around the eye) went unfiled, its traced shadow coming and going with the
		// camera's angle. Their main-view entries get previs's mark back (Hooks/CullGroups ForceVisible).
		constexpr float kPlayerSlack = 256.0f;  // the root's bound may be a frame older than a part's
		constexpr int   kPlayerDepth = 64;

		void ReadPlayer(FrameContext& a_context) noexcept
		{
			a_context.playerRoot = nullptr;
			if (g_tunables.cullPlayer) {
				return;
			}
			__try {
				const auto player = RE::PlayerCharacter::GetSingleton();
				const auto loaded = player ? player->loadedData : nullptr;
				if (const auto root = loaded ? loaded->data3D.get() : nullptr) {
					a_context.playerRoot = root;
					a_context.playerBound = root->worldBound;
				}
			} __except (EXCEPTION_EXECUTE_HANDLER) {
				a_context.playerRoot = nullptr;
			}
		}

		bool PlayerPart(const FrameContext& a_context, const RE::NiAVObject* a_object, const RE::NiBound& a_bound) noexcept
		{
			const auto root = a_context.playerRoot;
			if (!root || !a_object) {
				return false;
			}
			const auto& b = a_context.playerBound;
			if (b.fRadius > 0.0f && b.fRadius < 1.0e5f && a_bound.fRadius >= 0.0f && a_bound.fRadius < 1.0e5f) {
				const float dx = a_bound.center.x - b.center.x, dy = a_bound.center.y - b.center.y, dz = a_bound.center.z - b.center.z;
				const float reach = b.fRadius + a_bound.fRadius + kPlayerSlack;
				if (dx * dx + dy * dy + dz * dz > reach * reach) {
					return false;
				}
			}  // (an unusable bound, or NaN: walked)
			__try {
				auto node = a_object;
				for (int depth = 0; node && depth < kPlayerDepth; ++depth) {
					if (node == root) {
						return true;
					}
					node = node->parent;
				}
			} __except (EXCEPTION_EXECUTE_HANDLER) {
			}
			return false;
		}

		// What a node's subtree holds that must not go with the node: rejecting a node also drops the
		// lights beneath it (they then stop lighting visible surfaces), and an NPC walking under it can
		// leave the node's bound behind for a frame. Too large to scan counts as a light (conservative).
		enum SubtreeContents : std::uint8_t
		{
			kSubtreePlain = 0,
			kSubtreeLight = 1 << 0,
			kSubtreeActor = 1 << 1,
			kSubtreeAlwaysDraw = 1 << 2,  // a descendant the engine registers whatever the frustum says (flags bit 11)
		};

		std::uint8_t ScanSubtree(RE::NiAVObject* a_object, std::size_t a_budget = 512) noexcept
		{
			std::array<RE::NiNode*, 64> stack{};
			std::size_t                 top = 0;
			std::size_t                 visited = 0;
			std::uint8_t                found = kSubtreePlain;

			auto root = a_object->IsNode();
			if (!root) {
				return kSubtreePlain;
			}
			stack[top++] = root;
			while (top > 0) {
				const auto node = stack[--top];
				for (auto& child : node->children) {
					const auto object = child.get();
					if (!object) {
						continue;
					}
					if (++visited > a_budget) {
						return found | kSubtreeLight;
					}
					if ((object->GetFlags() >> 11) & 1) {
						found |= kSubtreeAlwaysDraw;
					}
					if (TypeOf(object) & kTypeLight) {
						return found | kSubtreeLight;
					}
					if (IsActorRoot(object)) {
						found |= kSubtreeActor;  // (its own parts are the actor's business)
						continue;
					}
					if (const auto childNode = object->IsNode()) {
						if (top == stack.size()) {
							return found | kSubtreeLight;
						}
						stack[top++] = childNode;
					}
				}
			}
			return found;
		}

		// ---- the test -------------------------------------------------------------------------

		enum class Verdict
		{
			kVisible,
			kHidden,
			kEdge,     // part of it is in the current view but was never rendered in the depth frame
			kOutside,  // not in the current view at all: the engine's frustum test drops it
			kBehind,   // behind the depth frame's camera
			kNear,     // reaches the camera: always visible
			kInvalid,  // bad bound
		};

		// Linear view depth of a Hi-Z value; +inf for cleared/sky texels.
		float LinearDepth(const HiZ::Camera& a_camera, float a_buffer) noexcept
		{
			const float ndc = (a_buffer - g_tunables.depthMin) / g_tunables.depthRange;
			if (a_buffer >= 0.999999f || ndc >= a_camera.depthA) {
				return std::numeric_limits<float>::infinity();
			}
			return a_camera.depthB / (ndc - a_camera.depthA);
		}

		using ShadowGeometry::SphereExtent;

		// Clips an NDC rectangle of the depth frame to the part the current camera can see.
		// kHidden here means "judge it": the clipped rectangle lies within the depth frame. With a_overhang (lights:
		// kLightOverhang; objects and sun shadows: fViewOverhang), a part in view beyond the frame by at most that much
		// is judged by the frame's edge (*a_overhung set). Lamp casters' shadow volumes stay strict.
		// a_sides (objects and sun shadows: FrameContext::overhang) overrides a_overhang per side {x0, x1, y0, y1}; past a_overhang
		// the frame's edge then holds only for a part no wider than the part inside the frame, per axis (v1.74, fViewOverhangTurn).
		Verdict ClipToView(const FrameContext& a_context, float& a_x0, float& a_x1, float& a_y0, float& a_y1, float a_overhang = 0.0f, bool* a_overhung = nullptr, const float* a_sides = nullptr) noexcept
		{
			a_x0 = std::max(a_x0, a_context.view[0]);
			a_x1 = std::min(a_x1, a_context.view[1]);
			a_y0 = std::max(a_y0, a_context.view[2]);
			a_y1 = std::min(a_y1, a_context.view[3]);
			if (a_x0 >= a_x1 || a_y0 >= a_y1) {
				return Verdict::kOutside;
			}
			// A visible part the depth frame never rendered: turning the camera brought it into view.
			if (a_x0 < -1.0f || a_x1 > 1.0f || a_y0 < -1.0f || a_y1 > 1.0f) {
				const float left = a_sides ? a_sides[0] : a_overhang, right = a_sides ? a_sides[1] : a_overhang;
				const float bottom = a_sides ? a_sides[2] : a_overhang, top = a_sides ? a_sides[3] : a_overhang;
				if (!(a_x0 >= -1.0f - left && a_x1 <= 1.0f + right && a_y0 >= -1.0f - bottom && a_y1 <= 1.0f + top)) {
					return Verdict::kEdge;
				}
				if (-1.0f - a_x0 > a_overhang || a_x1 - 1.0f > a_overhang || -1.0f - a_y0 > a_overhang || a_y1 - 1.0f > a_overhang) {
					const float outX = std::max(0.0f, -1.0f - a_x0) + std::max(0.0f, a_x1 - 1.0f);
					const float outY = std::max(0.0f, -1.0f - a_y0) + std::max(0.0f, a_y1 - 1.0f);
					if (outX > std::min(a_x1, 1.0f) - std::max(a_x0, -1.0f) || outY > std::min(a_y1, 1.0f) - std::max(a_y0, -1.0f)) {
						return Verdict::kEdge;  // (mostly past the frame: too little of it inside to judge the rest by)
					}
				}
				a_x0 = std::max(a_x0, -1.0f);
				a_x1 = std::min(a_x1, 1.0f);
				a_y0 = std::max(a_y0, -1.0f);
				a_y1 = std::min(a_y1, 1.0f);
				if (a_x0 >= a_x1 || a_y0 >= a_y1) {
					return Verdict::kEdge;  // (nothing of it inside the frame to judge by)
				}
				if (a_overhung) {
					*a_overhung = true;
				}
			}
			return Verdict::kHidden;
		}

		// A sphere (view-space center, radius) entirely outside the current view: behind the eye, or beyond one of
		// the view's four edge planes (planes through the eye; the view lies in front of the depth frame's eye).
		// Only decidable with the view known.
		bool SphereOutsideView(const FrameContext& a_context, float a_x, float a_y, float a_z, float a_radius) noexcept
		{
			if (!std::isfinite(a_context.view[0])) {
				return false;
			}
			const auto& camera = a_context.snapshot->camera;
			const auto  outsidePlane = [&](float a_scale, float a_coord, float a_edge, bool a_upper) {
				const float side = a_scale * a_coord - a_edge * a_z;  // >= 0 on the inner side of a lower edge
				const float reach = a_radius * std::sqrt(a_scale * a_scale + a_edge * a_edge);
				return a_upper ? side > reach : side < -reach;
			};
			return a_z + a_radius < 0.0f ||
			       outsidePlane(camera.scaleX, a_x, a_context.view[0], false) || outsidePlane(camera.scaleX, a_x, a_context.view[1], true) ||
			       outsidePlane(camera.scaleY, a_y, a_context.view[2], false) || outsidePlane(camera.scaleY, a_y, a_context.view[3], true);
		}

		// What a test read of the depth, for the verdict cache. Add records a level-0 texel rectangle read, widened by
		// the coarsest texel the query could have touched (the level where the rectangle spans <= 4 texels, as the
		// Snapshot queries choose it): which Hi-Z blocks the outcome rests on. AddHidden also records the evidence of a
		// hidden verdict: the rectangle whose texels were all nearer than the threshold, and the smallest such
		// threshold, so the verdict can be re-checked against a later depth with one coarse query (ViewValid).
		struct DepthRect
		{
			float x0{ 0.0f }, y0{ 0.0f }, x1{ 0.0f }, y1{ 0.0f };  // widened union of everything read
			bool  used{ false };
			float hx0{ 0.0f }, hy0{ 0.0f }, hx1{ 0.0f }, hy1{ 0.0f };  // union of the hidden evidence (not widened)
			float threshold{ 1.0f };                                    // buffer depth every texel of it was nearer than
			bool  hidden{ false };

			void Add(float a_x0, float a_y0, float a_x1, float a_y1) noexcept
			{
				float span = std::max(a_x1 - a_x0, a_y1 - a_y0);
				float texel = 1.0f;
				while (span > 4.0f && texel < 4096.0f) {
					span *= 0.5f;
					texel *= 2.0f;
				}
				a_x0 -= texel;
				a_y0 -= texel;
				a_x1 += texel;
				a_y1 += texel;
				if (!used) {
					x0 = a_x0;
					y0 = a_y0;
					x1 = a_x1;
					y1 = a_y1;
					used = true;
				} else {
					x0 = std::min(x0, a_x0);
					y0 = std::min(y0, a_y0);
					x1 = std::max(x1, a_x1);
					y1 = std::max(y1, a_y1);
				}
			}

			void AddHidden(float a_x0, float a_y0, float a_x1, float a_y1, float a_threshold) noexcept
			{
				Add(a_x0, a_y0, a_x1, a_y1);
				if (!hidden) {
					hx0 = a_x0;
					hy0 = a_y0;
					hx1 = a_x1;
					hy1 = a_y1;
					hidden = true;
				} else {
					hx0 = std::min(hx0, a_x0);
					hy0 = std::min(hy0, a_y0);
					hx1 = std::max(hx1, a_x1);
					hy1 = std::max(hy1, a_y1);
				}
				threshold = std::min(threshold, a_threshold);
			}
		};

		// With the verdict cache on, an object counts as hidden only with the cache's extra depth room (HiZ marks a
		// block changed only past half of it, so a reused verdict keeps a margin either way).
		float CacheLimit(const FrameContext& a_context, float a_limit) noexcept
		{
			return a_context.cacheEnabled ? (a_limit - HiZ::kCacheDepthSlack) / (1.0f + HiZ::kCacheDepthTolerance) : a_limit;
		}

		float CacheLimitFar(const FrameContext& a_context, float a_limit) noexcept
		{
			return a_context.cacheEnabled ? a_limit * (1.0f + HiZ::kCacheDepthTolerance) + HiZ::kCacheDepthSlack : a_limit;
		}

		// Diagnostic: where a kVisible verdict read the depth (the flip samples re-scan it at level 0).
		struct VisibleEvidence
		{
			bool            valid{ false };
			bool            rays{ false };
			float           x0{ 0.0f }, y0{ 0.0f }, x1{ 0.0f }, y1{ 0.0f };  // level-0 texel rectangle
			float           threshold{ 0.0f };
			float           nearest{ 0.0f };  // view depth of the sphere's nearest point
			HiZ::SphereRays sphere{};
		};

		// A light's reach tested (TestSphere): a thin overhang of the view judged by the frame's edge, and every Hi-Z level
		// refined down to single texels, so a far texel only counts where the light's rays pass (a coarse texel's far
		// corner elsewhere made whole lamps "visible": their shadow maps re-drawn for two frames).
		struct LightTest
		{
			float         overhang{ kLightOverhang };
			std::uint32_t refine{ 16 };
			bool          overhung{ false };    // out: part of it was judged by the frame's edge
			bool          emptyReach{ false };  // out: hidden though not behind the depth: no visible surface lies inside it
		};

		// Parallax (v1.88, bParallaxMargin). The depth was rendered from a camera parallaxMove units from this one. Against an
		// occluder at view depth d, an object at depth z then shifts on screen by up to parallaxMove * (1/d - 1/z) (tangent
		// units); the bound's growth by the move (dilateMove) covers only the 1/z part. So where the nearest surface over
		// the object's screen box lies nearer than the object, the box widens by that much: an object a near occluder's
		// edge has just uncovered (strafing past a pillar) is no longer judged by the old depth behind that edge, which
		// drew it a frame late. The object's own depth is unchanged. a_x0..a_y1: the box in NDC, before clipping.
		constexpr float kParallaxMinDepth = 16.0f;  // nearer surfaces count as this far (the near plane, first person)
		thread_local bool t_parallaxOff{ false };  // (diagnostic: EvaluateView's second run of the tests without the margin, SetKeptReasons)
		thread_local bool t_parallaxHit{ false };  // a test since EvaluateView's start widened its box

		float ParallaxTan(const FrameContext& a_context, float a_x0, float a_x1, float a_y0, float a_y1, float a_z) noexcept
		{
			if (t_parallaxOff || !(a_context.parallaxMove > 0.0f) || !(a_z > kParallaxMinDepth)) {
				return 0.0f;
			}
			const auto& snapshot = *a_context.snapshot;
			const float width = static_cast<float>(snapshot.width[0]);
			const float height = static_cast<float>(snapshot.height[0]);
			const float buffer = snapshot.MinDepth((a_x0 * 0.5f + 0.5f) * width, (0.5f - a_y1 * 0.5f) * height, (a_x1 * 0.5f + 0.5f) * width, (0.5f - a_y0 * 0.5f) * height);
			if (!(buffer >= 0.0f)) {
				return 0.0f;  // (no nearest pyramid)
			}
			float nearest = LinearDepth(snapshot.camera, buffer);
			if (!(nearest >= kParallaxMinDepth)) {
				nearest = kParallaxMinDepth;
			}
			if (!(nearest < a_z)) {
				return 0.0f;
			}
			Bump(kParallaxWidened);
			t_parallaxHit = true;
			return a_context.parallaxMove * (1.0f / nearest - 1.0f / a_z);
		}

		Verdict TestViewSpace(const FrameContext& a_context, const RE::NiBound& a_bound, float a_radius, DepthRect* a_rect = nullptr, LightTest* a_light = nullptr, VisibleEvidence* a_evidence = nullptr) noexcept
		{
			const auto& snapshot = *a_context.snapshot;
			const auto& camera = snapshot.camera;

			const float dx = a_bound.center.x - camera.origin[0];
			const float dy = a_bound.center.y - camera.origin[1];
			const float dz = a_bound.center.z - camera.origin[2];
			const float z = dx * camera.viewDir[0] + dy * camera.viewDir[1] + dz * camera.viewDir[2];
			const float x = dx * camera.viewRight[0] + dy * camera.viewRight[1] + dz * camera.viewRight[2];
			const float y = dx * camera.viewUp[0] + dy * camera.viewUp[1] + dz * camera.viewUp[2];
			// (a reused verdict must hold for a camera turned by the cache tolerance: at this depth that shifts the
			// object by up to this much)
			a_radius += std::max(z, 0.0f) * a_context.angularSlack;

			// With the current view known, anything entirely outside it is decided here, wherever it is
			// (beside or behind the camera too).
			if (SphereOutsideView(a_context, x, y, z, a_radius)) {
				return Verdict::kOutside;
			}
			if (z + a_radius < 0.0f) {
				return Verdict::kBehind;
			}
			const float nearest = z - a_radius;
			if (nearest < g_tunables.nearDistance) {
				return Verdict::kNear;
			}

			float x0, x1, y0, y1;
			SphereExtent(x, z, a_radius, x0, x1);
			SphereExtent(y, z, a_radius, y0, y1);
			x0 *= camera.scaleX;
			x1 *= camera.scaleX;
			y0 *= camera.scaleY;
			y1 *= camera.scaleY;
			// (v1.88: a near occluder's parallax since the depth frame widens the box, and the rays below)
			const float parallax = ParallaxTan(a_context, x0, x1, y0, y1, z);
			x0 -= parallax * camera.scaleX;
			x1 += parallax * camera.scaleX;
			y0 -= parallax * camera.scaleY;
			y1 += parallax * camera.scaleY;
			bool overhung = false;
			if (const auto clip = ClipToView(a_context, x0, x1, y0, y1, a_light ? a_light->overhang : g_tunables.viewOverhang, &overhung, a_light ? nullptr : a_context.overhang);
				clip != Verdict::kHidden) {
				return clip;
			}
			if (a_light) {
				a_light->overhung = overhung;
			}

			const float width = static_cast<float>(snapshot.width[0]);
			const float height = static_cast<float>(snapshot.height[0]);
			const float px0 = (x0 * 0.5f + 0.5f) * width;
			const float px1 = (x1 * 0.5f + 0.5f) * width;
			const float py0 = (0.5f - y1 * 0.5f) * height;
			const float py1 = (0.5f - y0 * 0.5f) * height;

			// Hidden when every occluder in the rectangle is nearer than the object's nearest point
			// (with tolerance). Compared in buffer depth so Hi-Z texels can be checked directly.
			const float limit = CacheLimit(a_context, (nearest - g_tunables.depthSlack) / (1.0f + g_tunables.depthTolerance));
			if (!(limit > 0.0f)) {
				return Verdict::kVisible;
			}
			if (camera.depthB >= 0.0f) {  // not standard Z: compare linear depths over the whole rectangle (no evidence recorded: never reused hidden)
				const float occluder = LinearDepth(camera, snapshot.MaxDepth(px0, py0, px1, py1));
				return occluder < limit ? Verdict::kHidden : Verdict::kVisible;
			}
			const float threshold = std::min(g_tunables.depthMin + g_tunables.depthRange * (camera.depthA + camera.depthB / limit), 0.999999f);

			// Only texels the sphere's own rays pass through can show it (not the rectangle's corners).
			HiZ::SphereRays rays{};
			rays.centerX = (camera.scaleX * x / z * 0.5f + 0.5f) * width;
			rays.centerY = (0.5f - camera.scaleY * y / z * 0.5f) * height;
			rays.depth = z;
			rays.radius = a_radius + parallax * z;
			rays.texelsPerTanX = camera.scaleX * width * 0.5f;
			rays.texelsPerTanY = camera.scaleY * height * 0.5f;
			rays.axisX = width * 0.5f;
			rays.axisY = height * 0.5f;
			if (!snapshot.AllNearer(px0, py0, px1, py1, threshold, a_light ? a_light->refine : kRefineLevels, &rays)) {
				// A light's reach (v1.55): what a lamp's shadow map needs is a visible surface inside the reach, not a reach
				// in front of the depth. A texel where nothing was drawn (the far plane: a crack through an interior's walls,
				// the sky) holds no surface to light, and neither does one whose surfaces all lie beyond the reach's far
				// side. Every texel of the rectangle must hold its surfaces in front of the reach's near side or beyond its
				// far side (the nearest and farthest pyramids, the same test the lamp casters' shadow volumes get). In the
				// v1.52-v1.54 runs one far-plane texel in an interior made the same lamp "visible" every second or so and its
				// whole shadow map was drawn again. Objects keep the strict test: one is visible through such a texel.
				if (a_light) {
					const float beyond = CacheLimitFar(a_context, (z + a_radius + g_tunables.depthSlack) * (1.0f + g_tunables.depthTolerance));
					const float farThreshold = std::min(g_tunables.depthMin + g_tunables.depthRange * (camera.depthA + camera.depthB / beyond), 0.999999f);
					// (only texels the reach's own rays pass through, as AllNearer just did: v1.55 scanned the whole screen box,
					// whose corners hold surfaces outside the sphere, and the rule held in 1 of 426 log intervals)
					// (and at level 0 the farthest *drawn* depth: v1.56 showed the flickering lamps' texel mixing a wall with
					// a crack into the void, whose farthest depth is the far plane: v1.57's Hi-Z keeps the wall's depth too)
					if (beyond > 0.0f && snapshot.NoSurfaceBetween(px0, py0, px1, py1, threshold, farThreshold, a_light->refine, &rays, true)) {
						a_light->emptyReach = true;
						return Verdict::kHidden;
					}
				}
				if (a_evidence) {
					*a_evidence = VisibleEvidence{ true, true, px0, py0, px1, py1, threshold, nearest, rays };
				}
				return Verdict::kVisible;
			}
			if (a_rect) {
				a_rect->AddHidden(px0, py0, px1, py1, threshold);
			}
			if (overhung && !a_light) {
				Bump(kOverhangHidden);
			}
			return Verdict::kHidden;
		}

		// Fallback when the view-space form couldn't be verified: project the bound's box corners.
		Verdict TestCorners(const FrameContext& a_context, const RE::NiBound& a_bound, float a_radius, DepthRect* a_rect = nullptr, LightTest* a_light = nullptr) noexcept
		{
			const auto& snapshot = *a_context.snapshot;
			const auto& camera = snapshot.camera;
			const auto& m = camera.viewProj;

			const float cx = a_bound.center.x - camera.posAdjust[0];
			const float cy = a_bound.center.y - camera.posAdjust[1];
			const float cz = a_bound.center.z - camera.posAdjust[2];

			float xmin = 1.0e30f, xmax = -1.0e30f, ymin = 1.0e30f, ymax = -1.0e30f, zmin = 1.0e30f;
			for (int i = 0; i < 8; ++i) {
				const float px = cx + ((i & 1) ? a_radius : -a_radius);
				const float py = cy + ((i & 2) ? a_radius : -a_radius);
				const float pz = cz + ((i & 4) ? a_radius : -a_radius);
				const float w = px * m[0][3] + py * m[1][3] + pz * m[2][3] + m[3][3];
				if (w < g_tunables.nearDistance) {
					return Verdict::kNear;
				}
				const float inv = 1.0f / w;
				xmin = std::min(xmin, (px * m[0][0] + py * m[1][0] + pz * m[2][0] + m[3][0]) * inv);
				xmax = std::max(xmax, (px * m[0][0] + py * m[1][0] + pz * m[2][0] + m[3][0]) * inv);
				ymin = std::min(ymin, (px * m[0][1] + py * m[1][1] + pz * m[2][1] + m[3][1]) * inv);
				ymax = std::max(ymax, (px * m[0][1] + py * m[1][1] + pz * m[2][1] + m[3][1]) * inv);
				zmin = std::min(zmin, (px * m[0][2] + py * m[1][2] + pz * m[2][2] + m[3][2]) * inv);
			}
			bool overhung = false;
			if (const auto clip = ClipToView(a_context, xmin, xmax, ymin, ymax, a_light ? a_light->overhang : g_tunables.viewOverhang, &overhung, a_light ? nullptr : a_context.overhang);
				clip != Verdict::kHidden) {
				return clip;
			}
			if (a_light) {
				a_light->overhung = overhung;
			}

			const float depth = g_tunables.depthMin + g_tunables.depthRange * std::clamp(zmin, 0.0f, 1.0f);
			const float width = static_cast<float>(snapshot.width[0]);
			const float height = static_cast<float>(snapshot.height[0]);
			const float px0 = (xmin * 0.5f + 0.5f) * width;
			const float py0 = (0.5f - ymax * 0.5f) * height;
			const float px1 = (xmax * 0.5f + 0.5f) * width;
			const float py1 = (0.5f - ymin * 0.5f) * height;
			const float threshold = depth - 1.0e-6f;
			if (!snapshot.AllNearer(px0, py0, px1, py1, threshold, kRefineLevels)) {
				return Verdict::kVisible;
			}
			if (a_rect) {
				a_rect->AddHidden(px0, py0, px1, py1, threshold);
			}
			if (overhung && !a_light) {
				Bump(kOverhangHidden);
			}
			return Verdict::kHidden;
		}

		// ---- first person drawn after the capture (v1.69, Hooks/FirstPersonOrder) ---------------------------
		// The depth frame then holds the world behind the weapon and no first-person surface, yet the weapon and arms
		// still receive light and shadow. They lie within kFirstPersonReach units of the eye (a long rifle's muzzle is
		// ~90): a light, or a shadow sweep, that comes that close is kept, as one crossing a first-person pixel was.
		constexpr float kFirstPersonReach = 128.0f;

		bool FirstPersonUnseen(const FrameContext& a_context) noexcept
		{
			return a_context.snapshot && a_context.snapshot->firstPersonAfter;
		}

		// Whether the sweep from a_center (view space: the eye at the origin) along the unit a_dir, radius a_radius +
		// t x a_spread for t in [0, a_tMax], comes within a_reach of the eye. Its distance to the eye less its radius
		// is convex in t, least where the offset along a_dir is a_spread x the perpendicular distance / sqrt(1 -
		// a_spread^2) (at a_tMax when the radius grows as fast as the distance).
		bool SweepNearEye(const float a_center[3], const float a_dir[3], float a_radius, float a_spread, float a_tMax, float a_reach) noexcept
		{
			const float along = ShadowGeometry::Dot(a_center, a_dir);
			const float perp2 = std::max(ShadowGeometry::Dot(a_center, a_center) - along * along, 0.0f);
			const float t = a_spread < 1.0f ? -along + a_spread * std::sqrt(perp2 / (1.0f - a_spread * a_spread)) : a_tMax;
			const float tc = std::clamp(t, 0.0f, a_tMax);
			const float offset = along + tc;
			const float reach = a_radius + tc * a_spread + a_reach;
			return perp2 + offset * offset < reach * reach;
		}

		// Whether a light's reach (world space) comes within kFirstPersonReach of the depth frame's eye.
		bool ReachNearEye(const FrameContext& a_context, const RE::NiPoint3& a_center, float a_radius) noexcept
		{
			const auto& camera = a_context.snapshot->camera;
			const float dx = a_center.x - camera.origin[0];
			const float dy = a_center.y - camera.origin[1];
			const float dz = a_center.z - camera.origin[2];
			const float reach = a_radius + a_context.dilateMove + kFirstPersonReach;
			return dx * dx + dy * dy + dz * dz < reach * reach;
		}

		// ---- sun shadows: the capsule a caster sweeps along the light --------------------------------------

		enum class SunVerdict
		{
			kNeeded,   // its shadow may fall on something visible
			kOutside,  // the sweep never reaches the current view within the cascade range
			kHidden,   // every visible surface where it could land is in front of it
			kEdge,     // needed: part of it may land in view where the depth frame has nothing (v1.73)
		};

		// Whether a caster's sun shadow can fall on anything visible. The shadow lies in the capsule swept from
		// the caster's bound along the light (its radius growing with the direction's uncertainty). Receivers are
		// in front of the eye, within the cascade range and inside the current view, so the sweep is clipped to
		// where its cross-section reaches that region; that part is then tested against the depth frame like an
		// object: hidden only if every visible surface over its screen box is in front of its nearest point.
		// a_far (v1.80): only the receivers of the sun's last cascade, the slab of view depth it covers (Sun::farPlanes).
		// A caster's sweep as TestSun starts it (and the entry walk's cheap shadow test, v1.94): the bound in the depth
		// frame's view space and its radius with the margins (the cache's move tolerance, the shadow maps' filtering, and
		// the cache's turn tolerance over the whole receiver range: a reused verdict must hold for a camera turned by that
		// much, which shifts a point at depth z by z x the slack). False: the bound or the camera is unusable (needed).
		struct SunSweep
		{
			float center[3]{};
			float radius{ 0.0f };
		};

		bool SunSweepStart(const FrameContext& a_context, const RE::NiBound& a_bound, SunSweep& a_out) noexcept
		{
			const auto& camera = a_context.snapshot->camera;
			const auto& sun = a_context.sun;
			const float radius = a_bound.fRadius;
			if (!camera.viewSpace || !(camera.depthB < 0.0f) || !(radius > 0.0f) || !(radius < 1.0e6f) ||
				!std::isfinite(a_bound.center.x) || !std::isfinite(a_bound.center.y) || !std::isfinite(a_bound.center.z)) {
				return false;
			}
			const float dx = a_bound.center.x - camera.origin[0];
			const float dy = a_bound.center.y - camera.origin[1];
			const float dz = a_bound.center.z - camera.origin[2];
			a_out.center[0] = dx * camera.viewRight[0] + dy * camera.viewRight[1] + dz * camera.viewRight[2];
			a_out.center[1] = dx * camera.viewUp[0] + dy * camera.viewUp[1] + dz * camera.viewUp[2];
			a_out.center[2] = dx * camera.viewDir[0] + dy * camera.viewDir[1] + dz * camera.viewDir[2];
			a_out.radius = radius + a_context.dilateMove + sun.margin + (std::isfinite(sun.reach) ? std::max(sun.reach, 0.0f) * a_context.angularSlack : 0.0f);
			return true;
		}

		// The sweep lies beyond one of the receiver planes it can only move away from: out of reach without the full
		// sweep (the cheap part of TestSun, where most casters end).
		bool SunSweepBeyondReach(const FrameContext& a_context, const SunSweep& a_sweep, bool a_far) noexcept
		{
			const auto& sun = a_context.sun;
			const auto* reachPlanes = a_far ? sun.farReachPlanes : sun.reachPlanes;
			const int   reachCount = a_far ? sun.farReachCount : sun.reachCount;
			for (int i = 0; i < reachCount; ++i) {
				const auto& plane = reachPlanes[i];
				if (ShadowGeometry::Dot(plane.n, a_sweep.center) + plane.d + a_sweep.radius < 0.0f) {
					return true;
				}
			}
			return false;
		}

		SunVerdict TestSun(const FrameContext& a_context, const RE::NiBound& a_bound, DepthRect* a_rect = nullptr, bool a_far = false) noexcept
		{
			const auto& snapshot = *a_context.snapshot;
			const auto& camera = snapshot.camera;
			const auto& sun = a_context.sun;
			SunSweep    sweep{};
			if (!SunSweepStart(a_context, a_bound, sweep)) {
				return SunVerdict::kNeeded;
			}
			const float (&center)[3] = sweep.center;
			const float r0 = sweep.radius;
			// (first person, which may lie outside the world view's planes: its own field of view, tested first)
			// (not in the last cascade's slab: first person is drawn next to the eye)
			if (!a_far && FirstPersonUnseen(a_context) && SweepNearEye(center, sun.dir, r0, sun.spread, 1.0e6f, kFirstPersonReach)) {
				return SunVerdict::kNeeded;
			}
			if (SunSweepBeyondReach(a_context, sweep, a_far)) {
				return SunVerdict::kOutside;
			}
			float t0 = 0.0f, t1 = 0.0f;
			if (!ShadowGeometry::SweepClip(center, sun.dir, r0, sun.spread, 1.0e6f, a_far ? sun.farPlanes : sun.planes, a_far ? sun.farPlaneCount : sun.planeCount, t0, t1)) {
				return SunVerdict::kOutside;
			}

			// Depth minus radius is linear along the capsule: its nearest point is at one of the ends (no stretch of it
			// is nearer than the whole).
			if (std::min(center[2] + t0 * sun.dir[2] - (r0 + t0 * sun.spread), center[2] + t1 * sun.dir[2] - (r0 + t1 * sun.spread)) < g_tunables.nearDistance) {
				return SunVerdict::kNeeded;
			}
			const float width = static_cast<float>(snapshot.width[0]);
			const float height = static_cast<float>(snapshot.height[0]);
			const auto  bufferDepth = [&](float a_z) { return std::min(g_tunables.depthMin + g_tunables.depthRange * (camera.depthA + camera.depthB / a_z), 0.999999f); };
			// The stretch [a_ta, a_tb] of the sweep: the convex hull of its end spheres, so it projects within the box around
			// both end spheres' projections (all in front of the eye), and its depth lies between the ends' nearest and
			// farthest points. a_extent: the box's larger side in NDC, before clipping to the view.
			const auto stretch = [&](float a_ta, float a_tb, float* a_extent, bool& a_overhung) {
				const float q0[3]{ center[0] + a_ta * sun.dir[0], center[1] + a_ta * sun.dir[1], center[2] + a_ta * sun.dir[2] };
				const float q1[3]{ center[0] + a_tb * sun.dir[0], center[1] + a_tb * sun.dir[1], center[2] + a_tb * sun.dir[2] };
				const float ra = r0 + a_ta * sun.spread;
				const float rb = r0 + a_tb * sun.spread;
				const float nearest = std::min(q0[2] - ra, q1[2] - rb);
				float       x0, x1, y0, y1, u0, u1, v0, v1;
				SphereExtent(q0[0], q0[2], ra, x0, x1);
				SphereExtent(q0[1], q0[2], ra, y0, y1);
				SphereExtent(q1[0], q1[2], rb, u0, u1);
				SphereExtent(q1[1], q1[2], rb, v0, v1);
				x0 = std::min(x0, u0) * camera.scaleX;
				x1 = std::max(x1, u1) * camera.scaleX;
				y0 = std::min(y0, v0) * camera.scaleY;
				y1 = std::max(y1, v1) * camera.scaleY;
				if (a_extent) {
					*a_extent = std::max(x1 - x0, y1 - y0);
				}
				// (v1.88: a near occluder's parallax since the depth frame, at the stretch's farthest depth)
				const float parallax = ParallaxTan(a_context, x0, x1, y0, y1, std::max(q0[2] + ra, q1[2] + rb));
				x0 -= parallax * camera.scaleX;
				x1 += parallax * camera.scaleX;
				y0 -= parallax * camera.scaleY;
				y1 += parallax * camera.scaleY;
				bool overhung = false;
				switch (ClipToView(a_context, x0, x1, y0, y1, g_tunables.viewOverhang, &overhung, a_context.overhang)) {
				case Verdict::kOutside:
					return SunVerdict::kOutside;
				case Verdict::kEdge:
					return SunVerdict::kEdge;
				default:
					break;
				}
				const float limit = CacheLimit(a_context, (nearest - g_tunables.depthSlack) / (1.0f + g_tunables.depthTolerance));
				if (!(limit > 0.0f)) {
					return SunVerdict::kNeeded;
				}
				const float farthest = std::max(q0[2] + ra, q1[2] + rb);
				const float limitFar = CacheLimitFar(a_context, (farthest + g_tunables.depthSlack) * (1.0f + g_tunables.depthTolerance));
				const float bx0 = (x0 * 0.5f + 0.5f) * width - 0.25f;
				const float by0 = (0.5f - y1 * 0.5f) * height - 0.25f;
				const float bx1 = (x1 * 0.5f + 0.5f) * width + 0.25f;
				const float by1 = (0.5f - y0 * 0.5f) * height + 0.25f;
				if (a_rect) {
					a_rect->Add(bx0, by0, bx1, by1);
				}
				// No visible surface inside the stretch's depth range over its screen box: at every texel the surfaces are
				// all in front of it or all behind it (the sky, cleared to the far plane, counts as behind: nothing there
				// receives a shadow).
				if (!snapshot.NoSurfaceBetween(bx0, by0, bx1, by1, bufferDepth(limit), bufferDepth(limitFar), kRefineLevels)) {
					return SunVerdict::kNeeded;
				}
				a_overhung = a_overhung || overhung;
				return SunVerdict::kHidden;
			};

			bool  overhung = false;
			float extent = 0.0f;
			const auto whole = stretch(t0, t1, &extent, overhung);
			const int  pieces = (whole == SunVerdict::kNeeded || whole == SunVerdict::kEdge) && extent > kSunPieceExtent ?
			                        static_cast<int>(std::min(std::ceil(extent / kSunPieceExtent), static_cast<float>(kSunPieces))) :
			                        1;
			if (pieces < 2) {
				if (whole == SunVerdict::kHidden && overhung) {
					Bump(kSunOverhang);
				}
				return whole;
			}
			// A long sweep (a low sun, the cascade range) projects to a box over much of the view with a depth range
			// that takes in most surfaces there, so the whole is nearly always needed. Its stretches cover it (their
			// union is the sweep) with smaller boxes and narrower ranges: hidden only if every one is.
			bool edge = false;
			for (int i = 0; i < pieces; ++i) {
				const float ta = t0 + (t1 - t0) * static_cast<float>(i) / static_cast<float>(pieces);
				const float tb = i + 1 == pieces ? t1 : t0 + (t1 - t0) * static_cast<float>(i + 1) / static_cast<float>(pieces);
				switch (stretch(ta, tb, nullptr, overhung)) {
				case SunVerdict::kNeeded:
					return SunVerdict::kNeeded;
				case SunVerdict::kEdge:
					edge = true;
					break;
				default:
					break;
				}
			}
			if (edge) {
				return SunVerdict::kEdge;
			}
			Bump(kSunPiecewise);
			if (overhung) {
				Bump(kSunOverhang);
			}
			return SunVerdict::kHidden;
		}

		// The per-side overhang a context's view gets (FrameContext::overhang): fViewOverhang, and with fViewOverhangTurn up to
		// that much where the view itself reaches past the depth frame (the strip a turn brought into view since the depth
		// was rendered: one frame of the turn at the v1.73 log's depth age of 1; the async worker's widened view's too).
		void SetOverhangs(FrameContext& a_context) noexcept
		{
			const float base = g_tunables.viewOverhang;
			const float turn = g_tunables.viewOverhangTurn;
			const bool  known = std::isfinite(a_context.view[0]);
			const float past[4]{ -1.0f - a_context.view[0], a_context.view[1] - 1.0f, -1.0f - a_context.view[2], a_context.view[3] - 1.0f };
			for (int i = 0; i < 4; ++i) {
				// (+ a hair: the clipped part's side equals the view's, and 1 + (view - 1) needn't round back to it)
				a_context.overhang[i] = turn > base && known ? std::max(base, std::min(turn, past[i] + 1.0e-5f)) : base;
			}
		}

		// Where sun-shadow receivers can be this frame (TestSun clips each caster's sweep to it): in front of the
		// eye, within reach, and inside the current view's edges (planes through the eye, as TestViewSpace uses).
		void SetSunPlanes(FrameContext& a_context) noexcept
		{
			auto& sun = a_context.sun;
			sun.planeCount = 0;
			sun.farPlaneCount = 0;
			sun.farReachCount = 0;
			if (sun.state != FrameContext::Sun::State::kOn || !a_context.snapshot) {
				sun.farOn = false;
				return;
			}
			const auto& camera = a_context.snapshot->camera;
			int         count = 0;
			sun.planes[count++] = { { 0.0f, 0.0f, 1.0f }, 0.0f };
			sun.planes[count++] = { { 0.0f, 0.0f, -1.0f }, sun.reach };
			if (std::isfinite(a_context.view[0])) {
				const auto edge = [&](int a_axis, float a_scale, float a_edge, bool a_upper) {
					const float length = std::sqrt(a_scale * a_scale + a_edge * a_edge);
					auto&       plane = sun.planes[count++];
					plane = {};
					plane.n[a_axis] = (a_upper ? -a_scale : a_scale) / length;
					plane.n[2] = (a_upper ? a_edge : -a_edge) / length;
				};
				edge(0, camera.scaleX, a_context.view[0], false);
				edge(0, camera.scaleX, a_context.view[1], true);
				edge(1, camera.scaleY, a_context.view[2], false);
				edge(1, camera.scaleY, a_context.view[3], true);
			}
			sun.planeCount = count;
			// Along the light a sweep only moves away from these planes (its radius growth included), so a caster
			// sphere already beyond one of them can't reach any receiver: SweepClip's answer, without the divisions.
			sun.reachCount = 0;
			for (int i = 0; i < count; ++i) {
				if (ShadowGeometry::Dot(sun.planes[i].n, sun.dir) + sun.spread <= 0.0f) {
					sun.reachPlanes[sun.reachCount++] = sun.planes[i];
				}
			}
			// The last cascade alone: its slab's view depths in place of the eye and the range, the same view edges.
			if (!sun.farOn || !(sun.farNear > 0.0f) || !(sun.farFar > sun.farNear)) {
				sun.farOn = false;
				return;
			}
			sun.farPlanes[0] = { { 0.0f, 0.0f, 1.0f }, -sun.farNear };
			sun.farPlanes[1] = { { 0.0f, 0.0f, -1.0f }, std::min(sun.farFar, sun.reach) };
			for (int i = 2; i < count; ++i) {
				sun.farPlanes[i] = sun.planes[i];
			}
			sun.farPlaneCount = count;
			for (int i = 0; i < count; ++i) {
				if (ShadowGeometry::Dot(sun.farPlanes[i].n, sun.dir) + sun.spread <= 0.0f) {
					sun.farReachPlanes[sun.farReachCount++] = sun.farPlanes[i];
				}
			}
		}

		// Where lamp-shadow receivers can be this frame: in front of the eye and inside the current view's edges
		// (as SetSunPlanes, without the cascade range: a lamp's reach bounds its sweep instead).
		void SetLampPlanes(FrameContext& a_context) noexcept
		{
			a_context.lampPlaneCount = 0;
			if (!a_context.snapshot || !a_context.snapshot->camera.viewSpace) {
				return;
			}
			const auto& camera = a_context.snapshot->camera;
			int         count = 0;
			a_context.lampPlanes[count++] = { { 0.0f, 0.0f, 1.0f }, 0.0f };
			if (std::isfinite(a_context.view[0])) {
				const auto edge = [&](int a_axis, float a_scale, float a_edge, bool a_upper) {
					const float length = std::sqrt(a_scale * a_scale + a_edge * a_edge);
					auto&       plane = a_context.lampPlanes[count++];
					plane = {};
					plane.n[a_axis] = (a_upper ? -a_scale : a_scale) / length;
					plane.n[2] = (a_upper ? a_edge : -a_edge) / length;
				};
				edge(0, camera.scaleX, a_context.view[0], false);
				edge(0, camera.scaleX, a_context.view[1], true);
				edge(1, camera.scaleY, a_context.view[2], false);
				edge(1, camera.scaleY, a_context.view[3], true);
			}
			a_context.lampPlaneCount = count;
		}

		// ---- mesh shapes: boxes in the depth frame's view space --------------------------------------

		// A view-space box (center, half extents) against the depth frame. Only the part beyond the near
		// distance can be drawn, so the box is clipped there first.
		Verdict TestViewBox(const FrameContext& a_context, const float a_center[3], const float a_halfIn[3], DepthRect* a_rect = nullptr) noexcept
		{
			const auto& snapshot = *a_context.snapshot;
			const auto& camera = snapshot.camera;
			const float zNear = g_tunables.nearDistance;
			const float zFar = a_center[2] + a_halfIn[2];
			if (zFar < zNear) {
				return Verdict::kOutside;  // entirely before the near plane or behind the eye
			}
			// (+ the cache's turn tolerance at the box's farthest depth)
			const float slack = zFar * a_context.angularSlack;
			const float a_half[3]{ a_halfIn[0] + slack, a_halfIn[1] + slack, a_halfIn[2] + slack };
			if (std::isfinite(a_context.view[0])) {
				const auto outsidePlane = [&](float a_scale, float a_coord, float a_extent, float a_edge, bool a_upper) {
					const float side = a_scale * a_coord - a_edge * a_center[2];
					const float reach = a_scale * a_extent + std::abs(a_edge) * a_half[2];
					return a_upper ? side > reach : side < -reach;
				};
				if (outsidePlane(camera.scaleX, a_center[0], a_half[0], a_context.view[0], false) ||
					outsidePlane(camera.scaleX, a_center[0], a_half[0], a_context.view[1], true) ||
					outsidePlane(camera.scaleY, a_center[1], a_half[1], a_context.view[2], false) ||
					outsidePlane(camera.scaleY, a_center[1], a_half[1], a_context.view[3], true)) {
					return Verdict::kOutside;
				}
			}
			const float zNearest = std::max(a_center[2] - a_half[2], zNear);
			// x/z and y/z over the clipped box peak at its corners.
			float x0 = std::numeric_limits<float>::infinity(), x1 = -x0, y0 = x0, y1 = -x0;
			for (const float z : { zNearest, zFar }) {
				for (const float sign : { -1.0f, 1.0f }) {
					const float x = camera.scaleX * (a_center[0] + sign * a_half[0]) / z;
					const float y = camera.scaleY * (a_center[1] + sign * a_half[1]) / z;
					x0 = std::min(x0, x);
					x1 = std::max(x1, x);
					y0 = std::min(y0, y);
					y1 = std::max(y1, y);
				}
			}
			// (v1.88: a near occluder's parallax since the depth frame, at the box's farthest depth: the widest shift)
			const float parallax = ParallaxTan(a_context, x0, x1, y0, y1, zFar);
			x0 -= parallax * camera.scaleX;
			x1 += parallax * camera.scaleX;
			y0 -= parallax * camera.scaleY;
			y1 += parallax * camera.scaleY;
			if (const auto clip = ClipToView(a_context, x0, x1, y0, y1, g_tunables.viewOverhang, nullptr, a_context.overhang); clip != Verdict::kHidden) {
				return clip;
			}
			const float limit = CacheLimit(a_context, (zNearest - g_tunables.depthSlack) / (1.0f + g_tunables.depthTolerance));
			if (!(limit > 0.0f) || camera.depthB >= 0.0f) {
				return Verdict::kVisible;
			}
			const float width = static_cast<float>(snapshot.width[0]);
			const float height = static_cast<float>(snapshot.height[0]);
			const float threshold = std::min(g_tunables.depthMin + g_tunables.depthRange * (camera.depthA + camera.depthB / limit), 0.999999f);
			// (a quarter texel of slack covers the render's sub-pixel jitter, as for spheres)
			const float px0 = (x0 * 0.5f + 0.5f) * width - 0.25f;
			const float py0 = (0.5f - y1 * 0.5f) * height - 0.25f;
			const float px1 = (x1 * 0.5f + 0.5f) * width + 0.25f;
			const float py1 = (0.5f - y0 * 0.5f) * height + 0.25f;
			if (!snapshot.AllNearer(px0, py0, px1, py1, threshold, kRefineLevels)) {
				return Verdict::kVisible;
			}
			if (a_rect) {
				a_rect->AddHidden(px0, py0, px1, py1, threshold);
			}
			return Verdict::kHidden;
		}

		// How NiTransform applies its rotation to a point: columns, R^T * v. Read in the engine (2026-10-01): a BSGeometry's
		// world bound is NiBound::Update(world bound, model bound, world transform) (0x141BB19A0, called by
		// BSGeometry::UpdateWorldBound 0x141BB0F40), which computes x' = sum_j c_j * R[j][0] (each component from a column of
		// the stored rows), the skinned path likewise. Up to v1.52 the reading was learned at run time from meshes whose
		// bound fits one reading only; the v1.52 interior run (precombined chunks with identity rotations) produced no such
		// mesh in the whole session ("votes 0/0"), so mesh shapes never ran there (~70-95 meshes a frame left to the
		// sphere). An object whose world bound doesn't land where R^T * c puts it is still left to the sphere; one whose
		// bound would fit the rows reading alone is counted (a layout misread would show there).
		std::atomic<std::uint64_t> g_rotationMisfits{ 0 };   // bound off its transform (since load)
		std::atomic<std::uint64_t> g_rotationRowsOnly{ 0 };  // ... of them, fitting R * v instead (should stay 0)

		bool WorldRotation(const RE::NiAVObject* a_object, const RE::NiBound& a_model, float a_rotate[3][3]) noexcept
		{
			const auto& transform = a_object->world;
			const auto& world = a_object->worldBound;
			const float scale = transform.scale;
			// The engine makes the world bound by moving the model bound with the world transform. Where the
			// radius doesn't match, the bound comes from somewhere else and the transform isn't the whole story.
			if (!(scale > 0.0f) || !(std::abs(world.fRadius - a_model.fRadius * scale) <= world.fRadius * 0.01f + 1.0f)) {
				return false;
			}
			const float c[3]{ a_model.center.x * scale, a_model.center.y * scale, a_model.center.z * scale };
			const float target[3]{ world.center.x - transform.translate.x, world.center.y - transform.translate.y, world.center.z - transform.translate.z };
			const float tolerance = world.fRadius * 0.01f + 1.0f;
			const auto  fits = [&](bool a_columns) {
				float error = 0.0f;
				for (int i = 0; i < 3; ++i) {
					float v = 0.0f;
					for (int j = 0; j < 3; ++j) {
						v += (a_columns ? transform.rotate.entry[j].pt[i] : transform.rotate.entry[i].pt[j]) * c[j];
					}
					error = std::max(error, std::abs(v - target[i]));
				}
				return error <= tolerance;
			};
			if (!fits(true)) {
				g_rotationMisfits.fetch_add(1, std::memory_order_relaxed);
				if (fits(false)) {
					g_rotationRowsOnly.fetch_add(1, std::memory_order_relaxed);
				}
				return false;
			}
			for (int i = 0; i < 3; ++i) {
				for (int j = 0; j < 3; ++j) {
					a_rotate[i][j] = transform.rotate.entry[j].pt[i];
				}
			}
			return true;
		}

		// NiAVObject::UpdateWorldData copies world into previousWorld before recomputing it, so the two differ
		// for an object that moved in its last update. Such an object can sit behind its own image in the
		// older depth frame, which a tight shape would take for an occluder. A previousWorld that was never
		// set (zero scale, or the default identity at the origin) says nothing.
		bool Moved(const RE::NiAVObject* a_object) noexcept
		{
			const auto& now = a_object->world;
			const auto& before = a_object->previousWorld;
			if (!(before.scale > 0.0f)) {
				return false;
			}
			bool same = now.scale == before.scale && now.translate.x == before.translate.x && now.translate.y == before.translate.y &&
			            now.translate.z == before.translate.z;
			bool identity = before.scale == 1.0f && before.translate.x == 0.0f && before.translate.y == 0.0f && before.translate.z == 0.0f;
			for (int i = 0; i < 3; ++i) {
				for (int j = 0; j < 3; ++j) {
					same = same && now.rotate.entry[i].pt[j] == before.rotate.entry[i].pt[j];
					identity = identity && before.rotate.entry[i].pt[j] == (i == j ? 1.0f : 0.0f);
				}
			}
			return !same && !identity;
		}

		// A mesh judged by its shape: the whole box first, then its occupied cells. Hidden once every
		// occupied cell is hidden or out of view; kept as soon as one isn't. a_hint (in/out) is the cell last seen
		// visible: the scan starts there, so a mesh that stays visible usually stops at its first cell.
		Verdict TestShape(const FrameContext& a_context, const RE::NiAVObject* a_object, const RE::NiBound& a_model, const MeshProxy::Shape& a_shape, std::uint32_t& a_cellsTested, std::uint8_t& a_hint, DepthRect* a_rect = nullptr) noexcept
		{
			float rotate[3][3];
			if (!WorldRotation(a_object, a_model, rotate)) {
				return Verdict::kInvalid;
			}
			const auto&  camera = a_context.snapshot->camera;
			const auto&  transform = a_object->world;
			const float* view[3]{ camera.viewRight, camera.viewUp, camera.viewDir };
			float        m[3][3], t[3];
			for (int i = 0; i < 3; ++i) {
				for (int j = 0; j < 3; ++j) {
					m[i][j] = transform.scale * (view[i][0] * rotate[0][j] + view[i][1] * rotate[1][j] + view[i][2] * rotate[2][j]);
				}
				t[i] = view[i][0] * (transform.translate.x - camera.origin[0]) + view[i][1] * (transform.translate.y - camera.origin[1]) +
				       view[i][2] * (transform.translate.z - camera.origin[2]);
			}
			const float dilate = a_context.dilateMove;
			const auto  box = [&](const float a_lo[3], const float a_size[3]) {
				float center[3], half[3];
				for (int i = 0; i < 3; ++i) {
					float c = t[i], h = dilate;
					for (int j = 0; j < 3; ++j) {
						const float local = a_lo[j] + a_size[j] * 0.5f;
						c += m[i][j] * local;
						h += std::abs(m[i][j]) * a_size[j] * 0.5f;
					}
					center[i] = c;
					half[i] = h;
				}
				return TestViewBox(a_context, center, half, a_rect);
			};

			const float size[3]{ a_shape.cell[0] * a_shape.dims[0], a_shape.cell[1] * a_shape.dims[1], a_shape.cell[2] * a_shape.dims[2] };
			const auto  whole = box(a_shape.min, size);
			if (whole == Verdict::kHidden || whole == Verdict::kOutside || a_shape.count == 0) {
				return whole;
			}
			bool                anyHidden = false;
			const std::uint32_t dims0 = a_shape.dims[0];
			const std::uint32_t dims1 = a_shape.dims[1];
			const std::uint32_t cells = dims0 * dims1 * a_shape.dims[2];
			const std::uint32_t first = a_hint < cells ? a_hint : 0u;
			for (std::uint32_t k = 0; k < cells; ++k) {
				auto bit = first + k;
				if (bit >= cells) {
					bit -= cells;
				}
				if (!(a_shape.occupied[bit >> 6] & (std::uint64_t{ 1 } << (bit & 63)))) {
					continue;
				}
				const auto  x = bit % dims0;
				const auto  y = (bit / dims0) % dims1;
				const auto  z = bit / (dims0 * dims1);
				const float lo[3]{ a_shape.min[0] + a_shape.cell[0] * static_cast<float>(x), a_shape.min[1] + a_shape.cell[1] * static_cast<float>(y), a_shape.min[2] + a_shape.cell[2] * static_cast<float>(z) };
				++a_cellsTested;
				const auto cell = box(lo, a_shape.cell);
				if (cell == Verdict::kHidden) {
					anyHidden = true;
				} else if (cell != Verdict::kOutside) {
					a_hint = static_cast<std::uint8_t>(bit);
					return cell;  // part of the mesh can be seen
				}
			}
			return anyHidden ? Verdict::kHidden : Verdict::kOutside;
		}

		Verdict Test(const FrameContext& a_context, const RE::NiBound& a_bound, DepthRect* a_rect = nullptr, LightTest* a_light = nullptr, VisibleEvidence* a_evidence = nullptr) noexcept
		{
			auto radius = a_bound.fRadius;
			if (!(radius > 0.0f) || !std::isfinite(radius) || radius > 1.0e6f ||
				!std::isfinite(a_bound.center.x) || !std::isfinite(a_bound.center.y) || !std::isfinite(a_bound.center.z)) {
				return Verdict::kInvalid;
			}
			// Grow the bound by how far the camera moved since the depth was rendered (parallax can
			// uncover it). The same growth covers the shift of the current view's footprint.
			radius += a_context.dilateMove;
			if (a_context.snapshot->camera.viewSpace) {
				return TestViewSpace(a_context, a_bound, radius, a_rect, a_light, a_evidence);
			}
			// (no view-space form: the turn tolerance is covered by the distance from the eye, which bounds the depth)
			const auto& eye = a_context.snapshot->camera.eye;
			const float dx = a_bound.center.x - eye[0], dy = a_bound.center.y - eye[1], dz = a_bound.center.z - eye[2];
			radius += std::sqrt(dx * dx + dy * dy + dz * dz) * a_context.angularSlack;
			return TestCorners(a_context, a_bound, radius, a_rect, a_light);
		}

		void CountVerdict(Verdict a_verdict) noexcept
		{
			switch (a_verdict) {
			case Verdict::kVisible:
				Bump(kVisible);
				break;
			case Verdict::kEdge:
				Bump(kEdge);
				break;
			case Verdict::kOutside:
				Bump(kOutside);
				break;
			case Verdict::kBehind:
				Bump(kBehind);
				break;
			case Verdict::kNear:
				Bump(kNear);
				break;
			case Verdict::kInvalid:
				Bump(kInvalid);
				break;
			default:
				break;
			}
		}

		std::atomic<float> g_lastRejected{ 0.0f };
		std::atomic<float> g_lastTested{ 0.0f };
		std::atomic<float> g_lastLights{ 0.0f };
		std::atomic<float> g_lastLightsRejected{ 0.0f };

		enum class Exemption : std::uint8_t
		{
			kNone,
			kType,
			kLight,
			kActor,
		};

		Exemption Classify(RE::NiAVObject* a_object, bool& a_isLight) noexcept
		{
			const auto type = TypeOf(a_object);
			a_isLight = (type & kTypeLight) != 0;
			if (!(type & kTypeCullable)) {
				return Exemption::kType;
			}
			if (!g_tunables.cullActors && InsideActor(a_object)) {
				return Exemption::kActor;
			}
			// A node that carries lights is never cut as a whole: its bound needn't cover their reach. Nor is
			// one carrying an NPC (whatever bCullActors says): the NPC is judged by its own, current bound.
			if (!a_isLight) {
				const auto contents = ScanSubtree(a_object);
				if (contents & kSubtreeLight) {
					return Exemption::kLight;
				}
				if (contents & kSubtreeActor) {
					return Exemption::kActor;
				}
			}
			return Exemption::kNone;
		}

		// A light's reach: its world position with the largest of the scene-graph bound radius, its own
		// model bound (NiLight +0x150) and its light radius, all scaled to world. The engine keeps the
		// radius in NiLight::spec.r (+0x138; the paraboloid shadow setup reads it from there too), and
		// it can be far larger than the bounds (742 vs 163 in the v1.6 run). Tiny reach means unknown:
		// never cull it.
		std::atomic<int> g_lightSamplesLogged{ 0 };

		bool LightReach(RE::NiAVObject* a_light, const RE::NiBound& a_bound, RE::NiBound& a_out) noexcept
		{
			const auto  light = reinterpret_cast<const CBRO::Engine::NiLightView*>(a_light);
			const float scale = std::isfinite(a_light->world.scale) && a_light->world.scale > 0.0f ? a_light->world.scale : 1.0f;
			const float model = light->modelBound.fRadius * scale;
			const float radius = light->spec.r * scale;
			const float world = a_bound.fRadius;
			const auto  valid = [](float a_value) { return std::isfinite(a_value) ? a_value : 0.0f; };
			a_out.center = a_light->world.translate;
			a_out.fRadius = std::max({ valid(model), valid(radius), valid(world) });

			if (g_lightSamplesLogged.load(std::memory_order_relaxed) < 12 && g_lightSamplesLogged.fetch_add(1) < 12) {
				char type[48]{};
				Util::TryGetRTTIName(a_light, type, sizeof(type));
				logger::info(
					"occlusion: light sample {} worldBound r={:.0f} modelBound r={:.0f} radius={:.0f} scale={:.2f} -> reach {:.0f}",
					type, world, light->modelBound.fRadius, light->spec.r, scale, a_out.fRadius);
			}
			return a_out.fRadius >= 16.0f && a_out.fRadius < 1.0e6f;
		}

		const FrameContext* CurrentContext() noexcept
		{
			if (!g_active.load(std::memory_order_relaxed)) {
				return nullptr;
			}
			const auto index = g_context.load(std::memory_order_acquire);
			if (index < 0) {
				return nullptr;
			}
			const auto& context = g_contexts[index];
			return context.cull && context.snapshot ? &context : nullptr;
		}

		// Looks up (and once per epoch re-classifies) an object's history entry.
		Entry* Lookup(const FrameContext& a_context, RE::NiAVObject* a_object, std::uint64_t& a_value) noexcept
		{
			const auto key = reinterpret_cast<std::uintptr_t>(a_object);
			const auto entry = FindEntry(key);
			if (!entry) {
				Bump(kTableFull);
				return nullptr;
			}
			// Re-classify each object once per epoch, staggered by address so it never happens all at once.
			const auto epoch = static_cast<std::uint64_t>(((a_context.clock + static_cast<std::uint32_t>((key >> 4) % kEpochFrames)) / kEpochFrames) & 0xFF);
			auto       value = entry->value.load(std::memory_order_relaxed);
			if (!(value & kClassified) || ((value >> kEpochShift) & 0xFF) != epoch) {
				bool       isLight = false;
				const auto exemption = Classify(a_object, isLight);
				const bool merged = exemption == Exemption::kNone && Hooks::CullGroups::IsMergeInstanced(a_object);
				value = (value & (kLowBits | (std::uint64_t{ 0xFF } << kHintShift))) | kClassified | (epoch << kEpochShift) |
				        (isLight ? kIsLight : 0) | (merged ? kIsMerged : 0);
				if (exemption != Exemption::kNone) {
					value |= kExempt | (static_cast<std::uint64_t>(exemption) << 42);  // bits 42-43: why, for the stats
				}
				entry->value.store(value, std::memory_order_relaxed);
			}
			a_value = value;
			return entry;
		}

		std::uint32_t NextStreak(std::uint64_t a_value, bool a_hidden, std::uint32_t a_clock) noexcept
		{
			if (!a_hidden) {
				return 0;
			}
			const auto streak = static_cast<std::uint32_t>((a_value >> 32) & 0xFF);
			return static_cast<std::uint32_t>(a_value) != a_clock ? std::min(streak + 1, 255u) : streak;
		}

		// A backup of an object's view streak that doesn't depend on which thread judged it (history table value bits
		// [0,40): the frame of its last hidden verdict and the streak then; merge-instanced meshes keep their own verdict
		// there). The streams carry the streak on the thread that judged the object, in walk order; when the engine's
		// jobs hand a block of objects to another thread, reorder a node's children, or a child was dropped with its
		// parent (not judged at all), the stream has no record and the object restarted its confirmation: drawn again
		// for a frame, a node's ~600 children at once in the v1.49 interior log ("confirming" max ~597 per interval,
		// ~54 records a frame found missing with the camera still).
		constexpr std::uint32_t kStreakBackupGap = 8;      // frames a backup stays usable
		constexpr std::uint32_t kStreakRefreshFrames = 4;  // reused and inherited hidden verdicts refresh it this often

		// The streak a hidden verdict continues from the backup (0: none usable).
		std::uint32_t BackupStreak(std::uint64_t a_value, std::uint32_t a_clock) noexcept
		{
			if (a_value & kIsMerged) {
				return 0;
			}
			const auto clock = static_cast<std::uint32_t>(a_value);
			const auto streak = static_cast<std::uint32_t>((a_value >> 32) & 0xFF);
			if (streak == 0 || a_clock - clock > kStreakBackupGap) {
				return 0;
			}
			return clock == a_clock ? streak : std::min(streak + 1, 255u);
		}

		void StoreStreak(Entry& a_entry, std::uint32_t a_clock, std::uint32_t a_streak) noexcept
		{
			auto value = a_entry.value.load(std::memory_order_relaxed);
			for (;;) {
				if (value & kIsMerged) {
					return;
				}
				const auto next = (value & ~kLowBits) | (static_cast<std::uint64_t>(std::min(a_streak, 255u)) << 32) | a_clock;
				if (next == value || a_entry.value.compare_exchange_weak(value, next, std::memory_order_relaxed)) {
					return;
				}
			}
		}

		bool RefreshTurn(const void* a_object, std::uint32_t a_clock) noexcept
		{
			return (a_clock + static_cast<std::uint32_t>(reinterpret_cast<std::uintptr_t>(a_object) >> 4)) % kStreakRefreshFrames == 0;
		}

		// A child dropped with its parent: hidden by implication (inside the parent's confirmed-hidden bound), so its
		// backup continues, at least at the confirmation length.
		void InheritHidden(const void* a_object, std::uint32_t a_clock, std::uint32_t a_confirmFrames) noexcept
		{
			if (!RefreshTurn(a_object, a_clock)) {
				return;
			}
			if (const auto entry = FindEntry(reinterpret_cast<std::uintptr_t>(a_object))) {
				StoreStreak(*entry, a_clock, std::max(BackupStreak(entry->value.load(std::memory_order_relaxed), a_clock), a_confirmFrames));
			}
		}

		bool Observing() noexcept
		{
			return g_tunables.observeOnly || g_observeOverride.load(std::memory_order_relaxed);
		}

		// ---- the Block::Add filter (an object's own entry) --------------------------------------

		// ---- kept-object dump (diagnostics, one sampled frame per location) ---------------------------
		// Records every object the main pass keeps on one frame (visible, near, edge, bad bound, exempt,
		// confirming) with its type, name, size and, for meshes, where its vertex data lives, so the
		// leftover draw calls can be attributed. Details are read while the object is certainly alive.

		struct KeptItem
		{
			const char*    reason{ "" };
			float          radius{ 0.0f };
			float          distance{ 0.0f };
			char           type[40]{};
			char           name[48]{};
			bool           geometry{ false };
			bool           skinned{ false };
			bool           dropped{ false };  // not kept: a mesh shape settled it (listed apart, to audit shapes)
			std::uint64_t  vertexDesc{ 0 };
			std::uintptr_t buffer{ 0 };
			std::uintptr_t data{ 0 };
			std::uint32_t  dataSize{ 0 };
			std::uint32_t  dataOffset{ 0 };
			std::uint32_t  vertices{ 0 };
			std::uint32_t  byteWidth{ 0 };
			std::uint32_t  usage{ 0 };
			// BSMergeInstancedTriShape only: its per-instance transform buffer (+0x170, 80-byte records that
			// start with an NiTransform relative to the node, per the precombine builder), to judge its
			// instances one by one later. Raw words, since the buffer object's layout isn't confirmed.
			bool           merged{ false };
			std::uintptr_t instanceObject{ 0 };
			std::uint64_t  instanceWords[5]{};
			std::uint32_t  instanceCount{ 0 };  // +0x34
			std::uint32_t  instanceKind{ 0 };   // +0x38
			std::uint32_t  lodTriangles[4]{};   // +0x1A0
			float          localTranslate[3]{};
			float          modelBound[4]{};
			std::uint32_t  instanceBytes{ 0 };
			std::uint32_t  instanceStride{ 0 };
			std::uint32_t  instanceBind{ 0 };
			std::uint32_t  instanceMisc{ 0 };
			std::uint32_t  instanceUsage{ 0 };
			std::int32_t   instanceWord{ -1 };  // which word held the ID3D11Buffer
			char           instanceModule[24]{};
		};

		std::atomic<std::uint32_t> g_dumpClock{ 0 };
		std::mutex                 g_keptLock;
		std::vector<KeptItem>      g_kept;
		std::uintptr_t             g_mergedVtable{ 0 };

		// BSGeometry: skinInstance +0x140, rendererData (BSGraphics::TriShape*) +0x148; TriShape:
		// vertexDesc +0, vertexBuffer +8; Buffer: ID3D11Buffer* +0, data +8, dataSize +0x34,
		// dataOffset +0x48; BSTriShape::numVertices +0x164.
		void DescribeGeometry(const RE::NiAVObject* a_object, KeptItem& a_item) noexcept
		{
			__try {
				const auto geometry = const_cast<RE::NiAVObject*>(a_object)->IsGeometry();
				if (!geometry) {
					return;
				}
				const auto base = reinterpret_cast<const std::byte*>(geometry);
				a_item.geometry = true;
				a_item.skinned = *reinterpret_cast<const std::uintptr_t*>(base + 0x140) != 0;
				a_item.vertices = *reinterpret_cast<const std::uint16_t*>(base + 0x164);
				const auto triShape = *reinterpret_cast<const std::byte* const*>(base + 0x148);
				if (!triShape) {
					return;
				}
				a_item.vertexDesc = *reinterpret_cast<const std::uint64_t*>(triShape);
				const auto vertexBuffer = *reinterpret_cast<const std::byte* const*>(triShape + 0x8);
				if (!vertexBuffer) {
					return;
				}
				a_item.buffer = *reinterpret_cast<const std::uintptr_t*>(vertexBuffer);
				a_item.data = *reinterpret_cast<const std::uintptr_t*>(vertexBuffer + 0x8);
				a_item.dataSize = *reinterpret_cast<const std::uint32_t*>(vertexBuffer + 0x34);
				a_item.dataOffset = *reinterpret_cast<const std::uint32_t*>(vertexBuffer + 0x48);
			} __except (EXCEPTION_EXECUTE_HANDLER) {
				a_item.geometry = false;
			}
		}

		void DescribeMerged(const RE::NiAVObject* a_object, KeptItem& a_item) noexcept
		{
			__try {
				const auto base = reinterpret_cast<const std::byte*>(a_object);
				if (!g_mergedVtable || *reinterpret_cast<const std::uintptr_t*>(base) != g_mergedVtable) {
					return;
				}
				a_item.merged = true;
				a_item.localTranslate[0] = a_object->local.translate.x;
				a_item.localTranslate[1] = a_object->local.translate.y;
				a_item.localTranslate[2] = a_object->local.translate.z;
				const auto& model = *reinterpret_cast<const RE::NiBound*>(base + 0x120);
				a_item.modelBound[0] = model.center.x;
				a_item.modelBound[1] = model.center.y;
				a_item.modelBound[2] = model.center.z;
				a_item.modelBound[3] = model.fRadius;
				for (int i = 0; i < 4; ++i) {
					a_item.lodTriangles[i] = *reinterpret_cast<const std::uint32_t*>(base + 0x1A0 + 4 * i);
				}
				const auto object = *reinterpret_cast<const std::byte* const*>(base + 0x170);
				a_item.instanceObject = reinterpret_cast<std::uintptr_t>(object);
				if (!object) {
					return;
				}
				for (int i = 0; i < 5; ++i) {
					a_item.instanceWords[i] = *reinterpret_cast<const std::uint64_t*>(object + 8 * i);
				}
				a_item.instanceCount = *reinterpret_cast<const std::uint32_t*>(object + 0x34);
				a_item.instanceKind = *reinterpret_cast<const std::uint32_t*>(object + 0x38);
			} __except (EXCEPTION_EXECUTE_HANDLER) {
			}
		}

		// Finds which of the instance buffer object's words is an ID3D11Buffer: only an object whose vtable
		// lies in a module named d3d11.dll (the runtime or a wrapper such as ENB's) is ever called.
		void DescribeInstanceBuffer(KeptItem& a_item) noexcept
		{
			for (int i = 0; i < 5; ++i) {
				const auto candidate = static_cast<std::uintptr_t>(a_item.instanceWords[i]);
				const auto vtable = Util::TryReadVtable(reinterpret_cast<const void*>(candidate));
				HMODULE    module = nullptr;
				if (!vtable || !GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT, reinterpret_cast<LPCWSTR>(vtable), &module)) {
					continue;
				}
				char path[MAX_PATH]{};
				GetModuleFileNameA(module, path, MAX_PATH);
				const std::string_view full(path);
				const auto             slash = full.find_last_of("\\/");
				const auto             file = slash == std::string_view::npos ? full : full.substr(slash + 1);
				strncpy_s(a_item.instanceModule, file.data(), _TRUNCATE);
				if (_stricmp(a_item.instanceModule, "d3d11.dll") != 0) {
					continue;
				}
				ID3D11Buffer* buffer = nullptr;
				if (FAILED(reinterpret_cast<IUnknown*>(candidate)->QueryInterface(__uuidof(ID3D11Buffer), reinterpret_cast<void**>(&buffer))) || !buffer) {
					continue;
				}
				D3D11_BUFFER_DESC desc{};
				buffer->GetDesc(&desc);
				buffer->Release();
				a_item.instanceWord = i;
				a_item.instanceBytes = desc.ByteWidth;
				a_item.instanceStride = desc.StructureByteStride;
				a_item.instanceBind = desc.BindFlags;
				a_item.instanceMisc = desc.MiscFlags;
				a_item.instanceUsage = static_cast<std::uint32_t>(desc.Usage);
				return;
			}
		}

		void RecordKept(const FrameContext& a_context, const RE::NiAVObject* a_object, const RE::NiBound& a_bound, const char* a_reason, bool a_dropped = false) noexcept
		{
			if (a_context.clock != g_dumpClock.load(std::memory_order_relaxed)) {
				return;
			}
			KeptItem item{};
			item.reason = a_reason;
			item.dropped = a_dropped;
			item.radius = a_bound.fRadius;
			const auto& camera = a_context.snapshot->camera;
			const float dx = a_bound.center.x - camera.eye[0];
			const float dy = a_bound.center.y - camera.eye[1];
			const float dz = a_bound.center.z - camera.eye[2];
			item.distance = std::sqrt(dx * dx + dy * dy + dz * dz);
			Util::TryGetRTTIName(a_object, item.type, sizeof(item.type));
			Util::TryGetObjectName(a_object, item.name, sizeof(item.name));
			DescribeGeometry(a_object, item);
			if (item.buffer) {
				D3D11_BUFFER_DESC desc{};
				reinterpret_cast<ID3D11Buffer*>(item.buffer)->GetDesc(&desc);
				item.byteWidth = desc.ByteWidth;
				item.usage = static_cast<std::uint32_t>(desc.Usage);
			}
			DescribeMerged(a_object, item);
			if (item.merged && item.instanceObject) {
				DescribeInstanceBuffer(item);
			}
			std::scoped_lock lock(g_keptLock);
			if (g_kept.size() < 8192) {
				g_kept.push_back(item);
			}
		}

		const char* VerdictName(Verdict a_verdict) noexcept
		{
			switch (a_verdict) {
			case Verdict::kVisible:
				return "visible";
			case Verdict::kEdge:
				return "edge";
			case Verdict::kNear:
				return "near";
			case Verdict::kInvalid:
				return "bad-bound";
			default:
				return "?";
			}
		}

		// ---- left-out dump (diagnostics: the frame before the kept dump's) ----------------------------
		// Every object the main view leaves out on one steady frame (verdicts reused and from the async map as usual),
		// with how it was left out, where its verdict came from, what a fresh test of its bound says against this
		// frame's depth, and whether that bound holds the mesh's own vertices (meshes with a CPU copy). The bound is all
		// CBRO and the engine's frustum test look at: a mesh reaching outside it can be hidden where it is seen.

		enum class LeftOutPath : std::uint8_t
		{
			kRejected,        // main-only group: confirmed hidden, its entry rejected
			kCasterRejected,  // group 0: neither the main view nor the sun needs it (rejected in every view)
			kDropped,         // group 0 or another shared group: left out of the main view's registration only
			kInherited,       // its parent was left out of the main view: so is it
			kSkipHidden,      // a top-level add never filed: confirmed hidden
			kSkipOutside,     // ... out of view
			kSkipCaster,      // ... group 0, neither view needs it
			kOutside,         // filed, judged out of view: the engine's own frustum test drops it from the main view
			kNodePruned,      // never filed: its cell node or container was left out whole (SkipCellNode; recorded on the dump frame only, v1.92.7)
		};
		constexpr const char* kLeftOutPathNames[]{ "rejected", "rejected in every view", "main view drop", "parent dropped", "never filed: hidden", "never filed: out of view", "never filed: no view needs it", "out of view (engine's frustum test)", "never filed: its node pruned whole" };

		enum class VerdictSource : std::uint8_t
		{
			kFresh,   // evaluated this frame
			kReused,  // last frame's record still held
			kAsync,   // the async worker's map
			kNone,    // not judged (inherited)
		};
		constexpr const char* kVerdictSourceNames[]{ "fresh", "reused", "async map", "not judged" };
		thread_local VerdictSource t_source{ VerdictSource::kNone };  // set by JudgeOrLookup / JudgeRecord

		struct LeftOutItem
		{
			const void*   object{ nullptr };
			const void*   parent{ nullptr };
			LeftOutPath   path{ LeftOutPath::kRejected };
			VerdictSource source{ VerdictSource::kNone };
			Outcome       outcome{ Outcome::kNone };
			std::uint8_t  verdict{ 0 };
			std::uint8_t  streak{ 0 };
			Verdict       now{ Verdict::kInvalid };  // a fresh test of the same bound against this frame's depth
			std::uint32_t fartherTexels{ 0 };        // ... when visible: level-0 texels beyond its nearest point
			bool          shaped{ false };           // ... the sphere couldn't settle it and the mesh has a shape:
			Verdict       shapeNow{ Verdict::kInvalid };  // what the shape test says now
			// (BruteShape, for shaped meshes crossing the ray window)
			std::int8_t   bruteRead{ -1 };  // -1 not checked, 0 the mesh couldn't be read, 1 read
			bool          bruteCached{ false }, bruteFresh{ false }, bruteSame{ false };
			std::uint32_t bruteTriangles{ 0 }, bruteInView{ 0 }, bruteVisible{ 0 }, bruteUnoccupied{ 0 };
			std::uint8_t  bruteDims[3]{}, bruteCount{ 0 };
			// v1.92.10: of the in-view triangles, those whose screen box crosses the ray window (the region a missing object
			// was seen in), how many the box test finds visible, and the first few in detail: where the test looked and what
			// the depth held there (a hidden verdict with a far surface over the box means the test, not the depth, is off).
			std::uint32_t bruteWindow{ 0 }, bruteWindowVisible{ 0 };
			struct WindowSample
			{
				float   zNear{ 0.0f }, zFar{ 0.0f };  // the triangle's view depth range
				float   ndc[4]{};                      // its screen box: x0, y0, x1, y1
				float   farthest{ 0.0f };              // view depth of the farthest surface over that box (Hi-Z)
				float   threshold{ 0.0f };             // view depth the hidden verdict compared against (0 when visible)
				Verdict verdict{ Verdict::kInvalid };
			};
			WindowSample  windowSample[3]{};
			std::uint8_t  windowSamples{ 0 };
			float         bound[4]{};
			float         ownOffset{ 0.0f };  // the object's own worldBound against the bound offered: center distance
			float         ownRadius{ 0.0f };  // ... and its radius
			float         distance{ 0.0f };
			float         ndc[2]{};
			bool          inFront{ false };
			// Meshes with a CPU copy of their vertices: how many lie outside the bound offered, and by how much.
			std::int32_t  vertices{ -1 };  // -1: not checked (no copy, skinned, merged, too many)
			std::uint32_t verticesInView{ 0 };  // ... of them inside the current view (in front of the near distance)
			bool          boundFits{ false };   // the world bound is the model bound moved by the world transform (WorldRotation)
			std::uint16_t descFlags{ 0 };       // the vertex desc's attribute flags (bit 9 instance, bit 6 skinned, bit 10 full precision)
			std::uint32_t verticesOutside{ 0 };
			float         excess{ 0.0f };
			float         vertexNdc[2]{};  // the vertex box's center on screen
			bool          vertexInFront{ false };
			float         occluder{ 0.0f };  // view depth of the farthest surface over its screen rectangle (Hi-Z, coarse level)
			float         viewDepth{ 0.0f };  // its center's view depth
			std::uint32_t refID{ 0 };
			std::uint32_t baseID{ 0 };
			std::uint8_t  baseType{ 0 };
			char          type[32]{};
			char          name[48]{};
			char          ancestry[128]{};  // parent < grandparent < ...: type 'name'
		};

		std::atomic<std::uint32_t> g_leftOutClock{ 0 };  // 0: no left-out dump pending
		std::vector<LeftOutItem>   g_leftOut;  // (under g_keptLock)

		bool LeftOutFrame(std::uint32_t a_clock) noexcept
		{
			const auto clock = g_leftOutClock.load(std::memory_order_relaxed);
			return clock != 0 && clock == a_clock;
		}
		constexpr std::uint32_t    kMaxCheckedVertices = 262144;

		// The same frame's depth as CBRO reads it, per 8x8 block of Hi-Z level 0: the farthest and the nearest surface
		// (view depth), and the camera, taken by the first RecordLeftOut of the frame.
		struct DepthPicture
		{
			std::uint32_t      width{ 0 }, height{ 0 };
			std::vector<float> farthest, nearest;
			float              eye[3]{}, dir[3]{};
		};
		std::atomic<std::uint32_t> g_pictureClock{ 0 };
		DepthPicture               g_picture;  // (under g_keptLock)
		HiZ::Camera                g_pictureCamera{};  // (under g_keptLock) the depth frame's camera, for the scene scan
		bool                       g_pictureCameraValid{ false };
		HiZ::Camera                g_rayCamera{};  // (under g_keptLock) the same camera, kept for the set diff's ray window (v1.92.10: the scan consumes the flag above)
		bool                       g_rayCameraValid{ false };
		std::unordered_set<const void*> g_registered;  // (under g_keptLock) objects the main accumulator registered on that frame
		// (under g_keptLock) objects offered to the main pass on that frame: bit 0 a Block::Add entry, bit 1 a top-level Group::Add
		std::unordered_map<const void*, std::uint8_t> g_offered;

		void NoteOffered(const FrameContext& a_context, const void* a_object, std::uint8_t a_how) noexcept
		{
			if (!LeftOutFrame(a_context.clock)) {
				return;
			}
			std::scoped_lock lock(g_keptLock);
			g_offered[a_object] |= a_how;
		}

		void TakeDepthPicture(const FrameContext& a_context) noexcept
		{
			const auto& snapshot = *a_context.snapshot;
			const auto& camera = snapshot.camera;
			DepthPicture picture{};
			picture.width = snapshot.width[0] / 8;
			picture.height = snapshot.height[0] / 8;
			picture.farthest.assign(static_cast<std::size_t>(picture.width) * picture.height, 0.0f);
			picture.nearest.assign(picture.farthest.size(), std::numeric_limits<float>::infinity());
			for (int i = 0; i < 3; ++i) {
				picture.eye[i] = camera.eye[i];
				picture.dir[i] = camera.viewDir[i];
			}
			const auto* farthest = snapshot.texels + snapshot.offset[0];
			const auto* nearest = snapshot.nearest ? snapshot.nearest + snapshot.offset[0] : nullptr;
			for (std::uint32_t y = 0; y < picture.height * 8; ++y) {
				for (std::uint32_t x = 0; x < picture.width * 8; ++x) {
					const auto index = static_cast<std::size_t>(y) * snapshot.width[0] + x;
					const auto block = static_cast<std::size_t>(y / 8) * picture.width + x / 8;
					picture.farthest[block] = std::max(picture.farthest[block], LinearDepth(camera, farthest[index]));
					if (nearest) {
						picture.nearest[block] = std::min(picture.nearest[block], LinearDepth(camera, nearest[index]));
					}
				}
			}
			std::scoped_lock lock(g_keptLock);
			g_picture = std::move(picture);
			g_pictureCamera = camera;
			g_pictureCameraValid = camera.viewSpace;
			g_rayCamera = camera;
			g_rayCameraValid = camera.viewSpace;
		}

		const char* FullVerdictName(Verdict a_verdict) noexcept
		{
			switch (a_verdict) {
			case Verdict::kHidden:
				return "hidden";
			case Verdict::kOutside:
				return "out of view";
			case Verdict::kBehind:
				return "behind";
			default:
				return VerdictName(a_verdict);
			}
		}

		bool ProjectToView(const HiZ::Camera& a_camera, float a_x, float a_y, float a_z, float a_ndc[2]) noexcept
		{
			const float dx = a_x - a_camera.origin[0], dy = a_y - a_camera.origin[1], dz = a_z - a_camera.origin[2];
			const float z = dx * a_camera.viewDir[0] + dy * a_camera.viewDir[1] + dz * a_camera.viewDir[2];
			const float x = dx * a_camera.viewRight[0] + dy * a_camera.viewRight[1] + dz * a_camera.viewRight[2];
			const float y = dx * a_camera.viewUp[0] + dy * a_camera.viewUp[1] + dz * a_camera.viewUp[2];
			if (!(z > 1.0f)) {
				return false;
			}
			a_ndc[0] = a_camera.scaleX * x / z;
			a_ndc[1] = a_camera.scaleY * y / z;
			return true;
		}

		// The reference owning the object (the nearest ancestor whose userData is a REFR or ACHR) and its base form.
		void FindReference(const RE::NiAVObject* a_object, LeftOutItem& a_item) noexcept
		{
			__try {
				const RE::NiAVObject* node = a_object;
				for (int depth = 0; node && depth < 32; ++depth, node = node->parent) {
					if (!node->userData) {
						continue;
					}
					const auto form = reinterpret_cast<const RE::TESForm*>(node->userData);
					const auto type = form->GetFormType();
					if (type != RE::ENUM_FORM_ID::kREFR && type != RE::ENUM_FORM_ID::kACHR) {
						continue;
					}
					a_item.refID = form->GetFormID();
					if (const auto base = reinterpret_cast<const RE::TESObjectREFR*>(form)->GetObjectReference()) {
						a_item.baseID = base->GetFormID();
						a_item.baseType = static_cast<std::uint8_t>(base->GetFormType());
					}
					return;
				}
			} __except (EXCEPTION_EXECUTE_HANDLER) {
			}
		}

		const RE::NiAVObject* TryParent(const RE::NiAVObject* a_object) noexcept
		{
			__try {
				return a_object->parent;
			} __except (EXCEPTION_EXECUTE_HANDLER) {
				return nullptr;
			}
		}

		void DescribeAncestry(const RE::NiAVObject* a_object, char* a_out, std::size_t a_size) noexcept
		{
			std::size_t used = 0;
			auto        node = TryParent(a_object);
			for (int depth = 0; node && depth < 3 && used + 1 < a_size; ++depth, node = TryParent(node)) {
				char type[32]{}, name[40]{};
				Util::TryGetRTTIName(node, type, sizeof(type));
				Util::TryGetObjectName(node, name, sizeof(name));
				const auto written = std::snprintf(a_out + used, a_size - used, "%s%s '%s'", depth ? " < " : "", type, name);
				if (written < 0) {
					break;
				}
				used = std::min(a_size - 1, used + static_cast<std::size_t>(written));
			}
		}

		// The mesh's CPU vertices placed in the world (the rotation reading the shape test verified) against the bound
		// offered: how many lie outside it, and the box they span (all reads guarded).
		void CheckVertices(const RE::NiAVObject* a_object, const float a_rotate[3][3], LeftOutItem& a_item, float a_lo[3], float a_hi[3], const HiZ::Camera* a_camera) noexcept
		{
			__try {
				const auto base = reinterpret_cast<const std::byte*>(a_object);
				if (*reinterpret_cast<const std::uintptr_t*>(base + 0x140) != 0) {
					return;  // skinned
				}
				const auto triShape = *reinterpret_cast<const std::byte* const*>(base + 0x148);
				if (!triShape) {
					return;
				}
				const auto desc = *reinterpret_cast<const std::uint64_t*>(triShape);
				a_item.descFlags = static_cast<std::uint16_t>(desc >> 44);
				const auto vertexBuffer = *reinterpret_cast<const std::byte* const*>(triShape + 0x8);
				if (!vertexBuffer) {
					return;
				}
				const auto          data = *reinterpret_cast<const std::byte* const*>(vertexBuffer + 0x8);
				const auto          bytes = *reinterpret_cast<const std::uint32_t*>(vertexBuffer + 0x34);
				const std::uint32_t stride = static_cast<std::uint32_t>(desc & 0xF) * 4;
				const std::uint32_t position = static_cast<std::uint32_t>((desc >> 2) & 0x3C);
				const bool          full = ((desc >> 44) & (1ull << 10)) != 0;
				const std::uint32_t count = *reinterpret_cast<const std::uint16_t*>(base + 0x164);
				if (!data || count == 0 || count > kMaxCheckedVertices || stride < position + (full ? 12u : 6u) ||
					static_cast<std::uint64_t>(count) * stride > bytes) {
					return;
				}
				const auto& transform = a_object->world;
				const float translate[3]{ transform.translate.x, transform.translate.y, transform.translate.z };
				const float slack = a_item.bound[3] + std::max(1.0f, a_item.bound[3] * 0.01f);
				for (int i = 0; i < 3; ++i) {
					a_lo[i] = std::numeric_limits<float>::max();
					a_hi[i] = -std::numeric_limits<float>::max();
				}
				for (std::uint32_t v = 0; v < count; ++v) {
					const auto* p = data + static_cast<std::size_t>(v) * stride + position;
					float       local[3];
					for (int k = 0; k < 3; ++k) {
						local[k] = full ? reinterpret_cast<const float*>(p)[k] : _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128(reinterpret_cast<const std::uint16_t*>(p)[k])));
					}
					float world[3];
					for (int i = 0; i < 3; ++i) {
						world[i] = translate[i] + transform.scale * (a_rotate[i][0] * local[0] + a_rotate[i][1] * local[1] + a_rotate[i][2] * local[2]);
						a_lo[i] = std::min(a_lo[i], world[i]);
						a_hi[i] = std::max(a_hi[i], world[i]);
					}
					const float dx = world[0] - a_item.bound[0], dy = world[1] - a_item.bound[1], dz = world[2] - a_item.bound[2];
					const float d = std::sqrt(dx * dx + dy * dy + dz * dz);
					if (d > slack) {
						++a_item.verticesOutside;
						a_item.excess = std::max(a_item.excess, d - a_item.bound[3]);
					}
					if (a_camera) {
						const float rel[3]{ world[0] - a_camera->origin[0], world[1] - a_camera->origin[1], world[2] - a_camera->origin[2] };
						const float vz = rel[0] * a_camera->viewDir[0] + rel[1] * a_camera->viewDir[1] + rel[2] * a_camera->viewDir[2];
						const float vx = rel[0] * a_camera->viewRight[0] + rel[1] * a_camera->viewRight[1] + rel[2] * a_camera->viewRight[2];
						const float vy = rel[0] * a_camera->viewUp[0] + rel[1] * a_camera->viewUp[1] + rel[2] * a_camera->viewUp[2];
						a_item.verticesInView += vz > g_tunables.nearDistance && std::abs(a_camera->scaleX * vx) <= vz && std::abs(a_camera->scaleY * vy) <= vz;
					}
				}
				a_item.vertices = static_cast<std::int32_t>(count);
			} __except (EXCEPTION_EXECUTE_HANDLER) {
				a_item.vertices = -1;
			}
		}

		// ---- kept in the middle of the screen (same frame): what the engine does with an object CBRO keeps --------------
		// For each kept entry whose bound center lies in the middle third of the screen: its fade node (itself, or a mesh's
		// shader property +0x48) and that node's fade and mesh-LOD state (FO4-ENGINE-NOTES 7.8, 7.9, 7.11). A node with
		// currentFade 0 files no meshes; a mesh-LOD level below 3 draws only the first segments of a BSMeshLODTriShape.
		struct KeptRegionItem
		{
			const void*   object{ nullptr };
			const void*   fadeNode{ nullptr };
			Hooks::CullGroups::GroupKind kind{ Hooks::CullGroups::GroupKind::kUnknown };
			float         bound[4]{};
			float         distance{ 0.0f };
			float         ndc[2]{};
			bool          fadeRead{ false };
			float         fadeAmount{ 0.0f }, currentFade{ 0.0f }, nearDistance{ 0.0f }, farDistance{ 0.0f }, boundRadius{ 0.0f }, lodBlend{ 0.0f };
			std::uint8_t  fadeType{ 0 }, lodState{ 0 }, lodLevel{ 0 }, lodPrevious{ 0 };
			std::uint64_t fadeFlags{ 0 };
			std::int64_t  unseenFrames{ 0 };  // frame counter - the node's last OnVisible stamp
			std::uint32_t lodSizes[3]{};      // BSMeshLODTriShape only
			bool          meshLOD{ false };
			std::uint32_t refID{ 0 };
			std::uint32_t baseID{ 0 };
			char          type[32]{};
			char          name[48]{};
			char          ancestry[128]{};
		};
		std::vector<KeptRegionItem> g_keptRegion;  // (under g_keptLock)

		bool ReadFadeState(const std::byte* a_node, std::uint32_t a_frame, KeptRegionItem& a_item) noexcept
		{
			__try {
				a_item.fadeAmount = *reinterpret_cast<const float*>(a_node + 0x118);
				a_item.fadeType = *reinterpret_cast<const std::uint8_t*>(a_node + 0x11C);
				a_item.lodState = *reinterpret_cast<const std::uint8_t*>(a_node + 0x11D);
				a_item.lodLevel = *reinterpret_cast<const std::uint8_t*>(a_node + 0x11E);
				a_item.lodPrevious = *reinterpret_cast<const std::uint8_t*>(a_node + 0x11F);
				a_item.lodBlend = *reinterpret_cast<const float*>(a_node + 0x13C);
				a_item.nearDistance = *reinterpret_cast<const float*>(a_node + 0x198);
				a_item.farDistance = *reinterpret_cast<const float*>(a_node + 0x19C);
				a_item.currentFade = *reinterpret_cast<const float*>(a_node + 0x1A0);
				a_item.boundRadius = *reinterpret_cast<const float*>(a_node + 0x1A8);
				a_item.unseenFrames = static_cast<std::int64_t>(a_frame) - static_cast<std::int64_t>(*reinterpret_cast<const std::uint32_t*>(a_node + 0x1B0));
				a_item.fadeFlags = *reinterpret_cast<const std::uint64_t*>(a_node + 0x108);
				return true;
			} __except (EXCEPTION_EXECUTE_HANDLER) {
				return false;
			}
		}

		const std::byte* ShaderFadeNode(const RE::NiAVObject* a_geometry) noexcept
		{
			__try {
				const auto shader = *reinterpret_cast<const std::byte* const*>(reinterpret_cast<const std::byte*>(a_geometry) + 0x138);
				return shader ? *reinterpret_cast<const std::byte* const*>(shader + 0x48) : nullptr;
			} __except (EXCEPTION_EXECUTE_HANDLER) {
				return nullptr;
			}
		}

		void ReadLODSizes(const RE::NiAVObject* a_geometry, KeptRegionItem& a_item) noexcept
		{
			__try {
				for (int i = 0; i < 3; ++i) {
					a_item.lodSizes[i] = *reinterpret_cast<const std::uint32_t*>(reinterpret_cast<const std::byte*>(a_geometry) + 0x170 + 4 * i);
				}
				a_item.meshLOD = true;
			} __except (EXCEPTION_EXECUTE_HANDLER) {
			}
		}

		void RecordKeptRegion(const FrameContext& a_context, const Hooks::CullGroups::BlockAdd& a_add) noexcept
		{
			if (!LeftOutFrame(a_context.clock) || !a_context.snapshot || !a_add.object || !a_add.bound) {
				return;
			}
			const auto& camera = a_context.snapshot->camera;
			KeptRegionItem item{};
			if (!camera.viewSpace || !ProjectToView(camera, a_add.bound->center.x, a_add.bound->center.y, a_add.bound->center.z, item.ndc) ||
				std::abs(item.ndc[0]) > 1.0f / 3.0f || std::abs(item.ndc[1]) > 1.0f / 3.0f) {
				return;
			}
			if (g_drops.Contains(a_add.object, DropSet::Tag(a_context.clock))) {
				return;  // (left out of the main view: the left-out dump has it)
			}
			item.object = a_add.object;
			item.kind = a_add.kind;
			item.bound[0] = a_add.bound->center.x;
			item.bound[1] = a_add.bound->center.y;
			item.bound[2] = a_add.bound->center.z;
			item.bound[3] = a_add.bound->fRadius;
			const float dx = item.bound[0] - camera.eye[0], dy = item.bound[1] - camera.eye[1], dz = item.bound[2] - camera.eye[2];
			item.distance = std::sqrt(dx * dx + dy * dy + dz * dz);
			Util::TryGetRTTIName(a_add.object, item.type, sizeof(item.type));
			Util::TryGetObjectName(a_add.object, item.name, sizeof(item.name));
			DescribeAncestry(a_add.object, item.ancestry, sizeof(item.ancestry));
			LeftOutItem ref{};
			FindReference(a_add.object, ref);
			item.refID = ref.refID;
			item.baseID = ref.baseID;

			static const auto frameCounter = reinterpret_cast<const std::uint32_t*>(CBRO::Engine::OG(734919).address());
			const std::string_view type(item.type);
			const std::byte* node = nullptr;
			if (type == "BSFadeNode"sv || type == "BSLeafAnimNode"sv) {
				node = reinterpret_cast<const std::byte*>(a_add.object);
			} else if (const_cast<RE::NiAVObject*>(a_add.object)->IsGeometry()) {
				node = ShaderFadeNode(a_add.object);
				if (type == "BSMeshLODTriShape"sv) {
					ReadLODSizes(a_add.object, item);
				}
			}
			static const auto defaultFade = CBRO::Engine::OG(244120).address();  // (the shader property's default +0x48 object, or a pointer to it)
			if (node && reinterpret_cast<std::uintptr_t>(node) != defaultFade && reinterpret_cast<std::uintptr_t>(node) != *reinterpret_cast<const std::uintptr_t*>(defaultFade)) {
				item.fadeNode = node;
				item.fadeRead = ReadFadeState(node, *frameCounter, item);
			}
			std::scoped_lock lock(g_keptLock);
			if (g_keptRegion.size() < 16384) {
				g_keptRegion.push_back(item);
			}
		}

		// A screen window (fRayWindowX0..Y1; the default is where the 2026-10-09 wall bottom went missing, lower left) as a
		// grid of rays from the depth frame's camera: whatever is drawn there has a bound one of them crosses within
		// fRayLength. Returns how far along the nearest such ray the bound's center lies, or a negative value when no ray
		// crosses it. (v1.92.9: from the INI, so a dump can be aimed at the screen region a missing object was seen in.)
		constexpr int kRaySteps = 6;
		const float*  RayWindow() noexcept { return Settings::Get().rayWindow; }
		float         RayLength() noexcept { return Settings::Get().rayLength; }

		float RayWindowHit(const HiZ::Camera& a_camera, float a_x, float a_y, float a_z, float a_radius) noexcept
		{
			if (!(a_radius > 0.0f) || !(a_camera.scaleX > 0.0f) || !(a_camera.scaleY > 0.0f)) {
				return -1.0f;
			}
			const float c[3]{ a_x - a_camera.origin[0], a_y - a_camera.origin[1], a_z - a_camera.origin[2] };
			float       best = -1.0f;
			for (int i = 0; i < kRaySteps; ++i) {
				for (int j = 0; j < kRaySteps; ++j) {
					const float nx = RayWindow()[0] + (RayWindow()[1] - RayWindow()[0]) * (static_cast<float>(i) + 0.5f) / kRaySteps;
					const float ny = RayWindow()[2] + (RayWindow()[3] - RayWindow()[2]) * (static_cast<float>(j) + 0.5f) / kRaySteps;
					float d[3], length = 0.0f;
					for (int k = 0; k < 3; ++k) {
						d[k] = a_camera.viewDir[k] + nx / a_camera.scaleX * a_camera.viewRight[k] + ny / a_camera.scaleY * a_camera.viewUp[k];
						length += d[k] * d[k];
					}
					length = std::sqrt(length);
					const float along = (c[0] * d[0] + c[1] * d[1] + c[2] * d[2]) / length;
					const float t = std::clamp(along, 0.0f, RayLength());
					float       miss = 0.0f;
					for (int k = 0; k < 3; ++k) {
						const float e = c[k] - t * d[k] / length;
						miss += e * e;
					}
					if (miss <= a_radius * a_radius && (best < 0.0f || t < best)) {
						best = t;
					}
				}
			}
			return best;
		}

		// Diagnostic (2026-10-09): a shaped left-out mesh crossing the ray window, checked triangle by triangle: each
		// triangle's view-space box through TestViewBox (as the shape's cells are), and whether it lies in a cell the
		// cached shape marks occupied. Also whether the cached shape equals one built from the mesh's data now.
		void BruteShape(const FrameContext& a_context, const RE::NiAVObject* a_object, LeftOutItem& a_item) noexcept
		{
			MeshProxy::Inspection look;
			MeshProxy::Inspect(const_cast<RE::NiAVObject*>(a_object), look);
			a_item.bruteRead = look.read ? 1 : 0;
			a_item.bruteCached = look.cached;
			a_item.bruteFresh = look.fresh;
			if (!look.read) {
				return;
			}
			const auto same = [](const MeshProxy::Shape& a_a, const MeshProxy::Shape& a_b) {
				return std::memcmp(a_a.min, a_b.min, sizeof(a_a.min)) == 0 && std::memcmp(a_a.cell, a_b.cell, sizeof(a_a.cell)) == 0 &&
				       std::memcmp(a_a.dims, a_b.dims, sizeof(a_a.dims)) == 0 && a_a.count == a_b.count && a_a.occupied[0] == a_b.occupied[0] &&
				       a_a.occupied[1] == a_b.occupied[1];
			};
			a_item.bruteSame = look.cached && look.fresh && same(look.cachedShape, look.freshShape);
			const auto& shape = look.cached ? look.cachedShape : look.freshShape;
			std::copy_n(shape.dims, 3, a_item.bruteDims);
			a_item.bruteCount = shape.count;
			const auto& model = *reinterpret_cast<const RE::NiBound*>(reinterpret_cast<const std::byte*>(a_object) + 0x120);
			float rotate[3][3];
			if (!WorldRotation(a_object, model, rotate)) {
				return;
			}
			const auto& camera = a_context.snapshot->camera;
			const auto& transform = a_object->world;
			const float* view[3]{ camera.viewRight, camera.viewUp, camera.viewDir };
			const auto   toView = [&](const float* a_p, float a_out[3]) {
				float world[3];
				for (int i = 0; i < 3; ++i) {
					world[i] = (i == 0 ? transform.translate.x : i == 1 ? transform.translate.y : transform.translate.z) - camera.origin[i] +
					           transform.scale * (rotate[i][0] * a_p[0] + rotate[i][1] * a_p[1] + rotate[i][2] * a_p[2]);
				}
				for (int i = 0; i < 3; ++i) {
					a_out[i] = view[i][0] * world[0] + view[i][1] * world[1] + view[i][2] * world[2];
				}
			};
			constexpr float kBig = std::numeric_limits<float>::max();
			const auto triangles = static_cast<std::uint32_t>(look.indices.size() / 3);
			a_item.bruteTriangles = triangles;
			for (std::uint32_t t = 0; t < triangles; ++t) {
				float lo[3]{ kBig, kBig, kBig }, hi[3]{ -kBig, -kBig, -kBig };
				float mlo[3]{ kBig, kBig, kBig }, mhi[3]{ -kBig, -kBig, -kBig };
				float ndc[4]{ kBig, kBig, -kBig, -kBig };  // (v1.92.10) the vertices' screen box, those beyond the near distance
				bool  valid = true;
				for (int k = 0; k < 3; ++k) {
					const auto index = look.indices[static_cast<std::size_t>(t) * 3 + k];
					if (index >= look.vertexCount) {
						valid = false;
						break;
					}
					const float* p = &look.positions[static_cast<std::size_t>(index) * 3];
					float        v[3];
					toView(p, v);
					for (int a = 0; a < 3; ++a) {
						lo[a] = std::min(lo[a], v[a]);
						hi[a] = std::max(hi[a], v[a]);
						mlo[a] = std::min(mlo[a], p[a]);
						mhi[a] = std::max(mhi[a], p[a]);
					}
					if (v[2] > g_tunables.nearDistance) {
						const float nx = camera.scaleX * v[0] / v[2], ny = camera.scaleY * v[1] / v[2];
						ndc[0] = std::min(ndc[0], nx);
						ndc[1] = std::min(ndc[1], ny);
						ndc[2] = std::max(ndc[2], nx);
						ndc[3] = std::max(ndc[3], ny);
					}
				}
				if (!valid) {
					continue;
				}
				const float center[3]{ (lo[0] + hi[0]) * 0.5f, (lo[1] + hi[1]) * 0.5f, (lo[2] + hi[2]) * 0.5f };
				const float half[3]{ (hi[0] - lo[0]) * 0.5f, (hi[1] - lo[1]) * 0.5f, (hi[2] - lo[2]) * 0.5f };
				DepthRect   rect;
				const auto  verdict = TestViewBox(a_context, center, half, &rect);
				if (verdict == Verdict::kOutside) {
					continue;
				}
				++a_item.bruteInView;
				a_item.bruteVisible += verdict != Verdict::kHidden;
				const float* window = RayWindow();
				if (ndc[2] >= ndc[0] && ndc[2] >= window[0] && ndc[0] <= window[1] && ndc[3] >= window[2] && ndc[1] <= window[3]) {
					++a_item.bruteWindow;
					a_item.bruteWindowVisible += verdict != Verdict::kHidden;
					if (a_item.windowSamples < 3) {
						auto& sample = a_item.windowSample[a_item.windowSamples++];
						sample.zNear = lo[2];
						sample.zFar = hi[2];
						std::copy_n(ndc, 4, sample.ndc);
						sample.verdict = verdict;
						const auto& snapshot = *a_context.snapshot;
						const float width = static_cast<float>(snapshot.width[0]), height = static_cast<float>(snapshot.height[0]);
						sample.farthest = LinearDepth(camera, snapshot.MaxDepth((ndc[0] * 0.5f + 0.5f) * width, (0.5f - ndc[3] * 0.5f) * height, (ndc[2] * 0.5f + 0.5f) * width, (0.5f - ndc[1] * 0.5f) * height));
						sample.threshold = verdict == Verdict::kHidden && rect.hidden ? LinearDepth(camera, rect.threshold) : 0.0f;
					}
				}
				if (shape.count == 0) {
					continue;  // (box only: every cell counts)
				}
				bool occupied = false;
				std::uint32_t c0[3], c1[3];
				for (int a = 0; a < 3; ++a) {
					const float inv = shape.cell[a] > 0.0f ? 1.0f / shape.cell[a] : 0.0f;
					const auto  last = static_cast<std::uint32_t>(std::max<int>(shape.dims[a], 1) - 1);
					c0[a] = std::min(last, static_cast<std::uint32_t>(std::max(0.0f, (mlo[a] - shape.min[a]) * inv)));
					c1[a] = std::min(last, static_cast<std::uint32_t>(std::max(0.0f, (mhi[a] - shape.min[a]) * inv)));
				}
				for (auto z = c0[2]; z <= c1[2] && !occupied; ++z) {
					for (auto y = c0[1]; y <= c1[1] && !occupied; ++y) {
						for (auto x = c0[0]; x <= c1[0] && !occupied; ++x) {
							const auto bit = x + shape.dims[0] * (y + shape.dims[1] * z);
							occupied = (shape.occupied[bit >> 6] >> (bit & 63)) & 1;
						}
					}
				}
				a_item.bruteUnoccupied += !occupied;
			}
		}

		void RecordLeftOut(const FrameContext& a_context, const RE::NiAVObject* a_object, const RE::NiBound& a_bound, LeftOutPath a_path, const Record* a_record) noexcept
		{
			if (Recorder::On()) {
				Recorder::NotePath(static_cast<std::uint8_t>(a_path));
			}
			if (!LeftOutFrame(a_context.clock) || !a_context.snapshot) {
				return;
			}
			if (g_pictureClock.exchange(a_context.clock) != a_context.clock) {
				TakeDepthPicture(a_context);
			}
			LeftOutItem item{};
			item.object = a_object;
			item.parent = a_object->parent;
			item.path = a_path;
			item.source = a_path == LeftOutPath::kInherited ? VerdictSource::kNone : t_source;
			if (a_record) {
				item.outcome = a_record->outcome;
				item.verdict = a_record->verdict;
				item.streak = a_record->streak;
			}
			item.bound[0] = a_bound.center.x;
			item.bound[1] = a_bound.center.y;
			item.bound[2] = a_bound.center.z;
			item.bound[3] = a_bound.fRadius;
			const auto& camera = a_context.snapshot->camera;
			const float dx = a_bound.center.x - camera.eye[0], dy = a_bound.center.y - camera.eye[1], dz = a_bound.center.z - camera.eye[2];
			item.distance = std::sqrt(dx * dx + dy * dy + dz * dz);
			item.inFront = camera.viewSpace && ProjectToView(camera, a_bound.center.x, a_bound.center.y, a_bound.center.z, item.ndc);
			if (item.inFront) {
				const auto& snapshot = *a_context.snapshot;
				item.viewDepth = dx * camera.viewDir[0] + dy * camera.viewDir[1] + dz * camera.viewDir[2];
				const float w = static_cast<float>(snapshot.width[0]), h = static_cast<float>(snapshot.height[0]);
				const float cx = (item.ndc[0] * 0.5f + 0.5f) * w, cy = (0.5f - item.ndc[1] * 0.5f) * h;
				const float rx = a_bound.fRadius / std::max(item.viewDepth, 1.0f) * camera.scaleX * w * 0.5f;
				const float ry = a_bound.fRadius / std::max(item.viewDepth, 1.0f) * camera.scaleY * h * 0.5f;
				item.occluder = LinearDepth(camera, snapshot.MaxDepth(cx - rx, cy - ry, cx + rx, cy + ry));
			}
			const auto& own = a_object->worldBound;
			const float ox = own.center.x - a_bound.center.x, oy = own.center.y - a_bound.center.y, oz = own.center.z - a_bound.center.z;
			item.ownOffset = std::sqrt(ox * ox + oy * oy + oz * oz);
			item.ownRadius = own.fRadius;

			VisibleEvidence evidence{};
			item.now = Test(a_context, a_bound, nullptr, nullptr, &evidence);
			if (item.now == Verdict::kVisible && evidence.valid) {
				item.fartherTexels = a_context.snapshot->ScanFarther(evidence.x0, evidence.y0, evidence.x1, evidence.y1, evidence.threshold, evidence.rays ? &evidence.sphere : nullptr).texels;
			}
			if (g_tunables.meshShapes && (item.now == Verdict::kVisible || item.now == Verdict::kNear || item.now == Verdict::kEdge) && a_bound.fRadius >= kShapeMinRadius &&
				camera.viewSpace && (TypeOf(a_object) & kTypeCullable)) {
				RE::NiBound model{};
				if (const auto shape = MeshProxy::Find(const_cast<RE::NiAVObject*>(a_object), model)) {
					std::uint32_t cells = 0;
					std::uint8_t  hint = 0;
					item.shaped = true;
					item.shapeNow = TestShape(a_context, a_object, model, *shape, cells, hint);
					// (2026-10-15: also every mesh its shape alone settles: Runtime Combiner chunks went missing in an interior
					// while their spheres were in view)
					if (RayWindowHit(camera, a_bound.center.x, a_bound.center.y, a_bound.center.z, a_bound.fRadius) >= 0.0f ||
						item.shapeNow == Verdict::kHidden || item.shapeNow == Verdict::kOutside) {
						BruteShape(a_context, a_object, item);
					}
				}
			}

			Util::TryGetRTTIName(a_object, item.type, sizeof(item.type));
			Util::TryGetObjectName(a_object, item.name, sizeof(item.name));
			DescribeAncestry(a_object, item.ancestry, sizeof(item.ancestry));
			FindReference(a_object, item);

			if (camera.viewSpace && const_cast<RE::NiAVObject*>(a_object)->IsGeometry() && Util::TryReadVtable(a_object) != g_mergedVtable) {
				float       rotate[3][3];
				const auto& model = *reinterpret_cast<const RE::NiBound*>(reinterpret_cast<const std::byte*>(a_object) + 0x120);
				// (2026-10-09: also when the bound doesn't fit the transform, with the transform's own rotation: a mesh drawn where
				// its bound isn't is exactly what this check is for)
				item.boundFits = WorldRotation(a_object, model, rotate);
				if (!item.boundFits) {
					const auto& world = a_object->world;
					for (int i = 0; i < 3; ++i) {
						for (int j = 0; j < 3; ++j) {
							rotate[i][j] = world.rotate.entry[j].pt[i];
						}
					}
				}
				float lo[3]{}, hi[3]{};
				CheckVertices(a_object, rotate, item, lo, hi, &camera);
				if (item.vertices > 0) {
					item.vertexInFront = ProjectToView(camera, (lo[0] + hi[0]) * 0.5f, (lo[1] + hi[1]) * 0.5f, (lo[2] + hi[2]) * 0.5f, item.vertexNdc);
				}
			}

			std::scoped_lock lock(g_keptLock);
			if (g_leftOut.size() < 65536) {
				g_leftOut.push_back(item);
			}
		}

		// Hidden streak for a hidden verdict now: consecutive only if the object tested hidden in the
		// previous frame too (a visible verdict writes nothing, so it breaks the chain by omission).
		std::uint32_t HiddenStreak(std::uint64_t a_value, std::uint32_t a_clock) noexcept
		{
			const auto last = static_cast<std::uint32_t>(a_value);
			const auto streak = static_cast<std::uint32_t>((a_value >> 32) & 0xFF);
			if (last == a_clock) {
				return std::max(streak, 1u);  // added twice this frame
			}
			return last + 1 == a_clock ? std::min(streak + 1, 255u) : 1u;
		}

		// Exempt types that tested hidden, sampled every 32nd frame, for the log (which kinds are kept); and the types
		// CBRO leaves out of group 0 entirely (never filed or rejected in every view), to name what another reader
		// of group 0 may be missing. A sampled frame counts thousands of objects (the v1.45 run: ~5,800 left out of
		// group 0 each), so the counts live in a small lock-free table keyed by vtable (a few dozen types): no mutex
		// and no allocation on the walk threads. Types stay once seen; the log takes only the counts.
		struct TypeCounts
		{
			static constexpr std::size_t kBits = 7;
			static constexpr std::size_t kSlots = std::size_t{ 1 } << kBits;
			struct Slot
			{
				std::atomic<std::uintptr_t> vtable{ 0 };
				std::atomic<std::uint32_t>  count{ 0 };
			};
			std::array<Slot, kSlots>   slots{};
			std::atomic<std::uint32_t> untracked{ 0 };  // more distinct types than slots

			void Add(std::uintptr_t a_vtable) noexcept
			{
				auto index = static_cast<std::size_t>(((a_vtable >> 3) * 0x9E3779B97F4A7C15ull) >> (64 - kBits));
				for (std::size_t probe = 0; probe < kSlots; ++probe, index = (index + 1) & (kSlots - 1)) {
					auto& slot = slots[index];
					auto  current = slot.vtable.load(std::memory_order_acquire);
					if (current == 0 && slot.vtable.compare_exchange_strong(current, a_vtable, std::memory_order_acq_rel)) {
						current = a_vtable;
					}
					if (current == a_vtable) {
						slot.count.fetch_add(1, std::memory_order_relaxed);
						return;
					}
				}
				untracked.fetch_add(1, std::memory_order_relaxed);
			}
		};
		TypeCounts g_exemptTypes;
		TypeCounts g_leftOutTypes;

		void SampleExemptType(const FrameContext& a_context, const RE::NiAVObject* a_object) noexcept
		{
			if (a_context.clock % 32 != 0) {
				return;
			}
			if (const auto vtable = Util::TryReadVtable(a_object)) {
				g_exemptTypes.Add(vtable);
			}
		}

		void SampleLeftOutType(const FrameContext& a_context, const RE::NiAVObject* a_object) noexcept
		{
			if (a_context.clock % 32 != 0) {
				return;
			}
			if (const auto vtable = Util::TryReadVtable(a_object)) {
				g_leftOutTypes.Add(vtable);
			}
		}

		// CPU time is sampled on every 8th frame (and scaled up in the report): reading the TSC twice per
		// object costs more than the rest of a quick verdict.
		constexpr std::uint32_t kTimingStride = 8;

		bool Timed(const FrameContext& a_context) noexcept
		{
			return a_context.clock % kTimingStride == 0;
		}

		// The split of the per-object time into its parts (record lookups, reuse checks, record writes) is timed on frames
		// of its own, halfway between the whole-test frames, so the parts' extra clock reads never inflate the whole
		// (v1.52 timed both on the same frames: most of its "decisions and counters" rest was those reads).
		bool BucketTimed(const FrameContext& a_context) noexcept
		{
			return a_context.clock % kTimingStride == kTimingStride / 2;
		}

		// The decision made at Group::Add for the object now being added (same thread): the Block::Add
		// that follows inside the engine's Group::Add reuses it instead of testing again.
		struct TopDecision
		{
			const RE::NiAVObject* object{ nullptr };
			std::uint32_t         clock{ 0 };
			const RE::NiBound*    result{ nullptr };
			bool                  forceVisible{ false };  // filed force-visible: the player's 3D (v1.89.1), the trace's out-of-view entries (v1.91)
		};
		thread_local TopDecision t_top;

		// The verdict of the top entry decided last on this thread (v1.98, bChildInherit): a child with the same bound (a
		// fade node with one mesh: 856 of the 1,367 let-through fade nodes a frame in the 2026-10-27 run) would get the same
		// sphere verdict from the same tests, so it takes the parent's instead of a lookup, a judgement and a record of its
		// own. Not when the child could get a mesh-shape test the node never had (big enough, the sphere visible or at the
		// edge), nor from an exemption, a light or a force-visible filing. The decision inherited is always "file it":
		// the engine's own frustum test still drops a child out of the view.
		struct TopRecord
		{
			const RE::NiAVObject* object{ nullptr };
			std::uint32_t         clock{ 0 };
			std::uint8_t          outcome{ 0 };
			std::uint8_t          verdict{ 0 };
			std::uint8_t          flags{ 0 };
			float                 bound[4]{};
		};
		// This frame's top decisions by object, per thread (direct-mapped): the engine files a node's children only when it
		// finishes the block the node sits in, hundreds of entries after the node's own decision (FO4-ENGINE-NOTES 5.2), so
		// the last decision alone never matched (v1.98 measured none).
		constexpr std::size_t kTopTableSize = 4096;
		thread_local std::array<TopRecord, kTopTableSize>* t_topTable{ nullptr };

		TopRecord& TopSlot(const RE::NiAVObject* a_object) noexcept
		{
			if (!t_topTable) [[unlikely]] {
				t_topTable = new std::array<TopRecord, kTopTableSize>();
			}
			const auto address = reinterpret_cast<std::uintptr_t>(a_object);
			return (*t_topTable)[((address >> 5) ^ (address >> 17)) & (kTopTableSize - 1)];
		}

		const TopRecord* TopOf(const FrameContext& a_context, const RE::NiAVObject* a_object) noexcept
		{
			if (!a_object) {
				return nullptr;
			}
			const auto& slot = TopSlot(a_object);
			return slot.object == a_object && slot.clock == a_context.clock ? &slot : nullptr;
		}

		// ---- stable-index decisions (v1.99.2, bStableIndex) --------------------------------------------------------
		// The engine finishes a group's node entries in the order they were filed (FO4OG 5.2: 1147875 runs 558046 over the
		// node blocks in order, each block's entries in index order, unbatched with no SubGroup), so the decision parent of
		// a child (the fade node filing it, or the node ForEachChild walks) is the next filed node that has children, or a
		// few further on. Each top-level node decision is recorded in that order, per group, and a child finds its parent's
		// record with a cursor that only moves forward: no hash, no probe. What the record answers (dropped from the main
		// view, filed force-visible for the trace, the player's, the verdict an inheriting child takes, the collection
		// root) cost four table lookups per parent before (the two drop sets, the top table, the collection cache; ~14% of
		// CBRO.dll's cull-stage samples in the 2026-10-27 v1.99 run, with the top table's write per filed entry). A parent
		// the cursor doesn't find within kParentLookAhead (a nested node, a group filed out of order) is answered by those
		// tables as before: the top-level drops and forced filings still go into the sets.
		enum ParentState : std::uint8_t
		{
			kParentJudged = 1 << 0,     // outcome, verdict, flags and bound hold its verdict (as its TopRecord would)
			kParentDropped = 1 << 1,    // a main-view drop (in g_drops)
			kParentForced = 1 << 2,     // filed force-visible for the trace (in g_traceForced)
			kParentPlayer = 1 << 3,     // the player's 3D
			kParentScolKnown = 1 << 4,  // kParentScolRoot answers for the collection root
			kParentScolRoot = 1 << 5,   // a static collection's root (its reference's base a SCOL)
		};

		struct ParentRecord
		{
			const RE::NiAVObject* object{ nullptr };
			float                 bound[4]{};
			std::uint8_t          outcome{ 0 };
			std::uint8_t          verdict{ 0 };
			std::uint8_t          flags{ 0 };
			std::uint8_t          state{ 0 };
		};

		struct ParentSequence
		{
			const void*               group{ nullptr };
			std::vector<ParentRecord> records;
			std::size_t               cursor{ 0 };
		};
		std::array<ParentSequence, 4> g_parentSequences;  // (main thread: written while the groups are filed, read while they are finished)
		DWORD                         g_mainThreadId{ 0 };
		constexpr std::size_t         kParentLookAhead = 256;
		constexpr std::size_t         kParentReserve = 8192;     // (records are not moved while the groups are finished)
		thread_local const void*      t_decisionGroup{ nullptr };  // the group a top-level decision files into (set by SkipTopLevelImpl's callers)
		// The nodes recorded this frame, one bit per hashed address (v1.99.3): the 2026-10-27 v1.99.2 run looked ~1,500-2,000
		// parents a frame up that were never recorded (a fade node nested in a filed node, met through ForEachChild), each
		// costing the cursor a scan of the look-ahead (~11% of CBRO.dll's cull-stage samples); a clear bit answers those.
		std::array<std::uint64_t, 512> g_parentPresence{};

		inline std::size_t ParentPresenceBit(const void* a_node) noexcept
		{
			return static_cast<std::size_t>(((reinterpret_cast<std::uintptr_t>(a_node) >> 4) * 0x9E3779B97F4A7C15ull) >> (64 - 15));
		}

		bool StableIndexOn() noexcept
		{
			return g_tunables.stableIndex && GetCurrentThreadId() == g_mainThreadId;
		}

		void ClearParentSequences() noexcept
		{
			g_parentPresence.fill(0);
			for (auto& sequence : g_parentSequences) {
				sequence.group = nullptr;
				sequence.records.clear();
				if (sequence.records.capacity() < kParentReserve) {
					sequence.records.reserve(kParentReserve);
				}
				sequence.cursor = 0;
			}
		}

		ParentSequence* ParentSequenceOf(const void* a_group, bool a_create) noexcept
		{
			if (!a_group) {
				return nullptr;
			}
			for (auto& sequence : g_parentSequences) {
				if (sequence.group == a_group) {
					return &sequence;
				}
				if (!sequence.group) {
					if (!a_create) {
						return nullptr;
					}
					sequence.group = a_group;
					return &sequence;
				}
			}
			return nullptr;
		}

		struct ParentRecordMemo
		{
			const RE::NiAVObject* parent{ nullptr };
			std::uint32_t         clock{ 0 };
			const ParentRecord*   record{ nullptr };
		};
		thread_local ParentRecordMemo t_parentRecord;

		// The record of a child's decision parent, in the group being finished (null: none found; the tables answer).
		// Siblings come one after another, so the answer is kept for them.
		const ParentRecord* ParentRecordOf(const RE::NiAVObject* a_parent, std::uint32_t a_clock) noexcept
		{
			if (!a_parent || !StableIndexOn()) {
				return nullptr;
			}
			if (t_parentRecord.parent == a_parent && t_parentRecord.clock == a_clock) {
				return t_parentRecord.record;
			}
			if (const auto bit = ParentPresenceBit(a_parent); !((g_parentPresence[bit >> 6] >> (bit & 63)) & 1)) {
				Bump(kStableParentNotRecorded);
				t_parentRecord = { a_parent, a_clock, nullptr };
				return nullptr;
			}
			const ParentRecord* found = nullptr;
			if (const auto sequence = ParentSequenceOf(Hooks::CullGroups::CurrentGroup(), false)) {
				const auto end = std::min(sequence->records.size(), sequence->cursor + kParentLookAhead);
				for (auto i = sequence->cursor; i < end; ++i) {
					if (sequence->records[i].object == a_parent) {
						found = &sequence->records[i];
						sequence->cursor = i + 1;
						break;
					}
				}
			}
			Bump(found ? kStableParentHits : kStableParentOutOfOrder);
			t_parentRecord = { a_parent, a_clock, found };
			return found;
		}

		// Whether the parent of an object being added was dropped from the main view this frame. The engine
		// adds a node's children one after another, so the answer is kept for the siblings (per thread).
		struct ParentDecision
		{
			const RE::NiAVObject* parent{ nullptr };
			std::uint32_t         clock{ 0 };
			bool                  dropped{ false };
		};
		thread_local ParentDecision t_parent;
		thread_local ParentDecision t_forcedParent;  // (g_traceForced: ParentForced)

		// The node whose decision a Block::Add child takes (v1.99): the fade node the engine is filing (AddP1FadeNode, its
		// flattened geometry: the child's own parent is one of the node's sub-nodes, which nothing decided), else the
		// child's parent. Before v1.99 a dropped fade node's geometry under a sub-node was judged one by one (674 a frame in
		// the 2026-10-27 recording) and each new sub-node cost the memos below a probe and a read of its own.
		const RE::NiAVObject* DecisionParent(const RE::NiAVObject* a_object) noexcept
		{
			if (g_tunables.childInherit) {
				if (const auto node = Hooks::CullGroups::FadeNodeFiling()) {
					return node;
				}
			}
			return a_object->parent;
		}

		bool ParentDropped(const RE::NiAVObject* a_object, std::uint32_t a_clock, std::uint32_t a_tag) noexcept
		{
			const auto parent = DecisionParent(a_object);
			if (!parent) {
				return false;
			}
			if (const auto record = ParentRecordOf(parent, a_clock)) {
				return (record->state & kParentDropped) != 0;
			}
			if (t_parent.parent != parent || t_parent.clock != a_clock) {
				t_parent = { parent, a_clock, g_drops.Contains(parent, a_tag) };
			}
			return t_parent.dropped;
		}

		// Whether the parent of an object being added was filed force-visible for the trace this frame (out of view: so is
		// the child; never hidden by implication).
		bool ParentForced(const RE::NiAVObject* a_object, std::uint32_t a_clock, std::uint32_t a_tag) noexcept
		{
			const auto parent = DecisionParent(a_object);
			if (!parent || !TraceFeed::On()) {
				return false;
			}
			if (const auto record = ParentRecordOf(parent, a_clock)) {
				return (record->state & kParentForced) != 0;
			}
			if (t_forcedParent.parent != parent || t_forcedParent.clock != a_clock) {
				t_forcedParent = { parent, a_clock, g_traceForced.Contains(parent, a_tag) };
			}
			return t_forcedParent.dropped;
		}

		// ---- the verdict cache -----------------------------------------------------------------------------
		// The v1.23 run put all of the per-object test time (1.7-2.0 ms per frame at the light spot) on the main
		// thread: every object the engine offers is tested from scratch every frame although, standing still,
		// nothing about it changed. So each object's last outcome is kept and reused while everything it depended
		// on still holds: the camera (and the depth frame's camera) within the cache tolerance of the epoch's
		// reference (Runtime keeps the epoch id; tested bounds are dilated by the tolerance and by depth x the turn
		// tolerance, so a verdict stays conservative anywhere inside it), the object's own bound unchanged (16-byte
		// compare), the sun's direction/state unchanged (its epoch id) and, for outcomes that rest on the depth, no
		// change in the 8x8 Hi-Z blocks the test read since its readback (HiZ marks the blocks whose farthest or
		// nearest depth changed against the previous readback). A hidden object reused for a frame counts that
		// frame toward its confirmation streak (the evidence still holds); one that was merely out of view keeps
		// its streak (that says nothing about the depth); one seen visible, near or at the edge starts over.
		//
		// The engine offers the objects in the same order every frame (the scene walk, then the block jobs), so the
		// cache is a per-thread sequential stream of last frame's records read with a cursor (a short look-ahead
		// resyncs after an insertion): no hashing, prefetch-friendly. One stream for the top-level adds
		// (Group::Add), one for the children (Block::Add from the jobs).

		// (Outcome, SunOutcome, RecordFlags and Record: Core/Verdict.h, shared with Core/Async)

		struct Stream
		{
			std::vector<Record> prev;
			std::vector<Record> cur;
			std::size_t         cursor{ 0 };
			std::uint32_t       clock{ 0 };

			void Begin(std::uint32_t a_clock)
			{
				if (clock != a_clock) {
					// A frame in which this thread judged nothing (a blocked frame, no depth yet) keeps the older
					// records: the validity checks still guard their reuse, and the streaks survive the gap.
					if (!cur.empty()) {
						prev.swap(cur);
					}
					cur.clear();
					cursor = 0;
					clock = a_clock;
				}
			}

			// Last frame's record for a_object if it is where the sequence says (or within a few slots after an
			// insertion); the cursor moves past it.
			const Record* Find(const void* a_object) noexcept
			{
				// The common case: the next record, or one of the next few (an object left the sequence). Then a
				// wider scan (a whole node's worth left it, e.g. a pruned container); a miss leaves the cursor.
				const auto count = prev.size();
				for (std::size_t k = 0; k < 512 && cursor + k < count; ++k) {
					if (prev[cursor + k].object == a_object) {
						const Record* found = &prev[cursor + k];
						cursor += k + 1;
						return found;
					}
				}
				return nullptr;
			}

			void Push(const Record& a_record)
			{
				if (cur.size() < 65536) {
					cur.push_back(a_record);
				}
			}

			void Clear()
			{
				prev.clear();
				cur.clear();
				cursor = 0;
			}
		};

		struct ThreadStreams
		{
			Stream        top;
			Stream        child;
			std::uint32_t generation{ 0 };
		};
		std::atomic<std::uint32_t> g_streamGeneration{ 1 };  // bumped by ResetHistory: every thread's streams start over

		ThreadStreams& Streams(const FrameContext& a_context)
		{
			thread_local ThreadStreams* streams = nullptr;  // (one per thread, for the process lifetime)
			if (!streams) [[unlikely]] {
				streams = new ThreadStreams();
			}
			if (const auto generation = g_streamGeneration.load(std::memory_order_relaxed); streams->generation != generation) {
				streams->generation = generation;
				streams->top.Clear();
				streams->child.Clear();
			}
			streams->top.Begin(a_context.clock);
			streams->child.Begin(a_context.clock);
			return *streams;
		}

		// The Hi-Z blocks a rectangle (widened) covers, into a record's block field.
		void SetBlocks(const FrameContext& a_context, std::uint8_t a_blocks[4], const DepthRect& a_rect) noexcept
		{
			if (!a_rect.used || a_context.blocksW == 0 || a_context.blocksH == 0) {
				return;
			}
			const auto block = [](float a_texel, std::uint32_t a_count) {
				return static_cast<std::uint8_t>(std::clamp(static_cast<int>(a_texel) / 8, 0, static_cast<int>(a_count) - 1));
			};
			a_blocks[0] = block(a_rect.x0, a_context.blocksW);
			a_blocks[1] = block(a_rect.y0, a_context.blocksH);
			a_blocks[2] = block(a_rect.x1, a_context.blocksW);
			a_blocks[3] = block(a_rect.y1, a_context.blocksH);
		}

		// A hidden view outcome's evidence into the record: the texel rectangle and threshold to re-check it by, and
		// the blocks it rests on.
		void SetEvidence(const FrameContext& a_context, Record& a_record, const DepthRect& a_rect) noexcept
		{
			if (!a_rect.hidden) {
				return;
			}
			const auto texel = [](float a_value, bool a_up) {
				return static_cast<std::uint16_t>(std::clamp(a_up ? std::ceil(a_value) : std::floor(a_value), 0.0f, 65535.0f));
			};
			a_record.rect[0] = texel(a_rect.hx0, false);
			a_record.rect[1] = texel(a_rect.hy0, false);
			a_record.rect[2] = texel(a_rect.hx1, true);
			a_record.rect[3] = texel(a_rect.hy1, true);
			a_record.threshold = a_rect.threshold;
			a_record.flags |= kRecordDepth;
			SetBlocks(a_context, a_record.blocks, a_rect);
		}

		// Whether any of the blocks moved in the invalidating direction (farthest farther, nearest nearer) since the
		// capture a_readback (HiZ's change map: a block's mark is the index of the capture that last changed it).
		bool BlocksChanged(const FrameContext& a_context, const std::uint8_t a_blocks[4], std::uint32_t a_readback) noexcept
		{
			if (a_blocks[0] == 255 || !a_context.blockChangedAt) {
				return true;  // nothing recorded: can't tell
			}
			for (std::uint32_t y = a_blocks[1]; y <= a_blocks[3]; ++y) {
				const auto* row = a_context.blockChangedAt + static_cast<std::size_t>(y) * a_context.blocksW;
				for (std::uint32_t x = a_blocks[0]; x <= a_blocks[2]; ++x) {
					if (row[x] > a_readback) {
						return true;
					}
				}
			}
			return false;
		}

		// The full view test, with the classification and the streak, into a fresh record. a_old is last frame's
		// record of the same object (streak continuity, the shape's cell hint) or null.
		// ---- diagnostic: what turns a confirmed-hidden verdict visible (flip samples, logged with the spike frames) ----
		enum class FlipKind : std::uint8_t
		{
			kObject,
			kPoint,
			kSpot,
			kCount
		};
		struct FlipSample
		{
			std::uint32_t              clock{ 0 };
			FlipKind                   kind{ FlipKind::kObject };
			std::uint8_t               verdict{ 0 };
			bool                       scanned{ false };
			float                      bound[4]{};
			float                      nearest{ 0.0f };
			std::uint32_t              readback{ 0 };
			std::uint32_t              heldSince{ 0 };  // objects: the depth the hidden verdict last held against
			HiZ::Snapshot::FartherScan scan{};
			char                       type[40]{};
		};
		constexpr std::size_t                   kFlipRing = 256;
		constexpr std::uint32_t                 kFlipSamplesPerFrame = 6;  // per kind
		std::array<FlipSample, kFlipRing>       g_flipRing{};
		std::atomic<std::uint32_t>              g_flipNext{ 0 };
		std::atomic<std::uint32_t>              g_flipClock{ 0 };  // the frame the per-frame counts below belong to
		std::array<std::atomic<std::uint32_t>, 3> g_flipTaken{};    // samples taken this frame, per kind
		std::array<std::atomic<std::uint64_t>, 3> g_flipCount{};    // flips, per kind (running totals)

		// A confirmed-hidden verdict turned into a kept one: counted, and a few per frame sampled with a level-0 scan of
		// the depth that made it visible. Any thread.
		void NoteFlip(const FrameContext& a_context, FlipKind a_kind, const RE::NiAVObject* a_object, const RE::NiBound& a_bound, Verdict a_verdict, std::uint32_t a_heldSince) noexcept
		{
			const auto kind = static_cast<std::size_t>(a_kind);
			g_flipCount[kind].fetch_add(1, std::memory_order_relaxed);
			if (g_flipClock.load(std::memory_order_relaxed) != a_context.clock || g_flipTaken[kind].fetch_add(1, std::memory_order_relaxed) >= kFlipSamplesPerFrame) {
				return;
			}
			FlipSample sample{};
			sample.clock = a_context.clock;
			sample.kind = a_kind;
			sample.verdict = static_cast<std::uint8_t>(a_verdict);
			sample.bound[0] = a_bound.center.x;
			sample.bound[1] = a_bound.center.y;
			sample.bound[2] = a_bound.center.z;
			sample.bound[3] = a_bound.fRadius;
			sample.readback = a_context.readback;
			sample.heldSince = a_heldSince;
			VisibleEvidence evidence{};
			if (a_verdict == Verdict::kVisible) {
				LightTest light{};
				(void)Test(a_context, a_bound, nullptr, a_kind == FlipKind::kObject ? nullptr : &light, &evidence);
			}
			if (evidence.valid) {
				sample.scanned = true;
				sample.nearest = evidence.nearest;
				sample.scan = a_context.snapshot->ScanFarther(evidence.x0, evidence.y0, evidence.x1, evidence.y1, evidence.threshold, evidence.rays ? &evidence.sphere : nullptr);
			}
			if (a_object) {
				Util::TryGetRTTIName(a_object, sample.type, sizeof(sample.type));
			}
			g_flipRing[g_flipNext.fetch_add(1, std::memory_order_relaxed) % kFlipRing] = sample;
		}

		void EvaluateView(const FrameContext& a_context, const Hooks::CullGroups::BlockAdd& a_add, const Record* a_old, Record& a_out, bool a_timed)
		{
			a_out = Record{};
			a_out.object = a_add.object;
			a_out.bound[0] = a_add.bound->center.x;
			a_out.bound[1] = a_add.bound->center.y;
			a_out.bound[2] = a_add.bound->center.z;
			a_out.bound[3] = a_add.bound->fRadius;
			a_out.viewEpoch = a_context.viewEpoch;
			a_out.readback = a_context.readback;

			Bump(kTested);
			DepthRect rect{};
			t_parallaxHit = false;
			auto      verdict = Test(a_context, *a_add.bound, &rect);
			// (last frame's verdict was edge: no depth for part of the object, so the hidden streak's backup stood through it)
			const bool edgeBefore = a_old && a_old->outcome == Outcome::kKept && a_old->verdict == static_cast<std::uint8_t>(Verdict::kEdge);

			// A big static mesh the sphere can't settle: judge it by its actual shape (MeshProxy decides
			// which meshes have one; a mesh that moved or sways is left to its sphere).
			const bool              reasons = g_keptReasons.load(std::memory_order_relaxed);  // (diagnostic, SetKeptReasons)
			const MeshProxy::Shape* shapeUsed = nullptr;                                     // ... the shape the verdict came through
			RE::NiBound             model{};
			if (g_tunables.meshShapes && (verdict == Verdict::kVisible || verdict == Verdict::kNear || verdict == Verdict::kEdge) &&
				a_add.bound->fRadius >= kShapeMinRadius && a_context.snapshot->camera.viewSpace && (TypeOf(a_add.object) & kTypeCullable)) {
				if (reasons) {
					a_out.flags |= kRecordShapeNone;  // (taken back below when a shape decides)
				}
				if (const auto shape = MeshProxy::Find(a_add.object, model)) {
					if (Moved(a_add.object)) {
						Bump(kShapeMoving);
					} else if (a_add.object->parent && (TypeOf(a_add.object->parent) & kTypeLeafAnim)) {
						Bump(kShapeAnimated);
					} else {
						Bump(kShapeTests);
						if (reasons) {
							a_out.flags = static_cast<std::uint8_t>((a_out.flags & ~kRecordShapeNone) | kRecordShaped);
							shapeUsed = shape;
						}
						std::uint32_t cells = 0;
						std::uint8_t  hint = a_old ? a_old->cellHint : std::uint8_t{ 0 };
						const auto    shapeStart = a_timed ? __rdtsc() : 0;
						const auto    shaped = TestShape(a_context, a_add.object, model, *shape, cells, hint, &rect);
						if (a_timed) {
							Bump(kCyclesShape, (__rdtsc() - shapeStart) * kTimingStride);
						}
						a_out.cellHint = hint;
						Bump(kShapeCells, cells);
						if (shaped == Verdict::kHidden || shaped == Verdict::kOutside) {
							Bump(shaped == Verdict::kHidden ? kShapeHidden : kShapeOutside);
							RecordKept(a_context, a_add.object, *a_add.bound, shaped == Verdict::kHidden ? "shape: hidden" : "shape: out of view", true);
							// Either way it goes through confirmation and the reject bound: the engine's own
							// frustum test would still pass its sphere, so "out of view" alone drops nothing.
							verdict = Verdict::kHidden;
						} else if (shaped == Verdict::kInvalid) {
							Bump(kShapeNoTransform);
						}
					}
				}
			}
			// (diagnostic, SetKeptReasons: the same tests again without the parallax margin where it widened them; hidden then
			// means the margin alone keeps the object)
			if (reasons && t_parallaxHit && (verdict == Verdict::kVisible || verdict == Verdict::kEdge)) {
				t_parallaxOff = true;
				auto bare = Test(a_context, *a_add.bound);
				if (shapeUsed && (bare == Verdict::kVisible || bare == Verdict::kNear || bare == Verdict::kEdge)) {
					std::uint32_t cells = 0;
					std::uint8_t  hint = a_out.cellHint;
					bare = TestShape(a_context, a_add.object, model, *shapeUsed, cells, hint);
				}
				t_parallaxOff = false;
				if (bare == Verdict::kHidden) {
					a_out.flags |= kRecordParallaxKept;
				}
			}
			a_out.verdict = static_cast<std::uint8_t>(verdict);

			if (verdict == Verdict::kOutside) {
				a_out.outcome = Outcome::kOutside;
				Bump(kOutside);
				if (TypeOf(a_add.object) & kTypeLight) {
					a_out.flags |= kRecordLight;  // (its light can reach into the view: never skipped as out of view)
				}
			} else if (verdict != Verdict::kHidden) {
				a_out.outcome = Outcome::kKept;
				CountVerdict(verdict);
				if (verdict == Verdict::kVisible || verdict == Verdict::kNear || verdict == Verdict::kEdge || verdict == Verdict::kInvalid) {
					RecordKept(a_context, a_add.object, *a_add.bound, VerdictName(verdict));
				}
			} else {
				std::uint64_t value = 0;
				const auto    entry = Lookup(a_context, a_add.object, value);
				if (!entry) {
					a_out.outcome = Outcome::kKept;
					a_out.flags |= kRecordNoCache;
				} else if (value & kExempt) {
					a_out.outcome = Outcome::kKept;
					switch (static_cast<Exemption>((value >> 42) & 3)) {
					case Exemption::kLight:
						Bump(kExemptLight);
						RecordKept(a_context, a_add.object, *a_add.bound, "hidden, kept: carries a light");
						break;
					case Exemption::kActor:
						Bump(kExemptActor);
						RecordKept(a_context, a_add.object, *a_add.bound, "hidden, kept: actor part");
						break;
					default:
						Bump(kExemptType);
						SampleExemptType(a_context, a_add.object);
						RecordKept(a_context, a_add.object, *a_add.bound, "hidden, kept: type");
						break;
					}
				} else {
					// A light reaches past its scene bound: it is hidden only if its whole reach is (never cached:
					// the reach's depth rectangle isn't tracked).
					bool hidden = true;
					if (value & kIsLight) {
						a_out.flags |= kRecordLight | kRecordNoCache;
						Bump(kLightsTested);
						RE::NiBound reach{};
						if (!LightReach(a_add.object, *a_add.bound, reach)) {
							Bump(kInvalid);
							hidden = false;
						} else if (FirstPersonUnseen(a_context) && ReachNearEye(a_context, reach.center, reach.fRadius)) {
							CountVerdict(Verdict::kNear);
							RecordKept(a_context, a_add.object, reach, "light reaches first person");
							hidden = false;
						} else if (const auto lightVerdict = Test(a_context, reach); lightVerdict != Verdict::kHidden) {
							CountVerdict(lightVerdict);
							RecordKept(a_context, a_add.object, reach, "light reaches the view");
							hidden = false;
						}
					}
					if (!hidden) {
						a_out.outcome = Outcome::kKept;
					} else {
						a_out.outcome = Outcome::kHidden;
						// Consecutive frames hidden: last frame's record continues the count when it was hidden (or only
						// out of view, which says nothing about the depth); anything seen starts over. With no record
						// (another thread or place in the walk judged it last frame, or it was dropped with its parent)
						// the table's backup continues it, and past an edge verdict too (v1.72: part of the object lay
						// where the depth frame has nothing, which says nothing either). In a pan the view overhangs the
						// depth frame by about the fViewOverhang strip, and an object straddling the leading edge goes
						// edge, hidden, edge as it comes and goes; v1.70's widened async view made it edge in every map
						// and v1.71's walk judged it hidden afresh each frame, so it stayed confirming, drawn.
						const auto backup = !a_old || edgeBefore ? BackupStreak(value, a_context.clock) : 0u;
						a_out.streak = a_old && (a_old->outcome == Outcome::kHidden || a_old->outcome == Outcome::kOutside) ?
						                   static_cast<std::uint8_t>(std::min<std::uint32_t>(a_old->streak + 1u, 255u)) :
						                   static_cast<std::uint8_t>(std::max(backup, 1u));
						if (backup) {
							Bump(kStreakFromBackup);
						}
						StoreStreak(*entry, a_context.clock, a_out.streak);
						if (a_out.streak < g_tunables.confirmFrames) {
							Bump(kConfirming);
							RecordKept(a_context, a_add.object, *a_add.bound, "hidden, confirming");
						} else if (Observing()) {
							Bump(kWouldReject);
						} else {
							Bump(kRejected);
							if (value & kIsLight) {
								Bump(kLightsRejected);
							}
						}
					}
				}
			}
			if (a_out.outcome == Outcome::kHidden) {
				SetEvidence(a_context, a_out, rect);  // (what a later frame re-checks the verdict by)
			} else {
				if (a_out.outcome == Outcome::kKept && a_old && a_old->outcome == Outcome::kHidden && a_old->streak >= g_tunables.confirmFrames) {
					NoteFlip(a_context, FlipKind::kObject, a_add.object, *a_add.bound, verdict, a_old->readback);  // (diagnostic)
				}
				// Seen (or out of view): the streak backup starts over too, if there may be one. (Not by an edge verdict: no
				// depth there, nothing seen.)
				if ((!a_old || a_old->outcome == Outcome::kHidden || edgeBefore) && verdict != Verdict::kEdge) {
					if (const auto entry = FindEntry(reinterpret_cast<std::uintptr_t>(a_add.object))) {
						StoreStreak(*entry, a_context.clock, 0);
					}
				}
			}
		}

		// The sun part (group 0: whether the cascades need the object's shadow), into the record.
		void EvaluateSun(const FrameContext& a_context, const RE::NiBound& a_bound, const Record* a_old, Record& a_out)
		{
			a_out.sunEpoch = a_context.sunEpoch;
			a_out.sunReadback = a_context.readback;
			a_out.sunBlocks[0] = 255;
			a_out.sunStreak = 0;
			a_out.flags &= ~(kRecordSunEdge | kRecordSunFar);
			switch (a_context.sun.state) {
			case FrameContext::Sun::State::kOff: {
				// The sun's cascades don't read group 0 this frame; the spot lights' group passes do (FO4-ENGINE-NOTES
				// 6.2a: the v1.31 interior run lost its lamp shadow maps when hidden entries were simply left out). So
				// with the sun off an entry the main view doesn't need is left out only when no recorded lamp (point or
				// spot, ShadowLights) can cast a visible shadow of it: the same shadow-volume test the point-light casters
				// get, per lamp within reach. Never reused across frames (the lamps and the depth move).
				const auto& lamps = ShadowLights::Lamps();
				if (!g_tunables.lampGroupTrim || !lamps.complete || lamps.count == 0 || lamps.overflow != 0 || !a_out.object) {
					Bump(kSunOff);
					a_out.sun = SunOutcome::kUnknown;
					return;
				}
				auto* caster = const_cast<RE::NiAVObject*>(static_cast<const RE::NiAVObject*>(a_out.object));
				for (std::uint32_t i = 0; i < lamps.count; ++i) {
					const auto& lamp = lamps.items[i];
					const float dx = a_bound.center.x - lamp.position.x, dy = a_bound.center.y - lamp.position.y, dz = a_bound.center.z - lamp.position.z;
					if (dx * dx + dy * dy + dz * dz > (lamp.reach + a_bound.fRadius) * (lamp.reach + a_bound.fRadius)) {
						continue;  // beyond this lamp's reach
					}
					Bump(kSunLampTests);
					const auto verdict = TestLampCasterIn(a_context, caster, lamp.position, lamp.reach, a_bound);
					if (verdict != LampVerdict::kOutside && verdict != LampVerdict::kMisses) {
						Bump(kSunLampNeeded);
						a_out.sun = SunOutcome::kUnknown;  // (needed: kept in group 0)
						return;
					}
				}
				Bump(kSunLampUnneeded);
				a_out.sun = SunOutcome::kLampUnneeded;
				return;
			}
			case FrameContext::Sun::State::kUnknown:
				a_out.sun = SunOutcome::kUnknown;
				return;
			default:
				break;
			}
			Bump(kSunTests);
			DepthRect  rect{};
			const auto verdict = TestSun(a_context, a_bound, &rect);
			if (verdict == SunVerdict::kOutside) {
				Bump(kSunOutside);
				a_out.sun = SunOutcome::kUnneeded;
				return;
			}
			// Needed, though maybe not by the sun's last cascade (v1.80): only receivers in the slab of view depth it covers
			// read it, so the same sweep is tested against those alone. Its boxes join the evidence (the record is judged
			// again when the depth there moves, SunValid).
			if ((verdict == SunVerdict::kNeeded || verdict == SunVerdict::kEdge) && a_context.sun.farOn) {
				Bump(kSunFarTests);
				const auto lastCascade = TestSun(a_context, a_bound, &rect, true);
				if (lastCascade == SunVerdict::kOutside || lastCascade == SunVerdict::kHidden) {
					Bump(kSunFarUnneeded);
					a_out.flags |= kRecordSunFar;
				}
			}
			if (rect.used) {
				SetBlocks(a_context, a_out.sunBlocks, rect);
			}
			// The shadow-behind streak, as the view's (v1.73): an edge outcome (part of the sweep lands in view where the
			// depth frame has nothing) says nothing about the depth, so it carries the streak it had and leaves the
			// table's backup standing; past it (or past a view edge record, which judged no shadow), or with no record, the
			// backup continues it. In a pan the async view's widened leading strip made
			// every caster sweeping across it "needed" in the map; the walk judged the shadow behind surfaces again but
			// restarted its streak each frame, so the cascades drew it for as long as the pan lasted.
			const bool carried = a_old && (a_old->sun == SunOutcome::kBehind || (a_old->flags & kRecordSunEdge));
			const bool viaBackup = !a_old || (a_old->flags & kRecordSunEdge) ||
			                       (a_old->sun == SunOutcome::kNone && a_old->outcome == Outcome::kKept && a_old->verdict == static_cast<std::uint8_t>(Verdict::kEdge));
			if (verdict == SunVerdict::kEdge) {
				Bump(kSunNeeded);
				a_out.sun = SunOutcome::kNeeded;
				a_out.flags |= kRecordSunEdge;
				a_out.sunStreak = carried ? a_old->sunStreak : std::uint8_t{ 0 };
				return;
			}
			const auto entry = a_out.object ? PeekEntry(reinterpret_cast<std::uintptr_t>(a_out.object)) : nullptr;
			if (verdict != SunVerdict::kHidden) {
				Bump(kSunNeeded);
				a_out.sun = SunOutcome::kNeeded;
				if (entry && (carried || viaBackup)) {
					entry->sun.store(0, std::memory_order_relaxed);
				}
				return;
			}
			a_out.sun = SunOutcome::kBehind;
			std::uint32_t streak = carried ? a_old->sunStreak + 1u : 0u;
			if (entry && viaBackup) {
				streak = std::max(streak, BackupStreak(entry->sun.load(std::memory_order_relaxed), a_context.clock));
			}
			a_out.sunStreak = static_cast<std::uint8_t>(std::clamp(streak, 1u, 255u));
			if (entry) {
				entry->sun.store((static_cast<std::uint64_t>(a_out.sunStreak) << 32) | a_context.clock, std::memory_order_relaxed);
			}
			Bump(a_out.sunStreak < g_tunables.confirmFrames ? kSunConfirming : kSunHidden);
		}

		bool SunUnneededIn(const Record& a_record) noexcept
		{
			return a_record.sun == SunOutcome::kUnneeded || a_record.sun == SunOutcome::kLampUnneeded ||
			       (a_record.sun == SunOutcome::kBehind && a_record.sunStreak >= g_tunables.confirmFrames);
		}

		// The spot lamps group 0 also serves with the sun up (v1.54). The sun's test says nothing about a spot lamp's
		// shadow, yet the spot lights' group passes read group 0 as the cascades do (FO4-ENGINE-NOTES 6.2a), and the night's
		// moon counts as the sun (every exterior frame of the v1.53 run had the cascades on, and kept spot lights filed
		// from group 0 in them). So an entry the sun doesn't need stays in group 0 while it lies within a recorded spot
		// lamp's reach, unless bLampGroupTrim's shadow-volume test shows that no such lamp's shadow of it can reach a
		// visible surface. Point lamps cull the scene graph itself and never read group 0. With the sun off EvaluateSun
		// already weighed every lamp. Gathered at each frame's begin from the lamps the shadow stage recorded last frame
		// (ShadowLights::Lamps); read-only for the rest of the frame.
		struct SpotLamps
		{
			bool                                                    known{ false };  // the list holds every spot lamp
			std::uint32_t                                           count{ 0 };
			std::array<ShadowLights::Lamp, ShadowLights::LampList::kMax> items{};
			float                                                   min[3]{};        // the box around every lamp's reach
			float                                                   max[3]{};
		};
		SpotLamps g_spotLamps;

		// Spot lamps left out of g_spotLamps this frame (v1.81): an emptied shadow map skips its group pass and drops its
		// registrations, so the group-0 entries kept for such a lamp only fed the sun's cascades (2026-10-09, London, facing a
		// wall: 2,922 entries a frame kept for 6 spot lamps 7k units behind it, all 6 maps emptied; cascades registered 8.7k
		// objects a frame against previs' 1.2k). A lamp is left out when last frame emptied its map with the same lit volume
		// as the frame before, and that volume still tests hidden or out of view against this frame's depth: the lamp loop
		// then tests the same volume against the same depth and empties it again. One that turns or moves is never left
		// out; one that does so this frame for the first time is counted (kSpotLampsLeftLit) when the next list shows its
		// map was drawn.
		std::array<ShadowLights::Lamp, ShadowLights::LampList::kMax> g_spotLeftOut{};
		std::uint32_t                                                 g_spotLeftOutCount{ 0 };
		std::atomic<int>                                              g_spotLeftLitLogged{ 0 };

		void CheckSpotLeftOut(const ShadowLights::LampList& a_lamps) noexcept
		{
			for (std::uint32_t k = 0; k < g_spotLeftOutCount; ++k) {
				const auto& out = g_spotLeftOut[k];
				for (std::uint32_t i = 0; i < std::min<std::uint32_t>(a_lamps.count, ShadowLights::LampList::kMax); ++i) {
					const auto& lamp = a_lamps.items[i];
					const float dx = lamp.position.x - out.position.x, dy = lamp.position.y - out.position.y, dz = lamp.position.z - out.position.z;
					if (!lamp.spot || dx * dx + dy * dy + dz * dz > 256.0f * 256.0f || std::abs(lamp.reach - out.reach) > 0.01f * out.reach) {
						continue;
					}
					if (!lamp.emptied) {
						Bump(kSpotLampsLeftLit);
						if (g_spotLeftLitLogged.load(std::memory_order_relaxed) < 8 && g_spotLeftLitLogged.fetch_add(1) < 8) {
							logger::info(
								"occlusion: spot lamp at ({:.0f},{:.0f},{:.0f}) reach {:.0f} left out of group 0's lamps, but its shadow map was drawn | lit volume then ({:.0f},{:.0f},{:.0f}) r {:.0f}, at its update ({:.0f},{:.0f},{:.0f}) r {:.0f}",
								out.position.x, out.position.y, out.position.z, out.reach, out.volume.center.x, out.volume.center.y, out.volume.center.z, out.volume.fRadius,
								lamp.volume.center.x, lamp.volume.center.y, lamp.volume.center.z, lamp.volume.fRadius);
						}
					}
					break;
				}
			}
		}

		// After this frame's context is published (TestSphere reads it).
		void GatherSpotLamps() noexcept
		{
			auto&       spots = g_spotLamps;
			const auto& lamps = ShadowLights::Lamps();
			CheckSpotLeftOut(lamps);
			g_spotLeftOutCount = 0;
			spots.known = lamps.complete && lamps.overflow == 0;
			spots.count = 0;
			for (std::uint32_t i = 0; i < std::min<std::uint32_t>(lamps.count, ShadowLights::LampList::kMax); ++i) {
				const auto& lamp = lamps.items[i];
				if (!lamp.spot) {
					continue;
				}
				if (g_tunables.emptiedSpotTrim && lamp.emptied && lamp.steady && !Observing()) {
					const auto verdict = TestSphere(lamp.volume.center, lamp.volume.fRadius);
					if (verdict == SphereVerdict::kHidden || verdict == SphereVerdict::kOutOfView) {
						Bump(kSpotLampsEmptied);
						g_spotLeftOut[g_spotLeftOutCount++] = lamp;
						continue;
					}
				}
				const float p[3]{ lamp.position.x, lamp.position.y, lamp.position.z };
				for (int k = 0; k < 3; ++k) {
					spots.min[k] = spots.count == 0 ? p[k] - lamp.reach : std::min(spots.min[k], p[k] - lamp.reach);
					spots.max[k] = spots.count == 0 ? p[k] + lamp.reach : std::max(spots.max[k], p[k] + lamp.reach);
				}
				spots.items[spots.count++] = lamp;
			}
		}

		// Whether a recorded spot lamp may shadow a visible surface with this group-0 entry (a_object null: within reach
		// is enough, for a node's whole set of entries).
		bool SpotLampMayNeed(const FrameContext& a_context, RE::NiAVObject* a_object, const RE::NiBound& a_bound) noexcept
		{
			const auto& spots = g_spotLamps;
			if (!spots.known) {
				Bump(kSpotLampsUnknown);
				return true;
			}
			if (spots.count == 0) {
				return false;
			}
			const float r = a_bound.fRadius;
			const float c[3]{ a_bound.center.x, a_bound.center.y, a_bound.center.z };
			for (int k = 0; k < 3; ++k) {
				if (c[k] + r < spots.min[k] || c[k] - r > spots.max[k]) {
					return false;  // (most entries: far from every lamp)
				}
			}
			for (std::uint32_t i = 0; i < spots.count; ++i) {
				const auto& lamp = spots.items[i];
				const float dx = c[0] - lamp.position.x, dy = c[1] - lamp.position.y, dz = c[2] - lamp.position.z;
				const float reach = lamp.reach + r;
				if (dx * dx + dy * dy + dz * dz > reach * reach) {
					continue;
				}
				if (!g_tunables.lampGroupTrim || !a_object) {
					return true;
				}
				const auto verdict = TestLampCasterIn(a_context, a_object, lamp.position, lamp.reach, a_bound);
				if (verdict != LampVerdict::kOutside && verdict != LampVerdict::kMisses) {
					return true;
				}
			}
			return false;
		}

		// Whether group 0 can do without an entry the main view doesn't need: the sun's verdict, then (sun up) the spot lamps.
		bool GroupZeroUnneeded(const FrameContext& a_context, const Record& a_record, RE::NiAVObject* a_object, const RE::NiBound& a_bound) noexcept
		{
			if (!SunUnneededIn(a_record)) {
				return false;
			}
			if (a_record.sun == SunOutcome::kLampUnneeded) {
				return true;  // (the sun-off verdict already weighed every lamp)
			}
			if (SpotLampMayNeed(a_context, a_object, a_bound)) {
				Bump(kSpotLampKept);
				// (v1.82) The sun's verdict stands for the cascades: they register it only for the lamp's sake otherwise.
				// 2026-10-09, London street, 3 lit spot lamps: 1,723 such entries a frame, cascades 11.2k registrations
				// against previs' 4.3k.
				if (g_tunables.spotOnlyCascadeTrim && a_object && a_context.sun.state == FrameContext::Sun::State::kOn) {
					Bump(g_sunDrops.Insert(a_object, DropSet::Tag(a_context.clock)) ? kSpotOnlyLeftOut : kDropFull);
				}
				return false;
			}
			return true;
		}

		// A group-0 entry the main view doesn't need, kept for its sun shadow (v1.80). With a low sun a caster's sweep runs
		// up to the whole cascade range, so EvaluateSun found it needed whenever it reached a visible surface anywhere:
		// one next to the camera kept it in every cascade whose box holds it, the last (largest) one included (2026-10-09,
		// a London building: the last cascade filed 1,185 objects a frame against previs' 323). That cascade is read only
		// by receivers in its slab of view depth (godrays read the first ones, FO4-ENGINE-NOTES 6.1), so an entry whose
		// sweep finds nothing visible there is left out of that cascade's registration alone (FarCascade).
		void NoteFarCascade(const FrameContext& a_context, const RE::NiAVObject* a_object, const Record& a_out) noexcept
		{
			if (!(a_out.flags & kRecordSunFar) || a_out.sun != SunOutcome::kNeeded || !a_context.sun.farOn) {
				return;
			}
			if (g_farDrops.Insert(a_object, DropSet::Tag(a_context.clock))) {
				Bump(kSunFarLeftOut);
			} else {
				Bump(kDropFull);
			}
		}

		// The last cascade's registration of a geometry: left out when it, or a node above it, was left out of that cascade
		// this frame (the entry is the geometry or one of its parents; the walk below a node never leaves its bound).
		constexpr int kFarParentDepth = 6;

		bool InFarDrops(const RE::NiAVObject* a_object, std::uint32_t a_tag) noexcept
		{
			__try {
				auto object = a_object;
				for (int depth = 0; object && depth < kFarParentDepth; ++depth) {
					if (g_farDrops.Contains(object, a_tag)) {
						return true;
					}
					object = object->parent;
				}
			} __except (EXCEPTION_EXECUTE_HANDLER) {
			}
			return false;
		}

		// Any cascade's registration of a geometry (v1.82): left out when it, or a node above it, is an entry kept in group 0
		// for spot lamps alone this frame. Siblings come one after another, so the answer for the parent is kept (per thread);
		// the walk stops at a grouping node (always-draw bit 11: never an entry).
		struct SunParentMemo
		{
			const RE::NiAVObject* parent{ nullptr };
			std::uint32_t         tag{ 0 };
			bool                  dropped{ false };
		};
		thread_local SunParentMemo t_sunParent;

		bool SunCascade(const RE::NiAVObject* a_object)
		{
			const auto context = CurrentContext();
			if (!context || !a_object) {
				return false;
			}
			const auto tag = DropSet::Tag(context->clock);
			bool       dropped = g_sunDrops.Contains(a_object, tag);
			if (!dropped) {
				__try {
					const auto parent = a_object->parent;
					if (t_sunParent.parent != parent || t_sunParent.tag != tag) {
						bool found = false;
						auto object = static_cast<const RE::NiAVObject*>(parent);
						for (int depth = 1; object && depth < kFarParentDepth && !((object->GetFlags() >> 11) & 1); ++depth) {
							if (g_sunDrops.Contains(object, tag)) {
								found = true;
								break;
							}
							object = object->parent;
						}
						t_sunParent = { parent, tag, found };
					}
					dropped = t_sunParent.dropped;
				} __except (EXCEPTION_EXECUTE_HANDLER) {
					return false;
				}
			}
			if (dropped) {
				Bump(kSpotOnlyRegistrations);
			}
			return dropped;
		}

		bool FarCascade(const RE::NiAVObject* a_object)
		{
			const auto context = CurrentContext();
			if (!context || !context->sun.farOn || !a_object || !InFarDrops(a_object, DropSet::Tag(context->clock))) {
				return false;
			}
			Bump(kSunFarRegistrations);
			return true;
		}

		enum class Reuse : std::uint8_t
		{
			kNo,
			kYes,
			kRechecked,  // a hidden outcome whose evidence was re-checked against the current depth (and held)
		};

		// Whether last frame's record can stand for this frame's view outcome. The camera epoch and the object's bound
		// must hold for any outcome. Then by outcome:
		//   - out of view: geometry alone (the epoch's dilation covers the camera);
		//   - kept (drawn): no depth change can make drawing unsafe, so it stands until its periodic turn comes
		//     (kKeptRecheckFrames), whatever the depth did. Up to v1.26 any change in its blocks re-evaluated it, and
		//     under a camera's idle sway (silhouettes moving a pixel a frame) that was every depth-reading object every
		//     frame: the v1.26 log had 2,700 of 3,100 evaluations a frame down to "depth changed" while standing still;
		//   - hidden: the evidence must still hold. It does when none of its blocks moved since the depth it was last
		//     verified against; otherwise the evidence rectangle is re-checked against the current depth at its own
		//     threshold (one coarse query; a pass is as good as a re-evaluation, and cheaper than one by the full test's
		//     refinement, rays and, for mesh shapes, every cell).
		// The camera's position on the frame each depth readback was judged by (main thread at BeginFrame, by readback
		// index; read by any thread). A hidden outcome rests on the screen box it was tested with, widened for the parallax
		// of that frame's move (ParallaxTan); judged from more than kReuseMove away, its box doesn't cover this frame's
		// shift of a near occluder's edge, so while the camera moves it is judged again (v1.88.1: within one cache epoch,
		// up to fCacheMove of travel, a reused hidden verdict kept an object a pillar's edge had uncovered undrawn).
		constexpr float kReuseMove = 0.5f;
		struct JudgedEye
		{
			std::uint32_t readback{ 0 };
			float         eye[3]{};
		};
		std::array<JudgedEye, 64> g_judgedEyes{};

		bool MovedSince(const FrameContext& a_context, std::uint32_t a_readback) noexcept
		{
			const auto& at = g_judgedEyes[a_readback & 63];
			if (at.readback != a_readback) {
				return true;  // (too old to tell)
			}
			const float dx = a_context.eye[0] - at.eye[0], dy = a_context.eye[1] - at.eye[1], dz = a_context.eye[2] - at.eye[2];
			return dx * dx + dy * dy + dz * dz > kReuseMove * kReuseMove;
		}

		Reuse ViewValid(const FrameContext& a_context, const Record& a_old, const RE::NiBound& a_bound) noexcept
		{
			if (a_old.flags & kRecordNoCache) {
				Bump(kCacheNoCache);
				return Reuse::kNo;
			}
			if (a_old.viewEpoch != a_context.viewEpoch) {
				Bump(kCacheEpoch);
				return Reuse::kNo;
			}
			if (a_old.bound[0] != a_bound.center.x || a_old.bound[1] != a_bound.center.y || a_old.bound[2] != a_bound.center.z || a_old.bound[3] != a_bound.fRadius) {
				Bump(kCacheBound);
				return Reuse::kNo;
			}
			switch (a_old.outcome) {
			case Outcome::kOutside:
				return Reuse::kYes;
			case Outcome::kKept:
				if ((a_context.clock + static_cast<std::uint32_t>(reinterpret_cast<std::uintptr_t>(a_old.object) >> 4)) % kKeptRecheckFrames == 0) {
					Bump(kCacheKeptRecheck);
					return Reuse::kNo;
				}
				return Reuse::kYes;
			case Outcome::kHidden:
				if (!(a_old.flags & kRecordDepth)) {
					Bump(kCacheDepth);  // no evidence recorded: can't tell
					return Reuse::kNo;
				}
				if (a_context.parallaxMove > 0.0f && a_old.readback != a_context.readback && MovedSince(a_context, a_old.readback)) {
					Bump(kCacheMoved);
					return Reuse::kNo;
				}
				if (a_old.readback == a_context.readback || !BlocksChanged(a_context, a_old.blocks, a_old.readback)) {
					return Reuse::kYes;
				}
				{
					const bool timed = BucketTimed(a_context);
					const auto start = timed ? __rdtsc() : 0;
					const bool held = a_context.snapshot->AllNearer(a_old.rect[0], a_old.rect[1], a_old.rect[2], a_old.rect[3], a_old.threshold, kRecheckRefine);
					if (timed) {
						Bump(kCyclesRecheck, (__rdtsc() - start) * kTimingStride);
					}
					if (held) {
						Bump(kCacheRecheck);
						return Reuse::kRechecked;
					}
				}
				Bump(kCacheDepth);
				return Reuse::kNo;
			default:
				return Reuse::kNo;
			}
		}

		// Whether a reused record's sun outcome still stands (its epoch already matched): geometry outcomes always; one
		// that rests on the depth (the shadow falls only behind visible surfaces) while none of the blocks its test
		// read moved since.
		bool SunValid(const FrameContext& a_context, const Record& a_old) noexcept
		{
			if (a_old.sun == SunOutcome::kLampUnneeded || (a_old.flags & kRecordSunEdge)) {
				return false;  // (rests on the lamps and the depth of its frame, or on depth not there yet: re-judged every frame)
			}
			if (a_old.sun != SunOutcome::kBehind) {
				// (a needed shadow the last cascade doesn't need rests on the depth in its slab, when its test read any)
				if (!(a_old.flags & kRecordSunFar) || a_old.sunBlocks[0] == 255) {
					return true;
				}
			} else if (a_old.sunBlocks[0] == 255) {
				return false;
			}
			return a_old.sunReadback == a_context.readback || !BlocksChanged(a_context, a_old.sunBlocks, a_old.sunReadback);
		}

		void ReuseView(const Record& a_old, Record& a_out) noexcept
		{
			a_out = a_old;
			Bump(kTested);
			Bump(kCacheHits);
			switch (a_out.outcome) {
			case Outcome::kOutside:
				Bump(kOutside);
				break;
			case Outcome::kHidden:
				a_out.streak = static_cast<std::uint8_t>(std::min<std::uint32_t>(a_out.streak + 1u, 255u));
				if (a_out.streak < g_tunables.confirmFrames) {
					Bump(kConfirming);
				} else if (Observing()) {
					Bump(kWouldReject);
				} else {
					Bump(kRejected);
				}
				break;
			default:
				CountVerdict(static_cast<Verdict>(a_out.verdict));
				break;
			}
		}

		void ReuseSun(const FrameContext& a_context, Record& a_out) noexcept
		{
			switch (a_out.sun) {
			case SunOutcome::kBehind:
				a_out.sunStreak = static_cast<std::uint8_t>(std::min<std::uint32_t>(a_out.sunStreak + 1u, 255u));
				Bump(a_out.sunStreak < g_tunables.confirmFrames ? kSunConfirming : kSunHidden);
				break;
			case SunOutcome::kUnneeded:
				Bump(kSunOutside);
				break;
			case SunOutcome::kNeeded:
				Bump(kSunNeeded);
				break;
			case SunOutcome::kUnknown:
				if (a_context.sun.state == FrameContext::Sun::State::kOff) {
					Bump(kSunOff);
				}
				break;
			default:
				break;
			}
		}

		// ---- Renderer Replacer's traced scene (Core/TraceFeed) ------------------------------------------------------
		// t_moved: the thread's last JudgeOrLookup found last frame's record with another bound (the object moved).
		// t_traceNode: the cell node (or root, container) the walk is expanding for the trace this frame; its entries are
		// its children or, through a container, its grandchildren (the walk's two levels under a cell's node 3).
		thread_local bool t_moved{ false };
		struct TraceNode
		{
			const RE::NiAVObject* node{ nullptr };
			std::uint32_t         clock{ 0 };
			bool                  all{ true };  // every entry (false: the tracer knows the node, only its entries it doesn't know; v2.0)
		};
		thread_local TraceNode t_traceNode{};

		// On its turn and, with a tracer that keeps what it is handed, not handed already (or its resend turn; v2.0).
		bool TraceTurnDue(const void* a_object, std::uint32_t a_clock) noexcept
		{
			if (!TraceFeed::Turn(a_object, a_clock)) {
				return false;
			}
			if (TraceFeed::Due(a_object, a_clock)) {
				return true;
			}
			Bump(kTraceKnownSkipped);
			return false;
		}

		// Whether an object about to be left out of every view is filed for the trace instead this frame: within its
		// radius, on its turn, in a frame its bound moved, or under a node walked for it. v2.0, a tracer that keeps what
		// it is handed: on its turn only the first time (TraceTurnDue), a mover every frame within the tracer's mover
		// radius and one frame in 8 farther, under a node walked for it only an entry it doesn't know (unless the walk is
		// the node's resend).
		bool TraceRefresh(const FrameContext& a_context, const RE::NiAVObject* a_object, const RE::NiBound& a_bound) noexcept
		{
			if (!TraceFeed::On() || Observing()) {
				return false;
			}
			if (t_moved) {
				if (!TraceFeed::InReach(a_bound, a_context.eye)) {
					return false;
				}
				if (!TraceFeed::MoverDue(a_object, a_bound, a_context.eye, a_context.clock)) {
					Bump(kTraceMoverHeld);
					return false;
				}
				Bump(kTraceMoved);
				return true;
			}
			const auto parent = a_object->parent;
			const bool underNode = t_traceNode.node && t_traceNode.clock == a_context.clock && parent &&
			                       (parent == t_traceNode.node || parent->parent == t_traceNode.node);
			if (!(underNode ? (t_traceNode.all || !TraceFeed::Known(a_object)) : TraceTurnDue(a_object, a_context.clock)) || !TraceFeed::InReach(a_bound, a_context.eye)) {
				return false;
			}
			Bump(kTraceFiled);
			return true;
		}

		// A main-only entry (the main accumulator alone registers its group) filed for the trace: a main-view drop. False
		// (not its turn, or the drop table full): left out as before.
		bool TraceDropFiled(const FrameContext& a_context, const RE::NiAVObject* a_object) noexcept
		{
			if (!g_drops.Insert(a_object, DropSet::Tag(a_context.clock))) {
				Bump(kDropFull);
				return false;
			}
			TraceFeed::Handed(a_object);
			return true;
		}

		bool TraceDrop(const FrameContext& a_context, const RE::NiAVObject* a_object, const RE::NiBound& a_bound) noexcept
		{
			if (!TraceRefresh(a_context, a_object, a_bound)) {
				return false;
			}
			if (!g_drops.Insert(a_object, DropSet::Tag(a_context.clock))) {
				Bump(kDropFull);
				return false;
			}
			TraceFeed::Handed(a_object);
			return true;
		}

		// An entry filed force-visible for the trace (out of the main view, refused there; Hooks::CullGroups ForceVisible, or
		// TopDecision::forceVisible at Group::Add): its children get the mark with it (ParentForced). The engine's frustum
		// test alone leaves an out-of-view entry out (FO4-ENGINE-NOTES 5.2: result byte or force-visible byte), so without the
		// mark no view would file it. Only the main view meets the children's marks: a lamp's or a cascade's group pass
		// rebuilds group 0's child list and tests it with its own planes (5.2, 6.2a); the entry itself it walks again.
		void MarkForced(const FrameContext& a_context, const RE::NiAVObject* a_object) noexcept
		{
			Bump(kTraceForced);
			if (!g_traceForced.Insert(a_object, DropSet::Tag(a_context.clock))) {
				Bump(kDropFull);
			}
		}

		// A child of an entry filed force-visible for the trace: the same mark (out of view as its parent, not hidden by
		// implication) and a main-view drop.
		void ForcedChild(const FrameContext& a_context, const RE::NiAVObject* a_object, const RE::NiBound& a_bound) noexcept
		{
			const auto tag = DropSet::Tag(a_context.clock);
			Bump(kTraceForcedChildren);
			if (!g_drops.Insert(a_object, tag) || !g_traceForced.Insert(a_object, tag)) {
				Bump(kDropFull);
			}
			Hooks::CullGroups::ForceVisible();
			RecordLeftOut(a_context, a_object, a_bound, LeftOutPath::kInherited, nullptr);
		}

		// An out-of-view entry no view files (a main-only group's; group 0's with the sun down, when no shadow pass is known
		// to file it) filed for the trace on its turn: a main-view drop the caller files force-visible. False: left as before.
		bool TraceOutsideFiled(const FrameContext& a_context, const RE::NiAVObject* a_object) noexcept
		{
			if (!g_drops.Insert(a_object, DropSet::Tag(a_context.clock))) {
				Bump(kDropFull);
				return false;
			}
			MarkForced(a_context, a_object);
			TraceFeed::Handed(a_object);
			return true;
		}

		bool TraceOutside(const FrameContext& a_context, const RE::NiAVObject* a_object, const RE::NiBound& a_bound) noexcept
		{
			return TraceRefresh(a_context, a_object, a_bound) && TraceOutsideFiled(a_context, a_object);
		}

		// A group-0 entry nothing needs (GroupZeroUnneeded) filed for the trace: a main-view drop and, with the sun up, out of
		// the sun's cascades (SunCascade, on while the feed is). Out of view (a_hidden false) the caller files it
		// force-visible (marked here). False: left out as before.
		bool TraceKeepFiled(const FrameContext& a_context, const RE::NiAVObject* a_object, const RE::NiBound& a_bound, bool a_hidden, const Record& a_record) noexcept
		{
			const bool sunUp = a_context.sun.state == FrameContext::Sun::State::kOn;
			const auto tag = DropSet::Tag(a_context.clock);
			if (!g_drops.Insert(a_object, tag) || (sunUp && !g_sunDrops.Insert(a_object, tag))) {
				Bump(kDropFull);
				return false;
			}
			if (!a_hidden) {
				MarkForced(a_context, a_object);
			}
			TraceFeed::Handed(a_object);
			RecordLeftOut(a_context, a_object, a_bound, LeftOutPath::kDropped, &a_record);
			return true;
		}

		bool TraceKeep(const FrameContext& a_context, const RE::NiAVObject* a_object, const RE::NiBound& a_bound, bool a_hidden, const Record& a_record) noexcept
		{
			return TraceRefresh(a_context, a_object, a_bound) && TraceKeepFiled(a_context, a_object, a_bound, a_hidden, a_record);
		}

		// The object's outcome for this frame: reused from its last record (a_old, or null) when everything it depended
		// on still holds, evaluated otherwise; the sun part too when a_wantSun and the main view doesn't need the
		// object.
		void JudgeRecord(const FrameContext& a_context, const Hooks::CullGroups::BlockAdd& a_add, const Record* old, bool a_wantSun, Record& a_out)
		{
			t_moved = old && (old->bound[0] != a_add.bound->center.x || old->bound[1] != a_add.bound->center.y || old->bound[2] != a_add.bound->center.z ||
			                   old->bound[3] != a_add.bound->fRadius);
			const bool    reusable = a_context.cacheEnabled && a_context.clock != g_dumpClock.load(std::memory_order_relaxed);
			const bool    timed = Timed(a_context);
			const bool    parts = BucketTimed(a_context);
			bool          reused = false;
			if (old && reusable) {
				const auto validStart = parts ? __rdtsc() : 0;
				const auto reuse = ViewValid(a_context, *old, *a_add.bound);
				if (parts) {
					Bump(kCyclesReuse, (__rdtsc() - validStart) * kTimingStride);
				}
				if (reuse != Reuse::kNo) {
					ReuseView(*old, a_out);
					if (reuse == Reuse::kRechecked) {
						a_out.readback = a_context.readback;  // the evidence was verified against this depth
					}
					reused = true;
					// (the streak backup, kept fresh for a record the stream may lose)
					if (a_out.outcome == Outcome::kHidden && RefreshTurn(a_add.object, a_context.clock)) {
						const auto backupStart = parts ? __rdtsc() : 0;
						if (const auto entry = FindEntry(reinterpret_cast<std::uintptr_t>(a_add.object))) {
							StoreStreak(*entry, a_context.clock, a_out.streak);
						}
						if (parts) {
							Bump(kCyclesRecord, (__rdtsc() - backupStart) * kTimingStride);
						}
					}
				}
			}
			if (!reused) {
				if (!old) {
					Bump(kCacheNew);
				}
				const auto start = timed ? __rdtsc() : 0;
				EvaluateView(a_context, a_add, old, a_out, timed);
				if (timed) {
					Bump(kCyclesEvaluate, (__rdtsc() - start) * kTimingStride);
				}
			}
			t_source = reused ? VerdictSource::kReused : VerdictSource::kFresh;  // (the left-out dump)
			if (a_wantSun && !(a_out.flags & kRecordLight)) {
				const bool unneededByView = a_out.outcome == Outcome::kOutside || (a_out.outcome == Outcome::kHidden && a_out.streak >= g_tunables.confirmFrames);
				const bool sunReusable = reused && old->sun != SunOutcome::kNone && old->sunEpoch == a_context.sunEpoch;
				if (!unneededByView) {
					a_out.sun = SunOutcome::kNone;
					a_out.flags &= ~kRecordSunFar;
				} else if (sunReusable && SunValid(a_context, *old)) {
					ReuseSun(a_context, a_out);
					// (the shadow streak's backup, kept fresh as the view's is, for a record the map loses or an edge)
					if (a_out.sun == SunOutcome::kBehind && RefreshTurn(a_add.object, a_context.clock)) {
						if (const auto entry = PeekEntry(reinterpret_cast<std::uintptr_t>(a_add.object))) {
							entry->sun.store((static_cast<std::uint64_t>(a_out.sunStreak) << 32) | a_context.clock, std::memory_order_relaxed);
						}
					}
				} else {
					if (sunReusable) {
						Bump(kCacheSunDepth);
					}
					const auto start = timed ? __rdtsc() : 0;
					EvaluateSun(a_context, *a_add.bound, old, a_out);
					if (timed) {
						Bump(kCyclesSun, (__rdtsc() - start) * kTimingStride);
					}
				}
			}
		}

		// The synchronous path: last frame's record comes from the thread's stream, and the new one goes into it.
		void Judge(const FrameContext& a_context, Stream& a_stream, const Hooks::CullGroups::BlockAdd& a_add, bool a_wantSun, Record& a_out)
		{
			const bool timed = BucketTimed(a_context);
			const auto lookupStart = timed ? __rdtsc() : 0;
			const auto old = a_stream.Find(a_add.object);
			if (timed) {
				Bump(kCyclesLookup, (__rdtsc() - lookupStart) * kTimingStride);
			}
			JudgeRecord(a_context, a_add, old, a_wantSun, a_out);
			const auto recordStart = timed ? __rdtsc() : 0;
			a_stream.Push(a_out);
			if (timed) {
				Bump(kCyclesRecord, (__rdtsc() - recordStart) * kTimingStride);
			}
		}

		// With asynchronous verdicts (Core/Async): the entry is recorded for the worker, and its outcome comes from
		// the worker's map (last frame's judgement of the same object with the same bound) when the map holds for
		// this frame's camera. What the map can't answer is judged here, as the synchronous path does, last frame's
		// record (if any) continuing the streaks: an object new to the walk or whose bound moved (animated, swaying),
		// and every object when the camera turned or moved past the map's margins. Up to v1.59 all of those were
		// simply kept: with a wobbling camera 4% of frames drew everything (~25,000 registrations against ~6,000).
		void JudgeOrLookupRecord(const FrameContext& a_context, Stream& a_stream, const Hooks::CullGroups::BlockAdd& a_add, bool a_wantSun, Record& a_out)
		{
			if (!Async::Enabled() || !a_context.asyncFrame) {
				Judge(a_context, a_stream, a_add, a_wantSun, a_out);
				return;
			}
			const bool timed = BucketTimed(a_context);
			const auto start = timed ? __rdtsc() : 0;
			Async::Record(a_add.object, *a_add.bound, a_add.kind);
			const auto record = Async::Find(a_add.object);
			if (timed) {
				Bump(kCyclesLookup, (__rdtsc() - start) * kTimingStride);
			}
			if (a_context.asyncValid && record && record->bound[0] == a_add.bound->center.x && record->bound[1] == a_add.bound->center.y &&
				record->bound[2] == a_add.bound->center.z && record->bound[3] == a_add.bound->fRadius) {
				// A verdict still unsettled is judged again with this frame's own view and depth, one capture newer than the
				// worker's: an object kept only because part of it lay where the worker's depth frame was never rendered (the
				// leading strip of a turn, and the strip its view was widened by), and a hidden one still confirming. The map
				// is a frame behind the walk, so in a turn every object entering the view stayed drawn a frame longer than
				// the walk would draw it (edge, then confirming, then hidden), and v1.70's widening kept every group-0 caster
				// in the strip beyond the view for the sun's cascades. Last frame's record continues the streak, as in the walk.
				const bool unsettled = (record->outcome == Outcome::kKept && record->verdict == static_cast<std::uint8_t>(Verdict::kEdge)) ||
				                       (record->outcome == Outcome::kHidden && record->streak < g_tunables.confirmFrames);
				if (!unsettled) {
					a_out = *record;
					Bump(kCacheHits);
					t_source = VerdictSource::kAsync;
					// (a shadow needed only as its sweep reaches the strip where the worker's depth frame has nothing, its
					// view outcome settled: the shadow alone is judged again, with its streak carried)
					if (a_wantSun && (record->flags & kRecordSunEdge)) {
						Bump(kWalkSunEdge);
						EvaluateSun(a_context, *a_add.bound, record, a_out);
					}
					return;
				}
				Bump(kWalkUnsettled);
				Record old = *record;
				old.flags |= kRecordNoCache;  // (evaluated, never reused: the view and depth it rests on are what changed)
				JudgeRecord(a_context, a_add, &old, a_wantSun, a_out);
				return;
			}
			if (!a_context.cull) {
				a_out = Record{};
				a_out.object = a_add.object;
				a_out.outcome = Outcome::kKept;
				return;
			}
			Bump(kWalkJudged);
			JudgeRecord(a_context, a_add, record, a_wantSun, a_out);
		}

		// A fade node the main view leaves out keeps its fade (v1.88.2, bKeepFadeState). BSFadeNode's fade update (id 928617,
		// from OnVisible) restarts the fade from 0 for a node unseen for over 20 frames (frameCounter +0x1B0 against the frame
		// number, id 734919), radius under 600, whose distance fade is at or below fFadeInThreshold: an object behind a
		// pillar for a third of a second faded in again from nothing when the pillar uncovered it ("popping up"). Previs
		// never let that happen: its entries get OnVisible every frame, even behind the camera. So the counter is kept
		// current, as OnVisible would; the fade value itself is left as it was. Main thread (the walk).
		constexpr std::uint64_t kFrameNumberID = 734919;
		constexpr std::size_t   kFadeFrameCounterOffset = 0x1B0;
		constexpr std::size_t   kFadeCurrentOffset = 0x1A0;       // BSFadeNode::currentFade
		constexpr std::size_t   kFadeRadiusOffset = 0x1A8;        // BSFadeNode::boundRadius (CheckFadeRadius; 0 on a chunk node)
		constexpr std::uint64_t kFadeCompleteBit = std::uint64_t{ 1 } << 37;  // flag bit 37: the fade-in completed

		const std::int32_t* FrameNumber() noexcept
		{
			static const auto frame = reinterpret_cast<const std::int32_t*>(CBRO::Engine::OG(kFrameNumberID).address());
			return frame;
		}

		// The counter only has to stay within 20 frames of the frame number, so a node is stamped on its turn, one frame in
		// kFadeStampFrames (v1.92). Up to v1.91 every left-out node was stamped every frame: ~7,000 stores a frame to engine
		// objects in a dense exterior (2026-10-26), each with a virtual IsFadeNode call, on the walk's critical path.
		constexpr std::uint32_t kFadeStampFrames = 8;

		bool FadeStampTurn(const void* a_object, std::uint32_t a_clock) noexcept
		{
			return (a_clock + static_cast<std::uint32_t>(reinterpret_cast<std::uintptr_t>(a_object) >> 4)) % kFadeStampFrames == 0;
		}

		// ---- the engine's fade data (BSFadeNode, FO4OG 7.8, 7.9) ----------------------------------------------------
		constexpr std::uint64_t kFadeMultTableID = 804710;
		constexpr std::uint64_t kFadeDistanceMultiplierID = 515159;
		constexpr std::uint64_t kFadeEnabledID = 1220201;
		constexpr std::uint64_t kFadeInThresholdID = 758162;  // fFadeInThreshold (the unseen rule's "near": id 928617)
		// BSFadeNode::OnVisible(NiCamera&, bool) (531348) and what it calls (FO4OG 7.8, 7.9; decompiled 2026-10-27)
		constexpr std::uint64_t kFadeUpdateID = 928617;             // the fade update for every type but 6: (node, fadeAmount +0x118, float[4] camera x y z lodAdjust)
		constexpr std::uint64_t kComputeFadeAmountID = 1040250;     // BSFadeNode::ComputeFadeAmount(float[4] camera, float& distance) -> fade
		constexpr std::uint64_t kDetermineMeshLODLevelID = 980443;  // BSFadeNode::DetermineMeshLODLevel(float distance)
		constexpr std::uint64_t kFadeUpdateGateID = 1445134;        // byte: 0 = OnVisible runs no fade update at all
		constexpr std::uint64_t kFadeOverrideID = 846038;           // float: -1 = none (type 6 then takes OnVisible's own branch)
		constexpr std::uint64_t kFadeDeltaID = 160595;              // float: the numerator of every fade step
		constexpr std::uint64_t kFadeInDivisorID = 538734;          // float: the fade-in step's divisor
		constexpr std::uint64_t kFadeStepCapID = 1415799;           // float: the most a fade moves in one update
		constexpr std::size_t   kCameraTranslateOffset = 0xA0;      // NiCamera world translate (OnVisible copies +0xA0, +0xA4, +0xA8, +0x194)
		using FadeUpdateFn = std::int64_t (*)(void*, float, const float*);
		using ComputeFadeAmountFn = float (*)(void*, const float*, float*);
		using DetermineMeshLODLevelFn = void (*)(void*, float);
		constexpr std::uint64_t kDrawWorldCameraID = 81406;  // global NiCamera*
		constexpr std::size_t   kCameraLODAdjustOffset = 0x194;

		struct FadeGlobals
		{
			const float*          multTable;
			const float*          distanceMultiplier;
			const std::uint8_t*   fading;
			const std::uintptr_t* camera;
			const float*          fadeInThreshold;
			const std::uint8_t*   updateGate;
			const float*          override;
			const float*          delta;
			const float*          inDivisor;
			const float*          stepCap;
			FadeUpdateFn          update;
			ComputeFadeAmountFn   computeFadeAmount;
			DetermineMeshLODLevelFn determineMeshLODLevel;
		};
		constexpr std::uint32_t   kFadeStartSamples = 8;  // fade-in starts named in the log per interval (v1.96.2)
		std::atomic<std::uint32_t> g_fadeStartSamples{ 0 };

		const FadeGlobals& FadeGlobalsOf() noexcept
		{
			static const FadeGlobals globals{
				reinterpret_cast<const float*>(CBRO::Engine::OG(kFadeMultTableID).address()),
				reinterpret_cast<const float*>(CBRO::Engine::OG(kFadeDistanceMultiplierID).address()),
				reinterpret_cast<const std::uint8_t*>(CBRO::Engine::OG(kFadeEnabledID).address()),
				reinterpret_cast<const std::uintptr_t*>(CBRO::Engine::OG(kDrawWorldCameraID).address()),
				reinterpret_cast<const float*>(CBRO::Engine::OG(kFadeInThresholdID).address()),
				reinterpret_cast<const std::uint8_t*>(CBRO::Engine::OG(kFadeUpdateGateID).address()),
				reinterpret_cast<const float*>(CBRO::Engine::OG(kFadeOverrideID).address()),
				reinterpret_cast<const float*>(CBRO::Engine::OG(kFadeDeltaID).address()),
				reinterpret_cast<const float*>(CBRO::Engine::OG(kFadeInDivisorID).address()),
				reinterpret_cast<const float*>(CBRO::Engine::OG(kFadeStepCapID).address()),
				reinterpret_cast<FadeUpdateFn>(CBRO::Engine::OG(kFadeUpdateID).address()),
				reinterpret_cast<ComputeFadeAmountFn>(CBRO::Engine::OG(kComputeFadeAmountID).address()),
				reinterpret_cast<DetermineMeshLODLevelFn>(CBRO::Engine::OG(kDetermineMeshLODLevelID).address())
			};
			return globals;
		}

		// v1.97.0: the fade of a left-out node is run, not only kept. With previs active, every one of its list entries is
		// force-visible in the finish loop (FO4OG 5.2 item 7) and every dynamic object visible from the camera position gets
		// BSFadeNode::OnVisible(camera) inside the query (FO4OG 5.5d; with precombines off that is nearly every static
		// reference): a node fades in, and out by distance, whether or not a view draws it, and is complete by the time it
		// enters the view. In a CBRO frame previs is suspended in the cull and the query is skipped (5.5e), so only drawn nodes
		// advanced; the v1.88.2 stamp alone froze the rest, and a node CBRO had hidden mid-fade, or had never shown, ran its
		// fade on screen when it came back (137 mid-fade nodes a frame strafing in the v1.96.1 run; each re-queues its
		// lighting every frame, 7.9). Now a left-out fade node gets what OnVisible would give it (FadeAdvance): the engine's own
		// update with the engine's own camera data, then the frame counter; not the lighting queue and not the filing, which
		// only a drawn node needs. A node mid-fade is updated every frame (as the query did); one complete, or never faded,
		// on its turn (1 frame in 8: a complete node only needs its counter within the 20-frame rule, a far node at 0 does not
		// move). Mid-fade nodes are remembered in a 64K-bit set so the other frames cost one bit test per left-out object.
		constexpr std::size_t kFadeSetWords = 1024;
		std::array<std::atomic<std::uint64_t>, kFadeSetWords> g_fadeSet{};
		std::atomic<std::uint32_t>                            g_fadeSetClock{ 0 };
		std::array<std::uintptr_t, 2>                         g_fadeVtables{};  // BSFadeNode, BSLeafAnimNode (other subclasses: the virtual call, on the turn)

		inline std::size_t FadeSetBit(std::uintptr_t a_node) noexcept
		{
			return static_cast<std::size_t>(((a_node >> 4) * 0x9E3779B97F4A7C15ull) >> (64 - 16));
		}

		// The fade update as BSFadeNode::OnVisible (531348) runs it (decompiled 2026-10-27): the skip for a complete bit-15
		// node, the decal fade's step, type 6's own branch (ComputeFadeAmount, DetermineMeshLODLevel, the fade-in step) or
		// id 928617 for the rest, the decal reset at fade 0, the frame counter. Returns false when OnVisible would not update.
		bool FadeAdvance(std::uintptr_t a_node) noexcept
		{
			const auto& g = FadeGlobalsOf();
			if (!*g.updateGate || !*g.camera) {
				return false;
			}
			auto&      flags = *reinterpret_cast<std::uint64_t*>(a_node + 0x108);
			auto&      currentFade = *reinterpret_cast<float*>(a_node + kFadeCurrentOffset);
			auto&      decalFade = *reinterpret_cast<float*>(a_node + 0x1A4);
			const auto fadeAmount = *reinterpret_cast<const float*>(a_node + 0x118);
			const bool bit15 = (flags >> 15) & 1;
			if (bit15 && fadeAmount == 1.0f && currentFade == 1.0f && decalFade == 1.0f) {
				return false;
			}
			const auto  camera = *g.camera;
			const float data[4]{
				*reinterpret_cast<const float*>(camera + kCameraTranslateOffset),
				*reinterpret_cast<const float*>(camera + kCameraTranslateOffset + 4),
				*reinterpret_cast<const float*>(camera + kCameraTranslateOffset + 8),
				*reinterpret_cast<const float*>(camera + kCameraLODAdjustOffset)
			};
			const float step = std::min(*g.delta / *g.inDivisor, *g.stepCap);
			if ((flags & kFadeCompleteBit) && decalFade < 1.0f) {
				decalFade = bit15 ? 1.0f : std::min(decalFade + step, 1.0f);
			}
			const auto type = *reinterpret_cast<const std::uint8_t*>(a_node + 0x11C);
			if (type == 6 && *g.override == -1.0f) {
				float distance = 0.0f;
				g.computeFadeAmount(reinterpret_cast<void*>(a_node), data, &distance);
				if (currentFade == 0.0f) {
					decalFade = 0.0f;
				}
				if ((flags >> 12) & 1) {
					g.determineMeshLODLevel(reinterpret_cast<void*>(a_node), distance);
				}
				if (!*g.fading) {
					*reinterpret_cast<float*>(a_node + 0x1B4) = 1.0f;
					decalFade = 1.0f;
					currentFade = 1.0f;
					flags |= kFadeCompleteBit;
				}
				if (!(flags & kFadeCompleteBit)) {
					currentFade = std::min(currentFade + step, 1.0f);
					if (currentFade >= 1.0f) {
						flags |= kFadeCompleteBit;
					}
				}
			} else {
				g.update(reinterpret_cast<void*>(a_node), fadeAmount, data);
				if (currentFade == 0.0f) {
					decalFade = 0.0f;
				}
			}
			*reinterpret_cast<std::int32_t*>(a_node + kFadeFrameCounterOffset) = *FrameNumber();
			return true;
		}

		// a_turn: the node's turn is taken now whatever the frame (v1.98: an entry left untouched under a lease is stamped at
		// its cluster's next walk, within the 20-frame rule).
		void KeepFadeSeen(RE::NiAVObject* a_object, std::uint32_t a_clock, bool a_turn = false) noexcept
		{
			const auto node = reinterpret_cast<std::uintptr_t>(a_object);
			const bool turn = a_turn || FadeStampTurn(a_object, a_clock);
			const auto bit = FadeSetBit(node);
			if (!turn && !((g_fadeSet[bit >> 6].load(std::memory_order_relaxed) >> (bit & 63)) & 1)) {
				return;  // not its turn and not known mid-fade: nothing to do this frame
			}
			auto setClock = g_fadeSetClock.load(std::memory_order_relaxed);
			if (a_clock - setClock >= 64 && g_fadeSetClock.compare_exchange_strong(setClock, a_clock, std::memory_order_relaxed)) {
				for (auto& word : g_fadeSet) {
					word.store(0, std::memory_order_relaxed);  // forgotten mid-fade nodes are found again on their next turn
				}
			}
			__try {
				const auto vtable = *reinterpret_cast<const std::uintptr_t*>(node);
				if (vtable != g_fadeVtables[0] && vtable != g_fadeVtables[1] && (!turn || !a_object->IsFadeNode())) {
					return;
				}
				const auto flags = *reinterpret_cast<const std::uint64_t*>(node + 0x108);
				const auto before = *reinterpret_cast<const float*>(node + kFadeCurrentOffset);
				if (flags & kFadeCompleteBit) {
					if (turn && FadeAdvance(node)) {  // complete: the counter within the 20-frame rule, the distance fade-out
						Bump(kFadeKeptSeen);
					}
					return;
				}
				if (before == 0.0f && !turn) {
					return;  // never faded (or faded out): started on its turn
				}
				if (!FadeAdvance(node)) {
					return;
				}
				Bump(kFadeAdvanced);
				const auto after = *reinterpret_cast<const float*>(node + kFadeCurrentOffset);
				if (before == 0.0f && after > 0.0f) {
					Bump(kFadeStarted);
				}
				if (*reinterpret_cast<const std::uint64_t*>(node + 0x108) & kFadeCompleteBit) {
					Bump(kFadeCompleted);
				} else if (after > 0.0f) {
					g_fadeSet[bit >> 6].fetch_or(std::uint64_t{ 1 } << (bit & 63), std::memory_order_relaxed);  // mid-fade: every frame until the next clear
				}
			} __except (EXCEPTION_EXECUTE_HANDLER) {
			}
		}

		// Whether a node is known mid-fade (the set), without touching it: an entry left untouched under a lease is visited
		// for its fade update then (v1.98).
		inline bool FadeMidWay(const void* a_object) noexcept
		{
			const auto bit = FadeSetBit(reinterpret_cast<std::uintptr_t>(a_object));
			return (g_fadeSet[bit >> 6].load(std::memory_order_relaxed) >> (bit & 63)) & 1;
		}

		// The walk's outcome for the entry, counted as decided (the turn table's own share of what the main view keeps).
		void JudgeOrLookup(const FrameContext& a_context, Stream& a_stream, const Hooks::CullGroups::BlockAdd& a_add, bool a_wantSun, Record& a_out)
		{
			const bool timed = BucketTimed(a_context);
			const auto start = timed ? __rdtsc() : 0;
			t_moved = false;
			JudgeOrLookupRecord(a_context, a_stream, a_add, a_wantSun, a_out);
			if (Recorder::On()) {
				Recorder::NoteJudged(a_out, static_cast<std::uint8_t>(t_source));
			}
			if (g_tunables.keepFadeState && a_add.object && (a_out.outcome == Outcome::kHidden || a_out.outcome == Outcome::kOutside)) {
				KeepFadeSeen(const_cast<RE::NiAVObject*>(a_add.object), a_context.clock);
			}
			if (a_out.outcome == Outcome::kKept && a_out.verdict == static_cast<std::uint8_t>(Verdict::kEdge)) {
				Bump(kFinalEdge);
			} else if (a_out.outcome == Outcome::kHidden && a_out.streak < g_tunables.confirmFrames) {
				Bump(kFinalConfirming);
			}
			if (timed) {
				Bump(kCyclesJudge, (__rdtsc() - start) * kTimingStride);
			}
		}

		// Whether the sun's cascades can do without a caster (node pruning, on a node's entries' sphere). "Outside"
		// is geometry alone and holds at once; "hidden" rests on the depth frame, so like an object's it must hold
		// for confirmFrames consecutive frames (streak in the history table).
		bool SunUnneeded(const FrameContext& a_context, RE::NiAVObject* a_object, const RE::NiBound& a_bound) noexcept
		{
			switch (a_context.sun.state) {
			case FrameContext::Sun::State::kOff:
				Bump(kSunOff);
				return false;  // (group 0 has other readers with the sun off: see EvaluateSun)
			case FrameContext::Sun::State::kUnknown:
				return false;
			default:
				break;
			}
			Bump(kSunTests);
			const auto verdict = TestSun(a_context, a_bound);
			if (verdict == SunVerdict::kOutside) {
				Bump(kSunOutside);
				return true;
			}
			if (verdict != SunVerdict::kHidden) {
				Bump(kSunNeeded);
				return false;
			}
			const auto entry = FindEntry(reinterpret_cast<std::uintptr_t>(a_object));
			if (!entry) {
				Bump(kTableFull);
				return false;
			}
			const auto streak = HiddenStreak(entry->sun.load(std::memory_order_relaxed), a_context.clock);
			entry->sun.store((static_cast<std::uint64_t>(streak) << 32) | a_context.clock, std::memory_order_relaxed);
			if (streak < g_tunables.confirmFrames) {
				Bump(kSunConfirming);
				return false;
			}
			Bump(kSunHidden);
			return true;
		}

		// Diagnostic (SetKeptReasons, Core/SetDiff): why the main view keeps a geometry entry, into its history entry's lamp
		// word (the lamp streak keeps the word's low bits). Walk threads, at the Block::Add filter, for every entry filed.
		void NoteKeptReason(const FrameContext& a_context, RE::NiAVObject* a_object, const Record& a_record) noexcept
		{
			const bool drawn = a_record.outcome == Outcome::kKept || (a_record.outcome == Outcome::kHidden && a_record.streak < g_tunables.confirmFrames);
			if (!drawn || !a_object->IsGeometry()) {
				return;
			}
			const auto entry = FindEntry(reinterpret_cast<std::uintptr_t>(a_object));
			if (!entry) {
				return;
			}
			const std::uint64_t reason = 1u | (static_cast<std::uint64_t>(a_record.verdict & 7u) << 1) |
			                             (static_cast<std::uint64_t>(static_cast<std::uint8_t>(a_record.outcome) & 3u) << 4) |
			                             (static_cast<std::uint64_t>((a_record.flags & kRecordShaped) != 0) << 6) |
			                             (static_cast<std::uint64_t>((a_record.flags & kRecordShapeNone) != 0) << 7) |
			                             (static_cast<std::uint64_t>((a_record.flags & kRecordParallaxKept) != 0) << 8) |
			                             (static_cast<std::uint64_t>(a_context.clock & 0x3FFu) << 10);
			const auto old = entry->lamp.load(std::memory_order_relaxed);
			entry->lamp.store((old & ~kReasonMask) | (reason << kReasonShift), std::memory_order_relaxed);
		}

		bool EntryOutside(const FrameContext& a_context, const RE::NiBound& a_bound) noexcept;
		bool ShadowBeyondReach(const FrameContext& a_context, const RE::NiBound& a_bound) noexcept;

		// A Block::Add child (the engine's push of a kept node's geometry: a fade node's geomArray records, a node's
		// children) got a full judgement each: the worker's map, the sun test, the drop sets, the records, the recorder.
		// The 2026-10-27 recording (Fallout London street, strafing): 3,900 judged children a frame, 2,050 of them out of
		// the view with a sun shadow no view needs (NPC head parts behind the camera, object LOD meshes, effect shapes),
		// ~150 ns each on the main thread, for a verdict the engine's own frustum test gives the main view anyway. So a
		// plain static mesh out of the view (the entry walk's own cheap tests: EntryOutside with the cache's margins,
		// ShadowBeyondReach, no spot lamp in reach) that isn't due for the trace is left out with no record, as the walk
		// leaves out such a top-level entry (v1.96, bChildCheapTests). Static: a tri-shape vtable, no skin instance, and a
		// world translation equal to previousWorld's (NiAVObject::UpdateWorldData copies the world into previousWorld
		// before each move, FO4-ENGINE-NOTES 7.10): anything animated or moved this frame (the trace feed refreshes
		// those), a node (its own bound is unmaintained), a light or a particle system takes the full path. Never in
		// observe-only mode, nor on the left-out dump's frame. A child so left out has no record: when it enters the view
		// it is judged afresh and, if hidden, confirms over confirmFrames first, as a walk-skipped entry does.
		constexpr std::size_t         kSkinInstanceOffset = 0x140;  // BSGeometry::skinInstance (Core/MeshProxy)
		std::array<std::uintptr_t, 3> g_staticShapeVtables{};

		bool CheapChildOutside(const FrameContext& a_context, const Hooks::CullGroups::BlockAdd& a_add) noexcept
		{
			if (!g_tunables.cheapChildren || !a_context.cull || Observing() || LeftOutFrame(a_context.clock)) {
				return false;
			}
			const auto object = a_add.object;
			const auto vtable = *reinterpret_cast<const std::uintptr_t*>(object);  // (the thunk read the object's flags already)
			if (vtable != g_staticShapeVtables[0] && vtable != g_staticShapeVtables[1] && vtable != g_staticShapeVtables[2]) {
				return false;
			}
			if (*reinterpret_cast<const std::uintptr_t*>(reinterpret_cast<std::uintptr_t>(object) + kSkinInstanceOffset) != 0) {
				return false;
			}
			const auto& now = object->world.translate;
			const auto& before = object->previousWorld.translate;
			if (now.x != before.x || now.y != before.y || now.z != before.z) {
				return false;
			}
			const auto& bound = *a_add.bound;
			if (!EntryOutside(a_context, bound)) {
				return false;
			}
			if (a_add.kind == Hooks::CullGroups::GroupKind::kSunShared && (!ShadowBeyondReach(a_context, bound) || SpotLampMayNeed(a_context, nullptr, bound))) {
				return false;
			}
			if (TraceFeed::On()) {
				const auto parent = object->parent;
				const bool underNode = t_traceNode.node && t_traceNode.clock == a_context.clock && parent && (parent == t_traceNode.node || parent->parent == t_traceNode.node);
				if ((underNode ? (t_traceNode.all || !TraceFeed::Known(object)) : TraceTurnDue(object, a_context.clock)) && TraceFeed::InReach(bound, a_context.eye)) {
					return false;
				}
			}
			t_source = VerdictSource::kNone;
			return true;
		}

		// ---- the trace feed's direct handover (v1.99.3, bTraceDirect, bDeferSunOnly) -----------------------------------
		// The 2026-10-27 v1.99.2 run: of the nodes the main view finished a frame, ~240 were filed for the trace alone (229
		// out of view, force-visible, 697 pieces with the mark) and ~377 were sun-only drops (hidden, kept in group 0 for the
		// cascades). For each the main view ran OnVisible, pushed every piece (AddP1, Block::Add, this filter, two drop-set
		// inserts per forced piece), tested and registered them, and CBRO refused every registration only to hand it to the
		// tracer; the forced ones were walked again by the sun traversal and the lamp passes (FO4OG 5.2 item 7). Now such an
		// entry is handed over here: a geometry as it stands; a fade node as the finish loop would register it (558046:
		// OnVisible with DrawWorld's camera, its fade and the LOD level the tracer reads; then, when the finish would call
		// AddP1FadeNode, the geomArray brought up to date and each record AddP1FadeNode would push, FO4OG 5.2, 7.8), to
		// the callback the refusals call. A trace filing no view needs is not filed at all; a sun-only drop is filed into
		// group 0 inside the cascade cull's window (FileDeferred, previs suspended), after the main view registered group 0:
		// the cascades and the lamps meet it, the main view never does. Any other node (ForEachChild's path) is filed as
		// before.
		constexpr std::uint64_t kOnVisibleID = 531348;         // BSFadeNode::OnVisible(NiCamera*, bool)
		constexpr std::uint64_t kUpdateGeomArrayID = 1054059;  // BSFadeNode::AtomicUpdateGeomArray(node, 1)
		constexpr std::size_t   kGeomArrayOffset = 0x168;      // geomArray records, 32 bytes each: bound +0, geometry +0x10, flags +0x18
		constexpr std::size_t   kGeomArrayCountOffset = 0x178;
		constexpr std::size_t   kGeomRecordSize = 0x20;
		constexpr std::size_t   kFadeAmountOffset = 0x118;
		constexpr std::size_t   kWorldRadiusOffset = 0xBC;
		using OnVisibleFn = void (*)(void*, std::uintptr_t, bool);
		using UpdateGeomArrayFn = std::uint32_t (*)(void*, std::uint32_t);

		struct HandoverFns
		{
			OnVisibleFn       onVisible;
			UpdateGeomArrayFn updateGeomArray;
		};

		const HandoverFns& HandoverFnsOf() noexcept
		{
			static const HandoverFns fns{
				reinterpret_cast<OnVisibleFn>(CBRO::Engine::OG(kOnVisibleID).address()),
				reinterpret_cast<UpdateGeomArrayFn>(CBRO::Engine::OG(kUpdateGeomArrayID).address())
			};
			return fns;
		}

		// A fade node as the main view's finish would take it: OnVisible, then (a_hand) its pieces to the tracer. Returns the
		// pieces handed, or -1 when the object is no fade node (nothing done). a_fadeOnly (v2.0, an entry no view files,
		// handed through the link): the fade update alone (FadeAdvance: its fade, LOD level and counter as OnVisible sets
		// them, without OnVisible's lighting queue, which only a drawn node needs). a_kind, a_flags: the link's record.
		int HandFadeNode(RE::NiAVObject* a_object, bool a_hand, bool a_fadeOnly, std::uint32_t a_kind, std::uint32_t a_flags) noexcept
		{
			const auto& fns = HandoverFnsOf();
			const auto& fade = FadeGlobalsOf();
			__try {
				const auto node = reinterpret_cast<std::uintptr_t>(a_object);
				if (!((*reinterpret_cast<const std::uint64_t*>(node + 0x108) >> 14) & 1) || !a_object->IsFadeNode()) {
					return -1;
				}
				const auto camera = *fade.camera;
				if (!camera) {
					return -1;
				}
				if (a_fadeOnly) {
					FadeAdvance(node);
				} else {
					fns.onVisible(a_object, camera, true);  // (as 558046 calls it in the main groups)
				}
				if (!a_hand) {
					return 0;
				}
				// (558046's gate for AddP1FadeNode: no piece is filed past it)
				const auto flags = *reinterpret_cast<const std::uint64_t*>(node + 0x108);
				const bool fading = ((flags >> 15) & 1) || !*fade.fading || *reinterpret_cast<const float*>(node + kFadeCurrentOffset) > 0.0f;
				const bool sized = ((flags >> 11) & 1) || *reinterpret_cast<const float*>(node + kWorldRadiusOffset) != 0.0f;
				if (!fading || !sized || (flags & 1) || !(*reinterpret_cast<const float*>(node + kFadeAmountOffset) > 0.0f)) {
					return 0;
				}
				fns.updateGeomArray(a_object, 1);
				const auto count = *reinterpret_cast<const std::uint32_t*>(node + kGeomArrayCountOffset);
				const auto records = *reinterpret_cast<const std::uintptr_t*>(node + kGeomArrayOffset);
				int        handed = 0;
				for (std::uint32_t i = 0; records && i < count; ++i) {
					const auto record = records + i * kGeomRecordSize;
					if (*reinterpret_cast<const std::uint32_t*>(record + 0x18) & 1) {
						continue;  // (AddP1FadeNode skips it)
					}
					if (const auto geometry = *reinterpret_cast<RE::NiAVObject* const*>(record + 0x10)) {
						TraceFeed::Take(geometry, a_object, a_kind, a_flags);
						++handed;
					}
				}
				return handed;
			} __except (EXCEPTION_EXECUTE_HANDLER) {
				return -1;
			}
		}

		// A top-level entry filed for the trace alone, on its turn, handed over instead (true; false: it isn't a geometry or a
		// fade node, or the feed takes nothing, and it is filed as before). a_flags: the link's record flags.
		bool HandEntry(RE::NiAVObject* a_object, std::uint32_t a_flags) noexcept
		{
			if (!g_tunables.traceDirect || !Hooks::CullGroups::TracerWanted() || GetCurrentThreadId() != g_mainThreadId) {
				return false;
			}
			const auto kind = t_moved ? CBRO_TRACE_MOVED : CBRO_TRACE_TAKE;
			if (a_object->IsGeometry()) {
				TraceFeed::Take(a_object, a_object, kind, a_flags);
				TraceFeed::Handed(a_object);
				Bump(kTraceHanded);
				Bump(kTraceHandedPieces);
				return true;
			}
			const auto handed = HandFadeNode(a_object, true, TraceFeed::Linked(), kind, a_flags);
			if (handed < 0) {
				return false;
			}
			TraceFeed::Handed(a_object);
			Bump(kTraceHanded);
			Bump(kTraceHandedPieces, static_cast<std::uint64_t>(handed));
			return true;
		}

		// A piece (Block::Add) filed for the trace alone, handed through the link instead (v2.0): a geometry, on the main
		// thread. True when handed: the caller rejects it (hidden) or lets the frustum test leave it out (out of view), with
		// no force mark and no main-view drop. Its anchor is the fade node filing it, else its parent.
		bool HandChild(const Hooks::CullGroups::BlockAdd& a_add, std::uint32_t a_flags) noexcept
		{
			if (!TraceFeed::Linked() || !Hooks::CullGroups::TracerWanted() || GetCurrentThreadId() != g_mainThreadId || !a_add.object->IsGeometry()) {
				return false;
			}
			const auto filing = Hooks::CullGroups::FadeNodeFiling();
			TraceFeed::Take(a_add.object, filing ? filing : a_add.object->parent, t_moved ? CBRO_TRACE_MOVED : CBRO_TRACE_TAKE, a_flags);
			TraceFeed::Handed(a_add.object);
			Bump(kTraceChildHanded);
			return true;
		}

		struct DeferredEntry
		{
			RE::NiAVObject* object;
			void*           group;
		};
		std::vector<DeferredEntry> g_deferred;  // (main thread: held back in the cull, filed in the cascade window)

		// Whether a sun-only drop's pieces go to the tracer this frame (they went every frame, as the refusals of their
		// registrations had): to a tracer that keeps what it is handed (v2.0) only the first time, when it moved (the mover
		// rule) or on its resend turn.
		bool SunOnlyHandDue(const FrameContext& a_context, const RE::NiAVObject* a_object, const RE::NiBound& a_bound) noexcept
		{
			if (!TraceFeed::Retains()) {
				return true;
			}
			if (t_moved) {
				return TraceFeed::MoverDue(a_object, a_bound, a_context.eye, a_context.clock);
			}
			if (!TraceFeed::Known(a_object) || TraceFeed::Due(a_object, a_context.clock)) {
				return true;
			}
			Bump(kSunOnlyKnown);
			return false;
		}

		// A group-0 entry the main view leaves out and the sun's cascades need (a confirmed-hidden caster): held back from the
		// main view, filed in the cascade cull's window; OnVisible now, as the main view's finish would have run it, and its
		// pieces to the tracer, as the refusals of their registrations handed them (SunOnlyHandDue). Only with the sun up
		// (the cascade cull opens the window every frame then) and for a geometry or a fade node.
		bool DeferSunOnly(const FrameContext& a_context, RE::NiAVObject* a_object, const RE::NiBound& a_bound) noexcept
		{
			if (!g_tunables.deferSunOnly || !t_decisionGroup || a_context.sun.state != FrameContext::Sun::State::kOn ||
				!Hooks::PrevisFeed::CascadeWindowReady() || GetCurrentThreadId() != g_mainThreadId) {
				return false;
			}
			const bool hand = Hooks::CullGroups::TracerWanted() && SunOnlyHandDue(a_context, a_object, a_bound);
			const auto kind = t_moved ? CBRO_TRACE_MOVED : CBRO_TRACE_TAKE;
			if (a_object->IsGeometry()) {
				if (hand) {
					TraceFeed::Take(a_object, a_object, kind, CBRO_TRACE_HIDDEN | CBRO_TRACE_SUN_ONLY);
					TraceFeed::Handed(a_object);
					Bump(kSunOnlyHandedPieces);
				}
			} else {
				const auto handed = HandFadeNode(a_object, hand, false, kind, CBRO_TRACE_HIDDEN | CBRO_TRACE_SUN_ONLY);
				if (handed < 0) {
					return false;
				}
				if (hand) {
					TraceFeed::Handed(a_object);
				}
				Bump(kSunOnlyHandedPieces, static_cast<std::uint64_t>(handed));
			}
			g_deferred.push_back({ a_object, const_cast<void*>(t_decisionGroup) });
			Bump(kSunOnlyDeferred);
			return true;
		}

		// The cascade cull's window opened (Hooks::PrevisFeed, previs suspended): the held-back entries into their group.
		void FileDeferred() noexcept
		{
			if (g_deferred.empty() || GetCurrentThreadId() != g_mainThreadId) {
				return;
			}
			for (const auto& entry : g_deferred) {
				Hooks::CullGroups::AddDirect(entry.group, entry.object, &entry.object->worldBound, 0);
			}
			Bump(kSunOnlyFiledLate, g_deferred.size());
			g_deferred.clear();
		}

		// ---- the Block::Add filter (children, and top-level entries not left out at Group::Add) -------------

		// The record of the fade node being filed (v1.99.1), looked up once per node: its geometry asks for it piece after
		// piece, and the top table's probe for each piece's own parent (a sub-node, never decided) was a miss each time.
		struct FilingTopMemo
		{
			const RE::NiAVObject* node{ nullptr };
			std::uint32_t         clock{ 0 };
			const TopRecord*      top{ nullptr };
		};
		thread_local FilingTopMemo t_filingTop;

		const TopRecord* FilingTop(const FrameContext& a_context, const RE::NiAVObject* a_node) noexcept
		{
			if (t_filingTop.node != a_node || t_filingTop.clock != a_context.clock) {
				t_filingTop = { a_node, a_context.clock, TopOf(a_context, a_node) };
			}
			const auto top = t_filingTop.top;
			return top && top->object == a_node && top->clock == a_context.clock ? top : nullptr;  // (the slot may have been taken since)
		}

		// Whether the entry takes its parent's decision (t_topRecord): the parent is the top entry decided last on this
		// thread, this frame, with this very bound, and its verdict is one the child's own tests would repeat.
		bool InheritTop(const FrameContext& a_context, const Hooks::CullGroups::BlockAdd& a_add) noexcept
		{
			if (!g_tunables.childInherit || Observing()) {
				return false;
			}
			// (a fade node's geometry comes flattened, FO4OG 5.3: a parent other than the node filing it is one of the node's
			// sub-nodes, which nothing decided; the node's record is asked for then, once per node)
			const auto       filing = Hooks::CullGroups::FadeNodeFiling();
			const auto       parent = a_add.object->parent;
			const bool       viaFade = filing && filing != parent;
			TopRecord        fromRecord{};
			const TopRecord* top = nullptr;
			if (const auto record = ParentRecordOf(viaFade ? filing : parent, a_context.clock)) {
				if (!(record->state & kParentJudged)) {
					return false;  // (forced, or not judged: as no TopRecord)
				}
				fromRecord.object = record->object;
				fromRecord.clock = a_context.clock;
				fromRecord.outcome = record->outcome;
				fromRecord.verdict = record->verdict;
				fromRecord.flags = record->flags;
				std::copy_n(record->bound, 4, fromRecord.bound);
				top = &fromRecord;
			} else {
				top = viaFade ? FilingTop(a_context, filing) : TopOf(a_context, parent);
			}
			if (!top) {
				return false;
			}
			const auto& bound = *a_add.bound;
			const bool  same = top->bound[0] == bound.center.x && top->bound[1] == bound.center.y && top->bound[2] == bound.center.z && top->bound[3] == bound.fRadius;
			bool        inherit = false;
			switch (static_cast<Outcome>(top->outcome)) {
			case Outcome::kOutside:
				// (a child's bound lies within its node's: out of the view too, and the engine's frustum test leaves it out of
				// the main view. Not in group 0, where the child's own sun reach decides: CheapChildOutside.)
				inherit = a_add.kind == Hooks::CullGroups::GroupKind::kMainOnly;
				break;
			case Outcome::kKept: {
				if (top->flags & kRecordLight) {
					break;
				}
				const auto verdict = static_cast<Verdict>(top->verdict);
				if (verdict == Verdict::kHidden) {
					break;  // kept by an exemption of the parent's
				}
				if (!same && !(top->bound[3] < kShapeMinRadius)) {
					break;  // a smaller bound under a big node: the child's own sphere may settle what the node's couldn't (2026-10-27 capture: 15% of such children hidden; 3% under small nodes)
				}
				if ((verdict == Verdict::kVisible || verdict == Verdict::kEdge) && g_tunables.meshShapes && bound.fRadius >= kShapeMinRadius) {
					break;  // the child's own shape may settle what the sphere couldn't
				}
				inherit = true;
				break;
			}
			default:
				break;
			}
			if (inherit && viaFade) {
				Bump(kChildInheritedFade);
			}
			return inherit;
		}

		// Whether a Block::Add child is part of the player's 3D: for a fade node's flattened geometry, the node's answer,
		// once per node (v1.99; its geometry can't be the player's unless it is).
		struct FilingPlayer
		{
			const RE::NiAVObject* node{ nullptr };
			std::uint32_t         clock{ 0 };
			bool                  player{ false };
		};
		thread_local FilingPlayer t_filingPlayer;

		bool ChildPlayerPart(const FrameContext& a_context, const Hooks::CullGroups::BlockAdd& a_add) noexcept
		{
			if (g_tunables.childInherit) {
				if (const auto node = Hooks::CullGroups::FadeNodeFiling()) {
					if (const auto record = ParentRecordOf(node, a_context.clock)) {
						return (record->state & kParentPlayer) != 0;
					}
					if (t_filingPlayer.node != node || t_filingPlayer.clock != a_context.clock) {
						t_filingPlayer = { node, a_context.clock, PlayerPart(a_context, node, node->worldBound) };
					}
					return t_filingPlayer.player;
				}
			}
			return PlayerPart(a_context, a_add.object, *a_add.bound);
		}

		// A child decided through the fade node filing it while its own parent is one of the node's sub-nodes (a count).
		void NoteFadeDecided(const Hooks::CullGroups::BlockAdd& a_add) noexcept
		{
			if (g_tunables.childInherit) {
				if (const auto node = Hooks::CullGroups::FadeNodeFiling(); node && node != a_add.object->parent) {
					Bump(kChildFadeDecided);
				}
			}
		}

		const RE::NiBound* FilterDecide(const FrameContext& a_context, const Hooks::CullGroups::BlockAdd& a_add)
		{
			if (ChildPlayerPart(a_context, a_add)) {
				Bump(kExemptPlayer);
				Hooks::CullGroups::ForceVisible();  // (filed whatever the frustum says, as previs files it)
				if (Recorder::On()) {
					Recorder::NoteFlags(RecorderFormat::kPlayer);
				}
				return nullptr;  // (nor a drop: its children and the main accumulator file it too)
			}
			auto&  streams = Streams(a_context);
			Record out{};
			switch (a_add.kind) {
			case Hooks::CullGroups::GroupKind::kMainOnly: {
				// (a child of an entry filed for the trace: main-only entries are main-view drops only then, TraceDrop, TraceOutside)
				if (ParentForced(a_add.object, a_context.clock, DropSet::Tag(a_context.clock)) && !IsActorRoot(a_add.object)) {
					NoteFadeDecided(a_add);
					ForcedChild(a_context, a_add.object, *a_add.bound);
					if (Recorder::On()) {
						Recorder::NoteFlags(RecorderFormat::kInherited);
					}
					return nullptr;
				}
				if (TraceFeed::On() && ParentDropped(a_add.object, a_context.clock, DropSet::Tag(a_context.clock)) && !IsActorRoot(a_add.object)) {
					NoteFadeDecided(a_add);
					Bump(kTraceInherited);
					if (!g_drops.Insert(a_add.object, DropSet::Tag(a_context.clock))) {
						Bump(kDropFull);
					}
					InheritHidden(a_add.object, a_context.clock, g_tunables.confirmFrames);
					RecordLeftOut(a_context, a_add.object, *a_add.bound, LeftOutPath::kInherited, nullptr);
					return nullptr;
				}
				if (InheritTop(a_context, a_add)) {
					Bump(kChildInherited);
					t_source = VerdictSource::kNone;
					if (Recorder::On()) {
						Recorder::NoteFlags(RecorderFormat::kInherited);
					}
					return nullptr;
				}
				if (CheapChildOutside(a_context, a_add)) {
					Bump(kChildCheapOutside);
					RecordLeftOut(a_context, a_add.object, *a_add.bound, LeftOutPath::kOutside, nullptr);
					return nullptr;  // (the engine's frustum test leaves it out of the main view)
				}
				JudgeOrLookup(a_context, streams.child, a_add, false, out);
				if (g_keptReasons.load(std::memory_order_relaxed)) {
					NoteKeptReason(a_context, a_add.object, out);  // (diagnostic)
				}
				if (out.outcome == Outcome::kHidden && out.streak >= g_tunables.confirmFrames && !Observing()) {
					if (TraceRefresh(a_context, a_add.object, *a_add.bound) && !HandChild(a_add, CBRO_TRACE_HIDDEN) && TraceDropFiled(a_context, a_add.object)) {
						RecordLeftOut(a_context, a_add.object, *a_add.bound, LeftOutPath::kDropped, &out);
						return nullptr;
					}
					RecordLeftOut(a_context, a_add.object, *a_add.bound, LeftOutPath::kRejected, &out);
					return &a_context.reject;
				}
				if (out.outcome == Outcome::kOutside) {
					// (handed through the link: filed as any out-of-view piece, the frustum test leaves it out)
					if (!Observing() && TraceRefresh(a_context, a_add.object, *a_add.bound) && !HandChild(a_add, CBRO_TRACE_OUTSIDE) &&
						TraceOutsideFiled(a_context, a_add.object)) {
						Hooks::CullGroups::ForceVisible();
						RecordLeftOut(a_context, a_add.object, *a_add.bound, LeftOutPath::kDropped, &out);
						return nullptr;
					}
					RecordLeftOut(a_context, a_add.object, *a_add.bound, LeftOutPath::kOutside, &out);
				}
				return nullptr;
			}
			case Hooks::CullGroups::GroupKind::kSunShared: {
				// Group 0 (the shadow casters; the sun's cascades read it too): rejected in every view only when neither
				// the main camera nor the sun's shadow needs the object; dropped from the main view alone when only the
				// shadow is needed. A child of a main-view drop inherits it (its parent's shadow was needed, so it stays
				// for the cascades).
				Bump(kShared);
				if ((a_add.object->GetFlags() >> 11) & 1) {
					return nullptr;  // the engine's always-draw mark: Block::Add forces such entries visible
				}
				const auto tag = DropSet::Tag(a_context.clock);
				if (ParentForced(a_add.object, a_context.clock, tag) && !IsActorRoot(a_add.object)) {
					NoteFadeDecided(a_add);
					ForcedChild(a_context, a_add.object, *a_add.bound);
					if (Recorder::On()) {
						Recorder::NoteFlags(RecorderFormat::kInherited);
					}
					return nullptr;
				}
				if (ParentDropped(a_add.object, a_context.clock, tag) && !IsActorRoot(a_add.object)) {
					NoteFadeDecided(a_add);
					Bump(kDropInherited);
					if (!g_drops.Insert(a_add.object, tag)) {
						Bump(kDropFull);
					}
					InheritHidden(a_add.object, a_context.clock, g_tunables.confirmFrames);
					RecordLeftOut(a_context, a_add.object, *a_add.bound, LeftOutPath::kInherited, nullptr);
					return nullptr;
				}
				if (InheritTop(a_context, a_add)) {
					Bump(kChildInherited);
					t_source = VerdictSource::kNone;
					if (Recorder::On()) {
						Recorder::NoteFlags(RecorderFormat::kInherited);
					}
					return nullptr;
				}
				if (CheapChildOutside(a_context, a_add)) {
					Bump(kChildCheapRejected);
					RecordLeftOut(a_context, a_add.object, *a_add.bound, LeftOutPath::kCasterRejected, nullptr);
					return &a_context.reject;
				}
				JudgeOrLookup(a_context, streams.child, a_add, true, out);
				if (g_keptReasons.load(std::memory_order_relaxed)) {
					NoteKeptReason(a_context, a_add.object, out);  // (diagnostic)
				}
				if (Observing()) {
					return nullptr;
				}
				const bool hiddenConfirmed = out.outcome == Outcome::kHidden && out.streak >= g_tunables.confirmFrames;
				if ((hiddenConfirmed || out.outcome == Outcome::kOutside) && !(out.flags & kRecordLight) && GroupZeroUnneeded(a_context, out, a_add.object, *a_add.bound)) {
					// (handed through the link: rejected in every view as without the feed)
					if (TraceRefresh(a_context, a_add.object, *a_add.bound) && !HandChild(a_add, hiddenConfirmed ? CBRO_TRACE_HIDDEN : CBRO_TRACE_OUTSIDE) &&
						TraceKeepFiled(a_context, a_add.object, *a_add.bound, hiddenConfirmed, out)) {
						if (!hiddenConfirmed) {
							Hooks::CullGroups::ForceVisible();
						}
						return nullptr;
					}
					Bump(kCasterRejected);
					SampleLeftOutType(a_context, a_add.object);
					RecordLeftOut(a_context, a_add.object, *a_add.bound, LeftOutPath::kCasterRejected, &out);
					return &a_context.reject;
				}
				if ((hiddenConfirmed || out.outcome == Outcome::kOutside) && !(out.flags & kRecordLight)) {
					NoteFarCascade(a_context, a_add.object, out);
				}
				if (hiddenConfirmed) {
					if (!g_drops.Insert(a_add.object, tag)) {
						Bump(kDropFull);
					} else {
						RecordLeftOut(a_context, a_add.object, *a_add.bound, LeftOutPath::kDropped, &out);
					}
				} else if (out.outcome == Outcome::kOutside) {
					// (with the sun up a shadow it may cast keeps it: the cascades file it within their range, 6.1; handed through the
					// link, it is filed as without the feed)
					if (a_context.sun.state != FrameContext::Sun::State::kOn && TraceRefresh(a_context, a_add.object, *a_add.bound) && !HandChild(a_add, CBRO_TRACE_OUTSIDE) &&
						TraceOutsideFiled(a_context, a_add.object)) {
						Hooks::CullGroups::ForceVisible();
						RecordLeftOut(a_context, a_add.object, *a_add.bound, LeftOutPath::kDropped, &out);
					} else {
						RecordLeftOut(a_context, a_add.object, *a_add.bound, LeftOutPath::kOutside, &out);
					}
				}
				return nullptr;
			}
			default: {
				// A group with other possible readers: only ever dropped from the main view.
				Bump(kShared);
				if ((a_add.object->GetFlags() >> 11) & 1) {
					return nullptr;
				}
				const auto tag = DropSet::Tag(a_context.clock);
				if (ParentDropped(a_add.object, a_context.clock, tag) && !IsActorRoot(a_add.object)) {
					NoteFadeDecided(a_add);
					Bump(kDropInherited);
					if (!g_drops.Insert(a_add.object, tag)) {
						Bump(kDropFull);
					}
					InheritHidden(a_add.object, a_context.clock, g_tunables.confirmFrames);
					RecordLeftOut(a_context, a_add.object, *a_add.bound, LeftOutPath::kInherited, nullptr);
					return nullptr;
				}
				JudgeOrLookup(a_context, streams.child, a_add, false, out);
				if (g_keptReasons.load(std::memory_order_relaxed)) {
					NoteKeptReason(a_context, a_add.object, out);  // (diagnostic)
				}
				if (out.outcome == Outcome::kHidden && out.streak >= g_tunables.confirmFrames && !Observing()) {
					if (!g_drops.Insert(a_add.object, tag)) {
						Bump(kDropFull);
					} else {
						RecordLeftOut(a_context, a_add.object, *a_add.bound, LeftOutPath::kDropped, &out);
					}
				}
				return nullptr;
			}
			}
		}

		// A static collection (SCOL) placed as its own meshes (precombines off, as Runtime Combiner runs the game): a piece's
		// bound need not hold its geometry. 2026-10-09, Fallout London, Folon_SCOL_MCHBlockGaljero: the lower part of a wall
		// beside the camera was one piece whose sphere (r=346) lay 26 units outside the view's left edge while its geometry
		// was in view; CBRO and the engine's own frustum test both dropped it, and previs (which files its list without a
		// frustum test) drew it. With precombines on the same wall is part of a precombined chunk with a sound bound. So a
		// piece of a collection is judged with the collection root's bound (the nearest ancestor whose userData is a
		// reference to a SCOL base, within four levels), and the engine files it with that bound too (Hooks/CullGroups
		// WidenBound): every view that tests it then tests the whole collection. (A root bound that doesn't enclose the
		// piece's is grown to: the bound used always holds both.) Siblings come one after another, so the root found for
		// a parent is kept (per thread, per frame); the bound handed out lives until the thread's next call (Block::Add
		// and the verdict cache copy it at once).
		struct ScolMemo
		{
			const RE::NiAVObject* parent{ nullptr };
			std::uint32_t         clock{ 0 };
			const RE::NiBound*    root{ nullptr };
		};
		thread_local ScolMemo    t_scol;
		thread_local RE::NiBound t_scolBound;

		const RE::NiBound* FindScolRoot(const RE::NiAVObject* a_parent, const RE::NiAVObject*& a_root, int& a_depth) noexcept
		{
			__try {
				const RE::NiBound*    bound = nullptr;
				const RE::NiAVObject* node = a_parent;
				for (int depth = 0; node && depth < 4; ++depth, node = node->parent) {
					if (!node->userData) {
						continue;
					}
					const auto form = reinterpret_cast<const RE::TESForm*>(node->userData);
					if (form->GetFormType() != RE::ENUM_FORM_ID::kREFR) {
						break;
					}
					const auto base = reinterpret_cast<const RE::TESObjectREFR*>(form)->GetObjectReference();
					if (base && base->GetFormType() == RE::ENUM_FORM_ID::kSCOL) {
						const auto& root = node->worldBound;
						if (root.fRadius > 0.0f && root.fRadius < 1.0e5f && std::isfinite(root.center.x) && std::isfinite(root.center.y) && std::isfinite(root.center.z)) {
							bound = &root;
							a_root = node;
							a_depth = depth;
						}
					}
					break;
				}
				return bound;
			} __except (EXCEPTION_EXECUTE_HANDLER) {
				return nullptr;
			}
		}

		// A parent's collection root, or that it has none (nearly every one: FindScolRoot was 0.12 ms a frame on the main thread
		// in the 2026-10-26 profile, three dependent reads per distinct parent, most of them misses), is remembered per thread
		// by its address and userData (the reference it stands for, which never changes while the node lives) for up to
		// kScolCacheFrames. v1.92.3 cached the misses alone and walked a root found again each frame (~270 pieces a frame, the
		// form reads: 0.05 ms listed 2026-10-27); v1.98 keeps the root too, and a hit is checked by the parent chain alone
		// (the node at the depth it was found, still with its reference's userData): a node freed since can't be reached so.
		struct ScolCacheEntry
		{
			const RE::NiAVObject* parent{ nullptr };
			std::uintptr_t        userData{ 0 };
			std::uint32_t         clock{ 0 };
			const RE::NiAVObject* root{ nullptr };  // null: no root (a miss)
			std::uintptr_t        rootUserData{ 0 };
			std::uint8_t          depth{ 0 };
		};
		constexpr std::size_t   kScolCacheSize = 8192;  // direct-mapped by address (v1.98.1: twice v1.98's; ~2,500 distinct parents a frame)
		constexpr std::uint32_t kScolCacheFrames = 64;

		const RE::NiBound* ScolRootStillThere(const RE::NiAVObject* a_parent, const ScolCacheEntry& a_slot) noexcept
		{
			__try {
				const RE::NiAVObject* node = a_parent;
				for (int depth = 0; node && depth < a_slot.depth; ++depth) {
					node = node->parent;
				}
				if (node != a_slot.root || node->userData != a_slot.rootUserData) {
					return nullptr;
				}
				const auto& root = node->worldBound;
				if (root.fRadius > 0.0f && root.fRadius < 1.0e5f && std::isfinite(root.center.x) && std::isfinite(root.center.y) && std::isfinite(root.center.z)) {
					return &root;
				}
				return nullptr;
			} __except (EXCEPTION_EXECUTE_HANDLER) {
				return nullptr;
			}
		}

		const RE::NiBound* ScolRootCached(const RE::NiAVObject* a_parent, std::uint32_t a_clock) noexcept
		{
			thread_local std::array<ScolCacheEntry, kScolCacheSize>* cache = nullptr;
			if (!cache) [[unlikely]] {
				cache = new std::array<ScolCacheEntry, kScolCacheSize>();
			}
			const auto address = reinterpret_cast<std::uintptr_t>(a_parent);
			auto&      slot = (*cache)[((address >> 5) ^ (address >> 18)) & (kScolCacheSize - 1)];
			const auto userData = a_parent->userData;
			if (slot.parent == a_parent && slot.userData == userData && a_clock - slot.clock < kScolCacheFrames) {
				if (!slot.root) {
					return nullptr;
				}
				Bump(kScolSlotHits);
				if (const auto bound = ScolRootStillThere(a_parent, slot)) {
					Bump(kScolRootHits);
					return bound;
				}
				Bump(kScolChainFails);
			}
			const RE::NiAVObject* root = nullptr;
			int                   depth = 0;
			const auto            bound = FindScolRoot(a_parent, root, depth);
			slot = { a_parent, userData, a_clock, bound ? root : nullptr, bound && root ? root->userData : 0, static_cast<std::uint8_t>(depth) };
			return bound;
		}

		// A collection root's bound when usable (as FindScolRoot takes it).
		const RE::NiBound* UsableScolBound(const RE::NiAVObject* a_root) noexcept
		{
			const auto& root = a_root->worldBound;
			return root.fRadius > 0.0f && root.fRadius < 1.0e5f && std::isfinite(root.center.x) && std::isfinite(root.center.y) && std::isfinite(root.center.z) ? &root : nullptr;
		}

		// Whether a node is a static collection's root: its userData a reference whose base is a SCOL (FindScolRoot's test at
		// the node itself; the walk asks once per epoch).
		bool IsScolReference(const RE::NiAVObject* a_object) noexcept
		{
			__try {
				const auto form = reinterpret_cast<const RE::TESForm*>(a_object->userData);
				if (!form || form->GetFormType() != RE::ENUM_FORM_ID::kREFR) {
					return false;
				}
				const auto base = reinterpret_cast<const RE::TESObjectREFR*>(form)->GetObjectReference();
				return base && base->GetFormType() == RE::ENUM_FORM_ID::kSCOL;
			} __except (EXCEPTION_EXECUTE_HANDLER) {
				return false;
			}
		}

		const RE::NiBound* ScolRootBound(const RE::NiAVObject* a_object, const RE::NiBound& a_bound, std::uint32_t a_clock) noexcept
		{
			const auto parent = DecisionParent(a_object);  // (v1.99: a fade node's geometry starts at the node, as its first ancestor with userData)
			if (!parent) {
				return nullptr;
			}
			if (t_scol.parent != parent || t_scol.clock != a_clock) {
				const auto record = ParentRecordOf(parent, a_clock);
				if (record && (record->state & kParentScolKnown)) {
					t_scol = { parent, a_clock, (record->state & kParentScolRoot) ? UsableScolBound(parent) : nullptr };  // (the walk's epoch classification)
				} else {
					t_scol = { parent, a_clock, ScolRootCached(parent, a_clock) };
				}
			}
			const auto root = t_scol.root;
			if (!root || !(a_bound.fRadius >= 0.0f)) {
				return nullptr;
			}
			const float dx = a_bound.center.x - root->center.x, dy = a_bound.center.y - root->center.y, dz = a_bound.center.z - root->center.z;
			const float d = std::sqrt(dx * dx + dy * dy + dz * dz);
			if (d + a_bound.fRadius <= root->fRadius) {
				return root;
			}
			if (d + root->fRadius <= a_bound.fRadius) {
				return nullptr;  // (the piece's own bound holds the root's: nothing to widen)
			}
			// The smallest sphere holding both.
			const float radius = (d + root->fRadius + a_bound.fRadius) * 0.5f;
			const float t = d > 0.0f ? (radius - root->fRadius) / d : 0.0f;
			t_scolBound.center = RE::NiPoint3{ root->center.x + dx * t, root->center.y + dy * t, root->center.z + dz * t };
			t_scolBound.fRadius = radius + 0.5f;
			return &t_scolBound;
		}

		// The recorder's note of a Block::Add decision (a child, or an unjudged filing: whatever the top-level decision didn't cover).
		void NoteChildDecision(const Hooks::CullGroups::BlockAdd& a_add, bool a_rejected) noexcept
		{
			std::uint16_t flags = a_rejected ? RecorderFormat::kLeftOut : 0;
			if (Hooks::CullGroups::ForceVisiblePending()) {
				flags |= RecorderFormat::kForced;
			}
			if ((a_add.object->GetFlags() >> 11) & 1) {
				flags |= RecorderFormat::kAlwaysDraw;
			}
			if (a_add.kind == Hooks::CullGroups::GroupKind::kMainOnly) {
				flags |= RecorderFormat::kGroup1;
			}
			if (Observing()) {
				flags |= RecorderFormat::kObserveOnly;
			}
			Recorder::NoteEntry(Recorder::Origin::kChild, a_add.object, *a_add.bound, static_cast<std::uint8_t>(a_add.kind), flags, 0);
		}

		thread_local std::uint32_t t_childSurvey{ 0 };

		const RE::NiBound* Filter(const Hooks::CullGroups::BlockAdd& a_in)
		{
			const auto context = CurrentContext();
			if (!context) {
				if (g_active.load(std::memory_order_relaxed)) {
					Bump(kNoContext);
				}
				return nullptr;
			}
			NoteOffered(*context, a_in.object, 1);  // (diagnostic, one frame per still location)
			// Decided at Group::Add already (same thread): not again.
			if (t_top.object == a_in.object && t_top.clock == context->clock) {
				t_top.object = nullptr;
				if (t_top.forceVisible) {
					Hooks::CullGroups::ForceVisible();
				}
				if (!t_top.result) {
					RecordKeptRegion(*context, a_in);  // (diagnostic, one frame per still location)
				}
				return t_top.result;
			}
			// (survey, v1.94.1: whose children the engine pushes, and what is filed unjudged; 1 add in 32 per thread since
			// v1.96: the parent's vtable and type were a cache miss per child, ~4,700 a frame on the main thread)
			if ((++t_childSurvey & 31) == 0) {
				static const auto niNodeVtable = RE::VTABLE::NiNode[0].address();
				const auto        parent = const_cast<RE::NiNode*>(a_in.object->parent);
				if (!parent) {
					Bump(kChildOther, 32);
				} else if (parent->IsFadeNode()) {
					Bump(kChildOfFade, 32);
				} else if (Util::TryReadVtable(parent) == niNodeVtable) {
					Bump(kChildOfNode, 32);
				} else {
					Bump(kChildOther, 32);
				}
			}
			// (the drop sets' slots for this object, in case the decision below inserts or looks it up: the judge in between
			// hides the table's miss; v1.92.3)
			g_drops.Prefetch(a_in.object);
			if (TraceFeed::On()) {
				g_traceForced.Prefetch(a_in.object);
			}
			if (RefreshTurn(a_in.object, context->clock)) {
				PrefetchEntry(a_in.object);  // (the streak backups write its history entry on its turn; v1.98)
			}
			auto add = a_in;
			if (g_tunables.scolRootBounds && !Observing()) {
				if (const auto root = ScolRootBound(a_in.object, *a_in.bound, context->clock); root && root != a_in.bound) {
					add.bound = root;
					Hooks::CullGroups::WidenBound(root);
					Bump(kScolWidened);
				}
			}
			const auto& a_add = add;
			const bool  recording = Recorder::On();
			if (recording) {
				Recorder::BeginDecision();
			}
			if (!Timed(*context)) {
				const auto result = FilterDecide(*context, a_add);
				if (recording) {
					NoteChildDecision(a_add, result != nullptr);
				}
				if (!result) {
					RecordKeptRegion(*context, a_add);
				}
				return result;
			}
			const auto start = __rdtsc();
			const auto result = FilterDecide(*context, a_add);
			Bump(kCycles, (__rdtsc() - start) * kTimingStride);
			if (recording) {
				NoteChildDecision(a_add, result != nullptr);
			}
			if (!result) {
				RecordKeptRegion(*context, a_add);
			}
			return result;
		}

		// The main accumulator registering an object while DrawWorld culls: leave out what was dropped.
		// (a_object is never dereferenced: only its address is looked up.)
		bool MainView(const RE::NiAVObject* a_object)
		{
			const auto context = CurrentContext();
			if (!context) {
				return false;
			}
			Bump(kRegistered);
			if (!g_drops.Contains(a_object, DropSet::Tag(context->clock))) {
				if (LeftOutFrame(context->clock)) {
					std::scoped_lock lock(g_keptLock);
					g_registered.insert(a_object);  // (the scene scan's "registered with the main view")
				}
				return false;
			}
			Bump(kRegistrationsDropped);
			if (Recorder::On()) {
				Recorder::NoteRegistration(a_object, true);
			}
			return true;
		}

		// Group::Add of a main-pass object into a main-only group or group 0 with no group markers pending: an
		// object nothing needs is left out entirely, so the engine never files, tests or walks it (with previs
		// off it would otherwise process every object in the loaded cells). In a main-only group that is an
		// object out of view or confirmed hidden; in group 0 its sun shadow must also be unneeded. Never for an
		// object the engine marks always-draw, nor a sky mesh (previs draws those whatever their bound), and never in
		// observe-only mode.
		// ---- survey counters (v1.92.4): what the engine does with the top-level fade nodes CBRO lets through ----------
		// A. Distance fade (FO4-ENGINE-NOTES 7.8, 7.9). ComputeFadeAmount (id 1040250) gives a node past its far range a fade
		// of 0 or below: the finish loop still runs OnVisible on it, but AddP1FadeNode then files none of its geometry
		// (currentFade 0, flag bit 15 clear). Such a node costs the walk a Group::Add, CBRO a verdict and the engine a
		// frustum test and OnVisible, for nothing drawn. Counted as ComputeFadeAmount computes it: s = the distance from
		// the world-bound centre to the camera x lodAdjust (NiCamera +0x194) / mult(type +0x11C; the table id 804710);
		// fade = 1 - (s - near) / (far - near), near / far (+0x198 / +0x19C) scaled by fDistanceMultiplier (id 515159);
		// 1 when fading (id 1220201) is off or mult <= 0; type 6 only fades in.
		// C. Single-record geometry lists: AddP1FadeNode pushes each geomArray record (count u32 at +0x178) through
		// ChildPush; a node with one record pushes one child whose bound equals the node's, so that child could take the
		// node's verdict without a lookup. Counted: kept nodes with one record, and the records of all kept nodes (the
		// frame's ChildPush adds from them: a check of the +0x178 read against the ChildPush hook's count).
		// One node in kFadeStampFrames a frame (the fade stamp's turn); the log scales the counts back.

		void CountFadeTop(const FrameContext& a_context, RE::NiAVObject* a_object, const RE::NiBound& a_bound, bool a_skipped, bool a_forced) noexcept
		{
			const auto multTable = FadeGlobalsOf().multTable;
			const auto distanceMultiplier = FadeGlobalsOf().distanceMultiplier;
			const auto fading = FadeGlobalsOf().fading;
			const auto camera = FadeGlobalsOf().camera;
			__try {
				if (!a_object->IsFadeNode()) {
					return;
				}
				const auto node = reinterpret_cast<const std::byte*>(a_object);
				const auto flags = *reinterpret_cast<const std::uint64_t*>(node + 0x108);
				const auto type = *reinterpret_cast<const std::uint8_t*>(node + 0x11C);
				const auto records = *reinterpret_cast<const std::uint32_t*>(node + 0x178);
				const auto nearRange = *reinterpret_cast<const float*>(node + 0x198);
				const auto farRange = *reinterpret_cast<const float*>(node + 0x19C);
				const auto currentFade = *reinterpret_cast<const float*>(node + 0x1A0);
				// the distance fade as ComputeFadeAmount (id 1040250) computes it: 1 unless the node fades by distance and is past its
				// near range; and the frame counter's age (OnVisible of this frame has not run yet: the walk precedes the finish loop)
				const float mult = multTable[type];
				const bool  byDistance = type != 6 && mult > 0.0f && *fading && *camera;
				float       fade = 1.0f;
				if (byDistance) {
					const float lodAdjust = *reinterpret_cast<const float*>(*camera + kCameraLODAdjustOffset);
					const float dx = a_bound.center.x - a_context.eye[0];
					const float dy = a_bound.center.y - a_context.eye[1];
					const float dz = a_bound.center.z - a_context.eye[2];
					const float s = std::sqrt(dx * dx + dy * dy + dz * dz) * lodAdjust / mult;
					const float nearScaled = *distanceMultiplier * nearRange;
					const float farScaled = *distanceMultiplier * farRange;
					if (nearScaled < s) {
						fade = 1.0f - (s - nearScaled) / (farScaled - nearScaled);
					}
				}
				const auto age = static_cast<std::int64_t>(*FrameNumber()) - static_cast<std::int64_t>(*reinterpret_cast<const std::int32_t*>(node + kFadeFrameCounterOffset));
				Bump(kFadeTopOffered);
				if (!a_skipped) {
					Bump(kFadeTopKept);
					Bump(kFadeTopKeptEntries, records);
					if (records == 1) {
						Bump(kFadeTopKeptOne);
					}
					// (v1.94 survey: BSFadeNode::OnVisible re-queues the node's lighting, one pass over the scene's light list,
					// when bit 51 is clear and bit 33 is set, or the node is mid-fade, or the scene node's byte +0x228 is set)
					if ((flags >> 33) & 1) {
						Bump(kFadeTopLightingBit33);
						if (a_forced) {
							Bump(kFadeTopLightingBit33Forced);
						}
					}
					if (currentFade > 0.0f && currentFade < 1.0f) {
						Bump(kFadeTopMidFade);
						if (a_forced) {
							Bump(kFadeTopMidFadeForced);
						}
						if (!(flags & kFadeCompleteBit)) {
							Bump(kFadeTopMidFadeIn);
							if (!(*reinterpret_cast<const float*>(node + kFadeRadiusOffset) < 600.0f)) {
								Bump(kFadeTopMidFadeInBig);
							}
							Bump(age == 1 ? kFadeTopMidFadeInContinuous : kFadeTopMidFadeInResumed);
							if (type == 3) {
								Bump(kFadeTopMidFadeInActor);
							}
						}
					}
					// A fade-in starts at this frame's OnVisible for a node at fade 0 with bit 37 clear (id 928617): the unseen rule
					// (frameCounter over 20 frames old) snaps it to 1 when near, else it stays 0 until near; stamped within 20 frames,
					// it fades in from 0 on screen when near. Which stamp made it "seen" is the question: OnVisible itself last frame
					// (a far node that just came within fFadeInThreshold: the engine's own distance fade-in, previs has the same), or
					// a stamp 2-20 frames ago while out of the main view (CBRO's keepFadeState, or a culling process's OnVisible).
					if (currentFade == 0.0f && !(flags & kFadeCompleteBit)) {
						const bool nearEnough = fade > *FadeGlobalsOf().fadeInThreshold;  // ("near" is a Windows macro)
						if (age > 20) {
							Bump(kFadeStartUnseen);
							if (nearEnough) {
								Bump(kFadeStartUnseenNear);
							}
						} else {
							Bump(age == 1 ? kFadeStartSeenLast : kFadeStartSeenRecent);
							if (nearEnough) {
								Bump(kFadeStartSeenNear);
								if (g_fadeStartSamples.fetch_add(1, std::memory_order_relaxed) < kFadeStartSamples) {
									char rtti[48]{};
									char name[48]{};
									Util::TryGetRTTIName(a_object, rtti, sizeof(rtti));
									Util::TryGetObjectName(a_object, name, sizeof(name));
									logger::info(
										"fade start sample (fades in from 0 on screen from this frame): {} '{}' type {} fade radius {:.0f} bound r {:.0f} at {:.0f} units, frame counter {} frames old, distance fade {:.2f}, records {}, flags {:#x}{}",
										rtti[0] ? rtti : "?", name, type, *reinterpret_cast<const float*>(node + kFadeRadiusOffset), a_bound.fRadius,
										std::sqrt((a_bound.center.x - a_context.eye[0]) * (a_bound.center.x - a_context.eye[0]) + (a_bound.center.y - a_context.eye[1]) * (a_bound.center.y - a_context.eye[1]) + (a_bound.center.z - a_context.eye[2]) * (a_bound.center.z - a_context.eye[2])),
										age, fade, records, flags, a_forced ? " (filed force-visible)" : "");
								}
							}
						}
					}
					if ((flags >> 51) & 1) {
						Bump(kFadeTopNoLighting);
					}
				}
				if (!byDistance) {
					return;  // (never fades out by distance)
				}
				if (!(fade <= 0.0f)) {
					return;  // the distance fade is above 0: drawn, or fading out
				}
				if ((flags >> 15) & 1) {
					Bump(kFadeTopPastFarBit15);
				} else if (currentFade > 0.0f) {
					Bump(kFadeTopFadingOut);
				} else {
					Bump(kFadeTopFaded);
					if (!a_skipped) {
						Bump(kFadeTopFadedKept);
						Bump(kFadeTopFadedKeptEntries, records);
					}
				}
			} __except (EXCEPTION_EXECUTE_HANDLER) {
			}
		}

		// What the caller already knows of the entry (v1.99: the walk classifies its entries itself, and Group::Add's hook
		// passes nothing): its player test and its sky test, so the decision doesn't read them again.
		enum KnownEntry : std::uint8_t
		{
			kKnownNone = 0,
			kKnownClassified = 1 << 0,
			kKnownPlayer = 1 << 1,
			kKnownSky = 1 << 2,
			kKnownScolChecked = 1 << 3,  // (v1.99.2) the entry's collection-root answer is kKnownScolRoot
			kKnownScolRoot = 1 << 4,
		};

		bool SkipTopLevelImpl(RE::NiAVObject* a_object, const RE::NiBound* a_bound, Hooks::CullGroups::GroupKind a_kind, std::uint8_t a_known);

		bool SkipTopLevel(RE::NiAVObject* a_object, const RE::NiBound* a_bound, Hooks::CullGroups::GroupKind a_kind)
		{
			t_decisionGroup = Hooks::CullGroups::AddingGroup();
			const bool skipped = SkipTopLevelImpl(a_object, a_bound, a_kind, kKnownNone);
			t_decisionGroup = nullptr;
			return skipped;
		}

		bool SkipTopLevelImpl(RE::NiAVObject* a_object, const RE::NiBound* a_bound, Hooks::CullGroups::GroupKind a_kind, std::uint8_t a_known)
		{
			const auto context = CurrentContext();
			if (!context) {
				return false;
			}
			const bool timed = Timed(*context);
			const auto start = timed ? __rdtsc() : 0;
			NoteOffered(*context, a_object, 2);  // (diagnostic, one frame per still location)
			g_drops.Prefetch(a_object);  // (as in Filter)
			if (TraceFeed::On()) {
				g_traceForced.Prefetch(a_object);
			}

			bool       skip = false;
			bool       force = false;  // filed force-visible for the trace (TraceOutside, TraceKeep)
			const bool known = (a_known & kKnownClassified) != 0;
			const bool player = known ? (a_known & kKnownPlayer) != 0 : PlayerPart(*context, a_object, *a_bound);
			const bool alwaysDraw = (a_object->GetFlags() >> 11) & 1;
			const bool sky = !player && !alwaysDraw && (known ? (a_known & kKnownSky) != 0 : Hooks::CullGroups::IsSky(a_object));
			const bool recording = Recorder::On();
			if (recording) {
				Recorder::BeginDecision();
			}
			Record out{};  // (kept for a same-bound child: t_topRecord)
			bool   judged = false;
			if (player) {
				Bump(kExemptPlayer);
			} else if (!alwaysDraw && !sky) {
				auto&      streams = Streams(*context);
				const bool group0 = a_kind == Hooks::CullGroups::GroupKind::kSunShared;
				const Hooks::CullGroups::BlockAdd add{ nullptr, a_object, a_bound, -1, 0, true, a_kind, 0 };
				judged = true;
				if (group0) {
					Bump(kShared);
				}
				JudgeOrLookup(*context, streams.top, add, group0, out);
				const bool hiddenConfirmed = out.outcome == Outcome::kHidden && out.streak >= g_tunables.confirmFrames;
				const bool light = (out.flags & kRecordLight) != 0;
				if (!Observing()) {
					if (group0) {
						if ((hiddenConfirmed || out.outcome == Outcome::kOutside) && !light && GroupZeroUnneeded(*context, out, a_object, *a_bound)) {
							const bool turn = TraceRefresh(*context, a_object, *a_bound);
							if (turn && HandEntry(a_object, hiddenConfirmed ? CBRO_TRACE_HIDDEN : CBRO_TRACE_OUTSIDE)) {
								RecordLeftOut(*context, a_object, *a_bound, LeftOutPath::kDropped, &out);  // (handed to the tracer: no view files it, v1.99.3)
								skip = true;
							} else if (turn && TraceKeepFiled(*context, a_object, *a_bound, hiddenConfirmed, out)) {
								force = !hiddenConfirmed;
							} else {
								Bump(kCasterSkipped);
								SampleLeftOutType(*context, a_object);
								RecordLeftOut(*context, a_object, *a_bound, LeftOutPath::kSkipCaster, &out);
								skip = true;
							}
						} else if (hiddenConfirmed) {
							NoteFarCascade(*context, a_object, out);
							if (DeferSunOnly(*context, a_object, *a_bound)) {
								RecordLeftOut(*context, a_object, *a_bound, LeftOutPath::kDropped, &out);  // (filed for the cascades later, v1.99.3)
								skip = true;
							} else if (!g_drops.Insert(a_object, DropSet::Tag(context->clock))) {
								Bump(kDropFull);
							} else {
								RecordLeftOut(*context, a_object, *a_bound, LeftOutPath::kDropped, &out);
							}
						} else if (out.outcome == Outcome::kOutside) {
							if (!light) {
								NoteFarCascade(*context, a_object, out);
							}
							// (with the sun up a shadow it may cast keeps it: the cascades file it within their range, 6.1; with the sun
							// down a lamp may need it: filed as without the feed, its pieces handed over directly, v1.99.3)
							if (context->sun.state != FrameContext::Sun::State::kOn && TraceRefresh(*context, a_object, *a_bound)) {
								force = !HandEntry(a_object, CBRO_TRACE_OUTSIDE) && TraceOutsideFiled(*context, a_object);
							}
							RecordLeftOut(*context, a_object, *a_bound, force ? LeftOutPath::kDropped : LeftOutPath::kOutside, &out);
						}
					} else if (hiddenConfirmed) {
						const bool turn = TraceRefresh(*context, a_object, *a_bound);
						if (turn && HandEntry(a_object, CBRO_TRACE_HIDDEN)) {
							RecordLeftOut(*context, a_object, *a_bound, LeftOutPath::kDropped, &out);  // (handed to the tracer, not filed, v1.99.3)
							skip = true;
						} else if (turn && TraceDropFiled(*context, a_object)) {
							RecordLeftOut(*context, a_object, *a_bound, LeftOutPath::kDropped, &out);
						} else {
							Bump(kSkippedHidden);
							RecordLeftOut(*context, a_object, *a_bound, LeftOutPath::kSkipHidden, &out);
							skip = true;
						}
					} else if (out.outcome == Outcome::kOutside) {
						const bool turn = TraceRefresh(*context, a_object, *a_bound);
						if (turn && HandEntry(a_object, CBRO_TRACE_OUTSIDE)) {
							RecordLeftOut(*context, a_object, *a_bound, LeftOutPath::kDropped, &out);  // (handed to the tracer, not filed, v1.99.3)
							skip = true;
						} else if (turn && TraceOutsideFiled(*context, a_object)) {
							force = true;
							RecordLeftOut(*context, a_object, *a_bound, LeftOutPath::kDropped, &out);
						} else if (!light) {
							// (a light's bound can be off-view while its light still reaches into view)
							Bump(kSkippedOutside);
							RecordLeftOut(*context, a_object, *a_bound, LeftOutPath::kSkipOutside, &out);
							skip = true;
						}
					}
				}
			}
			if (FadeStampTurn(a_object, context->clock)) {
				CountFadeTop(*context, a_object, *a_bound, skip, player || force);  // (survey, v1.92.4; the forced split v1.94.1)
			}
			if (recording) {
				const auto flags = static_cast<std::uint16_t>(
					(skip ? RecorderFormat::kLeftOut : 0) | ((player || force) ? RecorderFormat::kForced : 0) | (player ? RecorderFormat::kPlayer : 0) |
					(alwaysDraw ? RecorderFormat::kAlwaysDraw : 0) | (sky ? RecorderFormat::kSky : 0) |
					(a_kind == Hooks::CullGroups::GroupKind::kMainOnly ? RecorderFormat::kGroup1 : 0) | (Observing() ? RecorderFormat::kObserveOnly : 0));
				Recorder::NoteEntry(Recorder::Origin::kEngineTop, a_object, *a_bound, static_cast<std::uint8_t>(a_kind), flags, 0);  // (the walk's: SetWalkSite turns it into kWalkTop)
			}
			if (!skip) {
				t_top = { a_object, context->clock, nullptr, player || force };
			}
			// (v1.99.2: a node filed by a known group goes into its group's sequence instead of the top table)
			bool recorded = false;
			if (!skip && t_decisionGroup && StableIndexOn() && !a_object->IsGeometry()) {
				if (const auto sequence = ParentSequenceOf(t_decisionGroup, true)) {
					ParentRecord record{};
					record.object = a_object;
					if (judged && !force) {
						record.state |= kParentJudged;
						record.outcome = static_cast<std::uint8_t>(out.outcome);
						record.verdict = out.verdict;
						record.flags = out.flags;
						std::copy_n(out.bound, 4, record.bound);
					}
					if (g_drops.Contains(a_object, DropSet::Tag(context->clock))) {
						record.state |= kParentDropped;  // (inserted above: its line is in the cache)
					}
					if (force) {
						record.state |= kParentForced;
					}
					if (player) {
						record.state |= kParentPlayer;
					}
					if (a_known & kKnownScolChecked) {
						record.state |= kParentScolKnown | ((a_known & kKnownScolRoot) ? kParentScolRoot : 0);
					}
					sequence->records.push_back(record);
					const auto bit = ParentPresenceBit(a_object);
					g_parentPresence[bit >> 6] |= std::uint64_t{ 1 } << (bit & 63);
					recorded = true;
					Bump(kStableRecorded);
				}
			}
			if (recorded) {
			} else if (!skip && judged && !force) {
				auto& top = TopSlot(a_object);
				top.object = a_object;
				top.clock = context->clock;
				top.outcome = static_cast<std::uint8_t>(out.outcome);
				top.verdict = out.verdict;
				top.flags = out.flags;
				top.bound[0] = out.bound[0];
				top.bound[1] = out.bound[1];
				top.bound[2] = out.bound[2];
				top.bound[3] = out.bound[3];
			} else if (auto& top = TopSlot(a_object); top.object == a_object) {
				top.object = nullptr;  // (an earlier decision of this frame's on the object, left out or forced now)
			}
			if (timed) {
				Bump(kCycles, (__rdtsc() - start) * kTimingStride);
			}
			return skip;
		}

		// ---- whole cell nodes (the scene walk, before it adds a cell node's children one by one) --------------
		// A cell's child node 3 holds its precombined chunks and static refs (the walk adds them to group 0 one by
		// one). When that node lies entirely outside the view and its sun shadow can't reach the view either, it is
		// never walked: none of its objects is filed with any view. Its worldBound is the engine's own node bound
		// over its children, the same bound vanilla previs mode frustum-tests cell roots by (whole, as single entries
		// of group 1); the shadow condition is CBRO's addition. Only "outside" counts here, never "hidden behind
		// surfaces". Child 9's role is unknown: it is only counted.
		// What holds a node offered by a prune site. Only its ENTRIES matter: the objects the engine files one by one,
		// i.e. the node's children and, for a node the scene walk expands two levels (a cell's node 3, a_depth 2),
		// the children of its exact-NiNode children as well. An entry with the engine's always-draw bit (flags bit
		// 11) is registered whatever the frustum says (Block::Add forces it), and an actor's root moves: either holds
		// the node. The grouping nodes' own always-draw bits (the v1.20 run: cells' node 3 and the containers under
		// it all carry it) mean nothing, since the walk never files those nodes; they only stopped v1.17's and
		// v1.20's pruning. Lights don't hold either: the top-level early skip has left out light-carrying nodes
		// outside the view since v1.7 (lights live in the scene's light list, not in the culling groups). Scanned
		// once per epoch and cached in the node's history entry at bits [56,59) (such a node is never an entry, so
		// the object classification never shares its key). Too big to scan holds too.
		constexpr int  kNodeContentsShift = 56;
		std::uintptr_t g_niNodeVtable{ 0 };

		// The scans below read each child's vtable and one more line (its world bound, or its flags and userData), a miss
		// per child (0.11 ms a frame in the 2026-10-26 profile); the child a few slots ahead is fetched first (v1.92.3).
		constexpr std::size_t kScanPrefetch = 4;

		template <class T>
		void PrefetchChild(const T& a_children, std::size_t a_index, std::size_t a_count, std::size_t a_line) noexcept
		{
			if (a_index < a_count) {
				if (const auto ahead = reinterpret_cast<const char*>(a_children[static_cast<typename T::size_type>(a_index)].get())) {
					_mm_prefetch(ahead, _MM_HINT_T0);
					_mm_prefetch(ahead + a_line, _MM_HINT_T0);
				}
			}
		}

		std::uint8_t EntryScan(RE::NiAVObject* a_node, int a_depth) noexcept
		{
			const auto root = a_node->IsNode();
			if (!root) {
				return kSubtreePlain;
			}
			constexpr std::size_t kBudget = 8192;
			std::uint8_t          found = kSubtreePlain;
			std::size_t           visited = 0;
			const auto            entry = [&](RE::NiAVObject* a_object) {
				if ((a_object->GetFlags() >> 11) & 1) {
					found |= kSubtreeAlwaysDraw;
				}
				if (IsActorRoot(a_object)) {
					found |= kSubtreeActor;
				}
			};
			const auto& children = root->children;
			const auto  count = Util::UsedSlots(children);  // (the engine's bound, not size(): v1.92.11)
			for (std::uint16_t i = 0; i < count; ++i) {
				PrefetchChild(children, i + kScanPrefetch, count, 0x100);  // (vtable; flags, userData)
				const auto object = children[i].get();
				if (!object) {
					continue;
				}
				if (++visited > kBudget) {
					return found | kSubtreeLight;
				}
				const auto inner = a_depth >= 2 && Util::TryReadVtable(object) == g_niNodeVtable ? object->IsNode() : nullptr;
				if (!inner) {
					entry(object);
					continue;
				}
				const auto& grandchildren = inner->children;
				const auto  count2 = Util::UsedSlots(grandchildren);
				for (std::uint16_t j = 0; j < count2; ++j) {
					PrefetchChild(grandchildren, j + kScanPrefetch, count2, 0x100);
					const auto object2 = grandchildren[j].get();
					if (!object2) {
						continue;
					}
					if (++visited > kBudget) {
						return found | kSubtreeLight;
					}
					entry(object2);
				}
			}
			return found;
		}

		std::uint8_t NodeContents(const FrameContext& a_context, RE::NiAVObject* a_node, int a_depth) noexcept
		{
			const auto key = reinterpret_cast<std::uintptr_t>(a_node);
			const auto entry = FindEntry(key);
			if (!entry) {
				Bump(kTableFull);
				return kSubtreeLight;
			}
			const auto epoch = static_cast<std::uint64_t>(((a_context.clock + static_cast<std::uint32_t>((key >> 4) % kEpochFrames)) / kEpochFrames) & 0xFF);
			auto       value = entry->value.load(std::memory_order_relaxed);
			if (!(value & kClassified) || ((value >> kEpochShift) & 0xFF) != epoch) {
				const auto found = EntryScan(a_node, a_depth);
				value = (value & kLowBits) | kClassified | (epoch << kEpochShift) | (static_cast<std::uint64_t>(found & 7) << kNodeContentsShift);
				entry->value.store(value, std::memory_order_relaxed);
			}
			return static_cast<std::uint8_t>((value >> kNodeContentsShift) & 7);
		}

		// ---- node caches: the entries' slots and their clusters (v1.94 slots; v1.98 clusters and leases, bWalkLeases) ------
		// Per node a prune site offers (a cell's node 3, a container under it, a DrawWorld root, a cell's node 9), by child
		// slot: the object seen there, its bound at the last touch and its classification (container, light, actor, sky), so
		// a walk reads each entry's flags and bound only (v1.94). v1.98: the slots are grouped in runs of kClusterSlots,
		// each with the sphere around its members' bounds at their last touch and whether any member holds it. Every
		// frame used to touch every entry of every node offered (the walk's 5,979 and the scans' 4,925 in the 2026-10-27
		// run, a cache miss or two each: 1.1 ms of the main thread's 3.7 ms cull), to find the few that moved and the few
		// that meet the view. Now a cluster whose members all lay still at the last walk, held by nothing (an always-draw
		// entry, a light, an actor, the player's, a sky mesh, an unusable bound, a marker) and whose sphere lies outside
		// the view by the verdict's margins plus fWalkLeaseMargin, with no shadow or lamp needing it, is left out whole
		// without touching a member: the engine's child array alone says the slots still hold the same objects. Its
		// members are touched again at most iWalkLeaseFrames frames later (staggered, so a frame walks a part of them).
		// What such a frame would have found: a member moved meanwhile (by less than the margin: still outside; by more:
		// drawn up to the lease late, the one new risk, taken with the margin at 64 units and the lease at 4 frames), a
		// member mid-fade (its fade is run anyway: the set, FadeMidWay), the trace's turn (walked as an entry), a marker
		// pending (the cluster is walked). Node pruning reads the same clusters: a node whose clusters are all within
		// their lease and outside the view is left out whole on their spheres, its entries untouched.
		enum SlotKind : std::uint8_t
		{
			kSlotContainerKnown = 1 << 0,  // kSlotContainer decided for the slot's object
			kSlotContainer = 1 << 1,       // an exact NiNode under a cell's node 3: its members are the entries
			kSlotClassified = 1 << 2,      // kSlotLight, kSlotActor, kSlotSky decided for the slot's object, at `epoch`
			kSlotLight = 1 << 3,           // a light type (TypeOf): never left out as out of view
			kSlotActor = 1 << 4,           // an actor's root (userData a kACHR reference)
			kSlotSky = 1 << 5,             // a sky mesh
			kSlotGroup1 = 1 << 6,          // a root's child with flags bit 40 at the last walk: group 1 is its route
			kSlotScolRoot = 1 << 7,        // a static collection's root (IsScolReference, at `epoch`; v1.99.2)
		};

		// Why a slot's entry holds its cluster out of a lease, set at each touch (the sticky reasons: the transient ones,
		// in view, a shadow or lamp needing it, the trace's turn, are tested on the cluster's sphere each frame).
		enum SlotHold : std::uint8_t
		{
			kHoldBad = 1 << 0,         // bound unusable (unmaintained, huge, NaN): judged, never leased
			kHoldMoved = 1 << 1,       // the bound changed by more than kQuietMove since the touch before
			kHoldAlwaysDraw = 1 << 2,  // the engine's always-draw bit (flags bit 11)
			kHoldLight = 1 << 3,
			kHoldActor = 1 << 4,
			kHoldPlayer = 1 << 5,      // the player's 3D (filed force-visible for the trace every frame)
			kHoldSky = 1 << 6,
			kHoldMarker = 1 << 7,      // filed unjudged with a group marker
		};

		struct WalkSlot
		{
			const RE::NiAVObject* object{ nullptr };
			float                 bound[4]{};    // the bound at the last touch (center, radius): a change is a move
			bool                  seen{ false };  // bound valid
			bool                  scanChanged{ false };  // the pruning scan found the bound changed, and no walk has seen that yet
			std::uint8_t          kind{ 0 };
			std::uint8_t          epoch{ 0 };
			std::uint8_t          hold{ 0 };
			bool                  boundDirty{ true };  // the bound changed (or the slot is new) since its cluster's sphere was built (v1.99)
			bool                  handed{ false };     // its entry was handed to a tracer that keeps it (v2.0: a GONE record when it leaves)
		};
		static_assert(sizeof(WalkSlot) == 32);  // (two to a cache line: v1.99's certificates made it 64 and the walk's slot misses doubled; v1.99.1)

		struct WalkCluster
		{
			float         center[3]{};
			float         radius{ -1.0f };     // the sphere around the members' bounds at the last walk (< 0: no member)
			std::uint32_t walkedClock{ 0 };    // the frame its members were last touched, all of them
			std::uint8_t  members{ 0 };        // entries among its slots (containers and empty slots apart)
			std::uint8_t  hold{ 0 };           // the members' holds, together
			bool          valid{ false };      // touched whole since the slots were (re)built
			bool          walked{ false };     // ... by the walk (the members classified: light, actor, sky, player, always-draw), not the pruning scan alone
			std::uint16_t containers{ 0 };     // its slots holding containers, by bit (v1.99.1: a lease reads the masks, not the slots)
			std::uint16_t group1{ 0 };         // ... holding entries routed to group 1
			std::uint64_t children{ 0 };       // the signature of the child pointers it was finished with (ChildSignature)
		};

		struct WalkNodeCache
		{
			std::uint32_t            clock{ 0 };    // the last touch's frame (the sweep)
			std::uint32_t            stagger{ 0 };  // the node's share of the lease schedule
			bool                     traced{ false };  // walked whole for a tracer that keeps what it is handed (v2.0: its later trace walks take only what it doesn't know)
			std::vector<WalkSlot>    slots;         // by child slot
			std::vector<WalkCluster> clusters;      // by run of kClusterSlots slots
		};

		std::unordered_map<const void*, WalkNodeCache> g_walkNodes;  // (main thread)
		std::uint32_t                                  g_walkSweep{ 0 };
		constexpr std::uint32_t                        kWalkSweepFrames = 1024;  // a node not touched for this long is forgotten
		constexpr std::size_t                          kClusterSlots = 16;
		constexpr float                                kQuietMove = 8.0f;  // a bound shifting by less between touches (swaying) doesn't hold its cluster
		thread_local int                               t_walkDepth{ 0 };
		thread_local bool                              t_prunedFromCache{ false };  // (SkipCellNode: EntriesOutside answered from the clusters)

		// A slot's entry leaves it (another child took the slot, it was AppCulled, its node's cache is dropped): an entry
		// handed to a tracer that keeps it is gone for the tracer (v2.0, the link's GONE record).
		inline void SlotLeft(WalkSlot& a_slot) noexcept
		{
			if (a_slot.handed) {
				a_slot.handed = false;
				TraceFeed::Gone(a_slot.object);
				Bump(kTraceGone);
			}
		}

		void CacheLeft(WalkNodeCache& a_cache) noexcept
		{
			if (TraceFeed::Linked()) {
				for (auto& slot : a_cache.slots) {
					SlotLeft(slot);
				}
			}
		}

		WalkNodeCache& NodeCacheOf(const RE::NiNode* a_root, std::size_t a_count, std::uint32_t a_clock)
		{
			auto& cache = g_walkNodes[a_root];
			cache.clock = a_clock;
			if (cache.slots.size() != a_count) {
				// (v2.0: an entry handed to the tracer that still holds its slot stays handed: a detached child leaves its slot
				// null and AttachChild(child, false) appends at the count, FO4OG 7.13, 10.17, so a rebuilt node's entries mostly
				// keep their slots)
				std::vector<WalkSlot> slots(a_count);
				if (TraceFeed::Linked()) {
					const auto& children = a_root->children;
					for (std::size_t i = 0; i < cache.slots.size(); ++i) {
						auto& old = cache.slots[i];
						if (!old.handed) {
							continue;
						}
						if (i < a_count && children[static_cast<std::uint16_t>(i)].get() == old.object) {
							slots[i].object = old.object;
							slots[i].handed = true;
						} else {
							SlotLeft(old);
						}
					}
				}
				cache.slots = std::move(slots);
				cache.traced = false;
				cache.clusters.assign((a_count + kClusterSlots - 1) / kClusterSlots, WalkCluster{});
				cache.stagger = static_cast<std::uint32_t>(reinterpret_cast<std::uintptr_t>(a_root) >> 6);
				Bump(kWalkCachesRebuilt);
			}
			return cache;
		}

		inline bool UsableBound(const RE::NiBound& a_bound) noexcept
		{
			return a_bound.fRadius > 1.0f && a_bound.fRadius < 1.0e6f && std::isfinite(a_bound.center.x) && std::isfinite(a_bound.center.y) && std::isfinite(a_bound.center.z);
		}

		// The bound at a touch, into the slot: a shift past the quiet allowance since the touch before holds the cluster.
		// Returns whether the bound changed at all (the walk's own "moved").
		bool NoteSlotBound(WalkSlot& a_slot, const RE::NiBound& a_bound) noexcept
		{
			const bool changed = !a_slot.seen || a_slot.bound[0] != a_bound.center.x || a_slot.bound[1] != a_bound.center.y || a_slot.bound[2] != a_bound.center.z || a_slot.bound[3] != a_bound.fRadius;
			if (a_slot.seen && changed) {
				const float dx = a_bound.center.x - a_slot.bound[0], dy = a_bound.center.y - a_slot.bound[1], dz = a_bound.center.z - a_slot.bound[2];
				const float dr = a_bound.fRadius - a_slot.bound[3];
				if (!(dx * dx + dy * dy + dz * dz <= kQuietMove * kQuietMove) || !(std::fabs(dr) <= kQuietMove)) {
					a_slot.hold |= kHoldMoved;
				}
			}
			a_slot.bound[0] = a_bound.center.x;
			a_slot.bound[1] = a_bound.center.y;
			a_slot.bound[2] = a_bound.center.z;
			a_slot.bound[3] = a_bound.fRadius;
			a_slot.seen = true;
			if (changed) {
				a_slot.boundDirty = true;  // (its cluster's sphere is built again: FinishCluster)
			}
			return changed;
		}

		// Grows a_union (empty while !a_any) to hold a_bound.
		void GrowSphere(RE::NiBound& a_union, bool& a_any, const RE::NiBound& a_bound) noexcept
		{
			if (!a_any) {
				a_union = a_bound;
				a_any = true;
				return;
			}
			const float cx = a_bound.center.x - a_union.center.x;
			const float cy = a_bound.center.y - a_union.center.y;
			const float cz = a_bound.center.z - a_union.center.z;
			const float d = std::sqrt(cx * cx + cy * cy + cz * cz);
			if (d + a_bound.fRadius > a_union.fRadius) {
				if (d + a_union.fRadius <= a_bound.fRadius) {
					a_union = a_bound;
				} else {
					const float radius = (d + a_union.fRadius + a_bound.fRadius) * 0.5f;
					const float t = (radius - a_union.fRadius) / d;
					a_union.center.x += cx * t;
					a_union.center.y += cy * t;
					a_union.center.z += cz * t;
					a_union.fRadius = radius;
				}
			}
		}

		inline RE::NiBound SlotBound(const WalkSlot& a_slot) noexcept
		{
			return RE::NiBound{ RE::NiPoint3{ a_slot.bound[0], a_slot.bound[1], a_slot.bound[2] }, a_slot.bound[3] };
		}

		inline RE::NiBound ClusterBound(const WalkCluster& a_cluster) noexcept
		{
			return RE::NiBound{ RE::NiPoint3{ a_cluster.center[0], a_cluster.center[1], a_cluster.center[2] }, a_cluster.radius };
		}

		// A run of child slots' signature (v1.99.1): a lease compares the child array with the signature its cluster was
		// finished with, instead of each slot's object (a lease then costs the child array's lines, no slot: the slot
		// compares were ~7% of CBRO's cull-stage samples in the 2026-10-27 v1.99 run).
		inline std::uint64_t MixChild(const void* a_object, std::size_t a_index) noexcept
		{
			auto x = static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(a_object)) + a_index;
			x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
			x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
			return x ^ (x >> 31);
		}

		template <class Children>
		std::uint64_t ChildSignature(const Children& a_children, std::size_t a_first, std::size_t a_last) noexcept
		{
			std::uint64_t signature = 0;
			for (auto i = a_first; i < a_last; ++i) {
				signature += MixChild(a_children[static_cast<std::uint16_t>(i)].get(), i);
			}
			return signature;
		}

		// The cluster's sphere, members and hold from its slots, after every one of them was touched this frame.
		void FinishCluster(WalkNodeCache& a_cache, std::size_t a_index, std::uint32_t a_clock, bool a_walked) noexcept
		{
			auto&       cluster = a_cache.clusters[a_index];
			RE::NiBound sphere{};
			bool        any = false;
			std::uint8_t hold = 0;
			std::uint8_t members = 0;
			const auto first = a_index * kClusterSlots;
			const auto last = std::min(first + kClusterSlots, a_cache.slots.size());
			// (v1.99: the sphere is built again only when a member's bound changed or the members did; most frames none did)
			bool          rebuild = !cluster.valid;
			std::uint16_t containers = 0;
			std::uint16_t group1 = 0;
			std::uint64_t children = 0;
			for (auto i = first; i < last; ++i) {
				auto& slot = a_cache.slots[i];
				children += MixChild(slot.object, i);
				if (!slot.object) {
					continue;
				}
				const auto bit = static_cast<std::uint16_t>(1u << (i - first));
				if (slot.kind & kSlotContainer) {
					containers |= bit;
					continue;
				}
				if (slot.kind & kSlotGroup1) {
					group1 |= bit;
				}
				++members;
				hold |= slot.hold;
				if (!slot.seen) {
					hold |= kHoldBad;
				}
				if (slot.boundDirty) {
					slot.boundDirty = false;
					rebuild = true;
				}
			}
			if (members != cluster.members) {
				rebuild = true;
			}
			cluster.hold = hold;
			cluster.members = members;
			cluster.valid = true;
			cluster.walked = a_walked;
			cluster.walkedClock = a_clock;
			cluster.containers = containers;
			cluster.group1 = group1;
			cluster.children = children;
			if (!rebuild) {
				return;
			}
			for (auto i = first; i < last; ++i) {
				const auto& slot = a_cache.slots[i];
				if (slot.object && !(slot.kind & kSlotContainer) && slot.seen) {
					GrowSphere(sphere, any, SlotBound(slot));
				}
			}
			if (any) {
				cluster.center[0] = sphere.center.x;
				cluster.center[1] = sphere.center.y;
				cluster.center[2] = sphere.center.z;
				cluster.radius = sphere.fRadius;
			} else {
				cluster.radius = -1.0f;
			}
		}

		// A cluster is touched again at most leaseFrames frames after its last touch, on a frame of its own in the schedule.
		bool LeaseDue(const FrameContext& a_context, const WalkNodeCache& a_cache, std::size_t a_index, const WalkCluster& a_cluster) noexcept
		{
			if (!a_cluster.valid || a_context.clock - a_cluster.walkedClock > g_tunables.leaseFrames) {
				return true;
			}
			return (a_context.clock + a_cache.stagger + static_cast<std::uint32_t>(a_index)) % (g_tunables.leaseFrames + 1) == 0;
		}

		// The cluster's sphere lies entirely outside the view, by the verdict's margins and the lease margin on top.
		bool ClusterOutside(const FrameContext& a_context, const WalkCluster& a_cluster) noexcept
		{
			if (!(a_cluster.radius >= 0.0f)) {
				return false;
			}
			const auto& camera = a_context.snapshot->camera;
			if (!camera.viewSpace) {
				return false;
			}
			const float dx = a_cluster.center[0] - camera.origin[0];
			const float dy = a_cluster.center[1] - camera.origin[1];
			const float dz = a_cluster.center[2] - camera.origin[2];
			const float z = dx * camera.viewDir[0] + dy * camera.viewDir[1] + dz * camera.viewDir[2];
			const float x = dx * camera.viewRight[0] + dy * camera.viewRight[1] + dz * camera.viewRight[2];
			const float y = dx * camera.viewUp[0] + dy * camera.viewUp[1] + dz * camera.viewUp[2];
			return SphereOutsideView(a_context, x, y, z, a_cluster.radius + a_context.dilateMove + std::max(z, 0.0f) * a_context.angularSlack + g_tunables.leaseMargin);
		}

		// A grouping node's own worldBound is NOT maintained by the engine: the v1.20/v1.21 samples showed r=1 on every
		// cell node and container, and that is what their always-draw bit stands for ("never cull me by my bound").
		// v1.21 judged them by that bound and most of the scene vanished. So a node is judged by its ENTRIES' bounds
		// (the objects the engine files, whose bounds it culls by), all of them: skippable only if every entry lies
		// entirely outside the current view (with the usual dilation), with the sphere around all of them (for the
		// sun sweep) returned. An entry whose bound looks unmaintained too (radius <= 1, e.g. a nested container) or
		// invalid holds the node; the pass stops at the first entry that isn't outside. The scan fills the node's slots
		// and clusters as it goes (and a container's, two levels down), so that later frames can answer from the
		// clusters' spheres (CachedEntriesOutside) until their lease runs out.
		// One cluster of a node's slots read (v1.98.1; v1.98 read the whole node): each entry touched, its bound into the
		// slot and into the sphere; false at the first entry that meets the view or can't be judged by its bound (the node
		// is walked then, its remaining clusters untouched). A container found among the slots is scanned through
		// ScanEntries (two levels down at most). The cluster is finished once every slot was touched.
		bool ScanEntries(const FrameContext& a_context, RE::NiNode* a_root, int a_depth, RE::NiBound& a_union, bool& a_any, std::uint32_t& a_count) noexcept;
		bool CachedEntriesOutside(const FrameContext& a_context, RE::NiNode* a_root, int a_depth, RE::NiBound& a_union, bool& a_any, std::uint32_t& a_count, bool& a_usable) noexcept;

		bool ScanCluster(const FrameContext& a_context, RE::NiNode* a_root, WalkNodeCache& a_cache, std::size_t a_cluster, int a_depth, RE::NiBound& a_union, bool& a_any, std::uint32_t& a_count) noexcept
		{
			const auto& camera = a_context.snapshot->camera;
			const auto& children = a_root->children;
			const auto  count = a_cache.slots.size();
			const bool  resumed = g_tunables.walkLeases && a_cache.clusters[a_cluster].valid && a_context.clock - a_cache.clusters[a_cluster].walkedClock > 1;
			const auto  first = a_cluster * kClusterSlots;
			const auto  last = std::min(first + kClusterSlots, count);
			for (auto i = first; i < last; ++i) {
				PrefetchChild(children, i + kScanPrefetch, count, 0x80);  // (vtable; world bound)
				const auto object = children[static_cast<std::uint16_t>(i)].get();
				auto&      slot = a_cache.slots[i];
				if (slot.object != object) {
					SlotLeft(slot);
					slot = WalkSlot{};
					slot.object = object;
				}
				if (!object) {
					continue;
				}
				const auto inner = a_depth >= 2 && Util::TryReadVtable(object) == g_niNodeVtable ? object->IsNode() : nullptr;
				if (inner) {
					slot.kind |= static_cast<std::uint8_t>(kSlotContainerKnown | kSlotContainer);
					bool usable = true;
					if (g_tunables.walkLeases && CachedEntriesOutside(a_context, inner, 1, a_union, a_any, a_count, usable)) {
						continue;  // (its own clusters answered)
					}
					if ((g_tunables.walkLeases && usable) || !ScanEntries(a_context, inner, 1, a_union, a_any, a_count)) {
						return false;
					}
					continue;
				}
				slot.hold = 0;
				const auto& bound = object->worldBound;
				if (!UsableBound(bound)) {
					slot.hold |= kHoldBad;
					return false;  // unmaintained or bad: the node is walked normally
				}
				slot.scanChanged = NoteSlotBound(slot, bound) || slot.scanChanged;  // (the walk of this frame, if any, still sees the move)
				const float dx = bound.center.x - camera.origin[0];
				const float dy = bound.center.y - camera.origin[1];
				const float dz = bound.center.z - camera.origin[2];
				const float z = dx * camera.viewDir[0] + dy * camera.viewDir[1] + dz * camera.viewDir[2];
				const float x = dx * camera.viewRight[0] + dy * camera.viewRight[1] + dz * camera.viewRight[2];
				const float y = dx * camera.viewUp[0] + dy * camera.viewUp[1] + dz * camera.viewUp[2];
				if (!SphereOutsideView(a_context, x, y, z, bound.fRadius + a_context.dilateMove + std::max(z, 0.0f) * a_context.angularSlack)) {
					return false;
				}
				// (v1.96) The node may be left out whole, its entries never visited again while it is: a fade node among them
				// keeps its fade as the walk's skipped entries do (bKeepFadeState), or the engine restarts it from 0 when the
				// node comes back (a pop-in, and a lighting re-queue for every frame of the fade).
				if (g_tunables.keepFadeState) {
					KeepFadeSeen(object, a_context.clock, resumed);
				}
				GrowSphere(a_union, a_any, bound);
				++a_count;
			}
			FinishCluster(a_cache, a_cluster, a_context.clock, false);
			return true;
		}

		bool ScanEntries(const FrameContext& a_context, RE::NiNode* a_root, int a_depth, RE::NiBound& a_union, bool& a_any, std::uint32_t& a_count) noexcept
		{
			const auto count = Util::UsedSlots(a_root->children);  // (the engine's bound, not size(): v1.92.11)
			auto&      cache = NodeCacheOf(a_root, count, a_context.clock);
			for (std::size_t c = 0; c < cache.clusters.size(); ++c) {
				if (!ScanCluster(a_context, a_root, cache, c, a_depth, a_union, a_any, a_count)) {
					return false;
				}
			}
			return true;
		}

		// Node pruning answered from the clusters (v1.98): a cluster within its lease, held by nothing, its slots still
		// holding the objects its sphere was built from (the child array alone) answers by that sphere, no entry touched.
		// Any other cluster is read again, one at a time (v1.98.1: v1.98 gave the whole node up at the first such cluster,
		// and the stagger makes one due every frame in a node of five clusters or more). a_usable false: no cache for the
		// node yet, or its child count changed (a scan from scratch follows). False with a_usable: an entry, or a leased
		// cluster's sphere, meets the view.
		bool CachedEntriesOutside(const FrameContext& a_context, RE::NiNode* a_root, int a_depth, RE::NiBound& a_union, bool& a_any, std::uint32_t& a_count, bool& a_usable) noexcept
		{
			const auto found = g_walkNodes.find(a_root);
			if (found == g_walkNodes.end()) {
				a_usable = false;
				return false;
			}
			auto&       cache = found->second;
			const auto& children = a_root->children;
			const auto  count = Util::UsedSlots(children);
			if (cache.slots.size() != count) {
				a_usable = false;
				return false;
			}
			cache.clock = a_context.clock;
			for (std::size_t c = 0; c < cache.clusters.size(); ++c) {
				const auto& cluster = cache.clusters[c];
				const auto  first = c * kClusterSlots;
				const auto  last = std::min(first + kClusterSlots, static_cast<std::size_t>(count));
				// (v1.99.1: the child array against the signature the cluster was finished with, no slot read)
				const bool  leased = cluster.valid && !cluster.hold && !LeaseDue(a_context, cache, c, cluster) && ChildSignature(children, first, last) == cluster.children;
				if (!leased) {
					Bump(kCellNodeClustersRescanned);
					if (!ScanCluster(a_context, a_root, cache, c, a_depth, a_union, a_any, a_count)) {
						return false;
					}
					continue;
				}
				if (cluster.radius >= 0.0f && !ClusterOutside(a_context, cluster)) {
					return false;  // (meets the view: the node is walked, which touches its entries again)
				}
				for (auto i = first; i < last; ++i) {
					const auto object = children[static_cast<std::uint16_t>(i)].get();
					if (!object) {
						continue;
					}
					if (cluster.containers & (1u << (i - first))) {
						const auto inner = reinterpret_cast<RE::NiNode*>(object);
						bool       usable = true;
						if (CachedEntriesOutside(a_context, inner, 1, a_union, a_any, a_count, usable)) {
							continue;
						}
						if (usable || !ScanEntries(a_context, inner, 1, a_union, a_any, a_count)) {
							return false;  // (an entry of the container meets the view; or it had no cache yet and was scanned)
						}
						continue;
					}
					if (g_tunables.keepFadeState && FadeMidWay(object)) {
						KeepFadeSeen(object, a_context.clock);  // (its fade runs on: the one touch a lease allows)
					}
				}
				if (cluster.radius >= 0.0f) {
					GrowSphere(a_union, a_any, ClusterBound(cluster));
					a_count += cluster.members;
				}
			}
			return true;
		}

		bool EntriesOutside(const FrameContext& a_context, RE::NiAVObject* a_node, int a_depth, RE::NiBound& a_union, std::uint32_t& a_count) noexcept
		{
			const auto root = a_node->IsNode();
			if (!root || !std::isfinite(a_context.view[0])) {
				return false;
			}
			const auto& camera = a_context.snapshot->camera;
			if (!camera.viewSpace) {
				return false;
			}
			bool any = false;
			a_count = 0;
			t_prunedFromCache = false;
			if (g_tunables.walkLeases && !LeftOutFrame(a_context.clock)) {
				bool       usable = true;
				const bool outside = CachedEntriesOutside(a_context, root, a_depth, a_union, any, a_count, usable);
				if (usable) {
					t_prunedFromCache = outside && any;
					return outside && any;
				}
				any = false;
				a_count = 0;
			}
			return ScanEntries(a_context, root, a_depth, a_union, any, a_count) && any;
		}

		// The first few nodes offered, for the log: what they are.
		std::atomic<int> g_nodeSamples{ 0 };

		void SampleNode(RE::NiAVObject* a_node, std::uint32_t a_index, std::uint8_t a_contents, bool a_outside, std::uint32_t a_entries, float a_unionRadius)
		{
			if (g_nodeSamples.load(std::memory_order_relaxed) >= 8 || g_nodeSamples.fetch_add(1) >= 8) {
				return;
			}
			char type[48]{};
			char name[48]{};
			Util::TryGetRTTIName(a_node, type, sizeof(type));
			Util::TryGetObjectName(a_node, name, sizeof(name));
			const auto node = a_node->IsNode();
			logger::info(
				"node pruning: sample index {} type {} name '{}' flags {:#x} children {} own r={:.0f} (unmaintained: not used) contents {:#x} (1 too big, 2 actor entry, 4 always-draw entry) {}: {} entries outside, their sphere r={:.0f}",
				a_index, type[0] ? type : "?", name, a_node->GetFlags(), node ? Util::UsedSlots(node->children) : 0u, a_node->worldBound.fRadius, a_contents,
				a_outside ? "all entries outside the view" : "meets the view (or held)", a_entries, a_unionRadius);
		}

		// A node pruning would leave out whole (outside the view, its sun shadow unneeded), walked for the trace on its turn
		// with the sun up (TraceKeep files its entries force-visible and keeps them out of every view's draws). A container
		// under a node walked so is walked too (its entries are the node's grandchildren).
		bool TraceCellNode(const FrameContext& a_context, RE::NiAVObject* a_node, const RE::NiBound& a_bound) noexcept
		{
			if (!TraceFeed::On() || a_context.sun.state != FrameContext::Sun::State::kOn) {
				return false;
			}
			if (t_traceNode.node && t_traceNode.clock == a_context.clock && a_node->parent == t_traceNode.node) {
				return true;
			}
			if (!TraceFeed::Turn(a_node, a_context.clock) || !TraceFeed::InReach(a_bound, a_context.eye)) {
				return false;
			}
			// (v2.0: a node walked whole for a tracer that keeps what it is handed: its later trace walks take only the entries
			// the tracer doesn't know, but on the node's resend turn; known by the node's cache, not the known table, which the
			// tracer's forgetting of a piece clears up its parents)
			bool all = true;
			if (TraceFeed::Retains()) {
				if (const auto found = g_walkNodes.find(a_node); found != g_walkNodes.end()) {
					all = !found->second.traced || TraceFeed::ResendTurn(a_node, a_context.clock);
					found->second.traced = true;
				}
			}
			t_traceNode = { a_node, a_context.clock, all };
			return true;
		}

		// The left-out dump (v1.92.7): the entries of a node pruned whole, as EntriesOutside walked them, each recorded as
		// never filed with its own bound (the dump frame only).
		void RecordPrunedEntries(const FrameContext& a_context, RE::NiAVObject* a_node, int a_depth) noexcept
		{
			const auto root = a_node->IsNode();
			if (!root) {
				return;
			}
			t_source = VerdictSource::kNone;
			const auto record = [&](RE::NiAVObject* a_object) {
				RecordLeftOut(a_context, a_object, a_object->worldBound, LeftOutPath::kNodePruned, nullptr);
			};
			const auto& children = root->children;
			const auto  count = Util::UsedSlots(children);  // (the engine's bound, not size(): v1.92.11)
			for (std::uint16_t i = 0; i < count; ++i) {
				const auto object = children[i].get();
				if (!object) {
					continue;
				}
				const auto inner = a_depth >= 2 && Util::TryReadVtable(object) == g_niNodeVtable ? object->IsNode() : nullptr;
				if (!inner) {
					record(object);
					continue;
				}
				const auto& grandchildren = inner->children;
				const auto  count2 = Util::UsedSlots(grandchildren);
				for (std::uint16_t j = 0; j < count2; ++j) {
					if (const auto object2 = grandchildren[j].get()) {
						record(object2);
					}
				}
			}
		}

		// ---- the entry walk (v1.94, bEntryWalk) -------------------------------------------------------------------
		// A node a prune site offers that pruning can't leave out whole (an entry meets the view, or holds an always-draw
		// entry or an actor, or its sun shadow may reach the view) used to go back to the engine, which offered every one
		// of its entries to Group::Add one by one: the walk's loop and RTTI call, CBRO's Group::Add hook, and for each
		// entry the top-level decision (SkipTopLevel: the record lookup, the async candidate, the trace checks, the
		// counters), ~200 ns an entry in view or not. In a dense exterior some 4,400 of the 5,800 entries offered a frame
		// were out of view with no shadow needed (2026-10-26 profile: the cull stage spent ~1.5 ms in CBRO.dll and ~1.2 ms
		// in the engine's walk and adds). So CBRO walks such a node itself, in the engine's order and by its rules
		// (Hooks::PrevisFeed::EnumerateCandidates R3c, R3d, R4Child: a null or AppCulled child is skipped; an exact-NiNode
		// child of a cell's node 3 is a container whose members are the entries, without the AppCulled test; a root's
		// child goes to group 1 with flags bit 40 set, else group 0; a root that is a fade node is the engine's, filed
		// whole into the group array), and tells the engine to skip the node. An entry the cheap tests here can leave out
		// is never offered to anything: its bound unchanged since the last walk (the trace feed's "moved" rule), no light,
		// not always-draw, outside the view (the verdict's own test and margins), its sun shadow beyond the receivers'
		// reach planes (TestSun's first test) or, with the sun down, beyond every recorded lamp's reach (EvaluateSun's
		// sun-off rule with no lamp within reach), no spot lamp within reach, not the trace's turn, not the player's, not
		// a sky mesh, not a dump frame. Every other entry goes through the top-level decision as Group::Add would have
		// offered it (SkipTopLevel; its result reused by the Block::Add filter) and, when kept, is filed through the
		// engine's Group::Add (AddDirect). A group marker pending on the entry's block files it unjudged, as GroupAdd
		// does. Per node the slot cache keeps, by child slot, the object seen there, its bound at the last walk and its
		// classification (container, light), so a walk reads each entry's flags and bound only. Whole-node pruning runs
		// first, unchanged (SkipCellNode). Main thread, inside DrawWorld's cull.
		// The shadow of a caster bound is unneeded whatever the depth says: with the sun up its sweep lies beyond the
		// receivers' reach planes; with the sun down no recorded lamp (point or spot, the list complete) is within reach.
		// False: unknown here (the full path decides).
		bool ShadowBeyondReach(const FrameContext& a_context, const RE::NiBound& a_bound) noexcept
		{
			switch (a_context.sun.state) {
			case FrameContext::Sun::State::kOn: {
				SunSweep sweep{};
				if (!SunSweepStart(a_context, a_bound, sweep)) {
					return false;
				}
				if (FirstPersonUnseen(a_context) && SweepNearEye(sweep.center, a_context.sun.dir, sweep.radius, a_context.sun.spread, 1.0e6f, kFirstPersonReach)) {
					return false;
				}
				return SunSweepBeyondReach(a_context, sweep, false);
			}
			case FrameContext::Sun::State::kOff: {
				const auto& lamps = ShadowLights::Lamps();
				if (!g_tunables.lampGroupTrim || !lamps.complete || lamps.count == 0 || lamps.overflow != 0) {
					return false;
				}
				for (std::uint32_t i = 0; i < lamps.count; ++i) {
					const auto& lamp = lamps.items[i];
					const float dx = a_bound.center.x - lamp.position.x, dy = a_bound.center.y - lamp.position.y, dz = a_bound.center.z - lamp.position.z;
					const float reach = lamp.reach + a_bound.fRadius;
					if (dx * dx + dy * dy + dz * dz <= reach * reach) {
						return false;
					}
				}
				return true;
			}
			default:
				return false;
			}
		}

		// The bound lies entirely outside the view, by the verdict's own test and margins (Test, TestViewSpace: the
		// cache's move tolerance and its turn tolerance at the bound's depth). An unmaintained or bad bound is judged.
		bool EntryOutside(const FrameContext& a_context, const RE::NiBound& a_bound) noexcept
		{
			if (!(a_bound.fRadius > 1.0f) || !(a_bound.fRadius < 1.0e6f) || !std::isfinite(a_bound.center.x) || !std::isfinite(a_bound.center.y) || !std::isfinite(a_bound.center.z)) {
				return false;
			}
			const auto& camera = a_context.snapshot->camera;
			if (!camera.viewSpace) {
				return false;
			}
			const float dx = a_bound.center.x - camera.origin[0];
			const float dy = a_bound.center.y - camera.origin[1];
			const float dz = a_bound.center.z - camera.origin[2];
			const float z = dx * camera.viewDir[0] + dy * camera.viewDir[1] + dz * camera.viewDir[2];
			const float x = dx * camera.viewRight[0] + dy * camera.viewRight[1] + dz * camera.viewRight[2];
			const float y = dx * camera.viewUp[0] + dy * camera.viewUp[1] + dz * camera.viewUp[2];
			return SphereOutsideView(a_context, x, y, z, a_bound.fRadius + a_context.dilateMove + std::max(z, 0.0f) * a_context.angularSlack);
		}

		// A node's walk for the trace (v2.0): none, its entries the tracer doesn't know (a node it knows, on its turn), all.
		enum TraceWalk : std::uint8_t
		{
			kTraceWalkNone,
			kTraceWalkNew,
			kTraceWalkAll
		};

		// One entry of a walked node. a_group1: the engine's route for it. a_trace: the node is walked for the trace this
		// frame (kTraceWalkAll: every entry takes the full path). a_site: the engine site the add would have come from (the
		// audit). a_resumed: its cluster was left out whole since its last walk (a fade node's turn is made up).
		void WalkEntry(const FrameContext& a_context, RE::NiAVObject* a_object, WalkSlot& a_slot, bool a_group1, std::uint8_t a_trace, std::uintptr_t a_site, std::uint16_t a_loop, bool a_resumed)
		{
			const bool full = a_trace == kTraceWalkAll;
			Bump(kWalkEntries);
			const auto  group = Hooks::CullGroups::WalkGroup(a_group1);
			const auto  kind = a_group1 ? Hooks::CullGroups::GroupKind::kMainOnly : Hooks::CullGroups::GroupKind::kSunShared;
			const auto& bound = a_object->worldBound;
			Hooks::CullGroups::NoteEntryOffered(group, a_object, a_site);
			// (a move the pruning scan found this frame counts here too: the scan wrote the slot's bound before the walk)
			const bool scanMoved = a_slot.scanChanged;
			a_slot.hold = static_cast<std::uint8_t>(scanMoved ? (a_slot.hold & kHoldMoved) : 0);
			a_slot.scanChanged = false;
			a_slot.kind = static_cast<std::uint8_t>((a_slot.kind & ~kSlotGroup1) | (a_group1 ? kSlotGroup1 : 0));
			const bool moved = NoteSlotBound(a_slot, bound) || scanMoved;
			if (!UsableBound(bound)) {
				a_slot.hold |= kHoldBad;
			}
			// (v1.99: the group's own marker bytes first, then the entry's block: MarkerPending reads the entry's vtable, the one
			// line of it the walk's prefetch doesn't fetch, for a marker that is almost never pending)
			if (Hooks::CullGroups::MarkerPendingAny(group) && Hooks::CullGroups::MarkerPending(group, a_object)) {
				Bump(kWalkMarkerFiled);
				a_slot.hold |= kHoldMarker;
				if (Recorder::On()) {
					Recorder::NoteEntry(Recorder::Origin::kWalkMarker, a_object, bound, static_cast<std::uint8_t>(kind), static_cast<std::uint16_t>((moved ? RecorderFormat::kMoved : 0) | (a_group1 ? RecorderFormat::kGroup1 : 0)), a_loop);
				}
				Hooks::CullGroups::AddDirect(group, a_object, &bound, 0);  // (as GroupAdd files it: the Block::Add filter judges it)
				return;
			}
			const auto epoch = static_cast<std::uint8_t>(a_context.clock / kEpochFrames);
			if (!(a_slot.kind & kSlotClassified) || a_slot.epoch != epoch) {
				// (once per epoch: the type's table, the reference's form type, the shader property: a miss or two each)
				std::uint8_t kinds = kSlotClassified;
				if (TypeOf(a_object) & kTypeLight) {
					kinds |= kSlotLight;
				}
				if (IsActorRoot(a_object)) {
					kinds |= kSlotActor;
				}
				if (Hooks::CullGroups::IsSky(a_object)) {
					kinds |= kSlotSky;
				}
				if (g_tunables.stableIndex && IsScolReference(a_object)) {
					kinds |= kSlotScolRoot;
				}
				a_slot.kind = static_cast<std::uint8_t>((a_slot.kind & (kSlotContainerKnown | kSlotContainer | kSlotGroup1)) | kinds);
				a_slot.epoch = epoch;
			}
			const bool alwaysDraw = (a_object->GetFlags() >> 11) & 1;
			const bool light = (a_slot.kind & kSlotLight) != 0;
			const bool sky = (a_slot.kind & kSlotSky) != 0;
			const bool player = PlayerPart(a_context, a_object, bound);
			a_slot.hold |= static_cast<std::uint8_t>((alwaysDraw ? kHoldAlwaysDraw : 0) | (light ? kHoldLight : 0) | ((a_slot.kind & kSlotActor) ? kHoldActor : 0) | (sky ? kHoldSky : 0) | (player ? kHoldPlayer : 0));
			if (!full && !moved && !alwaysDraw && !light && !LeftOutFrame(a_context.clock) && EntryOutside(a_context, bound) &&
				ShadowBeyondReach(a_context, bound) && !SpotLampMayNeed(a_context, nullptr, bound) &&
				!(TraceFeed::On() && (a_trace == kTraceWalkNew ? !TraceFeed::Known(a_object) : TraceTurnDue(a_object, a_context.clock)) && TraceFeed::InReach(bound, a_context.eye)) &&
				!player && !sky) {
				Bump(kWalkSkipped);
				if (Recorder::On()) {
					Recorder::NoteEntry(Recorder::Origin::kWalkSkipped, a_object, bound, static_cast<std::uint8_t>(kind), static_cast<std::uint16_t>(RecorderFormat::kLeftOut | (a_group1 ? RecorderFormat::kGroup1 : 0)), a_loop);
				}
				if (g_tunables.keepFadeState) {
					KeepFadeSeen(a_object, a_context.clock, a_resumed);  // (as the top-level decision does for an entry out of view)
				}
				return;
			}
			Bump(kWalkOffered);
			PrefetchEntry(a_object);  // (the verdict path's streak backup and classification look its history entry up; v1.98)
			if (Recorder::On()) {
				Recorder::SetWalkSite(a_loop, static_cast<std::uint16_t>((moved ? RecorderFormat::kMoved : 0) | (full ? RecorderFormat::kTraceTurn : 0) | (alwaysDraw ? RecorderFormat::kAlwaysDraw : 0)));
			}
			// (its collection root, known when the reference was read: a root, or a reference to something else)
			const auto scol = static_cast<std::uint8_t>((a_slot.kind & kSlotScolRoot) ? (kKnownScolChecked | kKnownScolRoot) : (g_tunables.stableIndex && a_object->userData) ? kKnownScolChecked : 0);
			t_decisionGroup = group;
			const bool skipped = SkipTopLevelImpl(a_object, &bound, kind, static_cast<std::uint8_t>(kKnownClassified | (player ? kKnownPlayer : 0) | (sky ? kKnownSky : 0) | scol));
			t_decisionGroup = nullptr;
			if (TraceFeed::Linked() && TraceFeed::LastHanded() == a_object) {
				a_slot.handed = true;  // (a GONE record when it leaves the slot)
			}
			if (Recorder::On()) {
				Recorder::SetWalkSite(0, 0);  // (consumed by its note, or dropped with a decision that made none)
			}
			if (skipped) {
				return;
			}
			Bump(kWalkFiled);
			Hooks::CullGroups::AddDirect(group, a_object, &bound, 0);
		}

		bool SkipCellNode(RE::NiAVObject* a_node, std::uint32_t a_index);

		// The node's entries, in the engine's order and by its rules. False: the engine walks the node after all.
		bool WalkNodeEntries(const FrameContext& a_context, RE::NiAVObject* a_node, std::uint32_t a_index)
		{
			using Hooks::PrevisFeed::Site;
			const auto root = a_node->IsNode();
			if (!root) {
				return false;
			}
			const bool roots = a_index == Hooks::CullGroups::kRootIndex;
			if (roots && ((a_node->GetFlags() >> 14) & 1) && a_node->IsFadeNode()) {
				Bump(kWalkFadeRoots);
				return false;  // (the engine files such a root whole into the group array)
			}
			const bool containers = a_index == 3;                                   // a cell's node 3: exact-NiNode children are containers (R3c)
			const bool testCulled = a_index != Hooks::CullGroups::kContainerIndex;  // (a container's members get no AppCulled test, R3c)
			const auto site = Hooks::PrevisFeed::SiteAddress(roots ? Site::kR4Child : testCulled ? Site::kR3d : Site::kR3c);
			const bool traceWalk = t_traceNode.node && t_traceNode.clock == a_context.clock && (a_node == t_traceNode.node || a_node->parent == t_traceNode.node);
			const bool full = traceWalk && t_traceNode.all;
			const auto trace = static_cast<std::uint8_t>(full ? kTraceWalkAll : traceWalk ? kTraceWalkNew : kTraceWalkNone);
			const auto loop = static_cast<std::uint16_t>(roots ? 255 : a_index == Hooks::CullGroups::kContainerIndex ? 254 : a_index);  // (the recorder's site code)
			const auto& children = root->children;
			const auto  count = Util::UsedSlots(children);  // (the engine's bound, not size(): v1.92.11)
			auto&       cache = NodeCacheOf(root, count, a_context.clock);
			const bool  leases = g_tunables.walkLeases && !full && !LeftOutFrame(a_context.clock);
			const auto  group0 = Hooks::CullGroups::WalkGroup(false);
			const auto  group1 = Hooks::CullGroups::WalkGroup(true);
			// (v1.99.3: a cluster's child pointers and slots are fetched a cluster ahead; the slot reads were the walk's largest
			// stall in the v1.99.2 profile, ~9% of CBRO.dll's cull-stage samples)
			const auto fetchCluster = [&](std::size_t a_first) {
				const auto end = std::min(a_first + kClusterSlots, static_cast<std::size_t>(count));
				for (auto i = a_first; i < end; i += 8) {
					_mm_prefetch(reinterpret_cast<const char*>(&children[static_cast<std::uint16_t>(i)]), _MM_HINT_T0);
				}
				for (auto i = a_first; i < end; i += 2) {
					_mm_prefetch(reinterpret_cast<const char*>(&cache.slots[i]), _MM_HINT_T0);
				}
			};
			fetchCluster(0);
			for (std::size_t c = 0; c < cache.clusters.size(); ++c) {
				auto&      cluster = cache.clusters[c];
				const auto first = c * kClusterSlots;
				const auto last = std::min(first + kClusterSlots, static_cast<std::size_t>(count));
				if (last < count) {
					fetchCluster(last);
				}
				Bump(kWalkClusters);
				bool dormant = leases;
				if (dormant && !cluster.walked) {
					Bump(kWalkClusterNotWalked);
					dormant = false;
				}
				if (dormant && (cluster.hold || !(cluster.radius >= 0.0f))) {
					Bump(cluster.members == 0 ? kWalkClusterNoMembers : (cluster.hold & kHoldMarker) ? kWalkClusterHeldMarker : (cluster.hold & ~(kHoldBad | kHoldMoved)) ? kWalkClusterHeldKind : kWalkClusterHeldBad);
					dormant = false;
				}
				if (dormant && LeaseDue(a_context, cache, c, cluster)) {
					Bump(kWalkClusterDue);
					dormant = false;
				}
				if (dormant && ChildSignature(children, first, last) != cluster.children) {
					// (the slots no longer hold the objects the sphere was built from: told by the child array alone, v1.99.1)
					Bump(kWalkClusterSlotsChanged);
					dormant = false;
				}
				if (dormant && !ClusterOutside(a_context, cluster)) {
					Bump(kWalkClusterInView);
					dormant = false;
				}
				if (dormant) {
					const auto sphere = ClusterBound(cluster);
					if (!ShadowBeyondReach(a_context, sphere)) {
						Bump(kWalkClusterShadow);
						dormant = false;
					} else if (SpotLampMayNeed(a_context, nullptr, sphere)) {
						Bump(kWalkClusterLamp);
						dormant = false;
					} else if (Hooks::CullGroups::MarkerPendingAny(group0) || (roots && Hooks::CullGroups::MarkerPendingAny(group1))) {
						dormant = false;  // (the next entry filed must take the marker: as WalkEntry files such an entry)
						Bump(kWalkLeaseMarkerHeld);
					}
				}
				if (dormant) {
					Bump(kWalkClustersDormant);
					const bool traceReach = TraceFeed::On() && TraceFeed::InReach(ClusterBound(cluster), a_context.eye);
					for (auto i = first; i < last; ++i) {
						// (v1.99.1: the object from the child array, its kind from the cluster's masks: a slot is read only on the
						// trace's turn or for the recorder)
						const auto object = children[static_cast<std::uint16_t>(i)].get();
						if (!object) {
							continue;
						}
						const auto bit = static_cast<std::uint16_t>(1u << (i - first));
						if (cluster.containers & bit) {
							SkipCellNode(object, Hooks::CullGroups::kContainerIndex);  // (its own clusters decide)
							continue;
						}
						const bool group1Route = (cluster.group1 & bit) != 0;
						if (traceReach && (trace == kTraceWalkNew ? !TraceFeed::Known(object) : TraceTurnDue(object, a_context.clock)) &&
							TraceFeed::InReach(SlotBound(cache.slots[i]), a_context.eye)) {
							Bump(kWalkLeaseTraceVisits);
							WalkEntry(a_context, object, cache.slots[i], group1Route, trace, site, loop, true);
							continue;
						}
						Bump(kWalkEntries);
						Bump(kWalkLeased);
						Hooks::CullGroups::NoteEntryOffered(group1Route ? group1 : group0, object, site);
						if (Recorder::On()) {
							const auto kind = group1Route ? Hooks::CullGroups::GroupKind::kMainOnly : Hooks::CullGroups::GroupKind::kSunShared;
							Recorder::NoteEntry(Recorder::Origin::kWalkSkipped, object, SlotBound(cache.slots[i]), static_cast<std::uint8_t>(kind), static_cast<std::uint16_t>(RecorderFormat::kLeftOut | RecorderFormat::kLeased | (group1Route ? RecorderFormat::kGroup1 : 0)), loop);
						}
						if (g_tunables.keepFadeState && FadeMidWay(object)) {
							Bump(kWalkLeaseFadeVisits);
							KeepFadeSeen(object, a_context.clock);  // (its fade runs on: the one touch a lease allows)
						}
					}
					continue;
				}
				const bool resumed = leases && cluster.valid && a_context.clock - cluster.walkedClock > 1;
				for (auto i = first; i < last; ++i) {
					// (each entry's flags and bound lines are a first touch: the child a few slots ahead is fetched first)
					if (i + kScanPrefetch < count) {
						if (const auto ahead = reinterpret_cast<const char*>(children[static_cast<std::uint16_t>(i + kScanPrefetch)].get())) {
							_mm_prefetch(ahead + 0x80, _MM_HINT_T0);
							_mm_prefetch(ahead + 0x100, _MM_HINT_T0);
						}
					}
					const auto object = children[static_cast<std::uint16_t>(i)].get();
					auto&      slot = cache.slots[i];
					if (slot.object != object) {
						SlotLeft(slot);
						slot = WalkSlot{};
						slot.object = object;
					}
					if (!object) {
						continue;
					}
					const auto flags = object->GetFlags();
					if (testCulled && (flags & 1)) {
						Bump(kWalkCulled);
						SlotLeft(slot);  // (hidden from every view: gone for the tracer until shown again)
						// AppCulled: the engine's loop skips it before anything else. Its bound is read once for its cluster's
						// sphere (v1.98: Runtime Combiner's originals are many), never held by.
						slot.hold = 0;
						if (!slot.seen && UsableBound(object->worldBound)) {
							NoteSlotBound(slot, object->worldBound);
						}
						continue;
					}
					if (containers) {
						if (!(slot.kind & kSlotContainerKnown)) {
							slot.kind |= static_cast<std::uint8_t>(kSlotContainerKnown | (Util::TryReadVtable(object) == g_niNodeVtable ? kSlotContainer : 0));
						}
						if (slot.kind & kSlotContainer) {
							SkipCellNode(object, Hooks::CullGroups::kContainerIndex);  // (pruned whole, or its members walked here)
							continue;
						}
					}
					WalkEntry(a_context, object, slot, roots && ((flags >> 40) & 1), trace, site, loop, resumed);
				}
				FinishCluster(cache, c, a_context.clock, true);
			}
			Bump(kWalkNodes);
			return true;
		}

		// True: CBRO walked the node (the engine skips it). Timed as a whole, once per outermost node.
		bool WalkNode(const FrameContext& a_context, RE::NiAVObject* a_node, std::uint32_t a_index)
		{
			if (!Hooks::CullGroups::GroupAddHooked()) {
				return false;
			}
			if (a_context.clock - g_walkSweep >= kWalkSweepFrames) {
				g_walkSweep = a_context.clock;
				std::erase_if(g_walkNodes, [&](auto& a_entry) {
					if (a_context.clock - a_entry.second.clock < kWalkSweepFrames) {
						return false;
					}
					CacheLeft(a_entry.second);
					return true;
				});
			}
			const bool timed = t_walkDepth == 0 && Timed(a_context);
			const auto start = timed ? __rdtsc() : 0;
			++t_walkDepth;
			const bool walked = WalkNodeEntries(a_context, a_node, a_index);
			--t_walkDepth;
			if (timed) {
				Bump(kCyclesEntryWalk, (__rdtsc() - start) * kTimingStride);
			}
			return walked;
		}

		bool SkipCellNode(RE::NiAVObject* a_node, std::uint32_t a_index)
		{
			const auto context = CurrentContext();
			if (!context || Observing()) {
				return false;
			}
			if (a_index == 9) {
				Bump(kCellNodesOther);
				if (Recorder::On()) {
					Recorder::NoteNode(a_node, a_index, false, 0, 0, a_node->worldBound);
				}
				return g_tunables.entryWalk && WalkNode(*context, a_node, a_index);  // (never pruned whole; its entries walked by CBRO, v1.94)
			}
			const bool timed = Timed(*context);
			const auto start = timed ? __rdtsc() : 0;
			// A cell's node 3 is expanded two levels by the walk (its exact-NiNode children are containers); a
			// container or a DrawWorld root has its own children filed.
			const int depth = a_index == 3 ? 2 : 1;
			Bump(a_index == 3 ? kCellNodesSeen : a_index == Hooks::CullGroups::kContainerIndex ? kContainersSeen : kRootsSeen);
			// (the node's own always-draw bit means nothing here: the node itself is never an entry; its entries'
			// bits are what the scan looks for)
			if ((a_node->GetFlags() >> 11) & 1) {
				Bump(kCellNodesAlwaysDraw);
			}
			bool          skip = false;
			bool          outside = false;
			RE::NiBound   unionBound{};
			std::uint32_t entries = 0;
			// (never the node's own worldBound: unmaintained, see EntriesOutside)
			const auto contents = NodeContents(*context, a_node, depth);
			if (contents != 0) {
				Bump(kCellNodesHeld);
			} else if (!(outside = EntriesOutside(*context, a_node, depth, unionBound, entries))) {
				Bump(kCellNodesInView);
			} else if (!SunUnneeded(*context, a_node, unionBound)) {
				Bump(kCellNodesSunNeeded);
			} else if (SpotLampMayNeed(*context, nullptr, unionBound)) {
				Bump(kSpotLampNodesKept);  // (a spot lamp's group pass may file its entries: v1.54)
			} else if (TraceCellNode(*context, a_node, unionBound)) {
				Bump(kTraceCellNodes);  // (its entries come to SkipTopLevel and Block::Add, TraceKeep)
			} else {
				Bump(kCellNodesSkipped);
				Bump(kCellNodeEntriesSkipped, entries);
				if (t_prunedFromCache) {
					Bump(kCellNodesCached);
					Bump(kCellNodeEntriesCached, entries);
				}
				skip = true;
				if (LeftOutFrame(context->clock)) {
					RecordPrunedEntries(*context, a_node, depth);  // (the left-out dump, v1.92.7)
				}
			}
			if (Recorder::On()) {
				// (for a node, "player" stands for an actor among its entries)
				const auto flags = static_cast<std::uint16_t>(
					((contents & kSubtreeAlwaysDraw) ? RecorderFormat::kAlwaysDraw : 0) | ((contents & kSubtreeActor) ? RecorderFormat::kPlayer : 0) |
					((contents & kSubtreeLight) ? RecorderFormat::kLight : 0) | (outside ? RecorderFormat::kOutside : 0));
				Recorder::NoteNode(a_node, a_index, skip, entries, flags, unionBound);
			}
			if (g_nodeSamples.load(std::memory_order_relaxed) < 8) {
				SampleNode(a_node, a_index, contents, outside, entries, unionBound.fRadius);
			}
			if (timed) {
				const auto cycles = (__rdtsc() - start) * kTimingStride;
				Bump(kCycles, cycles);
				Bump(kCyclesNodeScan, cycles);
			}
			if (!skip && g_tunables.entryWalk) {
				return WalkNode(*context, a_node, a_index);  // (v1.94: the node's entries filed by CBRO itself, the engine's loop skipped)
			}
			return skip;
		}

		// ---- per-frame spread: how steady the culled set is from one frame to the next -----------------
		// Sampled once per culled frame at the cull stage's end (main thread, jobs done), from the running
		// counters. "Other views" = every accumulator's registrations (the hook count, which includes the sun's
		// cascades registered later in the previous frame and the lamps) less the main view's.

		struct Spread
		{
			double        min{ std::numeric_limits<double>::infinity() };
			double        max{ -std::numeric_limits<double>::infinity() };
			double        sum{ 0.0 };
			double        sumSq{ 0.0 };
			double        steps{ 0.0 };  // |value - previous frame's value|, summed
			std::uint32_t count{ 0 };
			double        last{ 0.0 };
			bool          hasLast{ false };

			void Add(double a_value) noexcept
			{
				min = std::min(min, a_value);
				max = std::max(max, a_value);
				sum += a_value;
				sumSq += a_value * a_value;
				if (hasLast) {
					steps += std::abs(a_value - last);
				}
				last = a_value;
				hasLast = true;
				++count;
			}

			std::string Describe() const
			{
				if (!count) {
					return "-";
				}
				const double avg = sum / count;
				const double sd = std::sqrt(std::max(0.0, sumSq / count - avg * avg));
				return std::format("{:.0f}/{:.0f}/{:.0f} sd {:.0f} step {:.0f}", min, avg, max, sd, count > 1 ? steps / (count - 1) : 0.0);
			}

			void Clear() noexcept { *this = Spread{}; }
		};

		struct FrameTotals
		{
			std::uint64_t registered{ 0 };
			std::uint64_t dropped{ 0 };
			std::uint64_t rejected{ 0 };
			std::uint64_t confirming{ 0 };
			std::uint64_t allRegistrations{ 0 };
		};
		FrameTotals   g_frameTotals;
		bool          g_frameTotalsKnown{ false };
		std::uint64_t g_edgeTotal{ 0 };
		std::uint64_t g_finalEdgeTotal{ 0 };
		std::uint64_t g_finalConfirmingTotal{ 0 };
		FrameSample   g_lastSample;
		std::uint64_t g_droppedTotal{ 0 };  // registrations dropped at emptied lamp shadow maps (running total)
		Spread        g_spreadKept;
		Spread        g_spreadOther;
		Spread        g_spreadRejected;
		Spread        g_spreadConfirming;

		// Diagnostic: frames whose drawn registrations (main view kept + lamp shadow maps filed) rise over the recent
		// median are logged with the flips behind them (a few sampled per kind) and the depth the frame judged by.
		struct DrawnHistory
		{
			std::array<float, 16>                   values{};
			std::uint32_t                           count{ 0 };
			std::uint32_t                           logged{ 0 };
			std::array<std::uint64_t, 3>            flipsThen{};
			std::uint32_t                           epoch{ 0 };
			std::uint32_t                           stillFrames{ 0 };
			std::uint32_t                           lastLogged{ 0 };
		};
		DrawnHistory            g_drawn;
		constexpr std::uint32_t kSpikeLogs = 16;         // per session
		constexpr float         kSpikeRise = 800.0f;     // registrations over the median (~400 draw calls)
		constexpr std::uint32_t kSpikeSettle = 120;      // still frames before any is logged
		constexpr std::uint32_t kSpikeSpacing = 30;      // frames between two logged

		void LogFlipSamples(std::uint32_t a_clock, FlipKind a_kind) noexcept
		{
			constexpr const char* kKinds[]{ "object", "point light", "spot light" };
			for (const auto& sample : g_flipRing) {
				if (sample.clock != a_clock || sample.kind != a_kind) {
					continue;
				}
				std::string scan;
				if (sample.scanned) {
					scan = sample.scan.texels ?
					           std::format(" | nearest at depth {:.0f}: level-0 texels beyond it {} (far plane {}, first-person {}; in the far-plane texels nearest surface {:.6f}, farthest drawn {:.6f}), farthest {:.6f}, first at ({:.0f},{:.0f})",
								   sample.nearest, sample.scan.texels, sample.scan.farPlane, sample.scan.firstPerson, sample.scan.farPlaneNearest, sample.scan.farPlaneDrawn, sample.scan.farthest, sample.scan.x, sample.scan.y) :
					           std::format(" | nearest at depth {:.0f}: no level-0 texel beyond it (a coarse texel alone)", sample.nearest);
				}
				logger::info(
					"  flip: {} {} at ({:.0f},{:.0f},{:.0f}) r {:.0f} -> {}{} | readback {}{}",
					kKinds[static_cast<std::size_t>(a_kind)], sample.type[0] ? sample.type : "", sample.bound[0], sample.bound[1], sample.bound[2], sample.bound[3],
					VerdictName(static_cast<Verdict>(sample.verdict)), scan, sample.readback,
					sample.heldSince ? std::format(" (hidden since readback {})", sample.heldSince) : std::string{});
			}
		}

		void NoteDrawn(double a_kept, double a_shadowFiled, double a_confirming) noexcept
		{
			const auto  context = CurrentContext();
			const float drawn = static_cast<float>(a_kept + a_shadowFiled);
			std::array<std::uint64_t, 3> flips{};
			for (std::size_t i = 0; i < flips.size(); ++i) {
				flips[i] = g_flipCount[i].load(std::memory_order_relaxed);
			}
			const auto flipsThen = std::exchange(g_drawn.flipsThen, flips);
			// Only a settled, still view: the same view epoch for a while and the view on the depth frame (the frames
			// after a load or a turn rise for good reasons; v1.50 spent its whole budget in a load's first second).
			const bool still = context && context->viewEpoch == g_drawn.epoch && context->view[0] == -1.0f && context->view[1] == 1.0f &&
			                   context->view[2] == -1.0f && context->view[3] == 1.0f;
			g_drawn.epoch = context ? context->viewEpoch : 0;
			if (!still) {
				g_drawn.stillFrames = 0;
				g_drawn.count = 0;
				return;
			}
			if (++g_drawn.stillFrames > kSpikeSettle && g_drawn.count >= g_drawn.values.size() && g_drawn.logged < kSpikeLogs &&
				context->clock - g_drawn.lastLogged >= kSpikeSpacing) {
				auto       recent = g_drawn.values;
				const auto n = recent.size();
				std::nth_element(recent.begin(), recent.begin() + n / 2, recent.end());
				const float median = recent[n / 2];
				if (drawn > median + kSpikeRise) {
					++g_drawn.logged;
					g_drawn.lastLogged = context->clock;
					const auto& snapshot = *context->snapshot;
					logger::info(
						"drawn-set spike {}/{}: clock {} | drawn registrations {:.0f} (recent median {:.0f}): main view kept {:.0f} (confirming {:.0f}), lamp shadow maps filed {:.0f} | flips: objects {}, point lights {}, spot lights {} | depth: readback {} from frame {} merged over {} | view [{:.3f},{:.3f}]x[{:.3f},{:.3f}], camera moved {:.1f}, view epoch {}",
						g_drawn.logged, kSpikeLogs, context->clock, drawn, median, a_kept, a_confirming, a_shadowFiled, flips[0] - flipsThen[0], flips[1] - flipsThen[1], flips[2] - flipsThen[2],
						context->readback, snapshot.frame, snapshot.mergedFrames, context->view[0], context->view[1], context->view[2], context->view[3], context->dilateMove, context->viewEpoch);
					LogFlipSamples(context->clock, FlipKind::kObject);
					LogFlipSamples(context->clock - 1, FlipKind::kPoint);
					LogFlipSamples(context->clock - 1, FlipKind::kSpot);
				}
			}
			g_drawn.values[g_drawn.count++ % g_drawn.values.size()] = drawn;
		}

		// ---- merge-instanced meshes (their instance entries) ------------------------------------

		// Hidden only if every instance is hidden (or outside the current view): the engine draws the
		// whole mesh as soon as one instance entry passes.
		bool DecideMerged(const FrameContext& a_context, const Hooks::CullGroups::BlockAdd& a_add, Entry& a_entry, std::uint64_t a_value)
		{
			Bump(kMerged);
			auto verdict = Test(a_context, *a_add.bound);  // the whole mesh: hidden there means hidden everywhere
			if (verdict == Verdict::kOutside) {
				a_entry.value.store(a_value & ~kMergedReject, std::memory_order_relaxed);
				return false;  // the frustum test drops it anyway
			}

			auto hint = static_cast<std::uint32_t>(a_value >> kHintShift);
			if (verdict != Verdict::kHidden) {
				const auto count = Hooks::CullGroups::InstanceCount(a_add.object);
				if (count > 0 && count <= kMaxInstanceTests) {
					verdict = Verdict::kHidden;
					// Start at the instance last seen visible: a visible mesh usually stops at the first test.
					for (std::uint32_t k = 0; k < count; ++k) {
						const auto i = (hint + k) % count;
						Bump(kInstanceTests);
						const auto instance = Test(a_context, *Hooks::CullGroups::InstanceBound(a_add.object, i));
						if (instance != Verdict::kHidden && instance != Verdict::kOutside) {
							verdict = instance;
							hint = i & 0xFF;
							break;
						}
					}
				}
			}

			const bool hidden = verdict == Verdict::kHidden;
			const auto streak = NextStreak(a_value, hidden, a_context.clock);
			const bool reject = hidden && streak >= g_tunables.confirmFrames && !Observing();
			if (hidden) {
				Bump(streak < g_tunables.confirmFrames ? kMergedConfirming : reject ? kMergedRejected : kWouldReject);
			}
			a_entry.value.store(
				(a_value & ~(kLowBits | kMergedReject | (std::uint64_t{ 0xFF } << kHintShift))) |
					(static_cast<std::uint64_t>(streak) << 32) | a_context.clock |
					(reject ? kMergedReject : 0) | (static_cast<std::uint64_t>(hint & 0xFF) << kHintShift),
				std::memory_order_relaxed);
			return reject;
		}

		void Instances(const Hooks::CullGroups::BlockAdd& a_add, std::uint32_t a_first, std::uint32_t a_end)
		{
			const auto context = CurrentContext();
			if (!context) {
				return;
			}
			const auto start = __rdtsc();
			Bump(kInstanceEntries, a_end - a_first);

			bool          reject = false;
			std::uint64_t value = 0;
			if (const auto entry = Lookup(*context, a_add.object, value); entry && (value & kIsMerged) && !(value & kExempt)) {
				if (a_add.startIndex < 0) {
					reject = DecideMerged(*context, a_add, *entry, value);
				} else {
					// A continuation in a new block: follow the decision made at the fresh add this frame.
					reject = (value & kMergedReject) && static_cast<std::uint32_t>(value) == context->clock;
				}
			}
			if (reject) {
				// (the whole mesh is filed when any instance entry passes: a main-view drop then, notes 5.2 item 6)
				t_moved = false;
				reject = !TraceDrop(*context, a_add.object, *a_add.bound);
			}
			if (reject) {
				for (auto i = a_first; i < a_end; ++i) {
					Hooks::CullGroups::WriteEntryBound(a_add.block, i, context->reject);
				}
				Bump(kInstanceEntriesRejected, a_end - a_first);
			}
			Bump(kCycles, __rdtsc() - start);
		}
	}

	void Install()
	{
		const auto& settings = Settings::Get();
		g_tunables.confirmFrames = settings.confirmFrames;
		g_tunables.nearDistance = settings.nearDistance;
		g_tunables.depthTolerance = settings.depthTolerance;
		g_tunables.depthSlack = settings.depthSlack;
		g_tunables.depthMin = settings.worldDepthMin;
		g_tunables.depthRange = settings.worldDepthMax - settings.worldDepthMin;
		g_tunables.observeOnly = settings.observeOnly;
		g_tunables.cullActors = settings.cullActors;
		g_tunables.cullPlayer = settings.cullPlayer;
		g_tunables.meshShapes = settings.meshShapes;
		g_tunables.lampGroupTrim = settings.lampGroupTrim;
		g_tunables.emptiedSpotTrim = settings.emptiedSpotTrim && settings.spotShadowCulling;
		g_tunables.spotOnlyCascadeTrim = settings.spotOnlyCascadeTrim;
		g_tunables.scolRootBounds = settings.scolRootBounds;
		g_tunables.keepFadeState = settings.keepFadeState;
		g_tunables.entryWalk = settings.entryWalk && settings.cellNodePruning;
		g_tunables.cheapChildren = settings.childCheapTests;
		g_tunables.childInherit = settings.childInherit;
		g_tunables.stableIndex = settings.stableIndex && settings.childInherit;
		g_tunables.traceDirect = settings.traceDirect;
		g_tunables.deferSunOnly = settings.deferSunOnly;
		if (g_tunables.deferSunOnly) {
			g_deferred.reserve(2048);
			Hooks::PrevisFeed::SetCascadeOpenCallback(&FileDeferred);
		}
		g_tunables.walkLeases = settings.walkLeases && settings.entryWalk && settings.cellNodePruning;
		g_tunables.leaseFrames = std::clamp(settings.walkLeaseFrames, 1u, 16u);
		g_tunables.leaseMargin = std::clamp(settings.walkLeaseMargin, 0.0f, 4096.0f);
		g_tunables.viewOverhang = settings.viewOverhang;
		g_tunables.viewOverhangTurn = settings.viewOverhangTurn;
		g_mergedVtable = RE::VTABLE::BSMergeInstancedTriShape[0].address();
		g_niNodeVtable = RE::VTABLE::NiNode[0].address();
		g_staticShapeVtables = { RE::VTABLE::BSTriShape[0].address(), RE::VTABLE::BSSubIndexTriShape[0].address(), RE::VTABLE::BSMeshLODTriShape[0].address() };
		g_fadeVtables = { RE::VTABLE::BSFadeNode[0].address(), RE::VTABLE::BSLeafAnimNode[0].address() };
		MeshProxy::Install();

		g_table = std::make_unique<Entry[]>(kTableSize);
		g_drops.Allocate();
		g_farDrops.Allocate();
		g_sunDrops.Allocate();
		g_traceForced.Allocate();
		Calibrate();
		Hooks::CullGroups::SetFilter(&Filter);
		Hooks::CullGroups::SetInstanceFilter(&Instances);
		Hooks::CullGroups::SetGroupFilter(&SkipTopLevel);
		Hooks::CullGroups::SetMainViewFilter(&MainView);
		Hooks::CullGroups::SetCellNodeFilter(&SkipCellNode);
		logger::info("occlusion: entry walk {} (bEntryWalk={}; needs bCellNodePruning=1 and the Group::Add hook)", g_tunables.entryWalk ? "on" : "off", settings.entryWalk);
		logger::info("occlusion: cheap child tests {} (bChildCheapTests={}): a static mesh child out of the view that no view needs is left out with no verdict", g_tunables.cheapChildren ? "on" : "off", settings.childCheapTests);
	}

	void NoteLightFlip(bool a_spot, const RE::NiPoint3& a_center, float a_radius, SphereVerdict a_verdict, SphereReason a_reason) noexcept
	{
		const auto context = CurrentContext();
		if (!context) {
			g_flipCount[static_cast<std::size_t>(a_spot ? FlipKind::kSpot : FlipKind::kPoint)].fetch_add(1, std::memory_order_relaxed);
			return;
		}
		// (the raw verdict again, for the sample: kVisible, or the reason it was unknown)
		const auto verdict = a_verdict == SphereVerdict::kVisible ? Verdict::kVisible :
		                     a_reason == SphereReason::kEdge      ? Verdict::kEdge :
		                     a_reason == SphereReason::kNear      ? Verdict::kNear :
		                                                            Verdict::kInvalid;
		RE::NiBound bound{};
		bound.center = a_center;
		bound.fRadius = a_radius;
		NoteFlip(*context, a_spot ? FlipKind::kSpot : FlipKind::kPoint, nullptr, bound, verdict, 0);
	}

	void EndFrameSample(std::uint64_t a_registrationsTotal, std::uint64_t a_droppedTotal)
	{
		const FrameTotals now{
			Total(kRegistered), Total(kRegistrationsDropped), Total(kRejected) + Total(kDropInherited),
			Total(kConfirming) + Total(kSunConfirming) + Total(kMergedConfirming), a_registrationsTotal
		};
		const bool culled = CurrentContext() != nullptr;
		const auto edge = Total(kEdge);
		const auto finalEdge = Total(kFinalEdge);
		const auto finalConfirming = Total(kFinalConfirming);
		g_lastSample = {};
		if (g_frameTotalsKnown && culled) {
			const auto   d = [](std::uint64_t a_now, std::uint64_t a_then) { return static_cast<double>(a_now - a_then); };
			const double registered = d(now.registered, g_frameTotals.registered);
			const double kept = registered - d(now.dropped, g_frameTotals.dropped);
			const double other = d(now.allRegistrations, g_frameTotals.allRegistrations) - registered;
			g_spreadKept.Add(kept);
			g_spreadOther.Add(other);
			g_spreadRejected.Add(d(now.rejected, g_frameTotals.rejected));
			g_spreadConfirming.Add(d(now.confirming, g_frameTotals.confirming));
			NoteDrawn(kept, other - d(a_droppedTotal, g_droppedTotal), d(now.confirming, g_frameTotals.confirming));
			g_lastSample = { true, static_cast<float>(kept), static_cast<float>(other - d(a_droppedTotal, g_droppedTotal)),
				static_cast<float>(d(now.confirming, g_frameTotals.confirming)), static_cast<float>(d(edge, g_edgeTotal)),
				static_cast<float>(d(finalEdge, g_finalEdgeTotal)), static_cast<float>(d(finalConfirming, g_finalConfirmingTotal)) };
		}
		g_edgeTotal = edge;
		g_finalEdgeTotal = finalEdge;
		g_finalConfirmingTotal = finalConfirming;
		g_frameTotals = now;
		g_droppedTotal = a_droppedTotal;
		g_frameTotalsKnown = culled;  // (two culled frames in a row give a per-frame delta)
	}

	FrameSample LastFrameSample() noexcept
	{
		return g_lastSample;
	}

	void BeginFrame(const FrameContext& a_context)
	{
		g_mainStats = &LocalStats();  // (BeginFrame runs on the main thread)
		Hooks::CullGroups::SetFrameEye(a_context.eye, a_context.cull && (a_context.eye[0] != 0.0f || a_context.eye[1] != 0.0f || a_context.eye[2] != 0.0f));
		g_mainThreadId = GetCurrentThreadId();
		ClearParentSequences();
		if (!g_deferred.empty()) {
			Bump(kSunOnlyDropped, g_deferred.size());  // (no cascade window last frame: never filed)
			g_deferred.clear();
		}
		if (const auto scene = static_cast<const std::uint8_t*>(Hooks::CullGroups::SceneNode()); scene && scene[0x228]) {
			Bump(kFrameLightingByte);  // (v1.94 survey: OnVisible re-queues every visible fade node's lighting while it is set)
		}
		g_flipClock.store(a_context.clock, std::memory_order_relaxed);  // (flip samples: a few per kind per frame)
		for (auto& taken : g_flipTaken) {
			taken.store(0, std::memory_order_relaxed);
		}
		const bool reset = g_resetRequested.exchange(false);
		if (const auto used = g_used.load(); reset) {
			ClearTable();
		} else if (used > kTableSize * 7 / 10) {
			// (rare: logged each time, so a run shows how often a long session compacts, and whether it ever clears)
			const auto kept = CompactTable(a_context.clock);
			if (kept > kTableSize / 2) {
				logger::info("occlusion history: table full ({} of {} objects, {} of them judged in the last {} frames); cleared, every streak restarts", used, kTableSize, kept, kTableKeepFrames);
				ClearTable();
			} else {
				logger::info("occlusion history: table full ({} of {} objects); compacted to the {} judged in the last {} frames", used, kTableSize, kept, kTableKeepFrames);
			}
		}
		// Drop tags repeat every kTagCycle frames: forget the old ones before a tag is reused.
		if (const auto cycle = a_context.clock / DropSet::kTagCycle; cycle != g_dropCycle) {
			g_dropCycle = cycle;
			g_drops.Clear();
			g_farDrops.Clear();
			g_sunDrops.Clear();
			g_traceForced.Clear();
		}
		// (the sets' presence bits for this frame's tag, before the context below publishes its clock)
		const auto dropTag = DropSet::Tag(a_context.clock);
		g_drops.BeginTag(dropTag);
		g_farDrops.BeginTag(dropTag);
		g_sunDrops.BeginTag(dropTag);
		g_traceForced.BeginTag(dropTag);
		MeshProxy::BeginFrame();

		if (a_context.parallaxMove > 0.0f || a_context.readback) {
			auto& at = g_judgedEyes[a_context.readback & 63];
			at.readback = a_context.readback;
			std::copy_n(a_context.eye, 3, at.eye);
		}
		const auto next = (g_context.load() + 1 + 2) % 2;
		g_contexts[next] = a_context;
		ReadPlayer(g_contexts[next]);
		SetOverhangs(g_contexts[next]);
		SetSunPlanes(g_contexts[next]);
		SetLampPlanes(g_contexts[next]);
		g_context.store(next, std::memory_order_release);
		GatherSpotLamps();  // (Runtime published last frame's lamps just before; its emptied lamps are re-tested on this context)
		Bump(a_context.cull ? kFramesCulling : kFramesBlocked);
	}

	void SetActive(bool a_active)
	{
		g_active.store(a_active);
	}

	void JudgeAsync(const FrameContext& a_context, const Hooks::CullGroups::BlockAdd& a_add, const Record* a_old, bool a_wantSun, Record& a_out)
	{
		if (!Timed(a_context)) {
			JudgeRecord(a_context, a_add, a_old, a_wantSun, a_out);
			return;
		}
		const auto start = __rdtsc();
		JudgeRecord(a_context, a_add, a_old, a_wantSun, a_out);
		Bump(kCycles, (__rdtsc() - start) * kTimingStride);  // (the worker's own stats block: "all threads", never the main thread's share)
	}

	void SetFarCascade(const void* a_accumulator) noexcept
	{
		Hooks::CullGroups::SetFarCascade(a_accumulator, a_accumulator ? &FarCascade : nullptr);
	}

	void SetSunCascades(const void* const* a_accumulators, std::uint32_t a_count) noexcept
	{
		const bool on = a_accumulators && a_count && (g_tunables.spotOnlyCascadeTrim || TraceFeed::On());
		Hooks::CullGroups::SetSunCascades(on ? a_accumulators : nullptr, on ? a_count : 0, on ? &SunCascade : nullptr);
	}

	void PrepareContext(FrameContext& a_context) noexcept
	{
		SetOverhangs(a_context);
		SetSunPlanes(a_context);
		SetLampPlanes(a_context);
	}

	void SetKeptReasons(bool a_on) noexcept
	{
		g_keptReasons.store(a_on, std::memory_order_relaxed);
	}

	KeptReason LastKeptReason(const void* a_object) noexcept
	{
		KeptReason out{};
		if (!g_table || !g_keptReasons.load(std::memory_order_relaxed)) {
			return out;
		}
		const auto entry = PeekEntry(reinterpret_cast<std::uintptr_t>(a_object));
		if (!entry) {
			return out;
		}
		const auto reason = (entry->lamp.load(std::memory_order_relaxed) & kReasonMask) >> kReasonShift;
		if (!(reason & 1u) || ((reason >> 10) & 0x3FFu) != (Clock() & 0x3FFu)) {
			return out;
		}
		out.valid = true;
		out.verdict = static_cast<std::uint8_t>((reason >> 1) & 7u);
		const auto outcome = static_cast<Outcome>((reason >> 4) & 3u);
		out.confirming = outcome == Outcome::kHidden;
		out.exempt = outcome == Outcome::kKept && out.verdict == static_cast<std::uint8_t>(Verdict::kHidden);
		out.shaped = ((reason >> 6) & 1u) != 0;
		out.shapeNone = ((reason >> 7) & 1u) != 0;
		out.parallaxKept = ((reason >> 8) & 1u) != 0;
		return out;
	}

	void SetObserveOnly(bool a_observeOnly)
	{
		g_observeOverride.store(a_observeOnly);
	}

	SphereVerdict TestSphere(const RE::NiPoint3& a_center, float a_radius, SphereReason* a_reason) noexcept
	{
		const auto reason = [&](SphereReason a_value) {
			if (a_reason) {
				*a_reason = a_value;
			}
		};
		reason(SphereReason::kNone);
		const auto context = CurrentContext();
		if (!context) {
			reason(SphereReason::kNoDepth);
			return SphereVerdict::kUnknown;
		}
		if (FirstPersonUnseen(*context) && ReachNearEye(*context, a_center, a_radius)) {
			reason(SphereReason::kNear);  // may light the first-person geometry
			return SphereVerdict::kUnknown;
		}
		RE::NiBound bound{};
		bound.center = a_center;
		bound.fRadius = a_radius;
		LightTest light{};
		switch (Test(*context, bound, nullptr, &light)) {
		case Verdict::kHidden:
			reason(light.emptyReach ? SphereReason::kEmptyReach : light.overhung ? SphereReason::kOverhang : SphereReason::kNone);
			return SphereVerdict::kHidden;
		case Verdict::kVisible:
			return SphereVerdict::kVisible;
		case Verdict::kOutside:
		case Verdict::kBehind:  // (entirely behind the eye: nothing visible is in it)
			return SphereVerdict::kOutOfView;
		case Verdict::kEdge:
			reason(SphereReason::kEdge);
			return SphereVerdict::kUnknown;
		case Verdict::kNear:
			reason(SphereReason::kNear);
			return SphereVerdict::kUnknown;
		default:
			reason(SphereReason::kInvalid);
			return SphereVerdict::kUnknown;
		}
	}

	std::uint32_t Clock() noexcept
	{
		const auto index = g_context.load(std::memory_order_acquire);
		return index < 0 ? 0 : g_contexts[index].clock;
	}

	// A lamp caster's shadow is the cone from the lamp through the caster's sphere, from the caster's far side out
	// to the lamp's reach: a sweep along (caster - lamp) whose radius grows with the distance (SweepClip). Clipped to
	// where receivers can be (in front of the eye, inside the view), the kept part is a segment between two spheres;
	// it shadows nothing visible if every visible surface over its screen box is in front of its nearest point
	// (the max pyramid), or behind its farthest point (the nearest pyramid). Either way no surface is inside the
	// volume. Depth verdicts hold only after confirmFrames consecutive frames, like an object's.
	LampVerdict TestLampCaster(RE::NiAVObject* a_caster, const RE::NiPoint3& a_lamp, float a_reach, const RE::NiBound& a_bound) noexcept
	{
		const auto context = CurrentContext();
		if (!context) {
			return LampVerdict::kUnknown;
		}
		return TestLampCasterIn(*context, a_caster, a_lamp, a_reach, a_bound);
	}

	LampVerdict TestLampCasterIn(const FrameContext& a_context, RE::NiAVObject* a_caster, const RE::NiPoint3& a_lamp, float a_reach, const RE::NiBound& a_bound) noexcept
	{
		const auto context = &a_context;
		if (!context->snapshot || context->lampPlaneCount == 0) {
			return LampVerdict::kUnknown;
		}
		const auto& snapshot = *context->snapshot;
		const auto& camera = snapshot.camera;
		const float radius = a_bound.fRadius;
		if (!camera.viewSpace || !(camera.depthB < 0.0f) || !(radius > 0.0f) || !(radius < 1.0e6f) || !(a_reach > 0.0f) ||
			!std::isfinite(a_bound.center.x) || !std::isfinite(a_bound.center.y) || !std::isfinite(a_bound.center.z)) {
			return LampVerdict::kUnknown;
		}
		const auto toView = [&](const RE::NiPoint3& a_p, float a_out[3]) {
			const float dx = a_p.x - camera.origin[0];
			const float dy = a_p.y - camera.origin[1];
			const float dz = a_p.z - camera.origin[2];
			a_out[0] = dx * camera.viewRight[0] + dy * camera.viewRight[1] + dz * camera.viewRight[2];
			a_out[1] = dx * camera.viewUp[0] + dy * camera.viewUp[1] + dz * camera.viewUp[2];
			a_out[2] = dx * camera.viewDir[0] + dy * camera.viewDir[1] + dz * camera.viewDir[2];
		};
		float lamp[3], center[3];
		toView(a_lamp, lamp);
		toView(a_bound.center, center);
		const float r = radius + context->dilateMove;
		const float d[3]{ center[0] - lamp[0], center[1] - lamp[1], center[2] - lamp[2] };
		const float dist = std::sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2]);
		if (dist <= r + 1.0f) {
			return LampVerdict::kNeeded;  // the lamp is inside the caster: shadow everywhere
		}
		if (dist - r >= a_reach) {
			return LampVerdict::kOutside;  // entirely beyond the lamp's reach
		}
		const float dir[3]{ d[0] / dist, d[1] / dist, d[2] / dist };
		const float spread = r / dist;             // the cone widens with the distance from the lamp
		const float tMax = a_reach - dist + r;     // no light beyond the reach: no shadow either
		if (FirstPersonUnseen(*context) && SweepNearEye(center, dir, r, spread, tMax, kFirstPersonReach)) {
			return LampVerdict::kNeeded;  // may shadow the first-person geometry
		}
		float       t0 = 0.0f, t1 = 0.0f;
		if (!ShadowGeometry::SweepClip(center, dir, r, spread, tMax, context->lampPlanes, context->lampPlaneCount, t0, t1)) {
			return LampVerdict::kOutside;
		}
		const float q0[3]{ center[0] + t0 * dir[0], center[1] + t0 * dir[1], center[2] + t0 * dir[2] };
		const float q1[3]{ center[0] + t1 * dir[0], center[1] + t1 * dir[1], center[2] + t1 * dir[2] };
		const float ra = r + t0 * spread;
		const float rb = r + t1 * spread;
		const float zMin = std::min(q0[2] - ra, q1[2] - rb);
		const float zMax = std::max(q0[2] + ra, q1[2] + rb);
		if (zMin < g_tunables.nearDistance) {
			return LampVerdict::kNeeded;  // reaches the camera: can't be judged
		}
		float x0, x1, y0, y1, u0, u1, v0, v1;
		SphereExtent(q0[0], q0[2], ra, x0, x1);
		SphereExtent(q0[1], q0[2], ra, y0, y1);
		SphereExtent(q1[0], q1[2], rb, u0, u1);
		SphereExtent(q1[1], q1[2], rb, v0, v1);
		x0 = std::min(x0, u0) * camera.scaleX;
		x1 = std::max(x1, u1) * camera.scaleX;
		y0 = std::min(y0, v0) * camera.scaleY;
		y1 = std::max(y1, v1) * camera.scaleY;
		switch (ClipToView(*context, x0, x1, y0, y1)) {
		case Verdict::kOutside:
			return LampVerdict::kOutside;
		case Verdict::kEdge:
			return LampVerdict::kNeeded;
		default:
			break;
		}
		const float width = static_cast<float>(snapshot.width[0]);
		const float height = static_cast<float>(snapshot.height[0]);
		const float px0 = (x0 * 0.5f + 0.5f) * width - 0.25f;
		const float py0 = (0.5f - y1 * 0.5f) * height - 0.25f;
		const float px1 = (x1 * 0.5f + 0.5f) * width + 0.25f;
		const float py1 = (0.5f - y0 * 0.5f) * height + 0.25f;
		const auto  bufferDepth = [&](float a_z) { return std::min(g_tunables.depthMin + g_tunables.depthRange * (camera.depthA + camera.depthB / a_z), 0.999999f); };

		const float limitBehind = (zMin - g_tunables.depthSlack) / (1.0f + g_tunables.depthTolerance);
		const float limitFront = (zMax + g_tunables.depthSlack) * (1.0f + g_tunables.depthTolerance);
		// No visible surface inside the volume's depth range at any texel of its box: each texel's surfaces are all in
		// front of the volume or all behind it (the sky counts as behind). Whole-box "all behind" or "all in front"
		// (v1.18) almost never held: a volume crossing a wall's silhouette into the sky failed both.
		if (!(limitBehind > 0.0f) || !snapshot.NoSurfaceBetween(px0, py0, px1, py1, bufferDepth(limitBehind), bufferDepth(limitFront), kRefineLevels)) {
			return LampVerdict::kNeeded;
		}
		const auto entry = FindEntry(reinterpret_cast<std::uintptr_t>(a_caster));
		if (!entry) {
			Bump(kTableFull);
			return LampVerdict::kNeeded;
		}
		const auto lampWord = entry->lamp.load(std::memory_order_relaxed);
		const auto streak = HiddenStreak(lampWord, context->clock);
		entry->lamp.store((lampWord & ~kLowBits) | (static_cast<std::uint64_t>(streak) << 32) | context->clock, std::memory_order_relaxed);  // (the high bits: the set-diff reasons)
		return streak < g_tunables.confirmFrames ? LampVerdict::kConfirming : LampVerdict::kMisses;
	}

	bool IsPlayerPart(const RE::NiAVObject* a_object, const RE::NiBound& a_bound) noexcept
	{
		const auto context = CurrentContext();
		return context && PlayerPart(*context, a_object, a_bound);
	}

	void CountLampVerdict(LampVerdict a_verdict) noexcept
	{
		switch (a_verdict) {
		case LampVerdict::kOutside:
			Bump(kLampVolumeOutside);
			break;
		case LampVerdict::kMisses:
			Bump(kLampVolumeMisses);
			break;
		case LampVerdict::kConfirming:
			Bump(kLampVolumeConfirming);
			break;
		case LampVerdict::kNeeded:
			Bump(kLampVolumeNeeded);
			break;
		default:
			Bump(kLampVolumeUnknown);
			break;
		}
	}

	bool ViewCone(ShadowGeometry::Cone& a_cone, float& a_push) noexcept
	{
		const auto context = CurrentContext();
		if (!context || !context->coneValid) {
			return false;
		}
		a_cone = context->cone;
		a_push = context->conePush;
		return true;
	}

	bool Deciding() noexcept
	{
		return Observing();
	}

	bool Active() noexcept
	{
		return g_active.load(std::memory_order_relaxed);
	}

	void ResetHistory()
	{
		g_resetRequested.store(true);
		g_streamGeneration.fetch_add(1, std::memory_order_relaxed);  // (every thread's verdict streams start over)
	}

	void RequestKeptDump()
	{
		// The left-out dump takes the next frame as it comes (verdicts reused as usual); the kept dump the one after,
		// on which nothing is reused (so every kept object is evaluated, and recorded).
		std::uint32_t expected = 0;
		if (g_dumpClock.compare_exchange_strong(expected, Clock() + 2) && Settings::Get().leftOutDump) {
			g_leftOutClock.store(Clock() + 1);
		}
	}

	float RayWindowHitOnDepthFrame(float a_x, float a_y, float a_z, float a_radius) noexcept
	{
		std::scoped_lock lock(g_keptLock);
		return g_rayCameraValid ? RayWindowHit(g_rayCamera, a_x, a_y, a_z, a_radius) : -1.0f;
	}

	namespace
	{
		// v1.92.10: the ray-window triangles of a left-out mesh (BruteShape), for the dump line.
		std::string RayWindowText(const LeftOutItem& a_item)
		{
			if (a_item.bruteWindow == 0) {
				return a_item.bruteRead > 0 ? std::string(" [ray window: no triangle crosses it]") : std::string{};
			}
			std::string text = std::format(" [ray window: {} triangles cross it, {} of them visible;", a_item.bruteWindow, a_item.bruteWindowVisible);
			for (std::uint8_t i = 0; i < a_item.windowSamples; ++i) {
				const auto& s = a_item.windowSample[i];
				text += std::format(" {}: view depth {:.0f}..{:.0f} at ndc ({:.3f},{:.3f})-({:.3f},{:.3f}), farthest surface over it {:.0f}, threshold {:.0f} -> {}{}",
					i + 1, s.zNear, s.zFar, s.ndc[0], s.ndc[1], s.ndc[2], s.ndc[3], s.farthest, s.threshold, FullVerdictName(s.verdict), i + 1 < a_item.windowSamples ? ";" : "");
			}
			return text + "]";
		}

		constexpr float kNearReach = 1500.0f;  // the left-out dump's and the scene scan's "near the eye" (bound within this)

		void LogLeftOut(std::vector<LeftOutItem>& a_items, std::uint32_t a_clock, const HiZ::Camera* a_camera)
		{
			constexpr const char* kOutcomeNames[]{ "none", "out of view", "hidden", "kept" };
			std::array<std::uint32_t, 9> byPath{};
			std::array<std::uint32_t, std::size(kVerdictSourceNames)> bySource{};
			std::map<std::string, std::uint32_t> byNow;
			std::uint32_t checked = 0, outsideBound = 0, ownMismatch = 0;
			const auto suspicious = [](const LeftOutItem& a_item) {
				const auto verdict = a_item.shaped && a_item.shapeNow != Verdict::kInvalid ? a_item.shapeNow : a_item.now;
				const bool seenNow = verdict != Verdict::kHidden && verdict != Verdict::kOutside && verdict != Verdict::kBehind;
				const bool ownOff = a_item.ownOffset > std::max(1.0f, a_item.bound[3] * 0.01f) || std::abs(a_item.ownRadius - a_item.bound[3]) > std::max(1.0f, a_item.bound[3] * 0.01f);
				const bool shapeWrong = a_item.bruteRead > 0 && (a_item.shapeNow == Verdict::kOutside ? a_item.bruteInView > 0 : a_item.bruteVisible > 0);
				return seenNow || a_item.verticesOutside > 0 || ownOff || shapeWrong;
			};
			for (const auto& item : a_items) {
				++byPath[static_cast<std::size_t>(item.path)];
				++bySource[static_cast<std::size_t>(item.source)];
				++byNow[FullVerdictName(item.now)];
				checked += item.vertices > 0;
				outsideBound += item.verticesOutside > 0;
				ownMismatch += item.ownOffset > std::max(1.0f, item.bound[3] * 0.01f) || std::abs(item.ownRadius - item.bound[3]) > std::max(1.0f, item.bound[3] * 0.01f);
			}
			std::string paths, sources, now;
			for (std::size_t i = 0; i < byPath.size(); ++i) {
				if (byPath[i]) {
					paths += std::format(" {} {} |", kLeftOutPathNames[i], byPath[i]);
				}
			}
			for (std::size_t i = 0; i < bySource.size(); ++i) {
				sources += std::format(" {} {}", kVerdictSourceNames[i], bySource[i]);
			}
			for (const auto& [name, count] : byNow) {
				now += std::format(" {} {}", name, count);
			}
			logger::info(
				"==== left out of the main view on one steady frame (clock {}): {} objects ====|{} verdicts:{} | a fresh test of the same bound now:{} | meshes checked against their bound: {}, with vertices outside it {} | own world bound differs from the bound offered: {}",
				a_clock, a_items.size(), paths, sources, now, checked, outsideBound, ownMismatch);

			// Suspicious first (seen by a fresh test, vertices outside the bound, a bound other than the object's own),
			// then the nearest ones whose bound center is on screen. Screen = NDC of the depth frame's camera, x right,
			// y up, [-1, 1].
			std::ranges::sort(a_items, [](const LeftOutItem& a_left, const LeftOutItem& a_right) { return a_left.distance < a_right.distance; });
			const auto line = [&](const LeftOutItem& a_item, const char* a_tag) {
				std::string vertices = "not checked";
				if (a_item.vertices > 0) {
					vertices = a_item.verticesOutside ?
					               std::format("{} of {} outside the bound by up to {:.0f}, their box at screen ({:.3f},{:.3f}){}", a_item.verticesOutside, a_item.vertices, a_item.excess,
									   a_item.vertexNdc[0], a_item.vertexNdc[1], a_item.vertexInFront ? "" : " behind the camera") :
					               std::format("all {} inside the bound", a_item.vertices);
					vertices += std::format(", {} in view", a_item.verticesInView);
				}
				vertices += std::format(" | bound fits its transform: {} | desc flags 0x{:X}", a_item.boundFits ? "yes" : "NO", a_item.descFlags);
				logger::info(
					"  {} {} ({}) | {} '{}' {:X} in {} {:X} | ref {:08X} base {:08X} (form type {}) | bound ({:.0f},{:.0f},{:.0f}) r={:.0f} dist={:.0f} screen ({:.3f},{:.3f}){} view depth {:.0f}, farthest surface over it {:.0f} | record {} ({}) streak {} | now {}{} | own world bound {:.0f} off, r={:.0f} | vertices: {}",
					a_tag, kLeftOutPathNames[static_cast<std::size_t>(a_item.path)], kVerdictSourceNames[static_cast<std::size_t>(a_item.source)], a_item.type, a_item.name,
					reinterpret_cast<std::uintptr_t>(a_item.object), a_item.ancestry, reinterpret_cast<std::uintptr_t>(a_item.parent), a_item.refID, a_item.baseID, a_item.baseType,
					a_item.bound[0], a_item.bound[1], a_item.bound[2], a_item.bound[3], a_item.distance, a_item.ndc[0], a_item.ndc[1], a_item.inFront ? "" : " (behind the camera)",
					a_item.viewDepth, a_item.occluder, kOutcomeNames[static_cast<std::size_t>(a_item.outcome) & 3],
					FullVerdictName(static_cast<Verdict>(a_item.verdict)), a_item.streak, FullVerdictName(a_item.now),
					(a_item.now == Verdict::kVisible ? std::format(" ({} level-0 texels beyond its nearest point)", a_item.fartherTexels) : std::string{}) +
						(a_item.shaped ? std::format(", by its mesh shape {}", FullVerdictName(a_item.shapeNow)) : std::string{}) +
							(a_item.bruteRead < 0 ? std::string{} :
							 a_item.bruteRead == 0 ? std::string(" [brute: mesh unreadable]") :
							 std::format(" [brute: cached {} fresh {} same {} | shape {}x{}x{} cells {} | triangles {}, in view {} (not hidden {}), in view in no occupied cell {}]",
								 a_item.bruteCached, a_item.bruteFresh, a_item.bruteSame, a_item.bruteDims[0], a_item.bruteDims[1], a_item.bruteDims[2], a_item.bruteCount,
								 a_item.bruteTriangles, a_item.bruteInView, a_item.bruteVisible, a_item.bruteUnoccupied)) +
								RayWindowText(a_item),
					a_item.ownOffset, a_item.ownRadius, vertices);
			};
			// First every one whose bound crosses the ray window (nearest along the rays first; may repeat below).
			if (a_camera) {
				std::vector<std::pair<float, const LeftOutItem*>> hits;
				for (const auto& item : a_items) {
					if (const float t = RayWindowHit(*a_camera, item.bound[0], item.bound[1], item.bound[2], item.bound[3]); t >= 0.0f) {
						hits.emplace_back(t, &item);
					}
				}
				std::ranges::sort(hits, [](const auto& a_left, const auto& a_right) { return a_left.first < a_right.first; });
				logger::info("==== left out, bound crossing the ray window (NDC x {:.2f}..{:.2f}, y {:.2f}..{:.2f}, up to {:.0f} along): {} ====",
					RayWindow()[0], RayWindow()[1], RayWindow()[2], RayWindow()[3], RayLength(), hits.size());
				for (std::size_t i = 0; i < hits.size() && i < 250; ++i) {
					line(*hits[i].second, "x");
				}
			}
			std::uint32_t printed = 0;
			for (auto& item : a_items) {
				if (printed >= 150) {
					break;
				}
				if (suspicious(item)) {
					line(item, "!");
					item.object = nullptr;  // (printed)
					++printed;
				}
			}
			// Then every one whose sphere reaches the view (center in front, its screen extent touching the frame), largest
			// on screen first (v1.92.7: up to v1.92.6 the middle third of the screen nearest first, which left out a far object
			// missing at the 2026-10-26 A/B spot).
			{
				std::vector<LeftOutItem*> inView;
				for (auto& item : a_items) {
					if (!item.object || !item.inFront || !(item.viewDepth > 0.0f)) {
						continue;
					}
					const float ex = a_camera ? item.bound[3] / item.viewDepth * a_camera->scaleX : 0.0f;
					const float ey = a_camera ? item.bound[3] / item.viewDepth * a_camera->scaleY : 0.0f;
					if (std::abs(item.ndc[0]) - ex <= 1.0f && std::abs(item.ndc[1]) - ey <= 1.0f) {
						inView.push_back(&item);
					}
				}
				std::ranges::sort(inView, [](const LeftOutItem* a_left, const LeftOutItem* a_right) {
					return a_left->bound[3] / std::max(a_left->viewDepth, 1.0f) > a_right->bound[3] / std::max(a_right->viewDepth, 1.0f);
				});
				logger::info("==== left out, sphere reaching the view: {} (largest on screen first, up to 400) ====", inView.size());
				printed = 0;
				for (auto* item : inView) {
					if (printed >= 400) {
						break;
					}
					line(*item, "-");
					item->object = nullptr;
					++printed;
				}
			}
			// Then the ones whose bound comes within kNearReach of the eye, wherever on (or off) the screen (2026-10-09: the
			// lower part of a wall beside the camera missing at the screen's left edge).
			printed = 0;
			for (const auto& item : a_items) {
				if (printed >= 150) {
					break;
				}
				if (item.object && item.distance - item.bound[3] <= kNearReach) {
					line(item, "~");
					++printed;
				}
			}
		}

		// Every mesh in the scene graph whose world bound comes within kNearReach of the eye (up to v1.81 a window around the
		// screen's upper middle: the 2026-10-05 spot's missing traffic-signal heads), with its path from the scene root
		// (child index, type, name), AppCulled ancestors, and whether the main accumulator registered it on the left-out
		// frame. Main thread, at a frame start.
		void ScanScene(const HiZ::Camera& a_camera, const std::unordered_set<const void*>& a_registered, const std::unordered_map<const void*, std::uint8_t>& a_offered)
		{
			const auto root = *reinterpret_cast<RE::NiNode* const*>(CBRO::Engine::OG(1327069).address());
			if (!root) {
				return;
			}
			struct Scan
			{
				const HiZ::Camera&                                   camera;
				const std::unordered_set<const void*>&               registered;
				const std::unordered_map<const void*, std::uint8_t>& offered;
				std::vector<std::string>                             path;
				std::vector<std::string>               lines;
				std::vector<std::pair<float, std::string>> rayLines;  // (bound crossing the ray window)
				std::uint32_t                          visited{ 0 }, found{ 0 }, foundRegistered{ 0 };
				int                                    culledAt{ -1 };

				void Visit(const RE::NiAVObject* a_object, int a_index)
				{
					if (!a_object || ++visited > 4'000'000 || path.size() >= 40) {
						return;
					}
					char type[32]{}, name[40]{};
					Util::TryGetRTTIName(a_object, type, sizeof(type));
					Util::TryGetObjectName(a_object, name, sizeof(name));
					const bool culled = (a_object->GetFlags() & 1) != 0;
					const auto how = offered.find(a_object);
					const auto mark = how == offered.end() ? "" : how->second == 1 ? "[entry]" : how->second == 2 ? "[top-level]" : "[top-level+entry]";
					path.push_back(std::format("{}:{}'{}'{}{}", a_index, type, name, culled ? "(AppCulled)" : "", mark));
					const int savedCulled = culledAt;
					if (culled && culledAt < 0) {
						culledAt = static_cast<int>(path.size()) - 1;
					}
					if (const auto node = const_cast<RE::NiAVObject*>(a_object)->IsNode()) {
						int i = 0;
						for (auto& child : node->children) {
							Visit(child.get(), i++);
						}
					} else if (const_cast<RE::NiAVObject*>(a_object)->IsGeometry()) {
						Check(a_object, type, name);
					}
					culledAt = savedCulled;
					path.pop_back();
				}

				void Check(const RE::NiAVObject* a_object, const char* a_type, const char* a_name)
				{
					const auto& bound = a_object->worldBound;
					const float dx = bound.center.x - camera.origin[0], dy = bound.center.y - camera.origin[1], dz = bound.center.z - camera.origin[2];
					const float rayHit = RayWindowHit(camera, bound.center.x, bound.center.y, bound.center.z, bound.fRadius);
					if ((!(std::sqrt(dx * dx + dy * dy + dz * dz) - bound.fRadius <= kNearReach) && rayHit < 0.0f) || !(bound.fRadius > 0.0f)) {
						return;
					}
					const float depth = dx * camera.viewDir[0] + dy * camera.viewDir[1] + dz * camera.viewDir[2];
					float       ndc[2]{ 9.0f, 9.0f };
					ProjectToView(camera, bound.center.x, bound.center.y, bound.center.z, ndc);  // (left at 9 behind the eye)
					++found;
					const bool registeredNow = registered.contains(a_object);
					foundRegistered += registeredNow;
					if (lines.size() >= 400 && (rayHit < 0.0f || rayLines.size() >= 600)) {
						return;
					}
					std::string chain;
					for (std::size_t i = 0; i < path.size(); ++i) {
						chain += (i ? " / " : "") + path[i];
					}
					std::string fade = "no fade node";
					if (const auto node = ShaderFadeNode(a_object)) {
						KeptRegionItem item{};
						static const auto frameCounter = reinterpret_cast<const std::uint32_t*>(CBRO::Engine::OG(734919).address());
						if (ReadFadeState(node, *frameCounter, item)) {
							fade = std::format("fade node {:X} currentFade {:.2f} fadeAmount {:.2f} unseen {} frames, mesh LOD level {}", reinterpret_cast<std::uintptr_t>(node),
								item.currentFade, item.fadeAmount, item.unseenFrames, item.lodLevel);
						}
					}
					auto text = std::format(
						"  {} | {} '{}' {:X} | bound ({:.0f},{:.0f},{:.0f}) r={:.0f} view depth {:.0f} screen ({:.3f},{:.3f}) | AppCulled {} | registered with the main view: {} | {} | path: {}",
						registeredNow ? "drawn" : "NOT REGISTERED", a_type, a_name, reinterpret_cast<std::uintptr_t>(a_object), bound.center.x, bound.center.y, bound.center.z, bound.fRadius,
						depth, ndc[0], ndc[1], culledAt < 0 ? std::string("no") : std::format("at path level {}", culledAt), registeredNow ? "yes" : "no", fade, chain);
					if (rayHit >= 0.0f && rayLines.size() < 600) {
						rayLines.emplace_back(rayHit, std::format("  x {:.0f} along |{}", rayHit, text));
					}
					if (lines.size() < 400) {
						lines.push_back(std::move(text));
					}
				}
			};
			Scan scan{ a_camera, a_registered, a_offered };
			scan.Visit(root, 0);
			logger::info(
				"==== scene scan: meshes whose bound comes within {:.0f} of the eye: {} ({} registered with the main view on the left-out frame, {} not) | {} objects visited ====",
				kNearReach, scan.found, scan.foundRegistered, scan.found - scan.foundRegistered, scan.visited);
			for (const auto& line : scan.lines) {
				logger::info("{}", line);
			}
			std::ranges::sort(scan.rayLines, [](const auto& a_left, const auto& a_right) { return a_left.first < a_right.first; });
			logger::info("==== scene scan: meshes whose bound crosses the ray window (NDC x {:.2f}..{:.2f}, y {:.2f}..{:.2f}): {} (nearest along the rays first) ====",
				RayWindow()[0], RayWindow()[1], RayWindow()[2], RayWindow()[3], scan.rayLines.size());
			for (const auto& [t, line] : scan.rayLines) {
				logger::info("{}", line);
			}
		}

		void LogKeptRegion(std::vector<KeptRegionItem>& a_items)
		{
			std::ranges::sort(a_items, [](const KeptRegionItem& a_left, const KeptRegionItem& a_right) { return a_left.distance < a_right.distance; });
			std::uint32_t faded = 0, lowLOD = 0;
			for (const auto& item : a_items) {
				faded += item.fadeRead && !(item.currentFade > 0.0f);
				lowLOD += item.fadeRead && item.lodLevel < 3 && ((item.fadeFlags >> 12) & 1);
			}
			logger::info(
				"==== kept in the middle third of the screen on the same frame: {} entries | their fade node at currentFade 0 (files no meshes): {} | mesh LOD level below 3 (mesh-LOD flag set): {} ====",
				a_items.size(), faded, lowLOD);
			constexpr const char* kKindNames[]{ "main-only", "group 0", "unknown" };
			std::uint32_t printed = 0;
			for (const auto& item : a_items) {
				if (printed++ >= 500) {
					break;
				}
				std::string fade = "no fade node";
				if (item.fadeRead) {
					const auto bit = [&](int a_bit) { return static_cast<int>((item.fadeFlags >> a_bit) & 1); };
					fade = std::format(
						"fade node {:X}: type {} fadeAmount {:.2f} currentFade {:.2f} near {:.0f} far {:.0f} boundRadius {:.0f} unseen {} frames | bits 12 {} 14 {} 15 {} 37 {} 38 {} | mesh LOD level {} (previous {}, state {}, blend {:.2f})",
						reinterpret_cast<std::uintptr_t>(item.fadeNode), item.fadeType, item.fadeAmount, item.currentFade, item.nearDistance, item.farDistance, item.boundRadius,
						item.unseenFrames, bit(12), bit(14), bit(15), bit(37), bit(38), item.lodLevel, item.lodPrevious, item.lodState, item.lodBlend);
				}
				const auto kind = static_cast<std::size_t>(item.kind);
				logger::info(
					"  kept {} | {} '{}' {:X} in {} | ref {:08X} base {:08X} | bound ({:.0f},{:.0f},{:.0f}) r={:.0f} dist={:.0f} screen ({:.3f},{:.3f}) | {}{}",
					kind < 3 ? kKindNames[kind] : "?", item.type, item.name, reinterpret_cast<std::uintptr_t>(item.object), item.ancestry, item.refID, item.baseID,
					item.bound[0], item.bound[1], item.bound[2], item.bound[3], item.distance, item.ndc[0], item.ndc[1], fade,
					item.meshLOD ? std::format(" | segments {}/{}/{}", item.lodSizes[0], item.lodSizes[1], item.lodSizes[2]) : std::string{});
			}
		}

		// The depth CBRO tested against on that frame, in view depth per 8x8 block of Hi-Z level 0 (rows top to bottom,
		// screen y from +1 down to -1; columns left to right): k = thousands of units, "  -  " = the far plane (sky).
		void LogDepthPicture(const DepthPicture& a_picture)
		{
			if (a_picture.farthest.empty()) {
				return;
			}
			logger::info(
				"==== depth on that frame, per block of {}x{} ({} columns x {} rows, NDC x = (column + 0.5) / {} * 2 - 1, y = 1 - (row + 0.5) / {} * 2), camera at ({:.0f},{:.0f},{:.0f}) looking ({:.3f},{:.3f},{:.3f}) ====",
				8, 8, a_picture.width, a_picture.height, a_picture.width, a_picture.height, a_picture.eye[0], a_picture.eye[1], a_picture.eye[2], a_picture.dir[0], a_picture.dir[1], a_picture.dir[2]);
			const auto cell = [](float a_depth) {
				if (!std::isfinite(a_depth) || a_depth > 1.0e6f) {
					return std::string("   - ");
				}
				return a_depth < 10000.0f ? std::format("{:5.1f}", a_depth / 1000.0f) : std::format("{:5.0f}", a_depth / 1000.0f);
			};
			for (const auto* plane : { &a_picture.farthest, &a_picture.nearest }) {
				logger::info("{} surface per block (k units):", plane == &a_picture.farthest ? "farthest" : "nearest");
				for (std::uint32_t y = 0; y < a_picture.height; ++y) {
					std::string row = std::format("  row {:2}:", y);
					for (std::uint32_t x = 0; x < a_picture.width; ++x) {
						row += cell((*plane)[static_cast<std::size_t>(y) * a_picture.width + x]);
					}
					logger::info("{}", row);
				}
			}
		}
	}

	void FlushKeptDump()
	{
		const auto dumpClock = g_dumpClock.load();
		if (dumpClock == 0 || Clock() <= dumpClock) {
			return;
		}
		std::vector<KeptItem>       items;
		std::vector<LeftOutItem>    leftOut;
		std::vector<KeptRegionItem> keptRegion;
		DepthPicture                picture;
		std::unordered_set<const void*> registered;
		std::unordered_map<const void*, std::uint8_t> offered;
		HiZ::Camera                 camera{};
		bool                        cameraValid = false;
		{
			std::scoped_lock lock(g_keptLock);
			items.swap(g_kept);
			leftOut.swap(g_leftOut);
			keptRegion.swap(g_keptRegion);
			registered.swap(g_registered);
			offered.swap(g_offered);
			picture = std::move(g_picture);
			g_picture = DepthPicture{};
			camera = g_pictureCamera;
			cameraValid = g_pictureCameraValid;
			g_pictureCameraValid = false;
		}
		if (const auto leftOutClock = g_leftOutClock.load(); leftOutClock != 0) {  // (bLeftOutDump)
			LogLeftOut(leftOut, leftOutClock, cameraValid ? &camera : nullptr);
			LogKeptRegion(keptRegion);
			if (cameraValid) {
				ScanScene(camera, registered, offered);
			}
			LogDepthPicture(picture);
		}
		g_leftOutClock.store(0);
		g_dumpClock.store(0);

		// Meshes a shape settled on this frame (not drawn): listed apart, largest first, so a wrongly hidden
		// mesh can be traced to its type and size.
		std::vector<KeptItem> dropped;
		std::erase_if(items, [&](const KeptItem& a_item) {
			if (a_item.dropped) {
				dropped.push_back(a_item);
			}
			return a_item.dropped;
		});
		if (!dropped.empty()) {
			std::ranges::sort(dropped, [](const KeptItem& a_left, const KeptItem& a_right) { return a_left.radius > a_right.radius; });
			std::map<std::string, std::uint32_t> byType;
			for (const auto& item : dropped) {
				++byType[std::format("{} ({})", item.type, item.reason)];
			}
			std::string text;
			for (const auto& [key, count] : byType) {
				text += std::format(" | {} x{}", key, count);
			}
			logger::info("==== settled by mesh shape on the same frame: {} meshes ===={}", dropped.size(), text);
			for (std::size_t i = 0; i < dropped.size() && i < 24; ++i) {
				const auto& item = dropped[i];
				logger::info(
					"  {:<20} {:<20} '{}' r={:.0f} dist={:.0f} verts={} | desc=0x{:X}",
					item.reason, item.type, item.name, item.radius, item.distance, item.vertices, item.vertexDesc);
			}
		}

		std::map<std::string, std::uint32_t> meshReasons;
		std::uint32_t                        meshes = 0, nodes = 0, withCpuData = 0, skinned = 0;
		for (const auto& item : items) {
			if (item.geometry) {
				++meshes;
				++meshReasons[item.reason];
				withCpuData += item.data != 0;
				skinned += item.skinned;
			} else {
				++nodes;
			}
		}
		std::string reasons;
		for (const auto& [reason, count] : meshReasons) {
			reasons += std::format(" {} {} |", reason, count);
		}
		logger::info("==== kept on one still frame: {} meshes (each is at least one draw), {} nodes (their children are judged on their own) ====", meshes, nodes);
		logger::info("meshes by reason:{}", reasons);
		logger::info("meshes with CPU-side vertex data: {} of {} | skinned: {}", withCpuData, meshes, skinned);

		// Grouped by type + name (precombined chunks have no name; LOD blocks are 'obj'/'obj-at'/'Land').
		std::map<std::string, std::pair<std::uint32_t, float>> groups;
		for (const auto& item : items) {
			if (item.geometry) {
				auto& group = groups[std::format("{} '{}'", item.type, item.name)];
				++group.first;
				group.second = std::max(group.second, item.radius);
			}
		}
		std::vector<std::pair<std::uint32_t, std::string>> ranked;
		for (const auto& [key, value] : groups) {
			ranked.emplace_back(value.first, std::format("{} x{} (max r {:.0f})", key, value.first, value.second));
		}
		std::ranges::sort(ranked, std::greater{});
		std::string text;
		for (std::size_t i = 0; i < ranked.size() && i < 16; ++i) {
			text += " | " + ranked[i].second;
		}
		logger::info("meshes by name (top 16):{}", text);

		std::ranges::sort(items, [](const KeptItem& a_left, const KeptItem& a_right) {
			if (a_left.geometry != a_right.geometry) {
				return a_left.geometry;
			}
			return a_left.radius > a_right.radius;
		});
		for (std::size_t i = 0; i < items.size() && i < 60; ++i) {
			const auto& item = items[i];
			if (!item.geometry) {
				break;
			}
			logger::info(
				"  {:<28} {:<24} '{}' r={:.0f} dist={:.0f} verts={} skinned={} | desc=0x{:X} buffer={:#x} data={:#x} size={} offset={} bytes={} usage={}",
				item.reason, item.type, item.name, item.radius, item.distance, item.vertices, item.skinned,
				item.vertexDesc, item.buffer, item.data, item.dataSize, item.dataOffset, item.byteWidth, item.usage);
		}

		// Merge-instanced meshes: where their instance transforms live (for judging instances one by one).
		std::uint32_t shown = 0;
		for (const auto& item : items) {
			if (!item.merged || shown >= 8) {
				continue;
			}
			++shown;
			logger::info(
				"  merge-instanced r={:.0f} dist={:.0f} verts={} lod tris {}/{}/{}/{} | local translate ({:.0f},{:.0f},{:.0f}) model bound ({:.1f},{:.1f},{:.1f}) r={:.0f} | instance object {:#x}: words {:#x} {:#x} {:#x} {:#x} {:#x}, +0x34 {} +0x38 {} | d3d buffer at word {} ({}): bytes {} stride {} bind {:#x} misc {:#x} usage {}",
				item.radius, item.distance, item.vertices, item.lodTriangles[0], item.lodTriangles[1], item.lodTriangles[2], item.lodTriangles[3],
				item.localTranslate[0], item.localTranslate[1], item.localTranslate[2], item.modelBound[0], item.modelBound[1], item.modelBound[2], item.modelBound[3],
				item.instanceObject, item.instanceWords[0], item.instanceWords[1], item.instanceWords[2], item.instanceWords[3], item.instanceWords[4],
				item.instanceCount, item.instanceKind, item.instanceWord, item.instanceModule, item.instanceBytes, item.instanceStride,
				item.instanceBind, item.instanceMisc, item.instanceUsage);
		}
	}

	TestTime TakeTestMilliseconds() noexcept
	{
		Calibrate();
		const auto total = Total(kCycles);
		const auto delta = total - g_cyclesReported;
		g_cyclesReported = total;
		const auto mainTotal = g_mainStats ? g_mainStats->values[kCycles].load(std::memory_order_relaxed) : 0;
		const auto mainDelta = mainTotal - g_mainCyclesReported;
		g_mainCyclesReported = mainTotal;
		const auto part = [](Counter a_counter) {
			const auto now = Total(a_counter);
			const auto since = now - g_reported[a_counter];
			g_reported[a_counter] = now;
			return since;
		};
		const auto   evaluate = part(kCyclesEvaluate);
		const auto   shape = part(kCyclesShape);
		const auto   sun = part(kCyclesSun);
		const auto   lookup = part(kCyclesLookup);
		const auto   reuse = part(kCyclesReuse);
		const auto   recheck = part(kCyclesRecheck);
		const auto   record = part(kCyclesRecord);
		const auto   nodeScan = part(kCyclesNodeScan);
		const auto mainPart = [](Counter a_counter) {
			const auto now = g_mainStats ? g_mainStats->values[a_counter].load(std::memory_order_relaxed) : 0;
			const auto since = now - g_mainReported[a_counter];
			g_mainReported[a_counter] = now;
			return since;
		};
		const auto   mainEvaluate = mainPart(kCyclesEvaluate);
		const auto   mainSun = mainPart(kCyclesSun);
		const auto   mainLookup = mainPart(kCyclesLookup);
		const auto   mainReuse = mainPart(kCyclesReuse);
		const auto   mainRecord = mainPart(kCyclesRecord);
		const auto   mainNodeScan = mainPart(kCyclesNodeScan);
		const auto   mainJudge = mainPart(kCyclesJudge);
		const auto   mainEntryWalk = mainPart(kCyclesEntryWalk);
		const double perMs = g_calibration.cyclesPerMs;
		const auto   ms = [perMs](std::uint64_t a_cycles) { return perMs > 0.0 ? static_cast<double>(a_cycles) / perMs : 0.0; };
		return { ms(delta), ms(mainDelta), ms(evaluate), ms(shape), ms(sun), ms(lookup), ms(reuse), ms(recheck), ms(record), ms(nodeScan),
			ms(mainEvaluate), ms(mainSun), ms(mainLookup), ms(mainReuse), ms(mainRecord), ms(mainNodeScan), ms(mainJudge), ms(mainEntryWalk) };
	}

	float RejectedPerFrame() noexcept
	{
		return g_lastRejected.load();
	}

	float TestedPerFrame() noexcept
	{
		return g_lastTested.load();
	}

	float LightsPerFrame() noexcept
	{
		return g_lastLights.load();
	}

	float LightsRejectedPerFrame() noexcept
	{
		return g_lastLightsRejected.load();
	}

	void LogStats(std::uint32_t a_frames)
	{
		const double frames = std::max(1u, a_frames);
		std::array<double, kCounterCount> delta{};
		for (std::size_t i = 0; i < kCounterCount; ++i) {
			if (i == kCycles || i == kCyclesEvaluate || i == kCyclesShape || i == kCyclesSun || i == kCyclesLookup || i == kCyclesReuse || i == kCyclesRecheck ||
				i == kCyclesRecord || i == kCyclesNodeScan || i == kCyclesJudge || i == kCyclesEntryWalk) {
				continue;  // reported by TakeTestMilliseconds
			}
			const auto total = Total(static_cast<Counter>(i));
			delta[i] = static_cast<double>(total - g_reported[i]);
			g_reported[i] = total;
		}
		const auto per = [&](Counter a_counter) { return delta[a_counter] / frames; };

		const bool observing = Observing();
		g_lastRejected.store(static_cast<float>(observing ? per(kWouldReject) : per(kRejected) + per(kMergedRejected) + per(kDropInherited)));
		g_lastTested.store(static_cast<float>(per(kTested) + per(kMerged)));
		g_lastLights.store(static_cast<float>(per(kLightsTested)));
		g_lastLightsRejected.store(static_cast<float>(per(kLightsRejected)));

		logger::info(
			"occlusion per frame: tested {:.0f} | rejected {:.0f}{} | confirming {:.0f} | visible {:.0f} | kept: edge {:.0f} near {:.0f} | not in view: outside {:.0f} behind {:.0f} | hidden but exempt: type {:.0f} light {:.0f} actor {:.0f} | the player's 3D, never judged {:.0f} | invalid {:.0f} | no-context {:.0f} | table-full {:.0f} | judged by the depth frame's edge through a view overhang of up to {:.3f}: hidden objects {:.0f}, sun shadows behind surfaces {:.0f} (evaluations, not reuses) | screen boxes widened for a near occluder's parallax {:.0f} | fade nodes left out, complete, updated and counter kept on their turn (1 frame in 8 each) {:.0f} | fades run while left out, by the engine's own update (as previs's query ran OnVisible on every object visible from the camera position): updates {:.0f}, started {:.0f}, completed {:.0f}",
			per(kTested), per(kRejected),
			observing ? std::format(" (decide-only: would reject {:.0f})", per(kWouldReject)) : std::string{},
			per(kConfirming), per(kVisible), per(kEdge), per(kNear), per(kOutside), per(kBehind),
			per(kExemptType), per(kExemptLight), per(kExemptActor), per(kExemptPlayer),
			per(kInvalid), per(kNoContext), per(kTableFull), g_tunables.viewOverhang, per(kOverhangHidden), per(kSunOverhang), per(kParallaxWidened), per(kFadeKeptSeen), per(kFadeAdvanced), per(kFadeStarted), per(kFadeCompleted));
		const auto sky = Hooks::CullGroups::TakeSkyCounts();
		logger::info(
			"occlusion merged meshes per frame: {:.0f} decided | hidden {:.0f} | confirming {:.0f} | instance tests {:.0f} | instance entries {:.0f}, rejected {:.0f} || previs-forced entries hidden {:.0f} || sky meshes given previs's force-visible mark {:.2f}, CBRO drops of them ignored {:.2f} || the player's 3D entries given it {:.1f}",
			per(kMerged), per(kMergedRejected), per(kMergedConfirming), per(kInstanceTests), per(kInstanceEntries), per(kInstanceEntriesRejected),
			static_cast<double>(Hooks::CullGroups::TakeForcedCleared()) / frames, static_cast<double>(sky.forced) / frames, static_cast<double>(sky.kept) / frames,
			static_cast<double>(Hooks::CullGroups::TakeForcedVisible()) / frames);
		const auto art = Hooks::CullGroups::TakeCellArtCounts();
		logger::info(
			"cell art node 9 per frame (AppCulled by the engine, previs draws its chunks by id): nodes {:.2f} | walked by the entry walk {:.2f} | children filed into group 0 whole {:.1f}",
			static_cast<double>(art.nodes) / frames, static_cast<double>(art.walked) / frames, static_cast<double>(art.filed) / frames);
		if (const auto rooms = Hooks::CullGroups::TakeOutsideRoomsCounts(); rooms.frames || rooms.roomFrames) {
			const auto perFrame = [](std::uint64_t a_count, std::uint64_t a_frames) { return a_frames ? static_cast<double>(a_count) / static_cast<double>(a_frames) : 0.0; };
			if (rooms.frames) {
				logger::info(
					"interior outside its rooms ({} frames with the camera in a room bound; the engine files only the rooms its portals show): filed into group 0 per such frame: unbound node {:.1f}, precombined art {:.1f} (node 9 AppCulled on {} frames), roomless multibounds {:.1f} (left out early, nothing needs them: {:.1f})",
					rooms.frames, perFrame(rooms.unbound, rooms.frames), perFrame(rooms.art, rooms.frames), rooms.artCulled, perFrame(rooms.roomless, rooms.frames), perFrame(rooms.outsideLeftOut, rooms.frames));
			}
			if (rooms.roomFrames) {
				logger::info(
					"interior rooms ({}; {} walks, {} with the camera outside every room) per walk: the camera's list {:.1f} of {:.1f} rooms | filed whole into group 1: rooms {:.1f}, objects {:.1f}; portal shared geometry nodes {:.1f}, objects {:.1f} (of these objects left out early, out of view or confirmed hidden: {:.1f}) | the walk's room and portal adds left out (their room filed whole) {:.1f}",
					Hooks::CullGroups::RoomBoundsOff() ? "room bounds off: every room and portal's shared geometry, none through portals" : "rooms the portals don't show", rooms.roomFrames, rooms.openRoomFrames,
					perFrame(rooms.roomsSeen, rooms.roomFrames), perFrame(rooms.roomsTotal, rooms.roomFrames), perFrame(rooms.hiddenRooms, rooms.roomFrames),
					perFrame(rooms.hiddenRoomObjects, rooms.roomFrames), perFrame(rooms.sharedNodes, rooms.roomFrames), perFrame(rooms.sharedObjects, rooms.roomFrames), perFrame(rooms.objectsLeftOut, rooms.roomFrames),
					perFrame(rooms.portalAddsSkipped, rooms.roomFrames));
			}
		}
		if (const auto view = Hooks::CullGroups::TakeViewFrustumCounts(); view.frames) {
			logger::info(
				"main view frustum (CBRO frames, bIgnoreOccluders): {} frames | the engine's occluders or portals on it, taken off: {} frames ({:.1f} ops each) | the portal graph's occluders outside every room {:.1f}",
				view.frames, view.cleared, view.cleared ? static_cast<double>(view.ops) / static_cast<double>(view.cleared) : 0.0,
				static_cast<double>(view.graphOccluders) / static_cast<double>(view.frames));
		}
		logger::info(
			"occlusion lights per frame: tested {:.0f} | rejected {:.0f} (point/spot lights whose whole reach is hidden)",
			per(kLightsTested), per(kLightsRejected));
		logger::info(
			"occlusion main view only (groups the sun's shadow cascades read too) per frame: entries {:.0f} | dropped with their parent {:.0f} | registrations {:.0f}, left out {:.0f} | drop table full {:.0f}",
			per(kShared), per(kDropInherited), per(kRegistered), per(kRegistrationsDropped), per(kDropFull));
		logger::info(
			"occlusion sun shadows per frame (group-0 objects the main view doesn't need): shadow tested {:.0f} | not needed: out of reach {:.0f}, behind surfaces {:.0f} | sun off: kept, no lamp known {:.0f}; lamp tests {:.0f}: no lamp's shadow reaches a visible surface {:.0f} (left out), a lamp's may {:.0f} | confirming {:.0f} | needed {:.0f} | behind surfaces only stretch by stretch (of the not needed) {:.0f} | last cascade: needed shadows tested {:.0f}, nothing visible in its slab {:.0f}, entries left out of it {:.0f} ({:.0f} registrations) || objects nothing needs: never filed {:.0f}, rejected {:.0f} | kept for a spot lamp within reach (sun up): entries {:.0f}, cell nodes {:.1f} (spot lamps not all known {:.0f}; {} spot lamps this frame) | emptied spot lamps given no entries {:.2f}, their maps drawn after all {:.2f} | entries kept for spot lamps alone left out of the sun's cascades {:.0f} ({:.0f} registrations)",
			per(kSunTests), per(kSunOutside), per(kSunHidden), per(kSunOff), per(kSunLampTests), per(kSunLampUnneeded), per(kSunLampNeeded), per(kSunConfirming), per(kSunNeeded), per(kSunPiecewise), per(kSunFarTests), per(kSunFarUnneeded), per(kSunFarLeftOut), per(kSunFarRegistrations), per(kCasterSkipped), per(kCasterRejected),
			per(kSpotLampKept), per(kSpotLampNodesKept), per(kSpotLampsUnknown), g_spotLamps.count, per(kSpotLampsEmptied), per(kSpotLampsLeftLit), per(kSpotOnlyLeftOut), per(kSpotOnlyRegistrations));
		logger::info(
			"occlusion early skips per frame (main-view-only groups, never filed with the engine): hidden {:.0f} | out of view {:.0f} | top-level adds considered {:.0f}",
			per(kSkippedHidden), per(kSkippedOutside), static_cast<double>(Hooks::CullGroups::TakeGroupAddsConsidered()) / frames);
		logger::info(
			"trace feed per frame (Renderer Replacer's scene; {}): refused registrations handed over {:.0f} | objects left out of every view otherwise, filed for it: on their turn {:.1f}, moved {:.1f}, main-only children with them {:.1f}, cell nodes walked {:.2f} (their sun-cascade registrations count with the spot-only ones above) | of those, out of view and filed force-visible {:.1f}, their children with the mark {:.1f}",
			TraceFeed::On() ? "on"sv : "off"sv, static_cast<double>(Hooks::CullGroups::TakeRefusedRegistrations()) / frames, per(kTraceFiled), per(kTraceMoved), per(kTraceInherited), per(kTraceCellNodes),
			per(kTraceForced), per(kTraceForcedChildren));
		logger::info("occlusion static collection pieces per frame (judged and filed with the collection's bound): {:.1f} (the root from the cache, its parent chain checked {:.1f}; parents found in the cache with a root {:.1f}, their chain check failed {:.1f})", per(kScolWidened), per(kScolRootHits), per(kScolSlotHits), per(kScolChainFails));
		logger::info(
			"occlusion node pruning per frame: offered by the scene walk: cell node 3 {:.1f}, container {:.1f}; by DrawWorld's root loop {:.1f} | skipped whole {:.1f} ({:.0f} entries never filed), of them on their clusters' cached spheres {:.1f} ({:.0f} entries untouched, {:.1f} clusters read again on the way) | walked: an entry meets the view {:.1f}, holds an always-draw entry or an actor {:.1f}, sun shadow may reach the view {:.1f} || cell node 9 seen {:.1f} (never pruned) | nodes with their own always-draw bit {:.1f}",
			per(kCellNodesSeen), per(kContainersSeen), per(kRootsSeen), per(kCellNodesSkipped), per(kCellNodeEntriesSkipped), per(kCellNodesCached), per(kCellNodeEntriesCached), per(kCellNodeClustersRescanned), per(kCellNodesInView), per(kCellNodesHeld), per(kCellNodesSunNeeded), per(kCellNodesOther), per(kCellNodesAlwaysDraw));
		logger::info(
			"entry walk per frame (bEntryWalk {}: nodes pruning can't leave out whole, walked by CBRO itself): nodes {:.1f} (fade-node roots left to the engine {:.2f}, slot caches rebuilt {:.2f}) | entries {:.0f}: left out by the cheap tests {:.0f}, offered to the top-level decision {:.0f}, of them filed {:.0f}; filed unjudged (a group marker pending) {:.1f} || leases (bWalkLeases {}, {} frames, margin {:.0f}): clusters {:.0f}, left out whole on their cached sphere {:.0f}, their entries untouched {:.0f} (visited for a fade mid-way {:.1f}, walked for the trace's turn {:.1f}); clusters walked for a pending marker {:.2f} | clusters walked, not leased: never finished by the walk {:.0f}, no entry of their own (containers only) {:.0f}, held by a bad or moved bound {:.0f}, by an always-draw, light, sky, actor or player member {:.0f}, by a marker filing {:.0f}, lease due {:.0f}, child array changed {:.0f}, sphere meets the view {:.0f}, sun shadow may reach {:.0f}, lamp may need {:.0f} | AppCulled children read {:.0f}",
			g_tunables.entryWalk ? "on" : "off", per(kWalkNodes), per(kWalkFadeRoots), per(kWalkCachesRebuilt), per(kWalkEntries), per(kWalkSkipped), per(kWalkOffered), per(kWalkFiled), per(kWalkMarkerFiled),
			g_tunables.walkLeases ? "on" : "off", g_tunables.leaseFrames, g_tunables.leaseMargin, per(kWalkClusters), per(kWalkClustersDormant), per(kWalkLeased), per(kWalkLeaseFadeVisits), per(kWalkLeaseTraceVisits), per(kWalkLeaseMarkerHeld),
			per(kWalkClusterNotWalked), per(kWalkClusterNoMembers), per(kWalkClusterHeldBad), per(kWalkClusterHeldKind), per(kWalkClusterHeldMarker), per(kWalkClusterDue), per(kWalkClusterSlotsChanged), per(kWalkClusterInView), per(kWalkClusterShadow), per(kWalkClusterLamp), per(kWalkCulled));
		logger::info("stable-index decisions per frame (bStableIndex {}): node decisions recorded in filing order {:.0f} | children's decision parents found by the cursor {:.0f}; not found: never recorded {:.1f} (no scan), recorded but not ahead of the cursor {:.1f} (the drop sets, the top table and the collection cache answered those)",
			g_tunables.stableIndex ? "on" : "off", per(kStableRecorded), per(kStableParentHits), per(kStableParentNotRecorded), per(kStableParentOutOfOrder));
		logger::info("trace handover per frame (bTraceDirect {}, bDeferSunOnly {}): entries filed for the trace alone handed over directly {:.1f}, with {:.0f} pieces | sun-only drops held back from the main view {:.1f}, their pieces handed {:.0f} (not again: the tracer keeps them {:.1f}); filed for the cascades in their window {:.1f}, never filed (no window) {:.2f}",
			g_tunables.traceDirect ? "on" : "off", g_tunables.deferSunOnly ? "on" : "off", per(kTraceHanded), per(kTraceHandedPieces), per(kSunOnlyDeferred), per(kSunOnlyHandedPieces), per(kSunOnlyKnown), per(kSunOnlyFiledLate), per(kSunOnlyDropped));
		{
			const auto link = TraceFeed::TakeCounts();
			const auto perCount = [&](std::uint64_t a_value) { return static_cast<double>(a_value) / frames; };
			logger::info("trace link per frame (v2: {}): left out on their turn, the tracer keeping them {:.0f} | movers held back past its mover radius {:.1f} | pieces filed for the trace alone handed through the link {:.1f} | entries handed that left their slot (GONE) {:.2f} || records: take {:.1f}, moved {:.1f}, gone {:.2f}; read by the tracer {:.1f}, lost unread {} | the tracer's forget calls {:.2f} ({:.2f} known entries forgotten), table cleared {}",
				TraceFeed::Linked() ? "the tracer keeps what it is handed, through the direct link"sv : TraceFeed::Retains() ? "the tracer keeps what it is handed, through the refused callback"sv : TraceFeed::On() ? "off: the tracer asks every turn"sv : "off"sv,
				per(kTraceKnownSkipped), per(kTraceMoverHeld), per(kTraceChildHanded), per(kTraceGone), perCount(link.records[CBRO_TRACE_TAKE]), perCount(link.records[CBRO_TRACE_MOVED]),
				perCount(link.records[CBRO_TRACE_GONE]), perCount(link.read), link.lost, perCount(link.forgetCalls), perCount(link.forgotten), link.cleared);
		}
		const auto sampled = [&](Counter a_counter) { return per(a_counter) * kFadeStampFrames; };
		logger::info(
			"top-level fade nodes per frame (survey, 1 node in {} sampled, scaled): offered {:.0f} | let through {:.0f}, their geometry records {:.0f} (their ChildPush adds), with one record {:.0f} | past the far fade range: fade still running out {:.0f}, flag bit 15 set {:.0f}, at fade 0 (no mesh of them filed) {:.0f}, of which let through {:.0f} with {:.0f} records | lighting re-queued among the let-through (OnVisible: flag bit 33 {:.0f}, of them filed force-visible for the trace or as the player's {:.0f}; mid-fade {:.0f}, force-visible {:.0f}, fading in {:.0f} (radius >= 600, no snap rule {:.0f}; stamped last frame {:.0f}, left out meanwhile {:.0f}; actors {:.0f}); bit 51 never {:.0f}) | frames with the scene node's lighting byte +0x228 set {:.0f}% || fade-in starts (let through at fade 0, bit 37 clear): frame counter over 20 frames old {:.0f} (near: snapped to 1 {:.0f}) | stamped within 20 frames {:.0f} (by OnVisible last frame {:.0f}, 2-20 frames ago {:.0f}; near: fade in from 0 on screen {:.0f})",
			kFadeStampFrames, sampled(kFadeTopOffered), sampled(kFadeTopKept), sampled(kFadeTopKeptEntries), sampled(kFadeTopKeptOne),
			sampled(kFadeTopFadingOut), sampled(kFadeTopPastFarBit15), sampled(kFadeTopFaded), sampled(kFadeTopFadedKept), sampled(kFadeTopFadedKeptEntries),
			sampled(kFadeTopLightingBit33), sampled(kFadeTopLightingBit33Forced), sampled(kFadeTopMidFade), sampled(kFadeTopMidFadeForced), sampled(kFadeTopMidFadeIn), sampled(kFadeTopMidFadeInBig), sampled(kFadeTopMidFadeInContinuous), sampled(kFadeTopMidFadeInResumed), sampled(kFadeTopMidFadeInActor), sampled(kFadeTopNoLighting), 100.0 * per(kFrameLightingByte),
			sampled(kFadeStartUnseen), sampled(kFadeStartUnseenNear), sampled(kFadeStartSeenLast) + sampled(kFadeStartSeenRecent), sampled(kFadeStartSeenLast), sampled(kFadeStartSeenRecent), sampled(kFadeStartSeenNear));
		g_fadeStartSamples.store(0, std::memory_order_relaxed);
		logger::info(
			"Block::Add entries per frame not decided at the top level (the engine's child pushes, and unjudged filings; 1 in 32 sampled, scaled): under a fade node {:.0f} | under a chunk container (exact NiNode) {:.0f} | under another node {:.0f} || left out by the cheap tests alone (a static mesh out of the view, no verdict): no view needs it {:.0f} | out of the main view, left to the engine's frustum test {:.0f} || decided with their kept top entry (the same bound, or a small entry; bChildInherit) {:.0f} (reached through the fade node filing them {:.0f}) | dropped or forced with the fade node filing them, their parent one of its sub-nodes {:.0f}",
			per(kChildOfFade), per(kChildOfNode), per(kChildOther), per(kChildCheapRejected), per(kChildCheapOutside), per(kChildInherited), per(kChildInheritedFade), per(kChildFadeDecided));
		logger::info("Group::Add callers of the main-pass top-level adds considered: {}", Hooks::CullGroups::TakeGroupAddSites(frames));
		logger::info(
			"occlusion lamp shadow volumes per frame (casters of point lights): outside the view {:.0f} | misses every visible surface {:.0f} | confirming {:.0f} | needed {:.0f} | unknown {:.0f}",
			per(kLampVolumeOutside), per(kLampVolumeMisses), per(kLampVolumeConfirming), per(kLampVolumeNeeded), per(kLampVolumeUnknown));
		logger::info(
			"verdict cache per frame: reused {:.0f} (hidden re-checked against the depth {:.0f}) | evaluated: new {:.0f}, camera epoch {:.0f}, bound changed {:.0f}, depth changed {:.0f}, hidden but judged from elsewhere (camera moving) {:.0f}, kept on its turn {:.0f}, never-reuse {:.0f} | sun re-evaluated under a reused view {:.0f} | hidden streaks continued from the backup {:.1f} | async: judged in the walk {:.0f} (no record with this bound, or the map's margins passed; the worker judges them again, so they count twice above), the map's edge and confirming verdicts judged again in the walk {:.0f}, its shadows needed only for its no-depth strip {:.0f}",
			per(kCacheHits), per(kCacheRecheck), per(kCacheNew), per(kCacheEpoch), per(kCacheBound), per(kCacheDepth), per(kCacheMoved), per(kCacheKeptRecheck), per(kCacheNoCache), per(kCacheSunDepth), per(kStreakFromBackup), per(kWalkJudged), per(kWalkUnsettled), per(kWalkSunEdge));
		logger::info(
			"per-frame spread over culled frames (min/avg/max, sd, avg change between consecutive frames): main view kept {} | other views' registrations {} | rejected {} | confirming (verdict flips) {}",
			g_spreadKept.Describe(), g_spreadOther.Describe(), g_spreadRejected.Describe(), g_spreadConfirming.Describe());
		g_spreadKept.Clear();
		g_spreadOther.Clear();
		g_spreadRejected.Clear();
		g_spreadConfirming.Clear();
		logger::info(
			"occlusion by mesh shape per frame: tested {:.0f} (sphere said visible/near/edge) | settled hidden {:.0f} | out of view {:.0f} | cells tested {:.0f} | left to the sphere: moved {:.1f}, swaying {:.1f}, transform unconfirmed {:.1f} | rotation: columns (the engine's) | bound off its transform since load {} (fitting rows instead {})",
			per(kShapeTests), per(kShapeHidden), per(kShapeOutside), per(kShapeCells), per(kShapeMoving), per(kShapeAnimated), per(kShapeNoTransform),
			g_rotationMisfits.load(std::memory_order_relaxed), g_rotationRowsOnly.load(std::memory_order_relaxed));
		MeshProxy::LogStats(a_frames);
		logger::info(
			"occlusion frames: culling {:.0f} / blocked {:.0f} (stale depth, camera jump, or inactive) | history {} objects",
			delta[kFramesCulling], delta[kFramesBlocked], g_used.load());

		const auto logTypes = [](TypeCounts& a_types, std::string_view a_what) {
			std::vector<std::pair<std::uint32_t, std::uintptr_t>> ranked;
			for (auto& slot : a_types.slots) {
				const auto vtable = slot.vtable.load(std::memory_order_acquire);
				if (const auto count = vtable ? slot.count.exchange(0, std::memory_order_relaxed) : 0u) {
					ranked.emplace_back(count, vtable);
				}
			}
			const auto untracked = a_types.untracked.exchange(0, std::memory_order_relaxed);
			if (ranked.empty()) {
				return;
			}
			std::ranges::sort(ranked, std::greater{});
			std::string text;
			for (std::size_t i = 0; i < ranked.size() && i < 10; ++i) {
				char                 name[64]{};
				const std::uintptr_t vtableOnly = ranked[i].second;  // an "object" whose first qword is the vtable
				Util::TryGetRTTIName(&vtableOnly, name, sizeof(name));
				text += std::format(" {} x{}", name[0] ? name : "?", ranked[i].first);
			}
			if (untracked) {
				text += std::format(" | untracked types x{}", untracked);
			}
			logger::info("{} (sampled every 32nd frame):{}", a_what, text);
		};
		logTypes(g_exemptTypes, "hidden but exempt by type"sv);
		logTypes(g_leftOutTypes, "left out of group 0 by type (never filed or rejected in every view)"sv);
	}
}
