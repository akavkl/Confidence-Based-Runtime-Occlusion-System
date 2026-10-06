#pragma once

#include "Core/HiZ.h"
#include "Core/ShadowGeometry.h"
#include "Core/Verdict.h"

namespace CBRO::Hooks::CullGroups
{
	struct BlockAdd;
}

// Main-pass occlusion decisions, made inside Block::Add (Hooks/CullGroups) for every entry that
// DrawWorld's culling groups receive (top-level and child adds). An object is hidden only after it
// tested hidden against the read-back Hi-Z for `confirmFrames` consecutive frames; one visible verdict
// restores it. How it is hidden depends on who else reads its group:
//   - group 0 (the shadow casters, which the sun's shadow cascades read too whenever previs is off): an
//     object whose sun shadow can't fall on anything visible (the capsule it sweeps along the light misses
//     the view, or lies behind visible surfaces) is rejected in every view, or never filed at all; one whose
//     shadow may be seen stays in the group, and only the main accumulator's registration of it is skipped.
//     Children of such a main-view drop are dropped with it.
//   - any other group not proven main-only: only ever dropped from the main view.
//   - main-only groups (non-casters, previs list, group array): the entry is rejected with a bound the
//     engine's frustum test drops (its subtree goes with it), or it is never filed at all.
// Nodes that carry lights or an NPC are never hidden as a whole (their meshes are still tested one by
// one); point and spot lights themselves are hidden when their whole influence sphere is. Merge-
// instanced meshes (precombined chunks) are hidden by rejecting all their instance entries once every
// instance is hidden. Shadow and other passes never lose an object.

namespace CBRO::Core::Occlusion
{
	// Light tests (TestSphere): how far (NDC) the current view may overhang the depth frame and still be judged by its
	// edge (Occlusion.cpp has why; objects and sun shadows take fViewOverhang).
	inline constexpr float kLightOverhang = 0.03f;

	// Per-frame inputs, prepared on the main thread right before DrawWorld culls.
	struct FrameContext
	{
		const HiZ::Snapshot* snapshot{ nullptr };
		std::uint32_t        clock{ 0 };          // advances once per frame; drives the confirmation streaks
		bool                 cull{ false };       // false: nothing is rejected this frame
		float                dilateMove{ 0.0f };  // camera translation since the depth frame (game units)
		RE::NiBound          reject{};            // bound the engine's frustum test is certain to reject

		// The current camera's view in the depth frame's NDC {x0, x1, y0, y1}: a conservative box
		// around where this frame's frustum lands on the depth frame's image plane. Only the part of
		// an object inside it can be seen this frame; if that part lies within the depth frame, the
		// rest of the object needn't have been rendered. Infinite when unknown (then an object must
		// lie fully inside the depth frame to be judged). The async worker's is widened toward the turn the next
		// frame may bring (Runtime), and its map is used only by a frame whose view lies inside.
		float view[4]{
			-std::numeric_limits<float>::infinity(), std::numeric_limits<float>::infinity(),
			-std::numeric_limits<float>::infinity(), std::numeric_limits<float>::infinity()
		};
		// Per side {x0, x1, y0, y1}: how far (NDC) a part in view past the depth frame is judged by its edge, for objects
		// and sun shadows (fViewOverhang; up to fViewOverhangTurn where the view itself reaches that far past it). Set
		// with the planes by BeginFrame and PrepareContext.
		float overhang[4]{};

		// The sun's shadow cascades (with previs off they read DrawWorld group 0 as extra views).
		struct Sun
		{
			enum class State : std::uint8_t
			{
				kUnknown,  // not verified: every group-0 object's shadow counts as needed
				kOff,      // the cascades don't read group 0 this frame (sun shadows off): no shadow is needed
				kOn,       // the cascades read group 0; the direction below is verified
			};
			State state{ State::kUnknown };
			float dir[3]{};        // the light's travel direction in the depth frame's view space (right, up, forward)
			float reach{ 0.0f };   // view depth beyond which nothing receives sun shadows (cascade range, padded)
			float spread{ 0.0f };  // sine of the direction's uncertainty (the engine eases it when the sun moves)
			float margin{ 0.0f };  // added to every caster's radius: the shadow maps' filtering

			// Where receivers can be, in the depth frame's view space (in front of the eye, within reach,
			// inside the current view): set by BeginFrame from the fields above.
			ShadowGeometry::Plane planes[6]{};
			int                   planeCount{ 0 };
			// Receiver planes a caster's sweep can only move away from (n.dir + spread <= 0): a sphere already
			// beyond one of them is out of reach without the full sweep.
			ShadowGeometry::Plane reachPlanes[6]{};
			int                   reachCount{ 0 };
		};
		Sun sun;

		// The current view as a world-space cone, for point-light shadow casters: the depth frame's camera
		// widened by the rotation since, with conePush covering the movement since.
		bool                 coneValid{ false };
		ShadowGeometry::Cone cone{};
		float                conePush{ 0.0f };

		// Where lamp-shadow receivers can be, in the depth frame's view space (in front of the eye, inside the
		// current view): set by BeginFrame; none when the depth frame has no view-space form.
		ShadowGeometry::Plane lampPlanes[5]{};
		int                   lampPlaneCount{ 0 };

		// The verdict cache (Occlusion): what this frame's reuse checks compare against (Runtime fills them).
		bool                 cacheEnabled{ false };
		std::uint32_t        viewEpoch{ 0 };     // changes when the camera, or the depth frame's camera, drifts past the cache tolerance
		std::uint8_t         sunEpoch{ 0 };      // changes with the sun's direction or state
		std::uint32_t        readback{ 0 };      // the snapshot's readback index
		const std::uint32_t* blockChangedAt{ nullptr };  // per 8x8 Hi-Z block: the readback at which its depth last changed
		std::uint32_t        blocksW{ 0 };
		std::uint32_t        blocksH{ 0 };
		float                angularSlack{ 0.0f };  // radians: tested bounds grow by their depth x this, covering the turn tolerance

		// Asynchronous verdicts (Core/Async): this frame judges through the worker (asyncFrame; else inside the walk as
		// v1.28 did) and may hide by the worker's map (asyncValid: its camera margins hold).
		bool                 asyncFrame{ false };
		bool                 asyncValid{ false };
	};

	void Install();
	void BeginFrame(const FrameContext& a_context);

	// Core/Async's worker: one candidate's record from the frame's context and its last record (or null), exactly as
	// the walk's own path would judge it (cache reuse, the view test, mesh shapes, lights, the sun when a_wantSun).
	// Any thread; the object is dereferenced (it must be alive).
	void JudgeAsync(const FrameContext& a_context, const Hooks::CullGroups::BlockAdd& a_add, const Record* a_old, bool a_wantSun, Record& a_out);
	// Derives a context's sun and lamp receiver planes from its view (BeginFrame does this for the frame's own).
	void PrepareContext(FrameContext& a_context) noexcept;

	// Once per frame at the cull stage's end (main thread, DrawWorld's jobs done), with the running total of
	// every accumulator's registrations (Hooks::CullGroups::ReadHookCalls): records how much the culled set
	// changes from frame to frame (LogStats prints the spread).
	// a_droppedTotal: registrations dropped at emptied lamp shadow maps (Hooks::CullGroups, running total). Frames whose
	// drawn set rises well over the recent median are logged with the flips behind them (diagnostic, bounded).
	void EndFrameSample(std::uint64_t a_registrationsTotal, std::uint64_t a_droppedTotal);
	// What the last EndFrameSample measured (valid: that frame and the one before were culled): main view kept, other
	// views' registrations (lamp maps' drops taken out), confirming verdicts and edge verdicts (walk and worker together).
	struct FrameSample
	{
		bool  valid{ false };
		float kept{ 0.0f };
		float other{ 0.0f };
		float confirming{ 0.0f };
		float edge{ 0.0f };
		float finalEdge{ 0.0f };        // the walk's own decisions only: kept as edge,
		float finalConfirming{ 0.0f };  // ... and hidden but still confirming
	};
	[[nodiscard]] FrameSample LastFrameSample() noexcept;

	void SetActive(bool a_active);
	void SetObserveOnly(bool a_observeOnly);  // runtime override of [Occlusion] bObserveOnly (diagnostics)
	[[nodiscard]] bool Active() noexcept;

	// Forget streaks and type decisions (load, mode switch). Applied at the next BeginFrame.
	void ResetHistory();

	void LogStats(std::uint32_t a_frames);

	// Sphere test against the current frame's depth, for other levers (shadow lights).
	enum class SphereVerdict
	{
		kUnknown,    // no usable depth this frame, partly off the depth frame, or reaches the camera: treat as visible
		kVisible,
		kHidden,     // every visible surface in its screen area is in front of it
		kOutOfView,  // entirely outside the current view (needs a known view footprint)
	};
	// Why (for the stats): kOverhang with kHidden, the others with kUnknown.
	enum class SphereReason : std::uint8_t
	{
		kNone,
		kOverhang,  // hidden, part of it judged by the depth frame's edge: the view overhangs the frame by a thin strip
		kNoDepth,   // no usable depth this frame
		kEdge,      // part of it is in view where the depth frame never rendered (a wider overhang)
		kNear,      // it reaches in front of the near distance (the camera is at or in it)
		kInvalid,   // a bad bound
		kEmptyReach,  // hidden though not behind the depth: no visible surface lies inside it (only nothing drawn, or surfaces beyond it)
	};
	// A light's whole reach: unlike an object's test, a thin strip where the current view overhangs the depth frame (a
	// camera turned a fraction of a degree since the depth was rendered) is taken to hold what the frame's edge holds,
	// and the Hi-Z is refined down to single texels (a far texel counts only where the light's rays pass). Since v1.55 a
	// reach also counts as hidden when no visible surface lies inside it (kEmptyReach): nothing in it can be lit.
	[[nodiscard]] SphereVerdict TestSphere(const RE::NiPoint3& a_center, float a_radius, SphereReason* a_reason = nullptr) noexcept;

	// Diagnostic (ShadowLights): an emptied lamp shadow map had to be drawn again (its light no longer judged unseen).
	void NoteLightFlip(bool a_spot, const RE::NiPoint3& a_center, float a_radius, SphereVerdict a_verdict, SphereReason a_reason) noexcept;

	// Whether a point light's caster can shadow anything visible this frame: its shadow volume (the cone from the
	// lamp through the caster's sphere, out to the lamp's reach) against the current depth (ShadowLights).
	enum class LampVerdict
	{
		kUnknown,     // no usable depth this frame: leave the caster to the engine
		kNeeded,      // may shadow a visible surface (or can't be judged: reaches the camera, crosses the view's edge)
		kOutside,     // the volume never enters the view, or lies beyond the lamp's reach
		kMisses,      // the volume lies entirely behind, or entirely in front of, every visible surface (confirmed)
		kConfirming,  // ... not yet for confirmFrames frames
	};
	[[nodiscard]] LampVerdict TestLampCaster(RE::NiAVObject* a_caster, const RE::NiPoint3& a_lamp, float a_reach, const RE::NiBound& a_bound) noexcept;
	// The same against a given context (the async worker's).
	[[nodiscard]] LampVerdict TestLampCasterIn(const FrameContext& a_context, RE::NiAVObject* a_caster, const RE::NiPoint3& a_lamp, float a_reach, const RE::NiBound& a_bound) noexcept;
	void                      CountLampVerdict(LampVerdict a_verdict) noexcept;  // for the stats (per-thread counters)

	[[nodiscard]] std::uint32_t Clock() noexcept;         // advances once per culled frame
	[[nodiscard]] bool          Deciding() noexcept;      // observe-only (config or diagnostic): decide but never hide

	// This frame's view cone for point-light shadow casters, if known (culling this frame).
	[[nodiscard]] bool ViewCone(ShadowGeometry::Cone& a_cone, float& a_push) noexcept;

	// CPU time spent in the per-object tests since the last call (ms): on all threads, and the main thread's share
	// (the part on the cull stage's critical path).
	struct TestTime
	{
		double all{ 0.0 };
		double mainThread{ 0.0 };
		double evaluate{ 0.0 };  // of `all`: full view evaluations (sphere tests, mesh shapes, lights), all threads
		double shape{ 0.0 };     // of `evaluate`: the mesh-shape tests
		double sun{ 0.0 };       // of `all`: sun-shadow evaluations
		// The rest of `all` (v1.52), all threads:
		double lookup{ 0.0 };    // last frame's record looked up (the thread's stream, or the worker's map)
		double reuse{ 0.0 };     // whether a record still holds (epoch, bound, depth blocks, re-check)
		double recheck{ 0.0 };   // of `reuse`: hidden verdicts' evidence re-checked against the current depth
		double record{ 0.0 };    // this frame's record written (the stream, the table's streak backup)
		double nodeScan{ 0.0 };  // cell-node pruning's scans of a node's entries
	};
	[[nodiscard]] TestTime TakeTestMilliseconds() noexcept;

	// Diagnostics: record every object the main pass keeps on the next frame, then log them (type,
	// name, size, vertex-data location) at the frame after. Call RequestKeptDump, then FlushKeptDump
	// at each cull begin.
	void RequestKeptDump();
	void FlushKeptDump();

	// Per-frame averages from the last LogStats (for the on-screen status).
	[[nodiscard]] float RejectedPerFrame() noexcept;
	[[nodiscard]] float TestedPerFrame() noexcept;
	[[nodiscard]] float LightsPerFrame() noexcept;
	[[nodiscard]] float LightsRejectedPerFrame() noexcept;
}
