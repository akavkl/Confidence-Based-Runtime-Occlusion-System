#pragma once

// Renderer Replacer's path tracer (D:/Projects/Renderer Replacer, src/Render/PathTracer/Scene.cpp) builds its scene from
// BSShaderAccumulator::RegisterObject (1447420): the main accumulator's registrations and the shadow maps' (the sun's
// cascades, the lamps). Its detour sits at that function's entry, beneath CBRO's slot-45 hook (Hooks/CullGroups), so it
// never sees a registration CBRO refuses, and an entry CBRO rejects or never files is registered nowhere: its
// reflections, bounces and traced shadows lost whatever CBRO culled from the screen. An instance's transform, skin pose
// and mesh-LOD level are read when it is registered.
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
//   block's force byte, as previs marks its entries), its children with the mark. With the sun up a group-0 entry whose
//   shadow may be needed is left to the cascades, which file it within their range.
// - (v1.99.3) such an object is handed over directly where it can be (a geometry; a fade node's pieces), not filed.
//
// v2 (CBRO 2.0), the direct link: on the 2026-10-27 runs the feed cost ~0.9 ms of CBRO's cull and ~2.4 ms of the
// tracer's scene intake a frame, nearly all of it handing over again, every frame or every turn, objects the tracer
// already held: the turns, the movers, the sun-only casters each frame, each through the engine's filing or a callback
// per piece. A tracer that asks for it (flags below) gets instead:
// - RETAINS: it keeps what it was handed while the object stands in the world. CBRO hands an object once (its first
//   turn: the known table remembers it), again only when it moves (every frame within moverRadius of the camera, one
//   frame in 8 farther) or once in resendFrames. The tracer calls CBRO_TraceFeedForget when it lets an instance go that
//   may come back (retired, hidden, its mesh released), CBRO_TraceFeedForgetAll when it drops its scene.
// - LINK: what CBRO hands over (the geometry of the entries it decided on: never filed, nothing refused) goes into a
//   batch of records the tracer reads once a frame (CBRO_TraceLinkRead, its frame thread, after the cull), each with
//   the entry it came from (its anchor), instead of a callback per piece. The entry walk adds a GONE record when an
//   entry it handed leaves its slot (detached, replaced, AppCulled) or its node is forgotten: the tracer drops the
//   anchor's instances without looking for them. Registrations CBRO refuses still go through `refused`.
// The rest of the time such an object stays out of the walk, as without the feed.

// The link's constants (v2): CBRO_TraceFeed::flags, CBRO_TraceRecord::kind and ::flags.
inline constexpr std::uint32_t CBRO_TRACE_FEED_RETAINS = 1;
inline constexpr std::uint32_t CBRO_TRACE_FEED_LINK = 2;
// v2.1.3: the tracer takes the sun cascades' registrations made in the batched cascade cull's jobs itself; without it a
// linked tracer gets them through the link (its own observer takes only the frame thread's)
inline constexpr std::uint32_t CBRO_TRACE_FEED_JOB_CASCADES = 4;
inline constexpr std::uint32_t CBRO_TRACE_TAKE = 0;      // `object`, a geometry: take it in as if registered this frame (first contact, resend)
inline constexpr std::uint32_t CBRO_TRACE_MOVED = 1;     // ... the same, its entry moved
inline constexpr std::uint32_t CBRO_TRACE_GONE = 2;      // `anchor` left the world as CBRO walks it: drop what came with it
inline constexpr std::uint32_t CBRO_TRACE_HIDDEN = 1;    // record flags: in the view, behind surfaces
inline constexpr std::uint32_t CBRO_TRACE_OUTSIDE = 2;   // record flags: out of the view
inline constexpr std::uint32_t CBRO_TRACE_SUN_ONLY = 4;  // record flags: a hidden caster the sun's cascades draw (filed into group 0 late)

extern "C"
{
	// The tracer's request. Renderer Replacer keeps a copy of this layout (src/Core/Registrations.cpp): grow it by
	// appending fields, gated on size.
	struct CBRO_TraceFeed
	{
		std::uint32_t size;           // sizeof(CBRO_TraceFeed); 24 for the v1 layout (up to `refused`)
		std::uint32_t refreshFrames;  // an object CBRO leaves out is filed for the tracer once in this many frames (rounded down to a power of two, 4..1024)
		float         radius;         // world units: an object wholly farther from the camera is never filed for it
		void (*refused)(std::uintptr_t a_accumulator, std::uintptr_t a_object);  // each refused registration, on the registering thread
		// v2 (CBRO 2.0)
		std::uint32_t flags;          // CBRO_TRACE_FEED_RETAINS, CBRO_TRACE_FEED_LINK, CBRO_TRACE_FEED_JOB_CASCADES
		std::uint32_t resendFrames;   // with RETAINS: an object handed already is handed again once in this many frames (rounded down to a power of two, refreshFrames..65536; 0: never)
		float         moverRadius;    // with RETAINS: an object that moved is handed every frame within this distance of the camera, one frame in 8 farther (0: every frame)
		std::uint32_t reserved;
	};
	static_assert(sizeof(CBRO_TraceFeed) == 40);

	// a LINK record (v2)
	struct CBRO_TraceRecord
	{
		std::uintptr_t object;  // TAKE, MOVED: a BSGeometry; GONE: 0
		std::uintptr_t anchor;  // the entry CBRO decided on (a fade node, a node, or the geometry itself)
		std::uint32_t  kind;    // CBRO_TRACE_TAKE, _MOVED, _GONE
		std::uint32_t  flags;   // CBRO_TRACE_HIDDEN, _OUTSIDE, _SUN_ONLY
	};
	static_assert(sizeof(CBRO_TraceRecord) == 24);

	struct CBRO_TraceBatch
	{
		const CBRO_TraceRecord* records;  // valid until CBRO's next cull begins
		std::uint32_t           count;
		std::uint32_t           lost;     // records dropped unread since the last read (CBRO then forgot everything: all come again)
	};

	// Exported. Null (or a null `refused`) turns the feed off. Main thread; takes effect at the next cull begin. False when
	// not taken (bTraceFeed=0, an older layout, CBRO not running).
	__declspec(dllexport) bool CBRO_SetTraceFeed(const CBRO_TraceFeed* a_feed);

	// Exported (v2, LINK). The records since the last read; the frame thread (DrawWorld's), after the cull. False (count
	// 0) when the link is off or called from another thread.
	__declspec(dllexport) bool CBRO_TraceLinkRead(CBRO_TraceBatch* a_out);

	// Exported (v2). The tracer let a_object's instance go while the object may still stand (or come back): a_object and
	// every parent up to the scene root are no longer known as handed, so their next turn hands them again. Any thread.
	__declspec(dllexport) void CBRO_TraceFeedForget(std::uintptr_t a_object);
	// Exported (v2). The tracer dropped its whole scene: nothing is known as handed any more. Any thread.
	__declspec(dllexport) void CBRO_TraceFeedForgetAll();
}

namespace CBRO::Core::TraceFeed
{
	void Install();  // after Hooks::CullGroups

	// Main thread at each cull begin, before the sun's cascades are read (Occlusion::SetSunCascades asks On()): this
	// frame's request.
	void BeginFrame() noexcept;
	[[nodiscard]] bool On() noexcept;       // this frame
	[[nodiscard]] bool Retains() noexcept;  // this frame: the tracer keeps what it was handed (v2)
	[[nodiscard]] bool Linked() noexcept;   // this frame: handovers go into the link's batch (v2)
	[[nodiscard]] bool ForwardsCascades() noexcept;  // this frame: the batched cascades' job-thread registrations go into the link (v2.1)

	// An object's turn this frame (one frame in refreshFrames, spread by address).
	[[nodiscard]] bool Turn(const void* a_object, std::uint32_t a_clock) noexcept;
	// Whether an object CBRO leaves out is filed for the tracer this frame by its turn alone: on its turn and, with a
	// retaining tracer, not handed already (or its resend turn).
	[[nodiscard]] bool Due(const void* a_object, std::uint32_t a_clock) noexcept;
	// A retaining tracer's resend turn for an object (one frame in resendFrames, by address; never without one).
	[[nodiscard]] bool ResendTurn(const void* a_object, std::uint32_t a_clock) noexcept;
	// The same for an object that moved: every frame, or with a retaining tracer within its mover radius of a_eye, else
	// one frame in 8.
	[[nodiscard]] bool MoverDue(const void* a_object, const RE::NiBound& a_bound, const float* a_eye, std::uint32_t a_clock) noexcept;
	// Whether any of a bound lies within the tracer's radius of a_eye.
	[[nodiscard]] bool InReach(const RE::NiBound& a_bound, const float* a_eye) noexcept;

	// The known table (v2, a retaining tracer; any thread, relaxed): an object filed or handed for the tracer is known
	// until the tracer forgets it, another object takes its slot (2^17 slots by address; the other is handed again on
	// its turn) or the table is cleared.
	[[nodiscard]] bool Known(const void* a_object) noexcept;
	void               Handed(const void* a_object) noexcept;
	void               Forget(const void* a_object) noexcept;
	// The object Handed() saw last on this thread (the entry walk marks its slot after a decision).
	[[nodiscard]] const void* LastHanded() noexcept;

	// A geometry handed over for an entry (a_anchor): into the link's batch on the main thread, staged on another thread
	// (v2.1, below), else through `refused` as a registration of the main accumulator. a_kind CBRO_TRACE_TAKE or _MOVED.
	void Take(RE::NiAVObject* a_geometry, const RE::NiAVObject* a_anchor, std::uint32_t a_kind, std::uint32_t a_flags) noexcept;
	// An entry the walk had handed left its slot (main thread): a GONE record, and it is no longer known.
	void Gone(const void* a_anchor) noexcept;

	// v2.1, the batched culling path (iCullingBatch; FO4OG 3): the main view's children are filed and the sun's cascades
	// registered by BSJobs jobs. A hand-over made on such a thread is staged and joins the link's batch on the main thread
	// (MergeStaged) once those jobs are done: DrawWorld's cull waits for its jobs before it returns, and CBRO waits for the
	// cascades' jobs at the shadow stage's begin. Both merges come before the tracer reads the batch (its composite).
	// A sun cascade's registration made on a job thread (Hooks::CullGroups' cascade forward, a linked tracer only), staged
	// as the engine's own filing (anchor 0): on the unbatched path the tracer takes the cascades' registrations from the
	// frame thread itself, which a job's are not. Any thread.
	void StageCascade(RE::NiAVObject* a_geometry) noexcept;
	// The staged records into the link's batch, in staging order (main thread, no job staging meanwhile).
	void MergeStaged() noexcept;
	struct StageCounts
	{
		std::uint64_t handovers{ 0 };  // staged hand-overs merged (an anchor: the entry walk's or a filed node's)
		std::uint64_t cascades{ 0 };   // staged cascade registrations merged (anchor 0)
		std::uint64_t lost{ 0 };       // not staged (the stage full): a hand-over went through `refused` instead, a cascade's was dropped
		std::uint64_t merges{ 0 };     // merges that found records
	};
	[[nodiscard]] StageCounts TakeStageCounts() noexcept;  // since the last call (main thread)

	struct Counts
	{
		std::uint64_t records[3]{};  // by kind, appended
		std::uint64_t read{ 0 }, lost{ 0 }, forgetCalls{ 0 }, forgotten{ 0 }, cleared{ 0 };
	};
	[[nodiscard]] Counts TakeCounts() noexcept;  // since the last call (main thread)
}
