#include "Core/TraceFeed.h"

#include "Hooks/CullGroups.h"
#include "Settings.h"

#include <algorithm>
#include <bit>

namespace CBRO::Core::TraceFeed
{
	namespace
	{
		using Refused = void (*)(std::uintptr_t, std::uintptr_t);

		constexpr std::uint32_t kMinRefresh = 4;
		constexpr std::uint32_t kMaxRefresh = 1024;
		constexpr std::uint32_t kMaxResend = 65536;
		constexpr std::uint32_t kMoverFarMask = 7;          // a mover past the tracer's mover radius: one frame in 8
		constexpr std::uint32_t kKnownBits = 17;            // the known table: 2^17 slots by address (1 MB)
		constexpr std::size_t   kMaxRecords = 1u << 16;     // the link's batch unread past this: dropped, everything forgotten
		constexpr std::size_t   kV1Size = 24;               // CBRO_TraceFeed up to `refused`

		bool g_allowed{ false };  // bTraceFeed, and the accumulator hook installed
		bool g_linkAllowed{ false };  // bTraceLink: v2's requests are taken (else a v2 request runs as v1)

		// The consumer's request (CBRO_SetTraceFeed, main thread), taken at the next cull begin.
		std::atomic<Refused>       g_requested{ nullptr };
		std::atomic<std::uint32_t> g_requestedMask{ 0 };
		std::atomic<float>         g_requestedRadius{ 0.0f };
		std::atomic<std::uint32_t> g_requestedFlags{ 0 };
		std::atomic<std::uint32_t> g_requestedResend{ 0 };  // a power of two, or 0: never
		std::atomic<float>         g_requestedMoverRadius{ 0.0f };

		// This frame's (BeginFrame; read by the walk on any thread, written only before it starts)
		std::atomic<Refused> g_refused{ nullptr };
		bool                 g_on{ false };
		bool                 g_retains{ false };
		bool                 g_linked{ false };
		bool                 g_jobCascades{ false };  // the linked tracer takes the batched cascades' job-thread registrations itself
		std::uint32_t        g_mask{ 0 };
		std::uint32_t        g_resend{ 0 };
		float                g_radius{ 0.0f };
		float                g_moverRadius{ 0.0f };
		DWORD                g_mainThread{ 0 };

		// the known table: an object (its address) per slot, 0 = none
		std::unique_ptr<std::atomic<std::uintptr_t>[]> g_known;

		// the link's batch (main thread): appended in the cull, read by the tracer after it; cleared at the next append or
		// cull begin once read
		std::vector<CBRO_TraceRecord> g_batch;
		bool                          g_batchRead{ false };
		std::uint32_t                 g_lost{ 0 };

		thread_local const void* t_lastHanded{ nullptr };

		// The stage (v2.1): records handed over on job threads. A thread reserves kStageChunk slots at a time (one shared
		// atomic add per chunk, not per record) and fills them in order; a reserved slot it never fills stays zero, which the
		// merge skips. The merge runs only after the jobs that stage are done (their waits are the happens-before), clears what
		// it read, and starts a new generation so that no thread writes into a chunk reserved before it.
		constexpr std::uint32_t                kStageRecords = 1u << 15;  // 786 KB
		constexpr std::uint32_t                kStageChunk = 64;
		std::unique_ptr<CBRO_TraceRecord[]>    g_stage;
		std::atomic<std::uint32_t>             g_stageTop{ 0 };
		std::atomic<std::uint32_t>             g_stageGeneration{ 1 };
		std::atomic<std::uint64_t>             g_stageLost{ 0 };
		StageCounts                            g_stageCounts{};  // (main thread: the merge's)
		struct StageCursor
		{
			std::uint32_t next{ 0 };
			std::uint32_t end{ 0 };
			std::uint32_t generation{ 0 };
		};
		thread_local StageCursor t_stage;

		bool Stage(const CBRO_TraceRecord& a_record) noexcept
		{
			if (!g_stage || !a_record.object) {
				return false;
			}
			const auto generation = g_stageGeneration.load(std::memory_order_acquire);
			if (t_stage.generation != generation || t_stage.next == t_stage.end) {
				const auto first = g_stageTop.fetch_add(kStageChunk, std::memory_order_relaxed);
				if (first >= kStageRecords) {
					t_stage = { 0, 0, generation };
					g_stageLost.fetch_add(1, std::memory_order_relaxed);
					return false;
				}
				t_stage = { first, std::min(first + kStageChunk, kStageRecords), generation };
			}
			g_stage[t_stage.next++] = a_record;
			return true;
		}

		Counts                     g_counts{};  // (main thread, but forgetting's)
		std::atomic<std::uint64_t> g_forgetCalls{ 0 };
		std::atomic<std::uint64_t> g_forgotten{ 0 };
		std::atomic<std::uint64_t> g_cleared{ 0 };

		void Forward(const void* a_accumulator, RE::NiAVObject* a_object)
		{
			if (const auto refused = g_refused.load(std::memory_order_relaxed)) {
				refused(reinterpret_cast<std::uintptr_t>(a_accumulator), reinterpret_cast<std::uintptr_t>(a_object));
			}
		}

		inline std::uint32_t TurnHash(const void* a_object) noexcept
		{
			return static_cast<std::uint32_t>(((reinterpret_cast<std::uintptr_t>(a_object) >> 4) * 0x9E3779B97F4A7C15ull) >> 32);
		}

		inline std::atomic<std::uintptr_t>& KnownSlot(const void* a_object) noexcept
		{
			const auto hash = ((reinterpret_cast<std::uintptr_t>(a_object) >> 4) * 0xD6E8FEB86659FD93ull) >> (64 - kKnownBits);
			return g_known[static_cast<std::size_t>(hash)];
		}

		void ClearKnown() noexcept
		{
			if (!g_known) {
				return;
			}
			for (std::size_t i = 0; i < (std::size_t{ 1 } << kKnownBits); ++i) {
				g_known[i].store(0, std::memory_order_relaxed);
			}
			g_cleared.fetch_add(1, std::memory_order_relaxed);
		}

		void Append(const CBRO_TraceRecord& a_record) noexcept
		{
			if (g_batchRead) {
				g_batch.clear();
				g_batchRead = false;
			}
			if (g_batch.size() >= kMaxRecords) {
				// (the tracer stopped reading: what it missed comes again on the turns)
				g_lost += static_cast<std::uint32_t>(g_batch.size());
				g_counts.lost += g_batch.size();
				g_batch.clear();
				ClearKnown();
			}
			g_batch.push_back(a_record);
			++g_counts.records[std::min<std::uint32_t>(a_record.kind, 2)];
		}
	}

	void Install()
	{
		const auto& settings = Settings::Get();
		g_allowed = settings.traceFeed;
		g_linkAllowed = settings.traceFeed && settings.traceLink;
		if (g_linkAllowed) {
			g_known = std::make_unique<std::atomic<std::uintptr_t>[]>(std::size_t{ 1 } << kKnownBits);
			g_batch.reserve(4096);
			g_stage = std::make_unique<CBRO_TraceRecord[]>(kStageRecords);  // (zeroed: an unfilled slot reads as none)
		}
		logger::info("trace feed: {}", !g_allowed ? "off (bTraceFeed=0)" :
		                               g_linkAllowed ? "on request (a path tracer's scene: Renderer Replacer); a tracer that keeps what it is handed gets each object once, through the direct link (bTraceLink=1)" :
		                                               "on request (a path tracer's scene: Renderer Replacer); every turn files again (bTraceLink=0)");
	}

	void BeginFrame() noexcept
	{
		g_mainThread = GetCurrentThreadId();
		const auto refused = g_allowed ? g_requested.load(std::memory_order_acquire) : nullptr;
		const bool on = refused != nullptr;
		const auto mask = g_requestedMask.load(std::memory_order_relaxed);
		const auto radius = g_requestedRadius.load(std::memory_order_relaxed);
		const auto flags = g_linkAllowed ? g_requestedFlags.load(std::memory_order_relaxed) : 0u;
		const bool retains = on && (flags & CBRO_TRACE_FEED_RETAINS) != 0;
		const bool linked = retains && (flags & CBRO_TRACE_FEED_LINK) != 0;
		const bool jobCascades = linked && (flags & CBRO_TRACE_FEED_JOB_CASCADES) != 0;
		const auto resend = g_requestedResend.load(std::memory_order_relaxed);
		const auto moverRadius = g_requestedMoverRadius.load(std::memory_order_relaxed);
		if (on != g_on || (on && (mask != g_mask || radius != g_radius || retains != g_retains || linked != g_linked || jobCascades != g_jobCascades || resend != g_resend || moverRadius != g_moverRadius))) {
			if (!on) {
				logger::info("trace feed: off");
			} else if (retains) {
				logger::info("trace feed: on, the tracer keeps what it is handed (v2): an object CBRO leaves out is handed once (its first turn: one frame in {}), again when it moves (every frame within {:.0f} units, one frame in {} farther) or {} | within {:.0f} units | {}",
					mask + 1, moverRadius, kMoverFarMask + 1, resend ? std::format("once in {} frames", resend) : std::string("never otherwise"), radius,
					linked ? jobCascades ? "handovers through the direct link (batched records with their entries, GONE when an entry handed leaves its slot); the batched cascades' job-thread registrations taken by the tracer itself" :
					                       "handovers through the direct link (batched records with their entries, GONE when an entry handed leaves its slot); the batched cascades' job-thread registrations forwarded through it" :
					         "handovers through the refused callback");
			} else {
				logger::info("trace feed: on (refused registrations handed over; what CBRO leaves out filed once every {} frames, and when it moves, within {:.0f} units)", mask + 1, radius);
			}
			if (retains != g_retains || !on) {
				ClearKnown();  // (a tracer that starts keeping, or a new one, holds nothing CBRO handed before)
			}
		}
		if (!linked || g_batchRead) {
			g_batch.clear();
			g_batchRead = false;
		}
		g_on = on;
		g_retains = retains;
		g_linked = linked;
		g_jobCascades = jobCascades;
		g_mask = mask;
		g_resend = resend;
		g_radius = radius;
		g_moverRadius = moverRadius;
		g_refused.store(refused, std::memory_order_relaxed);
		Hooks::CullGroups::SetRefusedObserver(on ? &Forward : nullptr);
	}

	bool On() noexcept
	{
		return g_on;
	}

	bool Retains() noexcept
	{
		return g_retains;
	}

	bool Linked() noexcept
	{
		return g_linked;
	}

	bool ForwardsCascades() noexcept
	{
		return g_linked && !g_jobCascades;
	}

	bool Turn(const void* a_object, std::uint32_t a_clock) noexcept
	{
		return ((TurnHash(a_object) + a_clock) & g_mask) == 0;
	}

	bool Due(const void* a_object, std::uint32_t a_clock) noexcept
	{
		const auto turn = TurnHash(a_object) + a_clock;
		if ((turn & g_mask) != 0) {
			return false;
		}
		if (!g_retains || !Known(a_object)) {
			return true;
		}
		return g_resend && (turn & (g_resend - 1)) == 0;
	}

	bool ResendTurn(const void* a_object, std::uint32_t a_clock) noexcept
	{
		return g_retains && g_resend && ((TurnHash(a_object) + a_clock) & (g_resend - 1)) == 0;
	}

	bool MoverDue(const void* a_object, const RE::NiBound& a_bound, const float* a_eye, std::uint32_t a_clock) noexcept
	{
		if (!g_retains || !(g_moverRadius > 0.0f)) {
			return true;
		}
		const float dx = a_bound.center.x - a_eye[0], dy = a_bound.center.y - a_eye[1], dz = a_bound.center.z - a_eye[2];
		const float reach = g_moverRadius + std::max(a_bound.fRadius, 0.0f);
		return dx * dx + dy * dy + dz * dz <= reach * reach || ((TurnHash(a_object) + a_clock) & kMoverFarMask) == 0;
	}

	bool InReach(const RE::NiBound& a_bound, const float* a_eye) noexcept
	{
		const float dx = a_bound.center.x - a_eye[0], dy = a_bound.center.y - a_eye[1], dz = a_bound.center.z - a_eye[2];
		const float reach = g_radius + std::max(a_bound.fRadius, 0.0f);
		return dx * dx + dy * dy + dz * dz <= reach * reach;
	}

	bool Known(const void* a_object) noexcept
	{
		return g_known && a_object && KnownSlot(a_object).load(std::memory_order_relaxed) == reinterpret_cast<std::uintptr_t>(a_object);
	}

	void Handed(const void* a_object) noexcept
	{
		t_lastHanded = a_object;
		if (g_retains && g_known && a_object) {
			KnownSlot(a_object).store(reinterpret_cast<std::uintptr_t>(a_object), std::memory_order_relaxed);
		}
	}

	void Forget(const void* a_object) noexcept
	{
		if (!g_known || !a_object) {
			return;
		}
		auto expected = reinterpret_cast<std::uintptr_t>(a_object);
		if (KnownSlot(a_object).compare_exchange_strong(expected, 0, std::memory_order_relaxed)) {
			g_forgotten.fetch_add(1, std::memory_order_relaxed);
		}
	}

	const void* LastHanded() noexcept
	{
		return t_lastHanded;
	}

	void Take(RE::NiAVObject* a_geometry, const RE::NiAVObject* a_anchor, std::uint32_t a_kind, std::uint32_t a_flags) noexcept
	{
		if (!a_geometry) {
			return;
		}
		if (g_linked) {
			const CBRO_TraceRecord record{ reinterpret_cast<std::uintptr_t>(a_geometry), reinterpret_cast<std::uintptr_t>(a_anchor), a_kind, a_flags };
			if (GetCurrentThreadId() == g_mainThread) {
				Append(record);
				return;
			}
			if (Stage(record)) {
				return;  // (a job thread of the batched path: merged after its jobs)
			}
		}
		Hooks::CullGroups::HandToTracer(a_geometry);
	}

	void StageCascade(RE::NiAVObject* a_geometry) noexcept
	{
		if (g_linked && a_geometry) {
			(void)Stage({ reinterpret_cast<std::uintptr_t>(a_geometry), 0, CBRO_TRACE_TAKE, 0 });
		}
	}

	void MergeStaged() noexcept
	{
		if (!g_stage) {
			return;
		}
		const auto top = std::min(g_stageTop.load(std::memory_order_acquire), kStageRecords);
		if (top == 0) {
			return;
		}
		for (std::uint32_t i = 0; i < top; ++i) {
			auto& record = g_stage[i];
			if (!record.object) {
				continue;  // (the unfilled rest of a thread's chunk)
			}
			if (g_linked) {
				Append(record);
				++(record.anchor ? g_stageCounts.handovers : g_stageCounts.cascades);
			}
			record = {};
		}
		++g_stageCounts.merges;
		g_stageTop.store(0, std::memory_order_relaxed);
		g_stageGeneration.fetch_add(1, std::memory_order_release);
	}

	StageCounts TakeStageCounts() noexcept
	{
		auto out = g_stageCounts;
		g_stageCounts = {};
		out.lost = g_stageLost.exchange(0, std::memory_order_relaxed);
		return out;
	}

	void Gone(const void* a_anchor) noexcept
	{
		if (!a_anchor) {
			return;
		}
		Forget(a_anchor);
		if (g_linked && GetCurrentThreadId() == g_mainThread) {
			Append({ 0, reinterpret_cast<std::uintptr_t>(a_anchor), CBRO_TRACE_GONE, 0 });
		}
	}

	Counts TakeCounts() noexcept
	{
		Counts out = g_counts;
		g_counts = {};
		out.forgetCalls = g_forgetCalls.exchange(0, std::memory_order_relaxed);
		out.forgotten = g_forgotten.exchange(0, std::memory_order_relaxed);
		out.cleared = g_cleared.exchange(0, std::memory_order_relaxed);
		return out;
	}

	// (the exports' access)
	namespace Exports
	{
		bool Read(CBRO_TraceBatch* a_out) noexcept
		{
			*a_out = { nullptr, 0, 0 };
			if (!g_linked || GetCurrentThreadId() != g_mainThread) {
				return false;
			}
			if (g_batchRead) {
				g_batch.clear();  // (read twice in a frame: nothing since)
			}
			a_out->records = g_batch.data();
			a_out->count = static_cast<std::uint32_t>(g_batch.size());
			a_out->lost = g_lost;
			g_lost = 0;
			g_batchRead = true;
			g_counts.read += g_batch.size();
			return true;
		}

		void ForgetChain(std::uintptr_t a_object) noexcept
		{
			if (!g_known || !a_object) {
				return;
			}
			g_forgetCalls.fetch_add(1, std::memory_order_relaxed);
			__try {
				auto node = reinterpret_cast<const RE::NiAVObject*>(a_object);
				for (int depth = 0; node && depth < 64; ++depth) {
					Forget(node);
					node = node->parent;
				}
			} __except (EXCEPTION_EXECUTE_HANDLER) {
			}
		}

		void ForgetAll() noexcept
		{
			ClearKnown();
		}
	}
}

bool CBRO_SetTraceFeed(const CBRO_TraceFeed* a_feed)
{
	using namespace CBRO::Core::TraceFeed;
	if (!a_feed || !a_feed->refused) {
		g_requested.store(nullptr, std::memory_order_release);
		return true;
	}
	if (a_feed->size < kV1Size || !g_allowed) {
		return false;
	}
	const auto refresh = std::bit_floor(std::clamp(a_feed->refreshFrames, kMinRefresh, kMaxRefresh));
	const bool v2 = a_feed->size >= sizeof(CBRO_TraceFeed);
	const auto resend = v2 && a_feed->resendFrames ? std::bit_floor(std::clamp(a_feed->resendFrames, refresh, kMaxResend)) : 0u;
	g_requestedMask.store(refresh - 1, std::memory_order_relaxed);
	g_requestedRadius.store(std::max(a_feed->radius, 0.0f), std::memory_order_relaxed);
	g_requestedFlags.store(v2 ? a_feed->flags : 0u, std::memory_order_relaxed);
	g_requestedResend.store(resend, std::memory_order_relaxed);
	g_requestedMoverRadius.store(v2 ? std::max(a_feed->moverRadius, 0.0f) : 0.0f, std::memory_order_relaxed);
	g_requested.store(a_feed->refused, std::memory_order_release);
	return true;
}

bool CBRO_TraceLinkRead(CBRO_TraceBatch* a_out)
{
	return a_out && CBRO::Core::TraceFeed::Exports::Read(a_out);
}

void CBRO_TraceFeedForget(std::uintptr_t a_object)
{
	CBRO::Core::TraceFeed::Exports::ForgetChain(a_object);
}

void CBRO_TraceFeedForgetAll()
{
	CBRO::Core::TraceFeed::Exports::ForgetAll();
}
