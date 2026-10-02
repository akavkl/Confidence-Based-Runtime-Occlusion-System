#include "Core/Async.h"

#include <cmath>
#include <condition_variable>
#include <thread>

namespace CBRO::Core::Async
{
	namespace
	{
		// The v1.36 run: a shared candidate ring (one contended atomic per entry from every walk thread) and a
		// 10 MB hash map (one DRAM access per lookup) cost the main thread 1.6 ms a frame, more than the synchronous
		// judgement it replaced. The engine offers the objects in the same order every frame on each thread (the
		// scene walk on the main thread, the block jobs on the workers), so each thread now keeps its own candidate
		// list, the worker writes that list's records in the same order, and next frame the thread reads them with a
		// cursor (a short look-ahead resyncs after an insertion): sequential, prefetch-friendly, no atomics. A hash
		// index over all records is the fallback for an object that left its thread's sequence.
		constexpr std::size_t kThreads = 32;        // walk threads (the main thread and the engine's block jobs)
		constexpr std::size_t kPerThread = 32768;   // candidates per thread per frame
		constexpr std::size_t kLookAhead = 64;      // sequential resync distance before the hash fallback
		constexpr std::size_t kIndexBits = 17;      // 131,072 index slots (object -> record)
		constexpr std::size_t kIndexSlots = std::size_t{ 1 } << kIndexBits;
		constexpr std::size_t kIndexProbe = 64;

		struct Candidate
		{
			RE::NiAVObject*             object{ nullptr };
			RE::NiBound                 bound{};
			Hooks::CullGroups::GroupKind kind{ Hooks::CullGroups::GroupKind::kUnknown };
		};

		// One thread's candidates of a frame and, once the worker ran, their records in the same order.
		struct Lane
		{
			Candidate*         items{ nullptr };
			Occlusion::Record* records{ nullptr };
			std::uint32_t      count{ 0 };     // candidates appended (the owning thread writes it during the cull)
			std::uint32_t      judged{ 0 };    // records written (the worker)
			std::uint32_t      overflow{ 0 };
		};

		// A frame's lanes (one per thread slot), double-buffered by frame parity: the hooks append to the current
		// parity's lanes and read the other parity's records (the worker's output for last frame's candidates).
		struct FrameLanes
		{
			std::array<Lane, kThreads> lanes{};
			Pose                       pose{};
			float                      moveMargin{ 0.0f };
			float                      turnMargin{ 0.0f };
			bool                       filled{ false };
		};

		// The fallback index: object -> record (generation-stamped, so a fill empties it by bumping the generation).
		struct Index
		{
			struct Slot
			{
				const void*              key{ nullptr };
				std::uint32_t            generation{ 0 };
				const Occlusion::Record* record{ nullptr };
			};
			Slot*         slots{ nullptr };
			std::uint32_t generation{ 0 };
			std::size_t   count{ 0 };
			std::size_t   full{ 0 };

			static std::size_t Hash(const void* a_key) noexcept
			{
				return static_cast<std::size_t>(((reinterpret_cast<std::uintptr_t>(a_key) >> 4) * 0x9E3779B97F4A7C15ull) >> (64 - kIndexBits));
			}

			void Begin() noexcept
			{
				if (++generation == 0) {
					generation = 1;
				}
				count = 0;
				full = 0;
			}

			void Insert(const void* a_key, const Occlusion::Record* a_record) noexcept
			{
				if (count >= kIndexSlots * 3 / 4) {
					++full;
					return;
				}
				auto index = Hash(a_key);
				for (std::size_t probe = 0; probe < kIndexProbe; ++probe, index = (index + 1) & (kIndexSlots - 1)) {
					auto& slot = slots[index];
					if (slot.generation != generation) {
						slot.key = a_key;
						slot.generation = generation;
						slot.record = a_record;
						++count;
						return;
					}
					if (slot.key == a_key) {
						slot.record = a_record;
						return;
					}
				}
				++full;
			}

			const Occlusion::Record* Find(const void* a_key) const noexcept
			{
				auto index = Hash(a_key);
				for (std::size_t probe = 0; probe < kIndexProbe; ++probe, index = (index + 1) & (kIndexSlots - 1)) {
					const auto& slot = slots[index];
					if (slot.generation != generation) {
						return nullptr;
					}
					if (slot.key == a_key) {
						return slot.record;
					}
				}
				return nullptr;
			}
		};

		struct Job
		{
			Occlusion::FrameContext context{};
			FrameLanes*             read{ nullptr };   // last frame's records (streak continuity, cache reuse)
			FrameLanes*             write{ nullptr };  // this frame's candidates -> records
			Index*                  readIndex{ nullptr };
			Index*                  writeIndex{ nullptr };
			double                  ms{ 0.0 };
		};

		bool                       g_enabled{ false };
		FrameLanes                 g_frames[2];
		Index                      g_indices[2];
		std::atomic<std::uint32_t> g_parity{ 0 };       // this frame's lanes = g_frames[parity]; records read = g_frames[parity ^ 1]
		std::atomic<std::uint32_t> g_frameStamp{ 0 };   // bumped at each cull begin: the threads reset their cursors
		std::atomic<bool>          g_readFilled{ false };  // g_frames[parity ^ 1] holds last frame's records (held for this camera or not)
		std::atomic<std::uint32_t> g_nextSlot{ 0 };

		std::thread             g_thread;
		std::mutex              g_mutex;
		std::condition_variable g_wake;
		std::condition_variable g_finished;
		Job                     g_job;
		bool                    g_hasJob{ false };
		bool                    g_busy{ false };
		bool                    g_jobPending{ false };

		// ---- per-thread state -------------------------------------------------------------------------------------
		struct ThreadState
		{
			int           slot{ -1 };
			std::uint32_t stamp{ 0 };    // the frame the cursor belongs to
			std::uint32_t cursor{ 0 };   // next record to try in the read lane
		};
		thread_local ThreadState t_state;

		int Slot() noexcept
		{
			if (t_state.slot < 0) [[unlikely]] {
				const auto slot = g_nextSlot.fetch_add(1, std::memory_order_relaxed);
				t_state.slot = slot < kThreads ? static_cast<int>(slot) : static_cast<int>(kThreads);  // kThreads = no lane
			}
			return t_state.slot;
		}

		// ---- stats (main thread except the worker's own) ---------------------------------------------------------
		struct Stats
		{
			std::uint32_t framesValid{ 0 };
			std::uint32_t framesInvalidMoved{ 0 };
			std::uint32_t framesInvalidTurned{ 0 };
			std::uint32_t framesNoMap{ 0 };
			std::uint32_t submitted{ 0 };
			std::uint32_t skippedBusy{ 0 };
			std::uint64_t candidates{ 0 };
			std::uint64_t overflow{ 0 };
			std::uint64_t indexFull{ 0 };
			std::uint32_t threadsSeen{ 0 };
			double        workerMs{ 0.0 };
			double        workerMaxMs{ 0.0 };
			double        waitMs{ 0.0 };
			double        waitMaxMs{ 0.0 };
			std::uint32_t waits{ 0 };
			double        moveMarginSum{ 0.0 };
			double        turnMarginSum{ 0.0 };
		};
		Stats                      g_stats;
		std::atomic<std::uint64_t> g_lookups{ 0 };
		std::atomic<std::uint64_t> g_hitsSequential{ 0 };
		std::atomic<std::uint64_t> g_hitsAhead{ 0 };
		std::atomic<std::uint64_t> g_hitsIndex{ 0 };

		double Now() noexcept
		{
			LARGE_INTEGER now{}, frequency{};
			QueryPerformanceCounter(&now);
			QueryPerformanceFrequency(&frequency);
			return static_cast<double>(now.QuadPart) * 1000.0 / static_cast<double>(frequency.QuadPart);
		}

		float Distance3(const float a[3], const float b[3]) noexcept
		{
			const float dx = a[0] - b[0], dy = a[1] - b[1], dz = a[2] - b[2];
			return std::sqrt(dx * dx + dy * dy + dz * dz);
		}

		float RotationAngle(const float a[3][3], const float b[3][3]) noexcept
		{
			float squared = 0.0f;
			for (int r = 0; r < 3; ++r) {
				for (int c = 0; c < 3; ++c) {
					const float d = a[r][c] - b[r][c];
					squared += d * d;
				}
			}
			return 2.0f * std::asin(std::min(1.0f, std::sqrt(squared) / 2.8284271f));
		}

		void RunJob(Job& a_job)
		{
			const auto start = Now();
			auto&      write = *a_job.write;
			auto&      index = *a_job.writeIndex;
			index.Begin();
			for (std::size_t s = 0; s < kThreads; ++s) {
				auto& lane = write.lanes[s];
				lane.judged = 0;
				if (!lane.items) {
					continue;
				}
				const auto count = std::min<std::uint32_t>(lane.count, kPerThread);
				const auto* readLane = a_job.read && a_job.read->filled ? &a_job.read->lanes[s] : nullptr;
				std::uint32_t cursor = 0;
				for (std::uint32_t i = 0; i < count; ++i) {
					const auto& candidate = lane.items[i];
					if (!candidate.object) {
						lane.records[i] = Occlusion::Record{};
						continue;
					}
					// Last frame's record: the same lane's sequence first (a cursor with a look-ahead), the index second.
					const Occlusion::Record* old = nullptr;
					if (readLane) {
						const auto judged = readLane->judged;
						for (std::uint32_t k = 0; k < kLookAhead && cursor + k < judged; ++k) {
							if (readLane->records[cursor + k].object == candidate.object) {
								old = &readLane->records[cursor + k];
								cursor += k + 1;
								break;
							}
						}
					}
					if (!old && a_job.readIndex) {
						old = a_job.readIndex->Find(candidate.object);
					}
					const Hooks::CullGroups::BlockAdd add{ nullptr, candidate.object, &candidate.bound, -1, 0, true, candidate.kind, 0 };
					auto&                             out = lane.records[i];
					Occlusion::JudgeAsync(a_job.context, add, old, candidate.kind == Hooks::CullGroups::GroupKind::kSunShared, out);
					index.Insert(candidate.object, &out);
				}
				lane.judged = count;
			}
			write.filled = true;
			a_job.ms = Now() - start;
		}

		void WorkerMain()
		{
			for (;;) {
				Job job;
				{
					std::unique_lock lock(g_mutex);
					g_wake.wait(lock, [] { return g_hasJob; });
					job = g_job;
					g_hasJob = false;
				}
				RunJob(job);
				{
					std::lock_guard lock(g_mutex);
					g_job.ms = job.ms;
					g_busy = false;
				}
				g_finished.notify_all();
			}
		}

		Lane& EnsureLane(FrameLanes& a_frame, int a_slot)
		{
			auto& lane = a_frame.lanes[static_cast<std::size_t>(a_slot)];
			if (!lane.items) {
				lane.items = new Candidate[kPerThread]();
				lane.records = new Occlusion::Record[kPerThread]();
			}
			return lane;
		}
	}

	void Install(bool a_enabled)
	{
		g_enabled = a_enabled;
		if (!a_enabled) {
			logger::info("async: off (bAsync=0): verdicts are evaluated inside the walk, on the walk's threads");
			return;
		}
		for (auto& index : g_indices) {
			index.slots = new Index::Slot[kIndexSlots]();
		}
		g_thread = std::thread(&WorkerMain);
		g_thread.detach();  // (lives with the process; a joinable global thread would terminate the process at exit)
		logger::info(
			"async: on (bAsync=1): each walk thread records its candidates in order and reads last frame's records of the same sequence; a worker judges them after the cull ({} threads x {} candidates a frame at most)",
			kThreads, kPerThread);
	}

	bool Enabled() noexcept
	{
		return g_enabled;
	}

	void Record(RE::NiAVObject* a_object, const RE::NiBound& a_bound, Hooks::CullGroups::GroupKind a_kind) noexcept
	{
		const auto slot = Slot();
		if (slot >= static_cast<int>(kThreads)) {
			return;
		}
		auto& lane = g_frames[g_parity.load(std::memory_order_acquire)].lanes[static_cast<std::size_t>(slot)];
		if (!lane.items) [[unlikely]] {
			// (allocated on first use by the thread itself: the lane is its own)
			lane.items = new Candidate[kPerThread]();
			lane.records = new Occlusion::Record[kPerThread]();
		}
		if (lane.count < kPerThread) {
			auto& item = lane.items[lane.count++];
			item.object = a_object;
			item.bound = a_bound;
			item.kind = a_kind;
		} else {
			++lane.overflow;
		}
	}

	bool BeginFrame(const Pose& a_pose) noexcept
	{
		if (!g_enabled) {
			return false;
		}
		// The worker is idle (Wait ran before the frame's end). Last frame's lanes hold this frame's records to read;
		// the other parity's lanes are cleared for this frame's candidates.
		{
			std::lock_guard lock(g_mutex);
			(void)g_busy;
		}
		const auto parity = g_parity.load(std::memory_order_acquire) ^ 1u;
		auto&      fill = g_frames[parity];
		for (auto& lane : fill.lanes) {
			lane.count = 0;
			lane.overflow = 0;
		}
		fill.filled = false;
		g_parity.store(parity, std::memory_order_release);
		g_frameStamp.fetch_add(1, std::memory_order_release);

		const auto& read = g_frames[parity ^ 1u];
		bool        valid = false;
		if (!read.filled) {
			++g_stats.framesNoMap;
		} else if (Distance3(a_pose.eye, read.pose.eye) > read.moveMargin) {
			++g_stats.framesInvalidMoved;
		} else if (RotationAngle(a_pose.rotate, read.pose.rotate) > read.turnMargin) {
			++g_stats.framesInvalidTurned;
		} else {
			++g_stats.framesValid;
			valid = true;
		}
		g_readFilled.store(read.filled, std::memory_order_release);
		return valid;
	}

	const Occlusion::Record* Find(const void* a_object) noexcept
	{
		if (!g_readFilled.load(std::memory_order_acquire)) {
			return nullptr;
		}
		const auto slot = Slot();
		if (slot >= static_cast<int>(kThreads)) {
			return nullptr;
		}
		g_lookups.fetch_add(1, std::memory_order_relaxed);
		const auto parity = g_parity.load(std::memory_order_acquire);
		const auto& lane = g_frames[parity ^ 1u].lanes[static_cast<std::size_t>(slot)];
		// A new frame: the cursor starts over.
		const auto stamp = g_frameStamp.load(std::memory_order_acquire);
		if (t_state.stamp != stamp) {
			t_state.stamp = stamp;
			t_state.cursor = 0;
		}
		if (lane.records) {
			const auto judged = lane.judged;
			auto&      cursor = t_state.cursor;
			if (cursor < judged && lane.records[cursor].object == a_object) {
				g_hitsSequential.fetch_add(1, std::memory_order_relaxed);
				return &lane.records[cursor++];
			}
			for (std::uint32_t k = 1; k < kLookAhead && cursor + k < judged; ++k) {
				if (lane.records[cursor + k].object == a_object) {
					g_hitsAhead.fetch_add(1, std::memory_order_relaxed);
					const auto* found = &lane.records[cursor + k];
					cursor += k + 1;
					return found;
				}
			}
		}
		if (const auto record = g_indices[parity ^ 1u].Find(a_object)) {
			g_hitsIndex.fetch_add(1, std::memory_order_relaxed);
			return record;
		}
		return nullptr;
	}

	void Submit(const Occlusion::FrameContext& a_context, const Pose& a_pose, float a_moveMargin, float a_turnMargin)
	{
		if (!g_enabled) {
			return;
		}
		std::unique_lock lock(g_mutex);
		if (g_busy) {  // (never, with Wait at the frame's end; counted in case)
			++g_stats.skippedBusy;
			return;
		}
		const auto parity = g_parity.load(std::memory_order_acquire);
		auto&      write = g_frames[parity];
		auto*      read = g_frames[parity ^ 1u].filled ? &g_frames[parity ^ 1u] : nullptr;
		write.pose = a_pose;
		write.moveMargin = a_moveMargin;
		write.turnMargin = a_turnMargin;
		std::uint64_t candidates = 0;
		std::uint32_t threads = 0;
		for (auto& lane : write.lanes) {
			if (lane.items) {
				++threads;
				candidates += std::min<std::uint32_t>(lane.count, kPerThread);
				g_stats.overflow += lane.overflow;
			}
		}
		g_job = Job{ a_context, read, &write, read ? &g_indices[parity ^ 1u] : nullptr, &g_indices[parity], 0.0 };
		g_hasJob = true;
		g_busy = true;
		g_jobPending = true;
		++g_stats.submitted;
		g_stats.candidates += candidates;
		g_stats.threadsSeen = std::max(g_stats.threadsSeen, threads);
		g_stats.moveMarginSum += a_moveMargin;
		g_stats.turnMarginSum += a_turnMargin;
		lock.unlock();
		g_wake.notify_one();
	}

	void Wait() noexcept
	{
		if (!g_enabled) {
			return;
		}
		std::unique_lock lock(g_mutex);
		if (g_busy) {  // (normally the worker is long done: its 1-3 ms against the render's 5+)
			const auto start = Now();
			g_finished.wait(lock, [] { return !g_busy; });
			const auto waited = Now() - start;
			++g_stats.waits;
			g_stats.waitMs += waited;
			g_stats.waitMaxMs = std::max(g_stats.waitMaxMs, waited);
		}
		if (g_jobPending) {
			g_jobPending = false;
			g_stats.workerMs += g_job.ms;
			g_stats.workerMaxMs = std::max(g_stats.workerMaxMs, g_job.ms);
			if (g_job.writeIndex) {
				g_stats.indexFull += g_job.writeIndex->full;
			}
		}
	}

	void Reset() noexcept
	{
		if (!g_enabled) {
			return;
		}
		Wait();
		std::lock_guard lock(g_mutex);
		for (auto& frame : g_frames) {
			frame.filled = false;
			for (auto& lane : frame.lanes) {
				lane.count = 0;
				lane.judged = 0;
				lane.overflow = 0;
			}
		}
		for (auto& index : g_indices) {
			index.Begin();
		}
		g_readFilled.store(false, std::memory_order_release);
	}

	void LogStats(std::uint32_t a_frames)
	{
		if (!g_enabled) {
			return;
		}
		const double frames = std::max(1u, a_frames);
		const auto   lookups = g_lookups.exchange(0);
		const auto   sequential = g_hitsSequential.exchange(0);
		const auto   ahead = g_hitsAhead.exchange(0);
		const auto   indexed = g_hitsIndex.exchange(0);
		const auto&  s = g_stats;
		const double jobs = std::max(1u, s.submitted);
		logger::info(
			"async per interval: map valid {} frames | not used: no map {}, camera moved past the margin {}, turned past it {} | jobs {} (skipped: worker busy {}) | candidates {:.0f}/job on {} threads, overflow {} | worker {:.2f} ms/job (max {:.2f}) | main thread waited {} times, {:.2f} ms avg (max {:.2f}) | lookups {:.0f}/frame: in sequence {:.0f}, by look-ahead {:.0f}, by index {:.0f}, missing {:.0f} | index full {} | margins: move {:.1f} units, turn {:.2f} deg",
			s.framesValid, s.framesNoMap, s.framesInvalidMoved, s.framesInvalidTurned, s.submitted, s.skippedBusy,
			static_cast<double>(s.candidates) / jobs, s.threadsSeen, s.overflow, s.workerMs / jobs, s.workerMaxMs,
			s.waits, s.waits ? s.waitMs / s.waits : 0.0, s.waitMaxMs,
			static_cast<double>(lookups) / frames, static_cast<double>(sequential) / frames, static_cast<double>(ahead) / frames, static_cast<double>(indexed) / frames,
			static_cast<double>(lookups - sequential - ahead - indexed) / frames, s.indexFull,
			s.moveMarginSum / jobs, s.turnMarginSum / jobs * 180.0 / 3.14159265);
		g_stats = {};
	}
}
