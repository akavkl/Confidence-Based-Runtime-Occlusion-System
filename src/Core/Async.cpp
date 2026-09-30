#include "Core/Async.h"

#include <cmath>
#include <condition_variable>
#include <thread>

namespace CBRO::Core::Async
{
	namespace
	{
		constexpr std::size_t kRing = 65536;   // candidates per frame (the walk offers ~10k at a dense exterior)
		constexpr std::size_t kMapBits = 17;   // 131,072 slots per map
		constexpr std::size_t kMapSlots = std::size_t{ 1 } << kMapBits;
		constexpr std::size_t kMapProbe = 64;

		struct Candidate
		{
			RE::NiAVObject*             object{ nullptr };
			RE::NiBound                 bound{};
			Hooks::CullGroups::GroupKind kind{ Hooks::CullGroups::GroupKind::kUnknown };
		};

		struct Ring
		{
			std::atomic<std::uint32_t> count{ 0 };
			Candidate*                 items{ nullptr };
		};

		// Open addressing keyed by the object pointer; a slot belongs to the fill whose generation it carries, so a
		// map is emptied by bumping its generation (no clearing).
		struct Map
		{
			struct Slot
			{
				const void*       key{ nullptr };
				std::uint32_t     generation{ 0 };
				Occlusion::Record record{};
			};
			Slot*         slots{ nullptr };
			std::uint32_t generation{ 0 };
			std::size_t   count{ 0 };
			std::size_t   full{ 0 };   // inserts refused (a full map: those objects stay drawn)
			Pose          pose{};
			float         moveMargin{ 0.0f };
			float         turnMargin{ 0.0f };
			bool          filled{ false };

			static std::size_t Hash(const void* a_key) noexcept
			{
				return static_cast<std::size_t>(((reinterpret_cast<std::uintptr_t>(a_key) >> 4) * 0x9E3779B97F4A7C15ull) >> (64 - kMapBits));
			}

			void Begin(const Pose& a_pose, float a_moveMargin, float a_turnMargin) noexcept
			{
				++generation;
				if (generation == 0) {
					generation = 1;
				}
				count = 0;
				full = 0;
				pose = a_pose;
				moveMargin = a_moveMargin;
				turnMargin = a_turnMargin;
				filled = false;
			}

			void Insert(const void* a_key, const Occlusion::Record& a_record) noexcept
			{
				if (count >= kMapSlots * 3 / 4) {
					++full;
					return;
				}
				auto index = Hash(a_key);
				for (std::size_t probe = 0; probe < kMapProbe; ++probe, index = (index + 1) & (kMapSlots - 1)) {
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
				for (std::size_t probe = 0; probe < kMapProbe; ++probe, index = (index + 1) & (kMapSlots - 1)) {
					const auto& slot = slots[index];
					if (slot.generation != generation) {
						return nullptr;
					}
					if (slot.key == a_key) {
						return &slot.record;
					}
				}
				return nullptr;
			}
		};

		struct Job
		{
			Occlusion::FrameContext context{};
			Pose                    pose{};
			float                   moveMargin{ 0.0f };
			float                   turnMargin{ 0.0f };
			Ring*                   ring{ nullptr };
			Map*                    read{ nullptr };   // last frame's records (streak continuity, cache reuse)
			Map*                    write{ nullptr };  // this frame's records
		};

		bool                    g_enabled{ false };
		Ring                    g_rings[2];
		std::atomic<Ring*>      g_fill{ nullptr };  // the ring the hooks append to
		Map                     g_maps[2];
		std::atomic<Map*>       g_current{ nullptr };  // the map the hooks read (null = none valid)
		Map*                    g_published{ nullptr };  // the last map the worker filled
		bool                    g_validThisFrame{ false };

		std::thread             g_thread;
		std::mutex              g_mutex;
		std::condition_variable g_wake;
		std::condition_variable g_finished;
		Job                     g_job;
		bool                    g_hasJob{ false };
		bool                    g_busy{ false };
		bool                    g_jobPending{ false };  // a submitted job not yet accounted for by Wait
		bool                    g_quit{ false };

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
			std::uint64_t ringOverflow{ 0 };
			std::uint64_t mapFull{ 0 };
			double        workerMs{ 0.0 };
			double        workerMaxMs{ 0.0 };
			double        waitMs{ 0.0 };
			double        waitMaxMs{ 0.0 };
			std::uint32_t waits{ 0 };
			double        moveMarginSum{ 0.0 };
			double        turnMarginSum{ 0.0 };
		};
		Stats g_stats;
		std::atomic<std::uint64_t> g_lookups{ 0 };
		std::atomic<std::uint64_t> g_hits{ 0 };

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
			const auto  start = Now();
			auto&       ring = *a_job.ring;
			const auto  count = std::min<std::size_t>(ring.count.load(std::memory_order_acquire), kRing);
			auto&       write = *a_job.write;
			const auto* read = a_job.read;
			write.Begin(a_job.pose, a_job.moveMargin, a_job.turnMargin);
			for (std::size_t i = 0; i < count; ++i) {
				const auto& candidate = ring.items[i];
				if (!candidate.object) {
					continue;
				}
				const Hooks::CullGroups::BlockAdd add{ nullptr, candidate.object, &candidate.bound, -1, 0, true, candidate.kind, 0 };
				const Occlusion::Record*          old = read ? read->Find(candidate.object) : nullptr;
				Occlusion::Record                 out{};
				Occlusion::JudgeAsync(a_job.context, add, old, candidate.kind == Hooks::CullGroups::GroupKind::kSunShared, out);
				write.Insert(candidate.object, out);
			}
			write.filled = true;
			ring.count.store(0, std::memory_order_release);
			const auto ms = Now() - start;
			// (the stats are the main thread's; the worker only reports its time through the job)
			a_job.moveMargin = static_cast<float>(ms);  // reused as the job's duration for the main thread
		}

		void WorkerMain()
		{
			for (;;) {
				Job job;
				{
					std::unique_lock lock(g_mutex);
					g_wake.wait(lock, [] { return g_hasJob || g_quit; });
					if (g_quit) {
						return;
					}
					job = g_job;
					g_hasJob = false;
				}
				RunJob(job);
				{
					std::lock_guard lock(g_mutex);
					g_job.moveMargin = job.moveMargin;  // the duration
					g_busy = false;
				}
				g_finished.notify_all();
			}
		}
	}

	void Install(bool a_enabled)
	{
		g_enabled = a_enabled;
		if (!a_enabled) {
			logger::info("async: off (bAsync=0): verdicts are evaluated inside the walk, on the walk's threads");
			return;
		}
		for (auto& ring : g_rings) {
			ring.items = new Candidate[kRing]();
		}
		for (auto& map : g_maps) {
			map.slots = new Map::Slot[kMapSlots]();
		}
		g_fill.store(&g_rings[0], std::memory_order_release);
		g_thread = std::thread(&WorkerMain);
		g_thread.detach();  // (lives with the process; a joinable global thread would terminate the process at exit)
		logger::info(
			"async: on (bAsync=1): the walk looks verdicts up in a map and records candidates; a worker judges them after the cull, into the map the next frame reads ({} candidates a frame at most, {} records a map)",
			kRing, kMapSlots);
	}

	bool Enabled() noexcept
	{
		return g_enabled;
	}

	void Record(RE::NiAVObject* a_object, const RE::NiBound& a_bound, Hooks::CullGroups::GroupKind a_kind) noexcept
	{
		auto* ring = g_fill.load(std::memory_order_acquire);
		if (!ring) {
			return;
		}
		const auto index = ring->count.fetch_add(1, std::memory_order_relaxed);
		if (index < kRing) {
			auto& item = ring->items[index];
			item.object = a_object;
			item.bound = a_bound;
			item.kind = a_kind;
		}
	}

	bool BeginFrame(const Pose& a_pose) noexcept
	{
		if (!g_enabled) {
			return false;
		}
		// The worker's last map becomes the one the hooks read (never while the worker fills it: Wait ran before the
		// frame's end, so the worker is idle here).
		{
			std::lock_guard lock(g_mutex);
			if (!g_busy && g_published && g_published->filled) {
				g_current.store(g_published, std::memory_order_release);
			}
		}
		const auto* map = g_current.load(std::memory_order_acquire);
		if (!map || !map->filled) {
			++g_stats.framesNoMap;
			g_validThisFrame = false;
			return false;
		}
		const float moved = Distance3(a_pose.eye, map->pose.eye);
		const float turned = RotationAngle(a_pose.rotate, map->pose.rotate);
		if (moved > map->moveMargin) {
			++g_stats.framesInvalidMoved;
			g_validThisFrame = false;
			return false;
		}
		if (turned > map->turnMargin) {
			++g_stats.framesInvalidTurned;
			g_validThisFrame = false;
			return false;
		}
		++g_stats.framesValid;
		g_validThisFrame = true;
		return true;
	}

	const Occlusion::Record* Find(const void* a_object) noexcept
	{
		const auto* map = g_current.load(std::memory_order_acquire);
		if (!map) {
			return nullptr;
		}
		g_lookups.fetch_add(1, std::memory_order_relaxed);
		const auto* record = map->Find(a_object);
		if (record) {
			g_hits.fetch_add(1, std::memory_order_relaxed);
		}
		return record;
	}

	void Submit(const Occlusion::FrameContext& a_context, const Pose& a_pose, float a_moveMargin, float a_turnMargin)
	{
		if (!g_enabled) {
			return;
		}
		auto* ring = g_fill.load(std::memory_order_acquire);
		const auto count = ring->count.load(std::memory_order_acquire);
		if (count > kRing) {
			g_stats.ringOverflow += count - kRing;
		}
		std::unique_lock lock(g_mutex);
		if (g_busy) {  // (never, with Wait at the frame's end; counted in case)
			++g_stats.skippedBusy;
			ring->count.store(0, std::memory_order_release);
			return;
		}
		// The hooks append to the other ring from now on; the worker owns this one until it finishes.
		g_fill.store(ring == &g_rings[0] ? &g_rings[1] : &g_rings[0], std::memory_order_release);
		g_rings[ring == &g_rings[0] ? 1 : 0].count.store(0, std::memory_order_release);
		auto* read = g_current.load(std::memory_order_acquire);
		auto* write = read == &g_maps[0] ? &g_maps[1] : &g_maps[0];
		if (!read) {
			write = &g_maps[0];
		}
		g_job = Job{ a_context, a_pose, a_moveMargin, a_turnMargin, ring, read, write };
		g_published = write;
		g_hasJob = true;
		g_busy = true;
		g_jobPending = true;
		++g_stats.submitted;
		g_stats.candidates += std::min<std::size_t>(count, kRing);
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
			const double ms = g_job.moveMargin;  // the job's duration, as RunJob left it
			g_stats.workerMs += ms;
			g_stats.workerMaxMs = std::max(g_stats.workerMaxMs, ms);
			if (g_published) {
				g_stats.mapFull += g_published->full;
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
		g_current.store(nullptr, std::memory_order_release);
		g_published = nullptr;
		for (auto& map : g_maps) {
			map.filled = false;
			++map.generation;
		}
		for (auto& ring : g_rings) {
			ring.count.store(0, std::memory_order_release);
		}
		g_validThisFrame = false;
	}

	void LogStats(std::uint32_t a_frames)
	{
		if (!g_enabled) {
			return;
		}
		const double frames = std::max(1u, a_frames);
		const auto   lookups = g_lookups.exchange(0);
		const auto   hits = g_hits.exchange(0);
		const auto&  s = g_stats;
		const double jobs = std::max(1u, s.submitted);
		logger::info(
			"async per interval: map valid {} frames | not used: no map {}, camera moved past the margin {}, turned past it {} | jobs {} (skipped: worker busy {}) | candidates {:.0f}/job, ring overflow {} | worker {:.2f} ms/job (max {:.2f}) | main thread waited {} times, {:.2f} ms avg (max {:.2f}) | lookups {:.0f}/frame, hits {:.0f} | map full {} | margins: move {:.1f} units, turn {:.2f} deg",
			s.framesValid, s.framesNoMap, s.framesInvalidMoved, s.framesInvalidTurned, s.submitted, s.skippedBusy,
			static_cast<double>(s.candidates) / jobs, s.ringOverflow, s.workerMs / jobs, s.workerMaxMs,
			s.waits, s.waits ? s.waitMs / s.waits : 0.0, s.waitMaxMs, static_cast<double>(lookups) / frames, static_cast<double>(hits) / frames, s.mapFull,
			s.moveMarginSum / jobs, s.turnMarginSum / jobs * 180.0 / 3.14159265);
		g_stats = {};
	}
}
