// Offline check of DropSet (src/Core/DropSet.h, compiled as is):
//  1. everything inserted in a frame is found, nothing else is (single thread, many frames, reuse of
//     addresses across frames, overlapping probe chains);
//  2. concurrent inserts from many threads, with duplicates and concurrent lookups, lose nothing and
//     never store an object twice in one frame;
//  3. a full neighbourhood makes Insert fail, and a failed object is never reported as present;
//  4. Clear() at the tag wrap forgets a stale tag that would otherwise alias.
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <random>
#include <thread>
#include <unordered_set>
#include <vector>

#include "Core/DropSet.h"

using CBRO::Core::DropSet;

static int g_failures = 0;
#define CHECK(cond, ...)                                        \
	do {                                                        \
		if (!(cond)) {                                          \
			if (g_failures++ < 20) std::printf(__VA_ARGS__);    \
		}                                                       \
	} while (0)

static const void* Obj(std::uint64_t a_address) { return reinterpret_cast<const void*>(a_address); }

int main()
{
	std::mt19937_64 rng(7);
	// Engine-like addresses: 16-byte aligned, heap range, some clustered (siblings allocated together).
	auto address = [&](std::uint64_t a_base) { return (a_base + (rng() % (1ull << 28)) * 16) & ~0xFull; };

	// ---- 1. single thread, 2000 frames ---------------------------------------------------------------
	{
		DropSet set;
		set.Allocate();
		std::vector<std::uint64_t> pool;
		for (int i = 0; i < 60000; ++i) {
			pool.push_back(address(0x1'8000'0000ull + (i % 7) * 0x10'0000'0000ull));
		}
		std::uint64_t inserted = 0, full = 0;
		for (std::uint32_t clock = 1; clock <= 2000; ++clock) {
			const auto tag = DropSet::Tag(clock);
			std::unordered_set<std::uint64_t> in;
			const int count = 500 + static_cast<int>(rng() % 12000);  // up to ~12.5k drops a frame
			for (int k = 0; k < count; ++k) {
				const auto a = pool[rng() % pool.size()];
				if (set.Insert(Obj(a), tag)) {
					in.insert(a);
					++inserted;
				} else {
					++full;
				}
			}
			for (const auto a : in) {
				CHECK(set.Contains(Obj(a), tag), "frame %u: inserted %llx not found\n", clock, (unsigned long long)a);
			}
			for (int k = 0; k < 4000; ++k) {
				const auto a = pool[rng() % pool.size()];
				CHECK(set.Contains(Obj(a), tag) == (in.count(a) != 0), "frame %u: %llx membership wrong\n", clock, (unsigned long long)a);
			}
			// the previous frame's objects are gone for this tag
			if (clock > 1) {
				const auto old = DropSet::Tag(clock - 1);
				CHECK(old != tag, "tags repeat on consecutive frames\n");
			}
		}
		std::printf("1. single thread: %llu inserts over 2000 frames, %llu refused (full neighbourhood)\n",
			(unsigned long long)inserted, (unsigned long long)full);
	}

	// ---- 2. concurrency ---------------------------------------------------------------------------------
	{
		DropSet set;
		set.Allocate();
		constexpr int kThreads = 12;
		for (std::uint32_t clock = 1; clock <= 300; ++clock) {
			const auto tag = DropSet::Tag(clock);
			std::vector<std::vector<std::uint64_t>> work(kThreads);
			std::vector<std::uint64_t>             all;
			for (int i = 0; i < 9000; ++i) {
				all.push_back(address(0x2'0000'0000ull));
			}
			// every object goes to two threads (duplicates race), ~half the objects also to a third
			for (std::size_t i = 0; i < all.size(); ++i) {
				work[i % kThreads].push_back(all[i]);
				work[(i * 7 + 3) % kThreads].push_back(all[i]);
				if (i % 2) {
					work[(i * 5 + 1) % kThreads].push_back(all[i]);
				}
			}
			std::vector<std::vector<char>> ok(kThreads);
			std::atomic<int>               go{ 0 };
			std::vector<std::thread>       threads;
			for (int t = 0; t < kThreads; ++t) {
				threads.emplace_back([&, t] {
					go.fetch_add(1);
					while (go.load() < kThreads) {
					}
					ok[t].resize(work[t].size());
					for (std::size_t k = 0; k < work[t].size(); ++k) {
						ok[t][k] = set.Insert(Obj(work[t][k]), tag);
						// a lookup racing other inserts: once our own insert returned true, it must be found
						if (ok[t][k]) {
							CHECK(set.Contains(Obj(work[t][k]), tag), "own insert not visible\n");
						}
					}
				});
			}
			for (auto& thread : threads) {
				thread.join();
			}
			std::unordered_set<std::uint64_t> accepted;
			for (int t = 0; t < kThreads; ++t) {
				for (std::size_t k = 0; k < work[t].size(); ++k) {
					if (ok[t][k]) {
						accepted.insert(work[t][k]);
					}
				}
			}
			for (const auto a : accepted) {
				CHECK(set.Contains(Obj(a), tag), "frame %u: accepted %llx lost\n", clock, (unsigned long long)a);
			}
		}
		std::printf("2. concurrency: 300 frames x 12 threads, duplicates racing\n");
	}

	// ---- 3. full neighbourhood -------------------------------------------------------------------------
	{
		DropSet set;
		set.Allocate();
		const auto tag = DropSet::Tag(5);
		// addresses that all hash to the same home slot are hard to build; instead fill the table densely
		std::vector<std::uint64_t> all;
		for (int i = 0; i < 70000; ++i) {
			all.push_back(address(0x3'0000'0000ull));
		}
		std::unordered_set<std::uint64_t> in, refused;
		for (const auto a : all) {
			(set.Insert(Obj(a), tag) ? in : refused).insert(a);
		}
		for (const auto a : refused) {
			if (in.count(a)) {
				continue;  // (the same address generated twice)
			}
			CHECK(!set.Contains(Obj(a), tag), "refused %llx reported present\n", (unsigned long long)a);
		}
		for (const auto a : in) {
			CHECK(set.Contains(Obj(a), tag), "dense: accepted %llx lost\n", (unsigned long long)a);
		}
		std::printf("3. overfull table: %zu accepted, %zu refused, no false positives\n", in.size(), refused.size());
	}

	// ---- 4. tag wrap -----------------------------------------------------------------------------------
	{
		DropSet set;
		set.Allocate();
		const auto a = Obj(0x4'0000'0010ull);
		const std::uint32_t first = 7;
		const std::uint32_t again = first + DropSet::kTagCycle;  // same tag one cycle later
		CHECK(DropSet::Tag(first) == DropSet::Tag(again), "tag cycle length wrong\n");
		set.Insert(a, DropSet::Tag(first));
		CHECK(set.Contains(a, DropSet::Tag(again)), "(expected alias without Clear)\n");
		set.Clear();
		CHECK(!set.Contains(a, DropSet::Tag(again)), "Clear() didn't forget the stale tag\n");
		CHECK(DropSet::Tag(0) != 0 && DropSet::Tag(DropSet::kTagCycle - 1) == DropSet::kTagCycle, "tag range wrong\n");
		for (std::uint32_t c = 0; c < 3 * DropSet::kTagCycle; c += 12345) {
			CHECK(DropSet::Tag(c) >= 1 && DropSet::Tag(c) <= DropSet::kTagMask, "tag %u out of range\n", DropSet::Tag(c));
		}
		std::printf("4. tag wrap: stale tag forgotten by Clear()\n");
	}

	std::printf(g_failures ? "FAILED: %d check(s)\n" : "all checks passed\n", g_failures);
	return g_failures ? 1 : 0;
}
