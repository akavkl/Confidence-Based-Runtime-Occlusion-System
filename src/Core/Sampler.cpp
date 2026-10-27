#include "Core/Sampler.h"

#include <Psapi.h>

#include <algorithm>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <thread>

namespace CBRO::Core::Sampler
{
	namespace
	{
		constexpr std::size_t   kMaxModules = 512;
		constexpr std::size_t   kRing = 1 << 17;      // samples kept between summaries (131k: ~50 s at 2.5 kHz)
		constexpr std::size_t   kStackBytes = 512;    // the top of the main thread's stack, copied per sample and scanned for the caller
		constexpr std::int64_t  kPeriod100ns = 4000;  // 400 us between samples
		constexpr double        kStaleMs = 100.0;     // no stage mark for this long (loading, a hitch, no world): not sampled
		constexpr std::size_t   kTop = 10;            // busiest addresses listed per stage
		constexpr double        kOwnShare = 0.05;     // CBRO.dll's addresses are listed in full for a stage where it holds at least this share
		constexpr double        kOwnMinShare = 0.002;  // ... each address down to this share of the stage
		constexpr std::size_t   kOwnTop = 64;         // ... at most this many
		constexpr double        kShownShare = 0.02;   // stages below this share of a mode's samples are not listed
		constexpr std::uint32_t kNoPhase = 0xFFFF'FFFFu;

		// System libraries: a sample in one is charged to its first non-system caller (see the header).
		constexpr std::string_view kSystemModules[]{
			"ntdll.dll", "kernel32.dll", "kernelbase.dll", "msvcp140.dll", "vcruntime140.dll", "vcruntime140_1.dll", "ucrtbase.dll",
			"d3d11.dll", "dxgi.dll", "d3dcompiler_47.dll", "dxcore.dll", "nvwgf2umx.dll", "nvldumdx.dll", "nvapi64.dll",
			"win32u.dll", "user32.dll", "gdi32.dll", "gdi32full.dll", "combase.dll", "rpcrt4.dll", "ntmarta.dll"
		};

		struct Module
		{
			std::uintptr_t base{ 0 };
			std::uintptr_t end{ 0 };
			bool           system{ false };
			char           name[48]{};
		};

		struct Sample
		{
			std::uintptr_t rip{ 0 };
			std::uintptr_t caller{ 0 };
			std::uint32_t  phase{ 0 };
		};

		bool                              g_enabled{ false };
		std::span<const std::string_view> g_modeNames;
		std::span<const std::string_view> g_bucketNames;
		std::int64_t                      g_frequency{ 1 };

		std::atomic<bool>          g_run{ false };     // the thread samples (false: parks for a summary)
		std::atomic<bool>          g_parked{ false };  // the thread's acknowledgement
		std::atomic<bool>          g_started{ false };
		std::atomic<std::uint32_t> g_phase{ kNoPhase };
		std::atomic<std::int64_t>  g_stamp{ 0 };
		HANDLE                     g_mainThread{ nullptr };

		// The sampler thread's while it runs, the main thread's while it is parked.
		Module*       g_modules{ nullptr };
		std::size_t   g_moduleCount{ 0 };
		Sample*       g_ring{ nullptr };
		std::uint32_t g_count{ 0 };
		std::uint32_t g_dropped{ 0 };  // the ring was full
		std::uint32_t g_failed{ 0 };   // the suspend or the context read failed
		// Every summary's full histogram (every charged address with 2+ samples, by mode and stage) goes to a CSV next to
		// the recordings, for tools/sampler_names.py to roll up by function: the log's top-10 and 0.2%-floor lists left
		// ~30% of CBRO.dll's cull samples and all of Fallout4.exe's tail unseen in the 2026-10-27 run (v1.96).
		std::filesystem::path g_csv;
		std::uint32_t         g_summaries{ 0 };

		std::int64_t Qpc() noexcept
		{
			LARGE_INTEGER now;
			QueryPerformanceCounter(&now);
			return now.QuadPart;
		}

		bool IsSystem(const char* a_name) noexcept
		{
			for (const auto& system : kSystemModules) {
				if (_stricmp(a_name, system.data()) == 0) {
					return true;
				}
			}
			return false;
		}

		// Main thread, the sampler parked: the loaded modules and their ranges.
		void RefreshModules() noexcept
		{
			HMODULE handles[kMaxModules];
			DWORD   needed = 0;
			if (!K32EnumProcessModules(GetCurrentProcess(), handles, sizeof(handles), &needed)) {
				return;
			}
			const auto  count = std::min<std::size_t>(needed / sizeof(HMODULE), kMaxModules);
			std::size_t used = 0;
			for (std::size_t i = 0; i < count; ++i) {
				MODULEINFO info{};
				if (!K32GetModuleInformation(GetCurrentProcess(), handles[i], &info, sizeof(info))) {
					continue;
				}
				auto& module = g_modules[used];
				module.base = reinterpret_cast<std::uintptr_t>(info.lpBaseOfDll);
				module.end = module.base + info.SizeOfImage;
				if (!K32GetModuleBaseNameA(GetCurrentProcess(), handles[i], module.name, static_cast<DWORD>(sizeof(module.name)))) {
					std::snprintf(module.name, sizeof(module.name), "module@%llX", static_cast<unsigned long long>(module.base));
				}
				module.system = IsSystem(module.name);
				++used;
			}
			g_moduleCount = used;
		}

		const Module* ModuleOf(std::uintptr_t a_address) noexcept
		{
			for (std::size_t i = 0; i < g_moduleCount; ++i) {
				if (a_address >= g_modules[i].base && a_address < g_modules[i].end) {
					return &g_modules[i];
				}
			}
			return nullptr;
		}

		// A return address follows a call: `E8 rel32`, or `FF /2` (2-7 bytes with its ModRM, SIB, displacement, REX).
		bool AfterCall(std::uintptr_t a_address) noexcept
		{
			__try {
				const auto* code = reinterpret_cast<const std::uint8_t*>(a_address);
				if (code[-5] == 0xE8) {
					return true;
				}
				for (int length = 2; length <= 7; ++length) {
					if (code[-length] == 0xFF && ((code[-length + 1] >> 3) & 7) == 2) {
						return true;
					}
				}
			} __except (EXCEPTION_EXECUTE_HANDLER) {
			}
			return false;
		}

		// Suspended: the top of the main thread's stack (memory reads only).
		std::size_t CopyStack(std::uintptr_t a_rsp, std::byte* a_out) noexcept
		{
			__try {
				std::memcpy(a_out, reinterpret_cast<const void*>(a_rsp), kStackBytes);
				return kStackBytes;
			} __except (EXCEPTION_EXECUTE_HANDLER) {
				return 0;
			}
		}

		void SamplerMain()
		{
			SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_ABOVE_NORMAL);
			HANDLE timer = CreateWaitableTimerExW(nullptr, nullptr, CREATE_WAITABLE_TIMER_HIGH_RESOLUTION, TIMER_ALL_ACCESS);
			if (!timer) {
				timer = CreateWaitableTimerExW(nullptr, nullptr, 0, TIMER_ALL_ACCESS);
			}
			alignas(16) CONTEXT   context{};
			alignas(16) std::byte stack[kStackBytes];
			for (;;) {
				if (timer) {
					LARGE_INTEGER due;
					due.QuadPart = -kPeriod100ns;
					SetWaitableTimer(timer, &due, 0, nullptr, nullptr, FALSE);
					WaitForSingleObject(timer, 10);
				} else {
					Sleep(1);
				}
				if (!g_run.load(std::memory_order_acquire)) {
					g_parked.store(true, std::memory_order_release);
					Sleep(1);
					continue;
				}
				g_parked.store(false, std::memory_order_release);  // (a park is acknowledged only between samples)
				const auto phase = g_phase.load(std::memory_order_relaxed);
				if (phase == kNoPhase || static_cast<double>(Qpc() - g_stamp.load(std::memory_order_relaxed)) * 1000.0 / static_cast<double>(g_frequency) > kStaleMs) {
					continue;
				}
				if (g_count >= kRing) {
					++g_dropped;
					continue;
				}
				// Suspended: the registers and the top of the stack, memory reads only.
				if (SuspendThread(g_mainThread) == static_cast<DWORD>(-1)) {
					++g_failed;
					continue;
				}
				context = {};
				context.ContextFlags = CONTEXT_CONTROL;
				const bool ok = GetThreadContext(g_mainThread, &context) != 0;
				const auto bytes = ok ? CopyStack(context.Rsp, stack) : 0;
				ResumeThread(g_mainThread);
				if (!ok) {
					++g_failed;
					continue;
				}
				// (resumed: the copy is ours to scan for the first return address)
				auto& sample = g_ring[g_count];
				sample.rip = context.Rip;
				sample.caller = 0;
				sample.phase = phase;
				for (std::size_t offset = 0; offset + 8 <= bytes; offset += 8) {
					std::uintptr_t value;
					std::memcpy(&value, stack + offset, sizeof(value));
					if (ModuleOf(value) && AfterCall(value)) {
						sample.caller = value;
						break;
					}
				}
				++g_count;
			}
		}

		// Main thread, the first mark: the thread sampled is this one.
		void Start()
		{
			g_mainThread = OpenThread(THREAD_SUSPEND_RESUME | THREAD_GET_CONTEXT | THREAD_QUERY_INFORMATION, FALSE, GetCurrentThreadId());
			if (!g_mainThread) {
				g_enabled = false;
				logger::warn("sampler: the main thread's handle could not be opened (error {}); off", GetLastError());
				return;
			}
			RefreshModules();
			g_started.store(true, std::memory_order_release);
			g_run.store(true, std::memory_order_release);
			std::thread(&SamplerMain).detach();
		}

		std::string Where(std::uintptr_t a_address)
		{
			if (const auto* module = ModuleOf(a_address)) {
				return std::format("{}+{:X}", module->name, a_address - module->base);
			}
			return std::format("{:X}", a_address);
		}
	}

	void Install(bool a_enabled, std::span<const std::string_view> a_modeNames, std::span<const std::string_view> a_bucketNames)
	{
		g_enabled = a_enabled;
		g_modeNames = a_modeNames;
		g_bucketNames = a_bucketNames;
		if (!a_enabled) {
			logger::info("sampler: off");
			return;
		}
		LARGE_INTEGER frequency;
		QueryPerformanceFrequency(&frequency);
		g_frequency = std::max<std::int64_t>(1, frequency.QuadPart);
		g_modules = new Module[kMaxModules];
		g_ring = new Sample[kRing];
		logger::info("sampler: on (bSampler=1): the main thread's instruction pointer sampled every {} us, filed by mode and stage; shares by module and the busiest addresses with each summary", kPeriod100ns / 10);
		{
			std::error_code             error;
			const std::filesystem::path dir{ "Data/CBRO_Recordings" };
			std::filesystem::create_directories(dir, error);
			SYSTEMTIME now{};
			GetLocalTime(&now);
			g_csv = dir / std::format("CBRO-samples-{:04}{:02}{:02}-{:02}{:02}{:02}.csv", now.wYear, now.wMonth, now.wDay, now.wHour, now.wMinute, now.wSecond);
			std::ofstream csv(g_csv, std::ios::out | std::ios::trunc);
			if (csv) {
				csv << "summary,frames,mode,stage,where,samples\n";
				logger::info("sampler: each summary's full histogram goes to {} (python -I tools\\sampler_names.py <csv> rolls it up by function)", g_csv.string());
			} else {
				logger::warn("sampler: cannot write {}: the histogram stays in the log's lists only", g_csv.string());
				g_csv.clear();
			}
		}
	}

	void SetPhase(int a_mode, std::size_t a_bucket) noexcept
	{
		if (!g_enabled) {
			return;
		}
		if (!g_started.load(std::memory_order_relaxed)) {
			Start();
			if (!g_enabled) {
				return;
			}
		}
		g_stamp.store(Qpc(), std::memory_order_relaxed);
		g_phase.store(a_mode < 0 ? kNoPhase : (static_cast<std::uint32_t>(a_mode) << 8) | static_cast<std::uint32_t>(a_bucket & 0xFF), std::memory_order_relaxed);
	}

	void LogStats(std::uint32_t a_frames)
	{
		if (!g_enabled || !g_started.load(std::memory_order_acquire)) {
			return;
		}
		g_run.store(false, std::memory_order_release);
		for (int spins = 0; !g_parked.load(std::memory_order_acquire); ++spins) {
			if (spins >= 200) {
				g_run.store(true, std::memory_order_release);
				logger::warn("sampler: the thread didn't park in 200 ms; this interval's samples kept for the next summary");
				return;
			}
			Sleep(1);
		}
		RefreshModules();

		const auto buckets = g_bucketNames.size();
		const auto phases = g_modeNames.size() * buckets;
		const auto phaseIndex = [&](std::uint32_t a_phase) -> std::size_t {
			const auto mode = a_phase >> 8;
			const auto bucket = a_phase & 0xFF;
			return mode < g_modeNames.size() && bucket < buckets ? mode * buckets + bucket : phases;
		};
		std::vector<std::uint32_t> phaseCount(phases + 1, 0);
		std::vector<std::uint32_t> moduleCount((phases + 1) * (kMaxModules + 1), 0);  // [phase][module index, or kMaxModules = no module]
		std::vector<std::uint64_t> keys;
		keys.reserve(g_count);
		for (std::uint32_t i = 0; i < g_count; ++i) {
			const auto& sample = g_ring[i];
			const auto  index = phaseIndex(sample.phase);
			const auto* self = ModuleOf(sample.rip);
			const auto* caller = sample.caller ? ModuleOf(sample.caller) : nullptr;
			const auto  charged = self && self->system && caller && !caller->system ? sample.caller : sample.rip;
			++phaseCount[index];
			++moduleCount[index * (kMaxModules + 1) + (self ? static_cast<std::size_t>(self - g_modules) : kMaxModules)];
			keys.push_back((static_cast<std::uint64_t>(index) << 48) | (charged & 0xFFFF'FFFF'FFFFull));
		}
		std::sort(keys.begin(), keys.end());

		const double frames = std::max(1u, a_frames);
		std::ofstream csv;
		if (!g_csv.empty()) {
			csv.open(g_csv, std::ios::out | std::ios::app);
		}
		++g_summaries;
		for (std::size_t mode = 0; mode < g_modeNames.size(); ++mode) {
			std::uint32_t modeTotal = 0;
			for (std::size_t bucket = 0; bucket < buckets; ++bucket) {
				modeTotal += phaseCount[mode * buckets + bucket];
			}
			if (modeTotal == 0) {
				continue;
			}
			std::string byStage;
			for (std::size_t bucket = 0; bucket < buckets; ++bucket) {
				byStage += std::format("{}{} {:.0f}%", bucket ? " | " : "", g_bucketNames[bucket], 100.0 * phaseCount[mode * buckets + bucket] / modeTotal);
			}
			logger::info("sampler: {}: {} samples ({:.1f} a frame over the interval's {} frames), by stage: {}", g_modeNames[mode], modeTotal, modeTotal / frames, a_frames, byStage);
			for (std::size_t bucket = 0; bucket < buckets; ++bucket) {
				const auto index = mode * buckets + bucket;
				const auto total = phaseCount[index];
				if (total < kShownShare * modeTotal || total == 0) {
					continue;
				}
				// the share by module (the sampled address itself), largest first
				std::vector<std::pair<std::uint32_t, std::size_t>> modules;
				for (std::size_t m = 0; m <= kMaxModules; ++m) {
					if (const auto n = moduleCount[index * (kMaxModules + 1) + m]) {
						modules.emplace_back(n, m);
					}
				}
				std::sort(modules.begin(), modules.end(), [](const auto& a, const auto& b) { return a.first > b.first; });
				std::string byModule;
				std::size_t shown = 0;
				for (const auto& [n, m] : modules) {
					if (shown++ == 6) {
						break;
					}
					byModule += std::format("{}{} {:.0f}%", byModule.empty() ? "" : ", ", m < kMaxModules ? g_modules[m].name : "no module", 100.0 * n / total);
				}
				// the busiest charged addresses
				std::vector<std::pair<std::uint32_t, std::uintptr_t>> top;
				const auto first = std::lower_bound(keys.begin(), keys.end(), static_cast<std::uint64_t>(index) << 48);
				const auto last = std::lower_bound(keys.begin(), keys.end(), static_cast<std::uint64_t>(index + 1) << 48);
				for (auto it = first; it != last;) {
					auto run = it;
					while (run != last && *run == *it) {
						++run;
					}
					top.emplace_back(static_cast<std::uint32_t>(run - it), static_cast<std::uintptr_t>(*it & 0xFFFF'FFFF'FFFFull));
					it = run;
				}
				std::sort(top.begin(), top.end(), [](const auto& a, const auto& b) { return a.first > b.first; });
				if (csv) {
					for (const auto& [n, address] : top) {
						if (n < 2) {
							break;
						}
						csv << g_summaries << ',' << a_frames << ',' << g_modeNames[mode] << ',' << g_bucketNames[bucket] << ',' << Where(address) << ',' << n << '\n';
					}
				}
				std::string busiest;
				for (std::size_t i = 0; i < top.size() && i < kTop; ++i) {
					busiest += std::format("{}{} {:.1f}%", i ? ", " : "", Where(top[i].second), 100.0 * top[i].first / total);
				}
				logger::info("sampler: {} / {} ({} samples, {:.0f}% of the mode): by module: {} | busiest: {}", g_modeNames[mode], g_bucketNames[bucket], total, 100.0 * total / modeTotal, byModule, busiest);
				// CBRO's own addresses in full (charged), where its share of the stage is worth it: tools/sampler_names.py rolls
				// them up by function (the top-10 list above shows only the hottest few of a spread-out share)
				const auto own = std::find_if(modules.begin(), modules.end(), [](const auto& a_entry) { return a_entry.second < kMaxModules && _stricmp(g_modules[a_entry.second].name, "CBRO.dll") == 0; });
				if (own != modules.end() && own->first >= kOwnShare * total) {
					const auto& module = g_modules[own->second];
					std::string list;
					std::size_t listed = 0;
					for (const auto& [n, address] : top) {  // (sorted by count)
						if (n < kOwnMinShare * total || listed >= kOwnTop) {
							break;
						}
						if (address < module.base || address >= module.end) {
							continue;
						}
						list += std::format("{}CBRO.dll+{:X} {:.1f}%", listed++ ? ", " : "", address - module.base, 100.0 * n / total);
					}
					logger::info("sampler: {} / {}: CBRO.dll by address ({:.0f}% of the stage, {} addresses): {}", g_modeNames[mode], g_bucketNames[bucket], 100.0 * own->first / total, listed, list);
				}
			}
		}
		if (g_dropped || g_failed || phaseCount[phases]) {
			logger::info("sampler: {} samples dropped (ring full), {} failed, {} without a stage", g_dropped, g_failed, phaseCount[phases]);
		}
		g_count = 0;
		g_dropped = 0;
		g_failed = 0;
		g_run.store(true, std::memory_order_release);
	}
}
