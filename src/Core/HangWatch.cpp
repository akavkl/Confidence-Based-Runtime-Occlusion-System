#include "Core/HangWatch.h"

#include <Psapi.h>
#include <TlHelp32.h>

#include <thread>

namespace CBRO::Core::HangWatch
{
	namespace
	{
		constexpr int         kStallSeconds = 15;
		constexpr int         kSecondSampleSeconds = 10;
		constexpr std::size_t kMaxModules = 1024;
		constexpr std::size_t kMaxThreads = 1024;
		constexpr std::size_t kStackBytes = 128 * 1024;
		constexpr std::size_t kMaxFrames = 64;
		constexpr std::size_t kReportBytes = 1 << 20;

		struct Module
		{
			std::uintptr_t base{ 0 };
			std::uintptr_t end{ 0 };
			char           name[64]{};
		};

		struct Sample
		{
			DWORD          id{ 0 };
			bool           ok{ false };
			std::uintptr_t start{ 0 };
			std::uintptr_t rip{ 0 };
			std::uintptr_t rsp{ 0 };
			std::uint32_t  frames{ 0 };
			std::uintptr_t frame[kMaxFrames]{};
		};

		using QueryThreadFn = LONG(NTAPI*)(HANDLE, ULONG, PVOID, ULONG, PULONG);
		constexpr ULONG kThreadStartAddress = 9;  // ThreadQuerySetWin32StartAddress

		std::atomic<std::uint64_t> g_beats{ 0 };
		std::atomic<DWORD>         g_mainThread{ 0 };
		bool                       g_installed{ false };
		std::wstring               g_path;
		QueryThreadFn              g_queryThread{ nullptr };

		// The watchdog thread's own (allocated at install, never while a thread is suspended).
		Module*        g_modules{ nullptr };
		std::size_t    g_moduleCount{ 0 };
		Sample*        g_samples{ nullptr };
		std::size_t    g_sampleCount{ 0 };
		DWORD*         g_ids{ nullptr };
		std::byte*     g_stack{ nullptr };
		char*          g_report{ nullptr };
		std::size_t    g_reportUsed{ 0 };
		alignas(16) CONTEXT g_context{};

		// While frames run (so no thread is stuck holding what this needs): the loaded modules and their ranges.
		void RefreshModules() noexcept
		{
			HMODULE handles[kMaxModules];
			DWORD   needed = 0;
			if (!K32EnumProcessModules(GetCurrentProcess(), handles, sizeof(handles), &needed)) {
				return;
			}
			const auto count = std::min<std::size_t>(needed / sizeof(HMODULE), kMaxModules);
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

		// A return address follows a call: `E8 rel32`, or `FF /2` (call through a register or memory, 2-7 bytes with
		// its ModRM, SIB, displacement and a REX prefix).
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

		// Suspended: the thread's registers and the top of its stack into the buffers (memory reads only).
		bool Capture(HANDLE a_thread, Sample& a_sample, std::size_t& a_bytes) noexcept
		{
			a_bytes = 0;
			g_context = {};
			g_context.ContextFlags = CONTEXT_CONTROL | CONTEXT_INTEGER;
			if (!GetThreadContext(a_thread, &g_context)) {
				return false;
			}
			a_sample.rip = g_context.Rip;
			a_sample.rsp = g_context.Rsp;
			MEMORY_BASIC_INFORMATION region{};
			if (!VirtualQuery(reinterpret_cast<const void*>(g_context.Rsp), &region, sizeof(region))) {
				return true;
			}
			const auto end = reinterpret_cast<std::uintptr_t>(region.BaseAddress) + region.RegionSize;
			const auto bytes = std::min<std::size_t>(kStackBytes, end > g_context.Rsp ? end - g_context.Rsp : 0) & ~std::size_t{ 7 };
			__try {
				std::memcpy(g_stack, reinterpret_cast<const void*>(g_context.Rsp), bytes);
				a_bytes = bytes;
			} __except (EXCEPTION_EXECUTE_HANDLER) {
			}
			return true;
		}

		void TakeSample() noexcept
		{
			g_sampleCount = 0;
			const auto self = GetCurrentThreadId();
			const auto process = GetCurrentProcessId();
			std::size_t ids = 0;
			const auto  snapshot = CreateToolhelp32Snapshot(TH32CS_SNAPTHREAD, 0);
			if (snapshot == INVALID_HANDLE_VALUE) {
				return;
			}
			THREADENTRY32 entry{};
			entry.dwSize = sizeof(entry);
			for (auto more = Thread32First(snapshot, &entry); more && ids < kMaxThreads; more = Thread32Next(snapshot, &entry)) {
				if (entry.th32OwnerProcessID == process && entry.th32ThreadID != self) {
					g_ids[ids++] = entry.th32ThreadID;
				}
			}
			CloseHandle(snapshot);

			for (std::size_t i = 0; i < ids; ++i) {
				auto& sample = g_samples[g_sampleCount];
				sample = {};
				sample.id = g_ids[i];
				const auto thread = OpenThread(THREAD_SUSPEND_RESUME | THREAD_GET_CONTEXT | THREAD_QUERY_INFORMATION, FALSE, sample.id);
				if (!thread) {
					continue;
				}
				if (g_queryThread) {
					PVOID start = nullptr;
					if (g_queryThread(thread, kThreadStartAddress, &start, sizeof(start), nullptr) == 0) {
						sample.start = reinterpret_cast<std::uintptr_t>(start);
					}
				}
				std::size_t bytes = 0;
				if (SuspendThread(thread) != static_cast<DWORD>(-1)) {
					sample.ok = Capture(thread, sample, bytes);
					ResumeThread(thread);
				}
				CloseHandle(thread);
				// (resumed: the copy is ours to scan)
				for (std::size_t offset = 0; offset + 8 <= bytes && sample.frames < kMaxFrames; offset += 8) {
					std::uintptr_t value;
					std::memcpy(&value, g_stack + offset, sizeof(value));
					if (ModuleOf(value) && AfterCall(value)) {
						sample.frame[sample.frames++] = value;
					}
				}
				++g_sampleCount;
			}
		}

		void Append(std::string_view a_text) noexcept
		{
			const auto room = kReportBytes - g_reportUsed;
			const auto size = std::min(room, a_text.size());
			std::memcpy(g_report + g_reportUsed, a_text.data(), size);
			g_reportUsed += size;
		}

		template <class... Args>
		void Appendf(std::format_string<Args...> a_format, Args&&... a_args) noexcept
		{
			char line[512];
			const auto result = std::format_to_n(line, sizeof(line), a_format, std::forward<Args>(a_args)...);
			Append({ line, static_cast<std::size_t>(result.out - line) });
		}

		void AppendAddress(std::uintptr_t a_address) noexcept
		{
			if (const auto module = ModuleOf(a_address)) {
				Appendf("{}+0x{:X}", module->name, a_address - module->base);
			} else {
				Appendf("0x{:X}", a_address);
			}
		}

		bool InCbro(const Sample& a_sample, const Module* a_cbro) noexcept
		{
			if (!a_cbro) {
				return false;
			}
			const auto in = [&](std::uintptr_t a_address) { return a_address >= a_cbro->base && a_address < a_cbro->end; };
			if (in(a_sample.rip) || in(a_sample.start)) {
				return true;
			}
			for (std::uint32_t i = 0; i < a_sample.frames; ++i) {
				if (in(a_sample.frame[i])) {
					return true;
				}
			}
			return false;
		}

		void WriteReport(int a_sample, std::uint64_t a_beats, int a_seconds) noexcept
		{
			g_reportUsed = 0;
			SYSTEMTIME now{};
			GetLocalTime(&now);
			Appendf(
				"==== CBRO hang watch {:02}:{:02}:{:02}.{:03}: no world frame for {} s (frames so far {}), sample {} of 2, {} threads ====\r\n",
				now.wHour, now.wMinute, now.wSecond, now.wMilliseconds, a_seconds, a_beats, a_sample, g_sampleCount);
			const auto main = g_mainThread.load(std::memory_order_relaxed);
			const Module* cbro = nullptr;
			for (std::size_t i = 0; i < g_moduleCount; ++i) {
				if (_stricmp(g_modules[i].name, "CBRO.dll") == 0) {
					cbro = &g_modules[i];
				}
			}
			// The main thread and every thread with a CBRO frame in full; then one line for each of the others.
			for (int pass = 0; pass < 2; ++pass) {
				for (std::size_t i = 0; i < g_sampleCount; ++i) {
					const auto& sample = g_samples[i];
					const bool  full = sample.id == main || InCbro(sample, cbro);
					if (full != (pass == 0)) {
						continue;
					}
					Appendf("thread {}{}{} | start ", sample.id, sample.id == main ? " (main)" : "", sample.ok ? "" : " (not read)");
					AppendAddress(sample.start);
					Append(" | rip ");
					AppendAddress(sample.rip);
					const std::uint32_t shown = full ? sample.frames : std::min<std::uint32_t>(sample.frames, 4);
					if (full) {
						Append("\r\n");
						for (std::uint32_t f = 0; f < shown; ++f) {
							Appendf("    {:2} ", f);
							AppendAddress(sample.frame[f]);
							Append("\r\n");
						}
					} else {
						for (std::uint32_t f = 0; f < shown; ++f) {
							Append(" < ");
							AppendAddress(sample.frame[f]);
						}
						Append("\r\n");
					}
				}
			}
			Append("\r\n");
			const auto file = CreateFileW(g_path.c_str(), FILE_APPEND_DATA, FILE_SHARE_READ, nullptr, OPEN_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
			if (file != INVALID_HANDLE_VALUE) {
				DWORD written = 0;
				WriteFile(file, g_report, static_cast<DWORD>(g_reportUsed), &written, nullptr);
				CloseHandle(file);
			}
		}

		void WatchMain()
		{
			std::uint64_t last = 0;
			int           stalled = 0;
			int           sampled = 0;
			int           sinceRefresh = 0;
			for (;;) {
				Sleep(1000);
				const auto beats = g_beats.load(std::memory_order_relaxed);
				if (beats != last) {
					last = beats;
					stalled = 0;
					sampled = 0;
					if (++sinceRefresh >= 10 || g_moduleCount == 0) {
						sinceRefresh = 0;
						RefreshModules();
					}
					continue;
				}
				if (beats == 0) {
					continue;  // (no world frame yet: the main menu, the first load)
				}
				++stalled;
				if ((sampled == 0 && stalled >= kStallSeconds) || (sampled == 1 && stalled >= kStallSeconds + kSecondSampleSeconds)) {
					++sampled;
					TakeSample();
					WriteReport(sampled, beats, stalled);
					logger::warn("hang watch: no world frame for {} s; threads sampled into CBRO-hang.log (sample {} of 2)", stalled, sampled);
				}
			}
		}
	}

	void Install()
	{
		if (g_installed) {
			return;
		}
		auto directory = F4SE::log::log_directory();
		if (!directory) {
			return;
		}
		g_installed = true;
		g_path = (*directory / "CBRO-hang.log").wstring();
		DeleteFileW(g_path.c_str());  // (one session's samples)
		g_queryThread = reinterpret_cast<QueryThreadFn>(GetProcAddress(GetModuleHandleW(L"ntdll.dll"), "NtQueryInformationThread"));
		g_modules = new Module[kMaxModules];
		g_samples = new Sample[kMaxThreads];
		g_ids = new DWORD[kMaxThreads];
		g_stack = new std::byte[kStackBytes];
		g_report = new char[kReportBytes];
		RefreshModules();
		std::thread(&WatchMain).detach();
		logger::info("hang watch: on (threads sampled into CBRO-hang.log after {} s without a world frame)", kStallSeconds);
	}

	void Beat() noexcept
	{
		if (g_mainThread.load(std::memory_order_relaxed) == 0) {
			g_mainThread.store(GetCurrentThreadId(), std::memory_order_relaxed);
		}
		g_beats.fetch_add(1, std::memory_order_relaxed);
	}
}
