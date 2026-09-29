#include "Probe/ProbeInternal.h"

#include "Util/Hooking.h"

#include <intrin.h>

// Wraps the culling virtuals of every NiCullingProcess-derived class in the exe (the MSVC RTTI
// scan finds exactly five). Slot layout checked offline against the 1.10.163 exe:
//   0x18 deleting destructor
//   0x19 Process(NiAVObject*)                                   reads arg->worldBound / arg->flags
//   0x1A Process(const NiCamera*, NiAVObject*, NiVisibleArray*) BSFadeNodeCuller: camera setter only
//   0x1B AppendVirtual(BSGeometry*)                             BS classes only (pure in the bases)
//   0x1C AppendNonAccum(NiAVObject*)
//   0x1D TestBaseVisibility(BSMultiBound*)                      calls bound->shape->WithinFrustum(planes)
//   0x1E TestBaseVisibility(BSOcclusionPlane*)
//   0x1F TestBaseVisibility(const NiBound*)                     reads bCustomCullPlanes (0x11F)
// The first probe run saw zero calls on 0x19/0x1A, and no code calls those bodies directly, so
// this run covers 0x19-0x1F and samples callers' return addresses to locate the real cull loop.

namespace CBRO::Probe::Cull
{
	namespace
	{
		struct ClassInfo
		{
			std::string_view name;
			std::uint64_t    vtableID;
			std::size_t      lastSlot;
		};

		constexpr std::array<ClassInfo, 5> kClasses{ {
			{ "NiCullingProcess"sv, 547317, 0x1A },
			{ "BSCullingProcess"sv, 556243, 0x1F },
			{ "BSGeometryListCullingProcess"sv, 1398386, 0x1F },
			{ "BSParabolicCullingProcess"sv, 845854, 0x1F },
			{ "BSFadeNodeCuller"sv, 1334420, 0x1A },
		} };

		constexpr std::size_t kFirstSlot = 0x19;
		constexpr std::size_t kProcessCameraSlot = 0x1A;
		constexpr std::size_t kTestBoundSlot = 0x1F;

		std::string_view SlotName(std::size_t a_slot) noexcept
		{
			switch (a_slot) {
			case 0x19:
				return "Process(obj)"sv;
			case 0x1A:
				return "Process(cam)"sv;
			case 0x1B:
				return "AppendVirtual"sv;
			case 0x1C:
				return "AppendNonAccum"sv;
			case 0x1D:
				return "TestVis(MultiBound)"sv;
			case 0x1E:
				return "TestVis(OcclusionPlane)"sv;
			case 0x1F:
				return "TestVis(NiBound)"sv;
			default:
				return "?"sv;
			}
		}

		struct HookSpec
		{
			std::size_t cls;
			std::size_t slot;
		};

		constexpr std::size_t kHookCount = [] {
			std::size_t count = 0;
			for (const auto& info : kClasses) {
				count += info.lastSlot - kFirstSlot + 1;
			}
			return count;
		}();

		constexpr auto kHooks = [] {
			std::array<HookSpec, kHookCount> hooks{};
			std::size_t                      i = 0;
			for (std::size_t c = 0; c < kClasses.size(); ++c) {
				for (std::size_t s = kFirstSlot; s <= kClasses[c].lastSlot; ++s) {
					hooks[i++] = { c, s };
				}
			}
			return hooks;
		}();

		// Every slot except TestVis(NiBound) takes a Gamebryo object as its first argument.
		constexpr bool HasObjectArg(std::size_t a_slot) noexcept
		{
			return a_slot != kTestBoundSlot;
		}

		constexpr std::size_t kThreadKinds = static_cast<std::size_t>(ThreadKind::kCount);
		constexpr std::size_t kMaxSamples = 16;

		struct Sample
		{
			std::uintptr_t key{ 0 };
			std::uint64_t  hits{ 0 };
			char           rtti[48]{};
		};

		struct alignas(64) HookStats
		{
			std::atomic<std::uint64_t>                           frameCalls{ 0 };
			std::atomic<std::uint64_t>                           intervalCalls{ 0 };
			std::array<std::atomic<std::uint64_t>, kThreadKinds> intervalThreads{};
			std::atomic<std::int64_t>                            sampleBudget{ 20000 };
			std::uint64_t                                        lastFrameCalls{ 0 };

			std::mutex          sampleLock;
			std::vector<Sample> callers;   // return addresses
			std::vector<Sample> argTypes;  // arg1 vtables
			int                 samplesLogged{ 0 };

			std::uintptr_t vtable{ 0 };
			std::uintptr_t thunk{ 0 };
			std::uintptr_t original{ 0 };
			bool           installed{ false };
		};
		std::array<HookStats, kHookCount> g_hooks{};

		constexpr std::size_t    kMaxCameras = 24;
		constexpr std::uintptr_t kNullCamera = 1;

		struct alignas(64) CameraSlot
		{
			std::atomic<std::uintptr_t> camera{ 0 };
			std::atomic<bool>           described{ false };
			char                        rtti[48]{};
			char                        name[64]{};

			std::atomic<std::uint32_t>                           classMask{ 0 };
			std::atomic<std::uint32_t>                           slotMask{ 0 };
			std::atomic<std::uint64_t>                           frameCalls{ 0 };
			std::atomic<std::uint64_t>                           intervalCalls{ 0 };
			std::array<std::atomic<std::uint64_t>, kThreadKinds> intervalThreads{};
			std::atomic<std::uint32_t>                           lastThread{ 0 };
			std::atomic<std::uintptr_t>                          lastProcess{ 0 };
			std::atomic<std::uint64_t>                           processSwitches{ 0 };
			std::atomic<std::uint64_t>                           lastSeenFrame{ ~0ull };
			std::uint64_t                                        lastFrameCalls{ 0 };
		};
		std::array<CameraSlot, kMaxCameras> g_cameras{};
		std::atomic<std::uint64_t>          g_cameraOverflow{ 0 };

		bool InGameImage(std::uintptr_t a_address) noexcept
		{
			const auto base = reinterpret_cast<std::uintptr_t>(GetModuleHandleW(nullptr));
			const auto dos = reinterpret_cast<const IMAGE_DOS_HEADER*>(base);
			const auto nt = reinterpret_cast<const IMAGE_NT_HEADERS64*>(base + dos->e_lfanew);
			return a_address >= base && a_address < base + nt->OptionalHeader.SizeOfImage;
		}

		std::string HookName(std::size_t a_hook)
		{
			const auto& spec = kHooks[a_hook];
			return std::format("{}[0x{:X}] {}", kClasses[spec.cls].name, spec.slot, SlotName(spec.slot));
		}

		CameraSlot* FindOrInsertCamera(std::uintptr_t a_camera) noexcept
		{
			for (auto& slot : g_cameras) {
				auto current = slot.camera.load(std::memory_order_acquire);
				if (current == a_camera) {
					return &slot;
				}
				if (current != 0) {
					continue;
				}
				if (slot.camera.compare_exchange_strong(current, a_camera, std::memory_order_acq_rel)) {
					if (a_camera == kNullCamera) {
						strncpy_s(slot.rtti, "(none)", _TRUNCATE);
					} else {
						TryGetRTTIName(reinterpret_cast<const void*>(a_camera), slot.rtti, sizeof(slot.rtti));
						TryGetObjectName(reinterpret_cast<const void*>(a_camera), slot.name, sizeof(slot.name));
					}
					slot.described.store(true, std::memory_order_release);
					return &slot;
				}
				if (current == a_camera) {
					return &slot;
				}
			}
			return nullptr;
		}

		void AddSample(std::vector<Sample>& a_samples, std::uintptr_t a_key, const void* a_object)
		{
			for (auto& sample : a_samples) {
				if (sample.key == a_key) {
					++sample.hits;
					return;
				}
			}
			if (a_samples.size() >= kMaxSamples) {
				return;
			}
			Sample sample{ a_key, 1 };
			if (a_object && !TryGetRTTIName(a_object, sample.rtti, sizeof(sample.rtti))) {
				strncpy_s(sample.rtti, "(no NiRTTI)", _TRUNCATE);
			}
			a_samples.push_back(sample);
		}

		void OnCall(std::size_t a_hook, const RE::NiCullingProcess* a_self, const void* a_arg1, std::uintptr_t a_returnAddress)
		{
			const auto& spec = kHooks[a_hook];
			auto&       stats = g_hooks[a_hook];
			const auto  tid = GetCurrentThreadId();
			const auto  threadKind = static_cast<std::size_t>(ClassifyThread(tid));

			stats.frameCalls.fetch_add(1, std::memory_order_relaxed);
			stats.intervalCalls.fetch_add(1, std::memory_order_relaxed);
			stats.intervalThreads[threadKind].fetch_add(1, std::memory_order_relaxed);

			if (stats.sampleBudget.load(std::memory_order_relaxed) > 0 &&
				stats.sampleBudget.fetch_sub(1, std::memory_order_relaxed) > 0) {
				const bool objectArg = HasObjectArg(spec.slot) && a_arg1;
				const auto argVtable = objectArg ? TryReadVtable(a_arg1) : 0;

				std::scoped_lock lock(stats.sampleLock);
				AddSample(stats.callers, a_returnAddress, nullptr);
				if (objectArg) {
					AddSample(stats.argTypes, argVtable, argVtable ? a_arg1 : nullptr);
				}
			}

			const bool pass = spec.slot == kProcessCameraSlot;
			// NiCullingProcess::camera at +0x18 (CommonLibF4-DM layout; the RD headers only forward-declare the class).
			auto       camera = pass ? reinterpret_cast<std::uintptr_t>(a_arg1) : *reinterpret_cast<const std::uintptr_t*>(reinterpret_cast<const std::byte*>(a_self) + 0x18);
			if (!camera) {
				camera = kNullCamera;
			}

			auto slot = FindOrInsertCamera(camera);
			if (!slot) {
				g_cameraOverflow.fetch_add(1, std::memory_order_relaxed);
				return;
			}

			slot->frameCalls.fetch_add(1, std::memory_order_relaxed);
			slot->intervalCalls.fetch_add(1, std::memory_order_relaxed);
			slot->intervalThreads[threadKind].fetch_add(1, std::memory_order_relaxed);
			slot->classMask.fetch_or(1u << spec.cls, std::memory_order_relaxed);
			slot->slotMask.fetch_or(1u << (spec.slot - kFirstSlot), std::memory_order_relaxed);
			slot->lastThread.store(tid, std::memory_order_relaxed);
			const auto self = reinterpret_cast<std::uintptr_t>(a_self);
			if (slot->lastProcess.load(std::memory_order_relaxed) != self) {
				slot->lastProcess.store(self, std::memory_order_relaxed);
				slot->processSwitches.fetch_add(1, std::memory_order_relaxed);
			}

			if (Tracing()) {
				const auto frame = State().frame.load(std::memory_order_relaxed);
				const bool firstTouch = slot->lastSeenFrame.exchange(frame, std::memory_order_relaxed) != frame;
				if (firstTouch || pass) {
					TraceEvent(std::format(
						"cull {} camera {} {} '{}' via {} process {} caller {} stage={}",
						pass ? "pass" : "first-touch", fmt_ptr(camera), slot->rtti, slot->name,
						HookName(a_hook), fmt_ptr(self), Util::DescribeCodeAddress(a_returnAddress), StageName(CurrentStage())));
				}
			}
		}

		using ProcessFn = std::uintptr_t (*)(void*, void*, void*, void*);

		template <std::size_t I>
		struct CullHook
		{
			static std::uintptr_t thunk(void* a_self, void* a1, void* a2, void* a3)
			{
				OnCall(I, static_cast<const RE::NiCullingProcess*>(a_self), a1, reinterpret_cast<std::uintptr_t>(_ReturnAddress()));
				return reinterpret_cast<ProcessFn>(g_hooks[I].original)(a_self, a1, a2, a3);
			}
		};

		template <std::size_t I>
		void InstallHook()
		{
			const auto& spec = kHooks[I];
			auto&       stats = g_hooks[I];
			const auto  name = std::format("cull:{}", HookName(I));

			const auto vtable = CBRO::Engine::OG(kClasses[spec.cls].vtableID).address();
			if (!InGameImage(vtable)) {
				logger::error("hook {}: vtable {} is outside Fallout4.exe; skipped", name, fmt_ptr(vtable));
				return;
			}
			const auto current = *reinterpret_cast<const std::uintptr_t*>(vtable + sizeof(std::uintptr_t) * spec.slot);
			if (!InGameImage(current)) {
				logger::error("hook {}: slot points outside Fallout4.exe ({}); skipped", name, Util::DescribeCodeAddress(current));
				return;
			}

			stats.vtable = vtable;
			stats.thunk = Util::FnAddr(&CullHook<I>::thunk);
			stats.original = Util::WriteVFunc(vtable, spec.slot, stats.thunk, name);
			stats.installed = stats.original != 0;
		}

		template <std::size_t... I>
		void InstallAll(std::index_sequence<I...>)
		{
			(InstallHook<I>(), ...);
		}

		std::string DescribeCaller(std::uintptr_t a_address)
		{
			auto        result = Util::DescribeCodeAddress(a_address);
			DWORD64     imageBase = 0;
			const auto  function = RtlLookupFunctionEntry(a_address, &imageBase, nullptr);
			if (function && imageBase == reinterpret_cast<DWORD64>(GetModuleHandleW(nullptr))) {
				result += std::format(" (chunk 0x{:X}+0x{:X})", function->BeginAddress, a_address - imageBase - function->BeginAddress);
			}
			return result;
		}

		void LogSamples(std::size_t a_hook, HookStats& a_stats)
		{
			std::scoped_lock lock(a_stats.sampleLock);
			const bool       exhausted = a_stats.sampleBudget.load() <= 0;
			if (a_stats.callers.empty() || a_stats.samplesLogged >= 2 || (a_stats.samplesLogged == 1 && !exhausted)) {
				return;
			}
			++a_stats.samplesLogged;

			logger::info("  samples for {} ({}):", HookName(a_hook), exhausted ? "first 20000 calls" : "so far");
			for (const auto& caller : a_stats.callers) {
				logger::info("    caller {} x{}", DescribeCaller(caller.key), caller.hits);
			}
			if (!a_stats.argTypes.empty()) {
				std::string types;
				for (const auto& type : a_stats.argTypes) {
					types += std::format(" {}x{}", type.rtti[0] ? type.rtti : "(null)", type.hits);
				}
				logger::info("    arg1 types:{}", types);
			}
		}
	}

	void Install()
	{
		InstallAll(std::make_index_sequence<kHookCount>{});
	}

	void VerifyChain()
	{
		std::size_t ours = 0;
		for (std::size_t i = 0; i < kHookCount; ++i) {
			const auto& stats = g_hooks[i];
			if (!stats.installed) {
				continue;
			}
			const auto current = *reinterpret_cast<const std::uintptr_t*>(stats.vtable + sizeof(std::uintptr_t) * kHooks[i].slot);
			if (current == stats.thunk) {
				++ours;
			} else {
				logger::warn("cull:{} slot now {} (replaced by another plugin)", HookName(i), Util::DescribeCodeAddress(current));
			}
		}
		logger::info("cull hooks still ours after all plugins loaded: {}/{}", ours, kHookCount);
	}

	void EndFrame(bool a_log)
	{
		for (auto& stats : g_hooks) {
			stats.lastFrameCalls = stats.frameCalls.exchange(0, std::memory_order_relaxed);
		}
		for (auto& slot : g_cameras) {
			slot.lastFrameCalls = slot.frameCalls.exchange(0, std::memory_order_relaxed);
		}

		if (!a_log) {
			return;
		}

		std::size_t active = 0;
		for (std::size_t i = 0; i < kHookCount; ++i) {
			auto&      stats = g_hooks[i];
			const auto interval = stats.intervalCalls.exchange(0);
			std::array<std::uint64_t, kThreadKinds> threads{};
			for (std::size_t t = 0; t < kThreadKinds; ++t) {
				threads[t] = stats.intervalThreads[t].exchange(0);
			}
			if (!interval) {
				continue;
			}
			++active;
			logger::info(
				"cull {}: last frame={} interval={} threads main/render/other={}/{}/{}",
				HookName(i), stats.lastFrameCalls, interval, threads[0], threads[1], threads[2]);
			LogSamples(i, stats);
		}
		if (!active) {
			logger::info("cull: no calls on any of the {} culling hooks this interval", kHookCount);
		}

		const auto worldCamera = State().worldCamera.load();
		for (auto& slot : g_cameras) {
			if (!slot.described.load(std::memory_order_acquire)) {
				continue;
			}
			const auto interval = slot.intervalCalls.exchange(0);
			std::array<std::uint64_t, kThreadKinds> threads{};
			for (std::size_t t = 0; t < kThreadKinds; ++t) {
				threads[t] = slot.intervalThreads[t].exchange(0);
			}
			const auto switches = slot.processSwitches.exchange(0);
			if (!interval) {
				continue;
			}

			std::string classes;
			const auto  classMask = slot.classMask.load();
			for (std::size_t c = 0; c < kClasses.size(); ++c) {
				if (classMask & (1u << c)) {
					if (!classes.empty()) {
						classes += '|';
					}
					classes += kClasses[c].name;
				}
			}
			std::string slots;
			const auto  slotMask = slot.slotMask.load();
			for (std::size_t s = kFirstSlot; s <= kTestBoundSlot; ++s) {
				if (slotMask & (1u << (s - kFirstSlot))) {
					if (!slots.empty()) {
						slots += '|';
					}
					slots += SlotName(s);
				}
			}

			const auto camera = slot.camera.load();
			logger::info(
				"  camera {} {} '{}'{}: last frame={} interval={} threads main/render/other={}/{}/{} lastTid={} classes={} slots={} processSwitches={}",
				fmt_ptr(camera), slot.rtti, slot.name, camera == worldCamera ? " [WORLD ROOT CAMERA]" : "",
				slot.lastFrameCalls, interval, threads[0], threads[1], threads[2], slot.lastThread.load(),
				classes, slots, switches);
		}
		if (const auto overflow = g_cameraOverflow.exchange(0)) {
			logger::warn("  {} cull calls not attributed: more than {} distinct cameras", overflow, kMaxCameras);
		}
	}
}
