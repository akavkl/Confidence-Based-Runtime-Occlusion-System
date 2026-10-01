#include "Probe/ProbeInternal.h"

#include "Hooks/CullGroups.h"
#include "Util/Hooking.h"

// FO4 1.10.163 does not cull the main camera through NiCullingProcess virtuals (run 2: zero calls).
// Static RE of the exe:
//   DrawWorld::Render_PreUI (984743) -> DrawWorld cull (718911) -> scene walk (1138818)
//   -> Group::Add (1175493) on DrawWorld's culling groups (1117782, 133326, 1328670, array 459440).
// A culling group is a non-polymorphic 0x170-byte struct: six frustum planes, then node and geometry
// block lists. Group::Add sorts the object into a 0x3A70-byte block (SoA bounds, up to 0x200 entries)
// through Block::Add (1143206). The culling-group pass (1147875) tests the blocks (inline on the calling
// thread with bCullingBatch 0; BSJobs only on the batched path) and adds visible children back through
// 357475 -> Block::Add; 626862 registers the result into an accumulator. Each block keeps group+0x158 at +0x3A60.
// This probe observes Group::Add and Block::Add (detoured in Hooks/CullGroups) to measure who feeds
// which group, on which threads, and how many main-pass entries CBRO dropped.

namespace CBRO::Probe::Groups
{
	namespace
	{
		constexpr std::size_t kGroupSize = 0x170;

		constexpr std::uint64_t                kDrawWorldCameraID = 81406;     // global NiCamera*
		constexpr std::uint64_t                kDrawWorldCullerAID = 865470;   // global BSGeometryListCullingProcess*
		constexpr std::uint64_t                kDrawWorldCullerBID = 1084947;  // global BSGeometryListCullingProcess*
		constexpr std::array<std::uint64_t, 3> kDrawWorldGroupIDs{ 1117782, 133326, 1328670 };
		constexpr std::uint64_t                kDrawWorldGroupArrayID = 459440;  // { group* data; ...; u32 count @+0x10 }

		constexpr std::size_t kThreadKinds = static_cast<std::size_t>(ThreadKind::kCount);
		constexpr std::size_t kMaxKeys = 32;
		constexpr std::size_t kMaxSamples = 16;

		struct Sample
		{
			std::uintptr_t key{ 0 };
			std::uint64_t  hits{ 0 };
			char           rtti[48]{};
		};

		struct alignas(64) KeyStats
		{
			std::atomic<std::uintptr_t>                          key{ 0 };
			std::atomic<bool>                                    ready{ false };
			std::uintptr_t                                       owner{ 0 };  // groups: group+0x158 at first sight
			std::atomic<std::uint64_t>                           frameCalls{ 0 };
			std::atomic<std::uint64_t>                           intervalCalls{ 0 };
			std::array<std::atomic<std::uint64_t>, kThreadKinds> intervalThreads{};
			std::atomic<std::int64_t>                            sampleBudget{ 20000 };
			std::atomic<std::uint64_t>                           lastSeenFrame{ ~0ull };
			std::uint64_t                                        lastFrameCalls{ 0 };

			std::mutex          sampleLock;
			std::vector<Sample> types;
			std::vector<Sample> callers;
			int                 samplesLogged{ 0 };
		};

		struct Table
		{
			std::string_view                 name;
			std::array<KeyStats, kMaxKeys>   keys{};
			std::atomic<std::uint64_t>       overflow{ 0 };
		};

		Table g_blocks{ "Block::Add by block owner (+0x3A60)" };
		Table g_groups{ "Group::Add by group" };

		bool g_installed{ false };
		std::atomic<std::uint64_t> g_mainSkipped{ 0 };

		KeyStats* FindOrInsert(Table& a_table, std::uintptr_t a_key, std::uintptr_t a_owner) noexcept
		{
			for (auto& slot : a_table.keys) {
				auto current = slot.key.load(std::memory_order_acquire);
				if (current == a_key) {
					return &slot;
				}
				if (current != 0) {
					continue;
				}
				if (slot.key.compare_exchange_strong(current, a_key, std::memory_order_acq_rel)) {
					slot.owner = a_owner;
					slot.ready.store(true, std::memory_order_release);
					return &slot;
				}
				if (current == a_key) {
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

		void Record(Table& a_table, std::string_view a_hook, std::uintptr_t a_key, std::uintptr_t a_owner, const void* a_object, std::uintptr_t a_returnAddress)
		{
			// key 0 is the empty-slot marker
			const auto key = a_key ? a_key : 1;
			auto       stats = FindOrInsert(a_table, key, a_owner);
			if (!stats) {
				a_table.overflow.fetch_add(1, std::memory_order_relaxed);
				return;
			}

			const auto tid = GetCurrentThreadId();
			stats->frameCalls.fetch_add(1, std::memory_order_relaxed);
			stats->intervalCalls.fetch_add(1, std::memory_order_relaxed);
			stats->intervalThreads[static_cast<std::size_t>(ClassifyThread(tid))].fetch_add(1, std::memory_order_relaxed);

			if (stats->sampleBudget.load(std::memory_order_relaxed) > 0 &&
				stats->sampleBudget.fetch_sub(1, std::memory_order_relaxed) > 0) {
				const auto       vtable = TryReadVtable(a_object);
				std::scoped_lock lock(stats->sampleLock);
				AddSample(stats->types, vtable, vtable ? a_object : nullptr);
				AddSample(stats->callers, a_returnAddress, nullptr);
			}

			if (Tracing()) {
				const auto frame = State().frame.load(std::memory_order_relaxed);
				if (stats->lastSeenFrame.exchange(frame, std::memory_order_relaxed) != frame) {
					char type[48]{};
					TryGetRTTIName(a_object, type, sizeof(type));
					TraceEvent(std::format(
						"{} first entry this frame: key {} object {} {} caller {} stage={}",
						a_hook, fmt_ptr(key), fmt_ptr(a_object), type, Util::DescribeCodeAddress(a_returnAddress), StageName(CurrentStage())));
				}
			}
		}

		void OnBlockAdd(const Hooks::CullGroups::BlockAdd& a_add, bool a_skipped)
		{
			if (a_skipped) {
				g_mainSkipped.fetch_add(1, std::memory_order_relaxed);
			}
			Record(g_blocks, "Block::Add", a_add.owner, 0, a_add.object, a_add.returnAddress);
		}

		void OnGroupAdd(void* a_group, RE::NiAVObject* a_object, std::uintptr_t a_owner, std::uintptr_t a_returnAddress)
		{
			Record(g_groups, "Group::Add", reinterpret_cast<std::uintptr_t>(a_group), a_owner, a_object, a_returnAddress);
		}
		std::uintptr_t ReadGlobalPointer(std::uint64_t a_id) noexcept
		{
			return *reinterpret_cast<const std::uintptr_t*>(CBRO::Engine::OG(a_id).address());
		}

		// Names the DrawWorld objects a key can be, plus its NiRTTI type if it is a Gamebryo object.
		std::string Label(std::uintptr_t a_key)
		{
			std::string label;
			if (a_key == ReadGlobalPointer(kDrawWorldCameraID)) {
				label += " =DrawWorld camera";
			}
			if (a_key == State().worldCamera.load()) {
				label += " =WorldRoot camera";
			}
			if (a_key == ReadGlobalPointer(kDrawWorldCullerAID)) {
				label += " =DrawWorld culler A";
			}
			if (a_key == ReadGlobalPointer(kDrawWorldCullerBID)) {
				label += " =DrawWorld culler B";
			}
			for (std::size_t i = 0; i < kDrawWorldGroupIDs.size(); ++i) {
				if (a_key == CBRO::Engine::OG(kDrawWorldGroupIDs[i]).address()) {
					label += std::format(" =DrawWorld group {}", i);
				}
			}
			const auto arrayBase = CBRO::Engine::OG(kDrawWorldGroupArrayID).address();
			const auto data = *reinterpret_cast<const std::uintptr_t*>(arrayBase);
			const auto count = *reinterpret_cast<const std::uint32_t*>(arrayBase + 0x10);
			if (data && a_key >= data && a_key < data + count * kGroupSize && (a_key - data) % kGroupSize == 0) {
				label += std::format(" =DrawWorld group array[{}]", (a_key - data) / kGroupSize);
			}

			char type[48]{};
			char name[64]{};
			if (a_key > 1 && TryGetRTTIName(reinterpret_cast<const void*>(a_key), type, sizeof(type))) {
				TryGetObjectName(reinterpret_cast<const void*>(a_key), name, sizeof(name));
				label += std::format(" [{} '{}']", type, name);
			}
			return label.empty() ? " (unknown)" : label;
		}

		std::string DescribeCaller(std::uintptr_t a_address)
		{
			auto       result = Util::DescribeCodeAddress(a_address);
			DWORD64    imageBase = 0;
			const auto function = RtlLookupFunctionEntry(a_address, &imageBase, nullptr);
			if (function && imageBase == reinterpret_cast<DWORD64>(GetModuleHandleW(nullptr))) {
				result += std::format(" (chunk 0x{:X})", function->BeginAddress);
			}
			return result;
		}

		void LogTable(Table& a_table)
		{
			bool any = false;
			for (auto& stats : a_table.keys) {
				if (!stats.ready.load(std::memory_order_acquire)) {
					continue;
				}
				const auto interval = stats.intervalCalls.exchange(0);
				std::array<std::uint64_t, kThreadKinds> threads{};
				for (std::size_t t = 0; t < kThreadKinds; ++t) {
					threads[t] = stats.intervalThreads[t].exchange(0);
				}
				if (!interval) {
					continue;
				}
				if (!any) {
					logger::info("{}:", a_table.name);
					any = true;
				}

				const auto key = stats.key.load();
				logger::info(
					"  key {}{}{}: last frame={} interval={} threads main/render/other={}/{}/{}",
					fmt_ptr(key), Label(key),
					stats.owner ? std::format(" owner(+0x158) {}{}", fmt_ptr(stats.owner), Label(stats.owner)) : std::string{},
					stats.lastFrameCalls, interval, threads[0], threads[1], threads[2]);

				std::scoped_lock lock(stats.sampleLock);
				const bool       exhausted = stats.sampleBudget.load() <= 0;
				if (stats.samplesLogged >= 2 || (stats.samplesLogged == 1 && !exhausted)) {
					continue;
				}
				++stats.samplesLogged;
				std::string types;
				for (const auto& type : stats.types) {
					types += std::format(" {}x{}", type.rtti[0] ? type.rtti : "(unreadable)", type.hits);
				}
				logger::info("    object types ({}):{}", exhausted ? "first 20000" : "so far", types);
				for (const auto& caller : stats.callers) {
					logger::info("    caller {} x{}", DescribeCaller(caller.key), caller.hits);
				}
			}
			if (!any) {
				logger::info("{}: no calls this interval", a_table.name);
			}
			if (const auto overflow = a_table.overflow.exchange(0)) {
				logger::warn("  {} calls not attributed: more than {} keys", overflow, kMaxKeys);
			}
		}
	}

	void Install()
	{
		Hooks::CullGroups::SetBlockObserver(&OnBlockAdd);
		Hooks::CullGroups::SetGroupObserver(&OnGroupAdd);
		g_installed = true;
	}
	void EndFrame(bool a_log)
	{
		for (auto* table : { &g_blocks, &g_groups }) {
			for (auto& stats : table->keys) {
				stats.lastFrameCalls = stats.frameCalls.exchange(0, std::memory_order_relaxed);
			}
		}
		if (!a_log) {
			return;
		}
		if (!g_installed) {
			return;
		}
		logger::info("main-pass entries dropped by CBRO this interval: {} (main owner {})", g_mainSkipped.exchange(0), fmt_ptr(Hooks::CullGroups::MainOwner()));
		LogTable(g_blocks);
		LogTable(g_groups);
	}
}
