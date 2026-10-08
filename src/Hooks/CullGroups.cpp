#include "Hooks/CullGroups.h"

#include "Hooks/PrevisFeed.h"
#include "Hooks/RenderStages.h"
#include "Settings.h"
#include "Util/Hooking.h"

#include <intrin.h>

namespace CBRO::Hooks::CullGroups
{
	namespace
	{
		constexpr std::uint64_t kBlockAddID = 1143206;
		constexpr std::uint64_t kGroupAddID = 1175493;
		constexpr std::uint64_t kChildPushID = 357475;
		constexpr std::uint64_t kMainGroupID = 1117782;         // DrawWorld group 0 (the struct itself): shadow casters
		constexpr std::uint64_t kGroup1ID = 133326;             // DrawWorld group 1: non-casters (main view only)
		constexpr std::uint64_t kGroup2ID = 1328670;            // DrawWorld group 2: the previs list (main view only)
		constexpr std::uint64_t kGroupArrayID = 459440;         // DrawWorld group array { group* data; ...; u32 count @+0x10 }
		constexpr std::uint64_t kMainAccumulatorID = 1211381;   // DrawWorld's BSShaderAccumulator* (global)
		constexpr std::uint64_t kAccumulatorVtableID = 357329;  // BSShaderAccumulator (the RTTI shows no subclass)
		constexpr std::size_t   kRegisterObjectSlot = 45;       // bool RegisterObject(BSGeometry*), called only via the vtable
		constexpr std::size_t   kGroupSize = 0x170;
		constexpr std::size_t   kGroupArrayCountOffset = 0x10;

		// The sun's cascades (see the header). Both paths start from Render_PreUI every frame and are checked by
		// reading their call bytes (nothing on them is patched).
		constexpr std::uint64_t kRenderPreUIID = 984743;
		constexpr std::size_t   kBatchedCallOffset = 0x175;        // Render_PreUI: call 339369
		constexpr std::uint64_t kBatchedCallerID = 339369;
		constexpr std::size_t   kBatchedSetupCallOffset = 0x185;   // 339369: call 432406 (update + cascade cull)
		constexpr std::uint64_t kBatchedSetupID = 432406;
		constexpr std::size_t   kUnbatchedCallOffset = 0x1BF;      // Render_PreUI: call 1108521
		constexpr std::uint64_t kUnbatchedCallerID = 1108521;
		constexpr std::size_t   kUnbatchedCullCallOffset = 0xE49;  // 1108521: call 1390075 (cascade cull, group 0)
		constexpr std::uint64_t kCascadeCullID = 1390075;
		constexpr std::uint64_t kDirLightUpdateID = 1242204;       // BSShadowDirectionalLight vtable slot 14
		constexpr std::size_t   kDirLightUpdateSlot = 14;
		constexpr std::uint64_t kGroupsEnabledID = 938585;   // byte: the value of the INI setting bCullingBatch:General
		constexpr std::uint64_t kDirShadowsID = 241042;      // byte
		constexpr std::uint64_t kShadowSceneNodeID = 1327069;     // ShadowSceneNode* the batched path takes its light from
		constexpr std::uint64_t kUnbatchedSceneNodeID = 879298;   // ShadowSceneNode* the unbatched path takes its light from
		constexpr std::uint64_t kDirLightVtableID = 97945;     // BSShadowDirectionalLight
		constexpr std::uint64_t kShadowRangeID = 777729;       // float
		constexpr std::size_t   kSunLightOffset = 0x208;       // ShadowSceneNode -> BSShadowDirectionalLight*
		constexpr std::size_t   kShadowCameraOffset = 0x2C0;   // BSShadowDirectionalLight -> NiCamera*
		constexpr std::size_t   kLocalRotateOffset = 0x30;     // NiAVObject::local.rotate (row 0 first)
		constexpr std::size_t   kLocalTranslateOffset = 0x60;  // NiAVObject::local.translate
		constexpr std::size_t   kCascadeCountOffset = 0x190;   // BSShadowDirectionalLight -> u32 cascade count
		constexpr std::size_t   kCascadesOffset = 0x198;       // ... -> its cascade records (0xF0 bytes each)
		constexpr std::size_t   kCascadeStride = 0xF0;
		constexpr std::size_t   kCascadeAccumulatorOffset = 0x48;  // record -> BSShaderAccumulator*
		constexpr std::size_t   kCascadeSlabOffsets[2]{ 0x58, 0x68 };  // record -> receiver slab NiPlanes (near, far)
		constexpr std::uint32_t kMaxCascades = 8;
		constexpr std::uint64_t kGodrayCascadesID = 542491;    // u32: godrays read the first min(count, this) cascades

		// The three loops that add a node's children to the main groups one by one (see the header), each patched
		// at the instruction that starts the loop with a switchable jump to a stub that asks the cell-node filter
		// first. What the stub's exits must find at their targets is verified before patching.
		enum class PruneSite : std::uint8_t
		{
			kWalkExterior,            // scene walk, exterior path: node in r14 (= rax), index in r15d, dx = 0
			kWalkOverride,            // scene walk, override-root path: node in r15 (= rax), index in r12d, dx = 0
			kDrawWorldRoot,           // DrawWorld's cull, registered-root loop: node in rsi (= rax, null if not a node), root slot in rbx
			kWalkContainer,           // scene walk, exterior path, a child of node 3 in rsi, NiNode's RTTI in rdi (the walk expands exact NiNodes)
			kWalkOverrideContainer,   // ... override-root path: the child in rsi, the RTTI in rbx
		};
		struct PruneSiteSpec
		{
			PruneSite                     kind;
			std::uint64_t                 functionID;
			std::size_t                   patchOffset;
			std::size_t                   resumeOffset;
			std::size_t                   skipOffset;
			std::size_t                   wholeOffset;  // kDrawWorldRoot only: where a root that isn't a node goes (added whole)
			std::span<const std::uint8_t> patchBytes;
			std::span<const std::uint8_t> resumeBytes;
			std::span<const std::uint8_t> skipBytes;
			const char*                   name;
		};
		constexpr std::uint64_t               kSceneWalkID = 1138818;
		constexpr std::uint64_t               kDrawWorldCullID = 718911;
		constexpr std::array<std::uint8_t, 7> kCmpChildren{ 0x66, 0x3B, 0x90, 0x32, 0x01, 0x00, 0x00 };                     // cmp dx,word [rax+132h]
		constexpr std::array<std::uint8_t, 6> kWalkExteriorJae{ 0x0F, 0x83, 0xBC, 0x00, 0x00, 0x00 };                       // jae -> the cell's next child
		constexpr std::array<std::uint8_t, 9> kWalkExteriorNext{ 0x41, 0x0F, 0xB7, 0x84, 0x24, 0x32, 0x01, 0x00, 0x00 };   // movzx eax,word [r12+132h]
		constexpr std::array<std::uint8_t, 6> kWalkOverrideJae{ 0x0F, 0x83, 0x9C, 0x00, 0x00, 0x00 };
		constexpr std::array<std::uint8_t, 5> kWalkOverrideNext{ 0x48, 0x8B, 0x7C, 0x24, 0x48 };                            // mov rdi,[rsp+48h] (then the next child)
		constexpr std::array<std::uint8_t, 9> kRootTest{ 0x48, 0x85, 0xC0, 0x0F, 0x84, 0x87, 0x00, 0x00, 0x00 };           // test rax,rax; je -> add the root whole
		constexpr std::array<std::uint8_t, 6> kRootResume{ 0x8B, 0x88, 0x08, 0x01, 0x00, 0x00 };                            // mov ecx,[rax+108h] (the flag-14 check)
		constexpr std::array<std::uint8_t, 4> kRootNext{ 0x48, 0x83, 0xC3, 0x08 };                                          // add rbx,8 (the next root)
		constexpr std::array<std::uint8_t, 3> kRootWhole{ 0x48, 0x8B, 0x13 };                                               // mov rdx,[rbx]
		constexpr std::array<std::uint8_t, 9> kContainerRtti{ 0x48, 0x8B, 0x06, 0x48, 0x8B, 0xCE, 0xFF, 0x50, 0x10 };     // mov rax,[rsi]; mov rcx,rsi; call [rax+10h] (GetRTTI)
		constexpr std::array<std::uint8_t, 3> kContainerCmpRdi{ 0x48, 0x3B, 0xC7 };                                        // cmp rax,rdi
		constexpr std::array<std::uint8_t, 3> kContainerCmpRbx{ 0x48, 0x3B, 0xC3 };                                        // cmp rax,rbx
		constexpr std::array<std::uint8_t, 8> kContainerNextR14{ 0x41, 0x0F, 0xB7, 0x86, 0x32, 0x01, 0x00, 0x00 };         // movzx eax,word [r14+132h] (node 3's next child)
		constexpr std::array<std::uint8_t, 8> kContainerNextR15{ 0x41, 0x0F, 0xB7, 0x87, 0x32, 0x01, 0x00, 0x00 };         // movzx eax,word [r15+132h]
		const std::array<PruneSiteSpec, 5>    kPruneSites{ {
			{ PruneSite::kWalkExterior, kSceneWalkID, 0x5B8, 0x5BF, 0x681, 0, kCmpChildren, kWalkExteriorJae, kWalkExteriorNext, "cullgroups:scene walk cell node (exterior path)" },
			{ PruneSite::kWalkOverride, kSceneWalkID, 0xA00, 0xA07, 0xAA4, 0, kCmpChildren, kWalkOverrideJae, kWalkOverrideNext, "cullgroups:scene walk cell node (override-root path)" },
			{ PruneSite::kDrawWorldRoot, kDrawWorldCullID, 0x309, 0x312, 0x3AE, 0x399, kRootTest, kRootResume, kRootNext, "cullgroups:DrawWorld cull registered root" },
			{ PruneSite::kWalkContainer, kSceneWalkID, 0x5F1, 0x5FA, 0x64A, 0, kContainerRtti, kContainerCmpRdi, kContainerNextR14, "cullgroups:scene walk container (exterior path)" },
			{ PruneSite::kWalkOverrideContainer, kSceneWalkID, 0xA31, 0xA3A, 0xA92, 0, kContainerRtti, kContainerCmpRbx, kContainerNextR15, "cullgroups:scene walk container (override-root path)" },
		} };


		// Which code adds main-pass top-level objects (Group::Add return addresses), for the log.
		struct SiteCount
		{
			std::atomic<std::uintptr_t> site{ 0 };
			std::atomic<std::uint64_t>  count{ 0 };
		};
		constexpr std::size_t               kSiteSlots = 32;  // (the v1.19 run filled 8 with rare sites before the main feeder appeared)
		std::array<SiteCount, kSiteSlots>   g_groupAddSites{};

		void RecordGroupAddSite(std::uintptr_t a_site) noexcept
		{
			for (auto& slot : g_groupAddSites) {
				auto current = slot.site.load(std::memory_order_relaxed);
				if (current == 0 && slot.site.compare_exchange_strong(current, a_site, std::memory_order_relaxed)) {
					current = a_site;
				}
				if (current == a_site) {
					slot.count.fetch_add(1, std::memory_order_relaxed);
					return;
				}
			}
		}
		// RegisterObject calls per accumulator (which views register how much), for the log: one table per mode
		// (0 previs, 1 CBRO), since the accumulator hook counts in both. Lamp shadow accumulators come and go at new
		// addresses (their cameras are pooled), so the tables are wide enough not to fill up with dead ones.
		std::array<std::array<SiteCount, 64>, 2> g_registrationSites{};
		std::atomic<int>                         g_registrationMode{ 1 };

		void RecordRegistration(std::uintptr_t a_accumulator) noexcept
		{
			for (auto& slot : g_registrationSites[static_cast<std::size_t>(g_registrationMode.load(std::memory_order_relaxed) & 1)]) {
				auto current = slot.site.load(std::memory_order_relaxed);
				if (current == 0 && slot.site.compare_exchange_strong(current, a_accumulator, std::memory_order_relaxed)) {
					current = a_accumulator;
				}
				if (current == a_accumulator) {
					slot.count.fetch_add(1, std::memory_order_relaxed);
					return;
				}
			}
		}

		// BSShaderAccumulator::renderMode (+0x560, CommonLibF4 header), read guarded: an accumulator may be gone.
		std::int32_t ReadRenderMode(std::uintptr_t a_accumulator) noexcept
		{
			__try {
				return static_cast<std::int32_t>(*reinterpret_cast<const std::uint32_t*>(a_accumulator + 0x560));
			} __except (EXCEPTION_EXECUTE_HANDLER) {
				return -1;
			}
		}

		std::string_view RenderModeName(std::int32_t a_mode) noexcept
		{
			switch (a_mode) {
			case 0x0:
				return "normal"sv;
			case 0xD:
				return "depth prepass"sv;
			case 0xE:
				return "occlusion map"sv;
			case 0xF:
				return "shadowmap"sv;
			case 0x10:
				return "shadowmap dir"sv;
			case 0x11:
				return "shadowmap PB"sv;
			case 0x12:
				return "localmap"sv;
			case 0x18:
				return "deferred gbuffer"sv;
			default:
				return ""sv;
			}
		}

		// Each function starts with `mov [rsp+N], rbx`: one 5-byte position-independent instruction.
		constexpr std::array<std::uint8_t, 5> kBlockAddPrologue{ 0x48, 0x89, 0x5C, 0x24, 0x08 };
		constexpr std::array<std::uint8_t, 5> kGroupAddPrologue{ 0x48, 0x89, 0x5C, 0x24, 0x20 };
		constexpr std::array<std::uint8_t, 5> kChildPushPrologue{ 0x48, 0x89, 0x5C, 0x24, 0x08 };

		constexpr std::size_t kBlockOwnerOffset = 0x3A60;
		constexpr std::size_t kBlockCountOffset = 0x3A68;
		constexpr std::size_t kBlockBoundsOffset = 0x60;
		// Per-entry bytes, 5 per entry: result, force-visible, then three group markers.
		constexpr std::size_t kEntryBytesOffset = 0x3060;
		constexpr std::size_t kEntryBytesStride = 5;
		constexpr std::size_t kObjectFlagsOffset = 0x108;  // NiAVObject::flags
		constexpr std::size_t kShaderPropertyOffset = 0x138;  // BSGeometry::properties[1]
		// Group::Add's current blocks and the pending group-marker bytes it sets on them.
		constexpr std::size_t kGroupGeometryBlockOffset = 0x150;
		constexpr std::size_t kGroupNodeBlockOffset = 0xD0;
		constexpr std::size_t kBlockMarkerBegin = 0x3A6C;
		constexpr std::size_t kBlockMarkerEnd = 0x3A6D;
		constexpr std::size_t kBlockMarkerExtra = 0x3A6E;
		constexpr std::size_t kGroupOwnerOffset = 0x158;

		// BSGeometry::type (+0x158) of meshes Block::Add expands into per-instance entries.
		constexpr std::size_t   kGeometryTypeOffset = 0x158;
		constexpr std::uint8_t  kMergeInstancedType = 0x0F;
		constexpr std::size_t   kInstancesOffset = 0x1C0;
		constexpr std::size_t   kInstanceCountOffset = 0x1D0;
		constexpr std::size_t   kInstanceStride = 0xF0;
		constexpr std::size_t   kInstanceBoundOffset = 0xC0;

		std::atomic<BlockFilter>    g_filter{ nullptr };
		std::atomic<InstanceFilter> g_instanceFilter{ nullptr };
		std::atomic<BlockObserver>  g_blockObserver{ nullptr };
		std::atomic<GroupObserver>  g_groupObserver{ nullptr };
		std::atomic<GroupFilter>    g_groupFilter{ nullptr };
		std::atomic<MainViewFilter> g_mainViewFilter{ nullptr };
		std::atomic<MainRegistrationObserver> g_mainRegistrationObserver{ nullptr };
		std::atomic<CellNodeFilter>     g_cellNodeFilter{ nullptr };
		std::atomic<bool>           g_mainCullActive{ false };
		std::atomic<const void*>    g_droppedAccumulator{ nullptr };
		std::atomic<const void*>    g_countedAccumulator{ nullptr };
		std::atomic<std::uint64_t>  g_countedRegistrations{ 0 };
		std::atomic<const void*>    g_filteredAccumulator{ nullptr };
		std::atomic<CasterFilter>   g_casterFilter{ nullptr };
		std::atomic<const void*>    g_farAccumulator{ nullptr };
		std::atomic<CasterFilter>   g_farFilter{ nullptr };

		std::atomic<std::uint64_t> g_forcedCleared{ 0 };
		std::atomic<std::uint64_t> g_skyForced{ 0 };   // sky entries given previs's force-visible mark (CBRO frames)
		std::atomic<std::uint64_t> g_skyKept{ 0 };     // ... and reject bounds CBRO's filter would have given them
		std::atomic<std::uint64_t> g_groupAddsConsidered{ 0 };

		std::uintptr_t g_mainGroup{ 0 };
		std::uintptr_t g_triShapeVtable{ 0 };
		std::uintptr_t g_skyShaderVtable{ 0 };
		std::uintptr_t g_group1{ 0 };
		std::uintptr_t g_group2{ 0 };
		std::uintptr_t g_groupArray{ 0 };
		std::uintptr_t g_mainAccumulator{ 0 };  // address of the global holding the accumulator pointer
		struct CallSite
		{
			std::uintptr_t site{ 0 };    // a `call rel32`
			std::uintptr_t target{ 0 };  // the engine function it calls
		};
		std::array<CallSite, 2> g_batchedPath{};    // Render_PreUI -> 339369 -> 432406
		std::array<CallSite, 2> g_unbatchedPath{};  // Render_PreUI -> 1108521 -> 1390075
		std::uintptr_t          g_dirLightUpdateSlot{ 0 };
		std::uintptr_t          g_dirLightUpdate{ 0 };
		std::uintptr_t          g_unbatchedSceneNode{ 0 };
		std::uintptr_t g_groupsEnabled{ 0 };
		std::uintptr_t g_dirShadows{ 0 };
		std::uintptr_t g_shadowSceneNode{ 0 };
		std::uintptr_t g_dirLightVtable{ 0 };
		std::uintptr_t g_shadowRange{ 0 };
		std::uintptr_t g_godrayCascades{ 0 };
		std::uintptr_t g_blockAddOriginal{ 0 };
		std::uintptr_t g_groupAddOriginal{ 0 };
		std::uintptr_t g_childPushOriginal{ 0 };
		std::uintptr_t g_registerOriginal{ 0 };
		bool           g_installed{ false };
		bool           g_ready{ false };

		// Taken out while CBRO is off, so the engine then runs its own code (SetHooksIn).
		Util::SwitchableHook g_blockAddHook;
		Util::SwitchableHook g_groupAddHook;
		Util::SwitchableHook g_childPushHook;
		Util::SwitchableHook g_registerHook;
		std::array<Util::SwitchableHook, 5> g_cellNodeHooks{};  // one per prune site

		// Hook calls, one block per thread (single writer), summed at log time.
		struct alignas(64) CallCounts
		{
			std::atomic<std::uint64_t> blockAdds{ 0 };
			std::atomic<std::uint64_t> groupAdds{ 0 };
			std::atomic<std::uint64_t> childPushes{ 0 };
			std::atomic<std::uint64_t> registrations{ 0 };
			std::atomic<std::uint64_t> dropped{ 0 };  // registrations into the dropped accumulator
		};
		constexpr std::size_t              kCountSlots = 64;
		std::array<CallCounts, kCountSlots> g_callCounts{};
		std::atomic<std::uint32_t>          g_nextCountSlot{ 0 };
		HookCalls                           g_callsReported{};  // totals at the last TakeHookCalls (main thread)

		CallCounts& LocalCounts() noexcept
		{
			thread_local CallCounts* counts = nullptr;  // constant-initialized: no thread-local init guard per call
			if (!counts) [[unlikely]] {
				counts = &g_callCounts[g_nextCountSlot.fetch_add(1, std::memory_order_relaxed) % kCountSlots];
			}
			return *counts;
		}

		void Count(std::atomic<std::uint64_t>& a_counter) noexcept
		{
			a_counter.store(a_counter.load(std::memory_order_relaxed) + 1, std::memory_order_relaxed);
		}

		// The group whose Group::Add or ChildPush is running on this thread (Block::Add's only callers).
		thread_local const void* t_group{ nullptr };

		using BlockAddFn = std::int32_t (*)(void*, RE::NiAVObject*, const RE::NiBound*, std::int32_t);
		using GroupAddFn = void (*)(void*, RE::NiAVObject*, const RE::NiBound*, std::uint32_t);
		// The flag bytes travel in r9 and 8-byte stack slots; all seven arguments are passed on untouched,
		// and RAX comes back unchanged (callers ignore it).
		using ChildPushFn = std::uint64_t (*)(void*, RE::NiAVObject*, const RE::NiBound*, std::uint64_t, std::uint64_t, std::uint64_t, void*);
		using RegisterFn = bool (*)(void*, RE::NiAVObject*);

		template <class T>
		T ReadAt(const void* a_base, std::size_t a_offset) noexcept
		{
			return *reinterpret_cast<const T*>(static_cast<const std::byte*>(a_base) + a_offset);
		}

		float* EntryBoundBase(const void* a_block, std::uint32_t a_index) noexcept
		{
			const auto base = reinterpret_cast<float*>(const_cast<std::byte*>(static_cast<const std::byte*>(a_block) + kBlockBoundsOffset));
			return base + (a_index & 3u) + (a_index >> 2) * 16u;
		}

		// Groups DrawWorld registers with the main accumulator only are main-only; group 0 is also read by the
		// sun's shadow cascades; anything else (or no group known) is unknown, which only ever costs culling,
		// never another view's objects.
		GroupKind ClassifyGroup(const void* a_group) noexcept
		{
			const auto group = reinterpret_cast<std::uintptr_t>(a_group);
			if (!group) {
				return GroupKind::kUnknown;
			}
			if (group == g_mainGroup) {
				return GroupKind::kSunShared;
			}
			if (group == g_group1 || group == g_group2) {
				return GroupKind::kMainOnly;
			}
			const auto data = ReadAt<std::uintptr_t>(reinterpret_cast<const void*>(g_groupArray), 0);
			const auto count = ReadAt<std::uint32_t>(reinterpret_cast<const void*>(g_groupArray), kGroupArrayCountOffset);
			const bool inArray = data && group >= data && group < data + static_cast<std::uintptr_t>(count) * kGroupSize && (group - data) % kGroupSize == 0;
			return inArray ? GroupKind::kMainOnly : GroupKind::kUnknown;
		}

		struct GroupScope
		{
			explicit GroupScope(const void* a_group) noexcept :
				saved(t_group)
			{
				t_group = a_group;
			}
			~GroupScope() { t_group = saved; }
			GroupScope(const GroupScope&) = delete;
			GroupScope& operator=(const GroupScope&) = delete;
			const void* saved;
		};

		// Sky meshes (BSSkyShaderProperty: stars, galaxy, moons, cloud layers) reach the main camera through group 1,
		// whose entries previs's Block::Add marks force-visible: they are drawn whatever their bound says. In a CBRO
		// frame previs is suspended, so the finish loop tests their bounds instead, and a sky mesh whose bound doesn't
		// follow the sky (the Visible Galaxy mod's "Galaxy" shape) vanished. Such an entry gets the mark back, and
		// CBRO's own filter never drops one. The first few are logged once each.
		void NoteSky(const RE::NiAVObject* a_object, const RE::NiBound* a_bound, GroupKind a_kind, std::uint8_t a_markBefore, std::uintptr_t a_caller) noexcept
		{
			static std::array<std::atomic<const void*>, 16> seen{};
			for (auto& slot : seen) {
				const void* current = slot.load(std::memory_order_relaxed);
				if (current == a_object) {
					return;
				}
				if (!current && slot.compare_exchange_strong(current, a_object, std::memory_order_relaxed)) {
					const char* name = a_object->name.c_str();
					logger::info(
						"cullgroups: sky mesh '{}' at Block::Add: group {} | bound ({:.0f},{:.0f},{:.0f}) r={:.1f} | flags 0x{:X} | force-visible mark {} -> {} | caller {:X}",
						name ? name : "", a_kind == GroupKind::kMainOnly ? "main-only (1/2/array)" : a_kind == GroupKind::kSunShared ? "0 (sun shared)" : "unknown",
						a_bound->center.x, a_bound->center.y, a_bound->center.z, a_bound->fRadius, ReadAt<std::uint64_t>(a_object, kObjectFlagsOffset),
						a_markBefore, a_kind == GroupKind::kMainOnly ? 1 : a_markBefore, a_caller - REL::Module::get().base());
					return;
				}
				if (slot.load(std::memory_order_relaxed) == a_object) {
					return;
				}
			}
		}

		bool IsSkyMesh(const RE::NiAVObject* a_object) noexcept
		{
			if (!a_object || !g_skyShaderVtable || ReadAt<std::uintptr_t>(a_object, 0) != g_triShapeVtable) {
				return false;
			}
			const auto shader = ReadAt<std::uintptr_t>(a_object, kShaderPropertyOffset);
			return shader && ReadAt<std::uintptr_t>(reinterpret_cast<const void*>(shader), 0) == g_skyShaderVtable;
		}

		std::int32_t BlockAddThunk(void* a_block, RE::NiAVObject* a_object, const RE::NiBound* a_bound, std::int32_t a_startIndex)
		{
			Count(LocalCounts().blockAdds);
			// Outside a frame CBRO culls this is a plain pass-through (in previs mode the hook is taken out), but for
			// the sky's force-visible mark.
			const bool active = g_mainCullActive.load(std::memory_order_relaxed);
			const auto observer = g_blockObserver.load(std::memory_order_relaxed);
			const bool sky = a_startIndex < 0 && IsSkyMesh(a_object);
			if (!active && !observer && !sky) {
				return reinterpret_cast<BlockAddFn>(g_blockAddOriginal)(a_block, a_object, a_bound, a_startIndex);
			}
			const auto filter = g_filter.load(std::memory_order_relaxed);
			const auto instanceFilter = g_instanceFilter.load(std::memory_order_relaxed);

			const auto owner = ReadAt<std::uintptr_t>(a_block, kBlockOwnerOffset);
			const bool mainPass = owner != 0 && owner == MainOwner();
			const BlockAdd add{
				a_block, a_object, a_bound, a_startIndex, owner, mainPass,
				mainPass ? ClassifyGroup(t_group) : GroupKind::kUnknown,
				reinterpret_cast<std::uintptr_t>(_ReturnAddress())
			};
			// CBRO only acts on DrawWorld's own cull (a later pass re-processing a group is left alone).
			const bool culling = add.mainPass && active;

			// Continuation calls (startIndex >= 0) only add more instance entries of a merged mesh.
			const RE::NiBound* replacement = nullptr;
			if (filter && culling && a_startIndex < 0 && a_object && a_bound) {
				replacement = filter(add);
				if (add.kind == GroupKind::kUnknown) {
					replacement = nullptr;  // (other readers possible: dropped from the main view only; the filter recorded it)
				}
			}
			if (sky && replacement) {
				g_skyKept.fetch_add(1, std::memory_order_relaxed);
				replacement = nullptr;
			}
			const bool forceSky = sky && add.mainPass && add.kind == GroupKind::kMainOnly;
			if (observer) {
				observer(add, replacement != nullptr);
			}

			const bool watchInstances = instanceFilter && culling && add.kind == GroupKind::kMainOnly && a_object;
			const auto before = (watchInstances || replacement || sky) ? ReadAt<std::uint32_t>(a_block, kBlockCountOffset) : 0u;
			const auto result = reinterpret_cast<BlockAddFn>(g_blockAddOriginal)(a_block, a_object, replacement ? replacement : a_bound, a_startIndex);
			if (sky && add.mainPass && before < 0x200 && ReadAt<std::uint32_t>(a_block, kBlockCountOffset) > before) {
				auto*      bytes = static_cast<std::uint8_t*>(a_block) + kEntryBytesOffset + before * kEntryBytesStride;
				const auto mark = bytes[1];
				if (forceSky && !mark) {
					bytes[1] = 1;  // force-visible, as previs's Block::Add sets it for group 1/2 entries
					g_skyForced.fetch_add(1, std::memory_order_relaxed);
				}
				NoteSky(a_object, a_bound, add.kind, mark, add.returnAddress);
			}
			if (replacement && before < 0x200) {
				// With previs active, Block::Add marks entries of some groups (group+0x16A == 0 ->
				// block+0x3A6F) force-visible, and the finish loop registers them whatever the frustum
				// test says. A rejected entry loses that mark, and its result byte is cleared in case no
				// frustum test runs for the block. An object's own always-draw flag (NiAVObject::flags
				// bit 11, the other source of the mark) is respected: that entry stays drawn.
				auto*      bytes = static_cast<std::uint8_t*>(a_block) + kEntryBytesOffset + before * kEntryBytesStride;
				const bool alwaysDraw = (ReadAt<std::uint64_t>(a_object, kObjectFlagsOffset) >> 11) & 1;
				if (bytes[1] && !alwaysDraw) {
					g_forcedCleared.fetch_add(1, std::memory_order_relaxed);
					bytes[0] = 0;  // result
					bytes[1] = 0;  // force-visible
				}
			}
			if (watchInstances) {
				const auto after = ReadAt<std::uint32_t>(a_block, kBlockCountOffset);
				const auto first = before + (a_startIndex < 0 ? 1u : 0u);  // a fresh add writes the object's own entry first
				if (after > first && after <= 0x200) {
					instanceFilter(add, first, after);
				}
			}
			return result;
		}

		void FileOutsideRooms() noexcept;
		bool InFiledRoom(const RE::NiAVObject* a_object) noexcept;
		bool                       g_interiorOutsideRooms{ false };
		std::uintptr_t             g_restrictedAddReturn{ 0 };  // the scene walk's in-a-room add (ResolveRestrictedAdd)
		std::uintptr_t             g_portalAddReturn{ 0 };      // its portal path's add (0 = off)

		void GroupAdd(void* a_group, RE::NiAVObject* a_object, const RE::NiBound* a_bound, std::uint32_t a_flags, std::uintptr_t a_returnAddress)
		{
			Count(LocalCounts().groupAdds);
			const bool active = g_mainCullActive.load(std::memory_order_relaxed);
			const auto observer = g_groupObserver.load(std::memory_order_relaxed);
			if (!active && !observer && !g_blockObserver.load(std::memory_order_relaxed)) {
				GroupScope scope(a_group);  // (the sky's force-visible mark needs the group: BlockAddThunk)
				reinterpret_cast<GroupAddFn>(g_groupAddOriginal)(a_group, a_object, a_bound, a_flags);
				return;
			}
			if (observer) {
				observer(a_group, a_object, ReadAt<std::uintptr_t>(a_group, kGroupOwnerOffset), a_returnAddress);
			}
			// Group::Add files geometry into the block at group+0x150 and nodes into group+0xD0; flag
			// bits set group markers (block+0x3A6C..0x3A6E) that the next entry picks up. Only a plain
			// add with nothing pending can be left out without moving a marker to another object, and
			// only in a group whose readers are all known.
			const auto filter = g_groupFilter.load(std::memory_order_relaxed);
			if (filter && a_flags == 0 && a_object && a_bound && active) {
				const auto owner = ReadAt<std::uintptr_t>(a_group, kGroupOwnerOffset);
				const auto kind = owner != 0 && owner == MainOwner() ? ClassifyGroup(a_group) : GroupKind::kUnknown;
				if (kind != GroupKind::kUnknown) {
					const auto block = ReadAt<const std::uint8_t*>(a_group, a_object->IsGeometry() ? kGroupGeometryBlockOffset : kGroupNodeBlockOffset);
					if (block && !block[kBlockMarkerBegin] && !block[kBlockMarkerEnd] && !block[kBlockMarkerExtra]) {
						const auto considered = g_groupAddsConsidered.fetch_add(1, std::memory_order_relaxed);
						if ((considered & 63) == 0) {  // sampled 1 in 64 (the log scales it back)
							RecordGroupAddSite(a_returnAddress);
						}
						if (filter(a_object, a_bound, kind)) {
							return;
						}
					}
				}
			}
			GroupScope scope(a_group);
			reinterpret_cast<GroupAddFn>(g_groupAddOriginal)(a_group, a_object, a_bound, a_flags);
		}

		void GroupAddThunk(void* a_group, RE::NiAVObject* a_object, const RE::NiBound* a_bound, std::uint32_t a_flags)
		{
			const auto returnAddress = reinterpret_cast<std::uintptr_t>(_ReturnAddress());
			if (returnAddress == g_portalAddReturn && InFiledRoom(a_object)) {
				return;  // (its room went whole into group 1 this frame)
			}
			GroupAdd(a_group, a_object, a_bound, a_flags, returnAddress);
			if (returnAddress == g_restrictedAddReturn && g_interiorOutsideRooms && reinterpret_cast<std::uintptr_t>(a_group) == g_group1) {
				FileOutsideRooms();  // (the walk DrawWorld runs: its group 1 here, so its group 0 is DrawWorld's)
			}
		}

		std::uint64_t ChildPushThunk(void* a_group, RE::NiAVObject* a_object, const RE::NiBound* a_bound, std::uint64_t a_flag, std::uint64_t a_arg5, std::uint64_t a_arg6, void* a_context)
		{
			Count(LocalCounts().childPushes);
			GroupScope scope(a_group);  // (also outside culled frames: the sky's force-visible mark needs the group)
			return reinterpret_cast<ChildPushFn>(g_childPushOriginal)(a_group, a_object, a_bound, a_flag, a_arg5, a_arg6, a_context);
		}

		bool RegisterObjectThunk(void* a_accumulator, RE::NiAVObject* a_object)
		{
			auto& counts = LocalCounts();
			Count(counts.registrations);
			if ((counts.registrations.load(std::memory_order_relaxed) & 15) == 0) {  // sampled 1 in 16 per thread (the log scales it back)
				RecordRegistration(reinterpret_cast<std::uintptr_t>(a_accumulator));
			}
			if (a_accumulator && a_accumulator == g_countedAccumulator.load(std::memory_order_relaxed)) {
				g_countedRegistrations.fetch_add(1, std::memory_order_relaxed);  // (offered: dropped ones too)
			}
			if (a_accumulator && a_accumulator == g_droppedAccumulator.load(std::memory_order_relaxed)) {
				Count(counts.dropped);
				return true;  // an emptied lamp shadow map (callers ignore the result)
			}
			if (a_accumulator && a_accumulator == g_filteredAccumulator.load(std::memory_order_relaxed)) {
				if (const auto filter = g_casterFilter.load(std::memory_order_relaxed); filter && filter(a_object)) {
					return true;  // a caster that can't shadow a visible pixel (callers ignore the result)
				}
			}
			if (a_accumulator && a_accumulator == g_farAccumulator.load(std::memory_order_relaxed)) {
				if (const auto filter = g_farFilter.load(std::memory_order_relaxed); filter && filter(a_object)) {
					return true;  // a caster whose shadow lands on nothing visible in the sun's last cascade
				}
			}
			if (const auto observer = g_mainRegistrationObserver.load(std::memory_order_relaxed);
				observer && a_accumulator && a_accumulator == *reinterpret_cast<void* const*>(g_mainAccumulator)) {
				observer(a_object);  // (both modes: the diagnostic compares them)
			}
			if (g_mainCullActive.load(std::memory_order_relaxed) && a_accumulator &&
				a_accumulator == *reinterpret_cast<void* const*>(g_mainAccumulator)) {
				if (const auto filter = g_mainViewFilter.load(std::memory_order_relaxed); filter && filter(a_object)) {
					return true;  // left out of the main view (callers ignore the result)
				}
			}
			return reinterpret_cast<RegisterFn>(g_registerOriginal)(a_accumulator, a_object);
		}

		// A cell's node 9 holds precombined art (mesh keys with bit 30, TESObjectCELL::AttachCombinedObjectArt+0x69D).
		// The engine has it AppCulled in some cells (the v1.67 scene scan, 2026-10-05: cells 12 and 17 of the A/B spot,
		// not cell 13; the writer isn't known). The previs-off walk skips an AppCulled node, so in CBRO frames its chunks
		// (traffic-signal heads, signs, bollards) were never filed, while previs draws them: its feed looks each chunk up
		// by id and tests only the chunk's own bit 0 (FO4-ENGINE-NOTES 5.5b, 7.12). So when the walk starts a cell's
		// node 3, an AppCulled node 9 of the same cell has its children filed into group 0 as the walk files a node 9's
		// (each child with bit 0 clear, whole, flags 0). Every CBRO frame (the hooks are in only then), culling or not.
		bool                       g_cellArtNode9{ false };
		bool                       g_cellNodePruning{ false };
		constexpr std::uint32_t    kCellArtNode = 9;
		std::atomic<std::uint64_t> g_culledArtNodes{ 0 };   // AppCulled node 9s found
		std::atomic<std::uint64_t> g_culledArtFiled{ 0 };   // their children filed

		void FileCulledCellArt(RE::NiAVObject* a_node3) noexcept
		{
			const auto cell = a_node3->parent;
			if (!cell || cell->children.size() <= kCellArtNode || cell->children[3].get() != a_node3) {
				return;
			}
			const auto art = cell->children[kCellArtNode].get();
			if (!art || !(art->GetFlags() & 1)) {
				return;  // (not AppCulled: the walk files it itself)
			}
			const auto node = art->IsNode();
			if (!node) {
				return;
			}
			g_culledArtNodes.fetch_add(1, std::memory_order_relaxed);
			std::uint64_t filed = 0;
			for (auto& child : node->children) {
				const auto object = child.get();
				if (object && !(object->GetFlags() & 1)) {
					AddDirect(reinterpret_cast<void*>(g_mainGroup), object, &object->worldBound, 0);
					++filed;
				}
			}
			g_culledArtFiled.fetch_add(filed, std::memory_order_relaxed);
		}

		// An interior whose camera stands in a room bound: the main culler's state byte +0x138 is clear (restricted),
		// and with previs inactive the scene walk (1138818) then files one global node (Group::Add at +0xAF2, that
		// branch's only add, return +0xAF7) and, through the portal path, only the rooms the portals show. Its
		// unrestricted branch (byte set; the override root, global 127974, is the interior cell's 3D) also files the
		// cell's child 3 (the portal graph's unbound node: statics and precombined chunks outside every room bound),
		// child 9 (precombined art) and child 7's children without a room into group 0. Previs ignores rooms and
		// draws them all, so in CBRO frames whatever lay outside every room bound was never drawn (2026-10-07: stairs
		// and a wall in a Fallout London interior; CBRO hid nothing, main view 601 registrations a frame against
		// previs' 802). So right after that add the three are filed into group 0 as the unrestricted branch files
		// them, node 9 also when AppCulled (as FileCulledCellArt), and CBRO judges them like any group-0 entry.
		// Every CBRO frame, culling or not. Main thread, inside the walk (GroupAddThunk).
		//
		// The rooms themselves (v1.77): the portal path files the camera's rooms (the entry's list, Group::Add at
		// +0xC28) and, through each of their portals that the culler passes, the node behind it with the portal's
		// clipped frustum (+0xD48). A room no portal shows is never filed, and Fallout London's portals miss rooms
		// that are in plain view (2026-10-07, same interior: the camera's list held 1 of 5 rooms, a wall was missing,
		// main view 227 registrations a frame against previs' ~470, CBRO hid 3). So every room not in the list goes
		// whole into DrawWorld's group 1 (main-only, as the engine's room groups and previs' list are), and the
		// portal path's add of a child of a room filed that way is left out (no object filed twice).
		std::uintptr_t             g_interiorRoot{ 0 };   // address of the global (127974)
		std::uintptr_t             g_mainCuller{ 0 };     // address of the global (865470)
		std::uintptr_t             g_niNodeRtti{ 0 };
		std::uintptr_t             g_frameCounter{ 0 };   // address of the engine's frame counter (734919)
		bool                       g_hiddenRooms{ false };
		constexpr std::uint64_t    kInteriorRootID = 127974;
		constexpr std::uint64_t    kMainCullerID = 865470;
		constexpr std::uint64_t    kNiNodeRttiID = 191219;
		constexpr std::uint64_t    kFrameCounterID = 734919;
		constexpr std::size_t      kRestrictedNodeLoad = 0xADE;  // mov rdx,[rip+disp32] (the global node it adds)
		constexpr std::size_t      kRestrictedAddCall = 0xAF2;   // call Group::Add
		constexpr std::size_t      kListedRoomAddCall = 0xC28;   // call Group::Add: a child of one of the camera's rooms
		constexpr std::size_t      kPortalAddCall = 0xD48;       // call Group::Add: a child of the node behind a portal
		constexpr std::size_t      kCullerStateOffset = 0x150;   // BSCullingProcess -> BSPortalGraphEntry
		constexpr std::size_t      kStateRoomListOffset = 0x18;  // entry: the camera's rooms (pointer array)
		constexpr std::size_t      kStateRoomCountOffset = 0x28; // ... their count (u32)
		constexpr std::size_t      kRoomSlot = 0x218 / 8;        // BSMultiBoundNode: its room (null = none)
		std::array<const RE::NiNode*, 64> g_filedRooms{};        // rooms filed whole this frame (main thread)
		std::size_t                g_filedRoomCount{ 0 };
		std::uint32_t              g_filedRoomsFrame{ 0 };
		std::atomic<std::uint64_t> g_hiddenRoomsFiled{ 0 };
		std::atomic<std::uint64_t> g_hiddenRoomObjects{ 0 };
		std::atomic<std::uint64_t> g_portalAddsSkipped{ 0 };
		std::atomic<std::uint64_t> g_restrictedFrames{ 0 };
		std::atomic<std::uint64_t> g_unboundFiled{ 0 };
		std::atomic<std::uint64_t> g_artFiled{ 0 };
		std::atomic<std::uint64_t> g_artCulled{ 0 };
		std::atomic<std::uint64_t> g_roomlessFiled{ 0 };
		std::atomic<std::uint64_t> g_roomsSeen{ 0 };
		std::atomic<std::uint64_t> g_roomsTotal{ 0 };

		void FileOutsideRooms() noexcept
		{
			const auto root = *reinterpret_cast<RE::NiAVObject* const*>(g_interiorRoot);
			const auto cell = root ? root->IsNode() : nullptr;
			if (!cell || (cell->GetFlags() & 1)) {
				return;
			}
			const auto group = reinterpret_cast<void*>(g_mainGroup);
			const auto child = [&](std::uint16_t a_index) -> RE::NiNode* {
				const auto object = cell->children.size() > a_index ? cell->children[a_index].get() : nullptr;
				return object ? object->IsNode() : nullptr;
			};
			const auto file = [&](RE::NiAVObject* a_object) { AddDirect(group, a_object, &a_object->worldBound, 0); };
			std::uint64_t unbound = 0, art = 0, roomless = 0, rooms = 0;
			if (const auto node = child(3); node && !(node->GetFlags() & 1)) {
				for (auto& entry : node->children) {
					const auto object = entry.get();
					if (!object || (object->GetFlags() & 1)) {
						continue;
					}
					if (reinterpret_cast<std::uintptr_t>(object->GetRTTI()) == g_niNodeRtti) {  // (exact NiNodes are expanded)
						for (auto& inner : static_cast<RE::NiNode*>(object)->children) {
							if (inner) {
								file(inner.get());
								++unbound;
							}
						}
					} else {
						file(object);
						++unbound;
					}
				}
			}
			if (const auto node = child(9)) {
				g_artCulled.fetch_add(node->GetFlags() & 1, std::memory_order_relaxed);
				for (auto& entry : node->children) {
					if (const auto object = entry.get(); object && !(object->GetFlags() & 1)) {
						file(object);
						++art;
					}
				}
			}
			// The camera's rooms (the entry's list) are the portal path's; every other room goes whole into group 1.
			const auto culler = *reinterpret_cast<const std::uintptr_t*>(g_mainCuller);
			const auto state = culler ? ReadAt<std::uintptr_t>(reinterpret_cast<const void*>(culler), kCullerStateOffset) : 0;
			const auto listed = state ? ReadAt<void* const*>(reinterpret_cast<const void*>(state), kStateRoomListOffset) : nullptr;
			const auto listedCount = listed ? ReadAt<std::uint32_t>(reinterpret_cast<const void*>(state), kStateRoomCountOffset) : 0u;
			const auto isListed = [&](const void* a_room) { return std::find(listed, listed + listedCount, a_room) != listed + listedCount; };
			std::uint64_t hiddenRooms = 0, roomObjects = 0;
			g_filedRoomCount = 0;
			g_filedRoomsFrame = *reinterpret_cast<const std::uint32_t*>(g_frameCounter);
			if (const auto node = child(7); node && !(node->GetFlags() & 1)) {
				for (auto& entry : node->children) {
					const auto object = entry.get();
					if (!object) {
						continue;
					}
					const auto multiBound = object->IsMultiBoundNode();
					using RoomFn = RE::NiAVObject* (*)(void*);
					const auto room = multiBound ? (*reinterpret_cast<RoomFn* const*>(multiBound))[kRoomSlot](multiBound) : nullptr;
					if (!room) {
						if (!(object->GetFlags() & 1)) {
							file(object);
							++roomless;
						}
						continue;
					}
					++rooms;
					const auto roomNode = room->IsNode();
					if (!g_hiddenRooms || isListed(room) || !roomNode || (roomNode->GetFlags() & 1) || g_filedRoomCount >= g_filedRooms.size()) {
						continue;
					}
					g_filedRooms[g_filedRoomCount++] = roomNode;
					++hiddenRooms;
					for (auto& inner : roomNode->children) {
						if (const auto item = inner.get(); item && !(item->GetFlags() & 1)) {
							AddDirect(reinterpret_cast<void*>(g_group1), item, &item->worldBound, 0);
							++roomObjects;
						}
					}
				}
			}
			g_roomsSeen.fetch_add(listedCount, std::memory_order_relaxed);
			g_roomsTotal.fetch_add(rooms, std::memory_order_relaxed);
			g_hiddenRoomsFiled.fetch_add(hiddenRooms, std::memory_order_relaxed);
			g_hiddenRoomObjects.fetch_add(roomObjects, std::memory_order_relaxed);
			g_restrictedFrames.fetch_add(1, std::memory_order_relaxed);
			g_unboundFiled.fetch_add(unbound, std::memory_order_relaxed);
			g_artFiled.fetch_add(art, std::memory_order_relaxed);
			g_roomlessFiled.fetch_add(roomless, std::memory_order_relaxed);
		}

		// The portal path is adding a_object: true when its parent is a room FileOutsideRooms filed whole this frame.
		bool InFiledRoom(const RE::NiAVObject* a_object) noexcept
		{
			if (!a_object || g_filedRoomCount == 0 || g_filedRoomsFrame != *reinterpret_cast<const std::uint32_t*>(g_frameCounter)) {
				return false;
			}
			const auto parent = a_object->parent;
			const auto end = g_filedRooms.begin() + static_cast<std::ptrdiff_t>(g_filedRoomCount);
			if (std::find(g_filedRooms.begin(), end, parent) == end) {
				return false;
			}
			g_portalAddsSkipped.fetch_add(1, std::memory_order_relaxed);
			return true;
		}

		// Checks the restricted branch's adds are where 1138818 had them (a global node's load, then calls to
		// Group::Add) before the Group::Add hook acts on their return addresses.
		bool ResolveRestrictedAdd() noexcept
		{
			const auto walk = CBRO::Engine::OG(kSceneWalkID).address();
			const auto load = walk + kRestrictedNodeLoad;
			const auto bytes = [](std::uintptr_t a_at) { return reinterpret_cast<const std::uint8_t*>(a_at); };
			const auto callsGroupAdd = [&](std::size_t a_offset) {
				const auto call = walk + a_offset;
				return bytes(call)[0] == 0xE8 && call + 5 + *reinterpret_cast<const std::int32_t*>(call + 1) == CBRO::Engine::OG(kGroupAddID).address();
			};
			if (bytes(load)[0] != 0x48 || bytes(load)[1] != 0x8B || bytes(load)[2] != 0x15 || !callsGroupAdd(kRestrictedAddCall)) {
				logger::warn("cullgroups: the scene walk's in-a-room add isn't at {} as expected: objects outside an interior's room bounds stay undrawn in CBRO frames", Util::DescribeCodeAddress(walk + kRestrictedAddCall));
				return false;
			}
			g_restrictedAddReturn = walk + kRestrictedAddCall + 5;
			g_interiorRoot = CBRO::Engine::OG(kInteriorRootID).address();
			g_mainCuller = CBRO::Engine::OG(kMainCullerID).address();
			g_niNodeRtti = CBRO::Engine::OG(kNiNodeRttiID).address();
			g_frameCounter = CBRO::Engine::OG(kFrameCounterID).address();
			g_hiddenRooms = callsGroupAdd(kListedRoomAddCall) && callsGroupAdd(kPortalAddCall);
			if (g_hiddenRooms) {
				g_portalAddReturn = walk + kPortalAddCall + 5;
			} else {
				logger::warn("cullgroups: the scene walk's portal adds aren't at {} / {} as expected: interior rooms no portal shows stay undrawn in CBRO frames",
					Util::DescribeCodeAddress(walk + kListedRoomAddCall), Util::DescribeCodeAddress(walk + kPortalAddCall));
			}
			return true;
		}

		// From a prune site's stub, with the node whose children are about to be added one by one (a cell's child node
		// 3 or 9 at the scene walk, or a root registered with DrawWorld's cull: index kRootIndex); main thread, inside
		// DrawWorld's cull. True leaves the node out (none of its objects is filed with any view).
		bool CellNodeThunk(RE::NiAVObject* a_node, std::uint32_t a_index)
		{
			if (!a_node) {
				return false;
			}
			if (a_index == 3 && g_cellArtNode9) {
				FileCulledCellArt(a_node);
			}
			if (!g_cellNodePruning || !g_mainCullActive.load(std::memory_order_relaxed)) {
				return false;
			}
			const auto filter = g_cellNodeFilter.load(std::memory_order_relaxed);
			return filter && filter(a_node, a_index);
		}

		// Builds a site's stub (in the trampoline: code only) and patches the site. The stub calls CellNodeThunk with
		// the node and index the loop keeps in callee-saved registers, then either continues with the next node (skip)
		// or restores the clobbered registers, repeats the replaced instruction and resumes. rsp is 16-byte aligned at
		// every site (both functions: entry 8 mod 16, three pushes, an even sub), so `sub rsp,20h` gives the call its
		// home space and the alignment the ABI wants.
		bool InstallPruneSite(const PruneSiteSpec& a_site, Util::SwitchableHook& a_hook)
		{
			const auto base = CBRO::Engine::OG(a_site.functionID).address();
			const auto patch = base + a_site.patchOffset;
			const auto resume = base + a_site.resumeOffset;
			const auto skip = base + a_site.skipOffset;
			const auto whole = a_site.wholeOffset ? base + a_site.wholeOffset : 0;
			const auto matches = [](std::uintptr_t a_at, std::span<const std::uint8_t> a_bytes) {
				return std::memcmp(reinterpret_cast<const void*>(a_at), a_bytes.data(), a_bytes.size()) == 0;
			};
			if (!matches(resume, a_site.resumeBytes) || !matches(skip, a_site.skipBytes) || (whole && !matches(whole, kRootWhole))) {
				logger::error("hook {}: {} isn't laid out as expected around the loop; not patched", a_site.name, Util::DescribeCodeAddress(patch));
				return false;
			}

			std::vector<std::uint8_t> code;
			const auto emit = [&](std::initializer_list<std::uint8_t> a_bytes) { code.insert(code.end(), a_bytes); };
			const auto emit64 = [&](std::uint64_t a_value) {
				for (int i = 0; i < 8; ++i) {
					code.push_back(static_cast<std::uint8_t>(a_value >> (8 * i)));
				}
			};
			const auto emitJumpTo = [&](std::uintptr_t a_target) {
				emit({ 0xFF, 0x25, 0x00, 0x00, 0x00, 0x00 });  // jmp [rip+0]
				emit64(a_target);
			};
			const auto emitCall = [&](std::initializer_list<std::uint8_t> a_loadArgs) {
				emit({ 0x48, 0x83, 0xEC, 0x20 });  // sub rsp,20h
				emit(a_loadArgs);                  // rcx = the node, edx = the index
				emit({ 0x48, 0xB8 });              // mov rax,imm64
				emit64(Util::FnAddr(&CellNodeThunk));
				emit({ 0xFF, 0xD0 });              // call rax
				emit({ 0x48, 0x83, 0xC4, 0x20 });  // add rsp,20h
				emit({ 0x84, 0xC0 });              // test al,al
				emit({ 0x75, 0x00 });              // jne skip (patched below)
				return code.size() - 1;
			};
			std::size_t jzWhole = 0;
			std::size_t jneResume = 0;
			std::size_t jneSkip = 0;
			const bool  container = a_site.kind == PruneSite::kWalkContainer || a_site.kind == PruneSite::kWalkOverrideContainer;
			const bool  rdi = a_site.kind == PruneSite::kWalkContainer;  // which register holds NiNode's RTTI
			switch (a_site.kind) {
			case PruneSite::kWalkExterior:
				jneSkip = emitCall({ 0x49, 0x8B, 0xCE, 0x41, 0x8B, 0xD7 });  // mov rcx,r14; mov edx,r15d
				emit({ 0x49, 0x8B, 0xC6 });                                    // mov rax,r14        rax and dx as the walk left them,
				emit({ 0x33, 0xD2 });                                          // xor edx,edx
				emit({ 0x66, 0x3B, 0x90, 0x32, 0x01, 0x00, 0x00 });            // cmp dx,word [rax+132h]   then the replaced instruction
				break;
			case PruneSite::kWalkOverride:
				jneSkip = emitCall({ 0x49, 0x8B, 0xCF, 0x41, 0x8B, 0xD4 });  // mov rcx,r15; mov edx,r12d
				emit({ 0x49, 0x8B, 0xC7 });                                    // mov rax,r15
				emit({ 0x33, 0xD2 });
				emit({ 0x66, 0x3B, 0x90, 0x32, 0x01, 0x00, 0x00 });
				break;
			case PruneSite::kDrawWorldRoot:
				emit({ 0x48, 0x85, 0xC0 });  // test rax,rax        the root as a node, or null
				emit({ 0x74, 0x00 });        // jz whole            (not a node: the engine adds it whole)
				jzWhole = code.size() - 1;
				jneSkip = emitCall({ 0x48, 0x8B, 0xCE, 0xBA, 0xFF, 0xFF, 0xFF, 0xFF });  // mov rcx,rsi; mov edx,kRootIndex
				emit({ 0x48, 0x8B, 0xC6 });                                                // mov rax,rsi (the test and jump were done above)
				break;
			case PruneSite::kWalkContainer:
			case PruneSite::kWalkOverrideContainer:
				emit({ 0x48, 0x8B, 0x06 });                     // mov rax,[rsi]
				emit({ 0x48, 0x8B, 0xCE });                     // mov rcx,rsi
				emit({ 0xFF, 0x50, 0x10 });                     // call [rax+10h]     NiObject::GetRTTI (the replaced instructions)
				emit(rdi ? std::initializer_list<std::uint8_t>{ 0x48, 0x3B, 0xC7 } : std::initializer_list<std::uint8_t>{ 0x48, 0x3B, 0xC3 });  // cmp rax,rdi|rbx
				emit({ 0x75, 0x00 });                           // jne resume         not an exact NiNode: rax intact, the walk's own compare repeats
				jneResume = code.size() - 1;
				jneSkip = emitCall({ 0x48, 0x8B, 0xCE, 0xBA, 0xFE, 0xFF, 0xFF, 0xFF });  // mov rcx,rsi; mov edx,kContainerIndex
				emit(rdi ? std::initializer_list<std::uint8_t>{ 0x48, 0x8B, 0xC7 } : std::initializer_list<std::uint8_t>{ 0x48, 0x8B, 0xC3 });  // mov rax,rdi|rbx (so the walk's compare passes)
				break;
			}
			const auto resumeLabel = code.size();
			emitJumpTo(resume);
			const auto skipLabel = code.size();
			emitJumpTo(skip);
			code[jneSkip] = static_cast<std::uint8_t>(skipLabel - (jneSkip + 1));
			if (container) {
				code[jneResume] = static_cast<std::uint8_t>(resumeLabel - (jneResume + 1));
			}
			if (whole) {
				const auto wholeLabel = code.size();
				emitJumpTo(whole);
				code[jzWhole] = static_cast<std::uint8_t>(wholeLabel - (jzWhole + 1));
			}

			const auto stub = Util::WriteStub(code);
			if (!stub) {
				logger::error("hook {}: no trampoline room for its stub; not patched", a_site.name);
				return false;
			}
			return Util::PatchJumpSwitchable(a_hook, patch, a_site.patchBytes, stub, a_site.name);
		}

		bool InstallCellNodePrune()
		{
			bool any = false;
			for (std::size_t i = 0; i < kPruneSites.size(); ++i) {
				any = InstallPruneSite(kPruneSites[i], g_cellNodeHooks[i]) || any;
			}
			return any;
		}

		// Whether the `call rel32` at a_site still targets a_target (the site was readable when resolved). CBRO's own
		// timing wrapper on the sun-cascades site (Hooks/RenderStages, a pass-through chained to the engine's function)
		// counts as the engine's.
		bool CallsTo(std::uintptr_t a_site, std::uintptr_t a_target) noexcept
		{
			const auto* code = reinterpret_cast<const std::uint8_t*>(a_site);
			if (!a_site || code[0] != 0xE8) {
				return false;
			}
			const auto current = a_site + 5 + *reinterpret_cast<const std::int32_t*>(code + 1);
			if (current == a_target) {
				return true;
			}
			// (write_call<5> makes the site call a jump stub in the trampoline that lands on the thunk; v1.28 compared the
			// stub with the thunk itself, so the path always read as changed and the sun culling never ran)
			using RenderStages::Stage;
			const auto landing = Util::FollowJumps(current);
			const auto thunk = RenderStages::ThunkOf(Stage::kSunCascades);
			if (thunk != 0 && landing == thunk && RenderStages::OriginalOf(Stage::kSunCascades) == a_target) {
				return true;
			}
			// Likewise CBRO's cascade-window wrapper on 1108521+0xE49 (Hooks/PrevisFeed), chained to the engine's cull.
			const auto window = PrevisFeed::CascadeCullThunkAddress();
			return window != 0 && landing == window && PrevisFeed::CascadeCullPrevious() == a_target;
		}

		bool Intact(const std::array<CallSite, 2>& a_path) noexcept
		{
			return CallsTo(a_path[0].site, a_path[0].target) && CallsTo(a_path[1].site, a_path[1].target);
		}

		// A plugin that wraps a site after CBRO did fails CallsTo, though the engine's function still runs beneath it:
		// CBRO's own hook under that wrapper is what calls it. Shader Replacer v0.1 wraps both unbatched sites; read as
		// "path changed" on every frame, that kept sun-shadow trimming off for the whole session (2026-10-08: the
		// cascades registered 12.3k objects a frame against previs's 2.1k). So an unbatched site also counts as the
		// engine's when CBRO's hook there ran since the previous check (the last frame's cascades: they run after the
		// cull) and chains to the engine's function. CBRO's cascade wrapper sits inside the engine's stage function, so
		// its call proves that function ran too, in either load order. Without directional shadows the stage never
		// reaches the cascade cull's call. The batched path has no CBRO hook to give such proof.
		std::uint64_t g_stageEntriesSeen{ 0 };
		std::uint64_t g_cascadeCallsSeen{ 0 };

		bool UnbatchedIntact(bool a_dirShadows) noexcept
		{
			using RenderStages::Stage;
			const auto entries = RenderStages::Entries(Stage::kSunCascades);
			const auto calls = PrevisFeed::CascadeCullCalls();
			const bool stageRan = entries != g_stageEntriesSeen && RenderStages::OriginalOf(Stage::kSunCascades) == g_unbatchedPath[0].target;
			const bool cullRan = calls != g_cascadeCallsSeen && PrevisFeed::CascadeCullPrevious() == g_unbatchedPath[1].target;
			g_stageEntriesSeen = entries;
			g_cascadeCallsSeen = calls;
			return (stageRan || cullRan || CallsTo(g_unbatchedPath[0].site, g_unbatchedPath[0].target)) &&
			       (!a_dirShadows || cullRan || CallsTo(g_unbatchedPath[1].site, g_unbatchedPath[1].target));
		}

		// Once per session, when the path stays unverified for a while (a single frame can lack the proof, e.g. the
		// first one with directional shadows on): which call isn't the engine's.
		std::uint32_t           g_pathFailures{ 0 };
		bool                    g_pathWarned{ false };
		constexpr std::uint32_t kPathWarnFrames = 60;

		void WarnPathChanged(bool a_batched)
		{
			const auto& path = a_batched ? g_batchedPath : g_unbatchedPath;
			const auto  describe = [](const CallSite& a_site) {
				return std::format(
					"{} -> {} {}", Util::DescribeCodeAddress(a_site.site), Util::DescribeCodeAddress(Util::ReadCall5Target(a_site.site)),
					CallsTo(a_site.site, a_site.target) ? "(engine's)" : "(CHANGED)");
			};
			logger::warn(
				"cullgroups: sun cascades: the {} path is not verified as the engine's for {} culled frames in a row: {}, {}; light update slot {}. "
				"Sun-shadow trimming stays off while it lasts (group 0 keeps every object for the cascades)",
				a_batched ? "batched" : "unbatched", kPathWarnFrames, describe(path[0]), describe(path[1]),
				*reinterpret_cast<const std::uintptr_t*>(g_dirLightUpdateSlot) == g_dirLightUpdate ? "(engine's)" : "(CHANGED)");
		}

		// The sun's two cascade paths (see the constants), located and logged once; ReadSun re-reads them.
		void ResolveSunPath()
		{
			const auto preUI = CBRO::Engine::OG(kRenderPreUIID).address();
			const auto batchedCaller = CBRO::Engine::OG(kBatchedCallerID).address();
			const auto unbatchedCaller = CBRO::Engine::OG(kUnbatchedCallerID).address();
			g_batchedPath = { CallSite{ preUI + kBatchedCallOffset, batchedCaller }, CallSite{ batchedCaller + kBatchedSetupCallOffset, CBRO::Engine::OG(kBatchedSetupID).address() } };
			g_unbatchedPath = { CallSite{ preUI + kUnbatchedCallOffset, unbatchedCaller }, CallSite{ unbatchedCaller + kUnbatchedCullCallOffset, CBRO::Engine::OG(kCascadeCullID).address() } };
			g_dirLightUpdateSlot = g_dirLightVtable + sizeof(std::uintptr_t) * kDirLightUpdateSlot;
			g_dirLightUpdate = CBRO::Engine::OG(kDirLightUpdateID).address();
			g_unbatchedSceneNode = CBRO::Engine::OG(kUnbatchedSceneNodeID).address();
			const auto describe = [](const std::array<CallSite, 2>& a_path) {
				return std::format(
					"{} -> {} {}, {} -> {} {}",
					Util::DescribeCodeAddress(a_path[0].site), Util::DescribeCodeAddress(Util::ReadCall5Target(a_path[0].site)), CallsTo(a_path[0].site, a_path[0].target) ? "(engine's)" : "(CHANGED)",
					Util::DescribeCodeAddress(a_path[1].site), Util::DescribeCodeAddress(Util::ReadCall5Target(a_path[1].site)), CallsTo(a_path[1].site, a_path[1].target) ? "(engine's)" : "(CHANGED)");
			};
			logger::info("cullgroups: sun cascades, batched path: {}", describe(g_batchedPath));
			logger::info("cullgroups: sun cascades, unbatched path: {}", describe(g_unbatchedPath));
			logger::info(
				"cullgroups: sun light update slot {} {}",
				Util::DescribeCodeAddress(*reinterpret_cast<const std::uintptr_t*>(g_dirLightUpdateSlot)),
				*reinterpret_cast<const std::uintptr_t*>(g_dirLightUpdateSlot) == g_dirLightUpdate ? "(engine's)" : "(CHANGED)");
		}
	}

	void SetFilter(BlockFilter a_filter)
	{
		g_filter.store(a_filter);
	}

	void SetInstanceFilter(InstanceFilter a_filter)
	{
		g_instanceFilter.store(a_filter);
	}

	void SetGroupFilter(GroupFilter a_filter)
	{
		g_groupFilter.store(a_filter);
	}

	void SetMainViewFilter(MainViewFilter a_filter)
	{
		g_mainViewFilter.store(a_filter);
	}

	void SetMainRegistrationObserver(MainRegistrationObserver a_observer)
	{
		g_mainRegistrationObserver.store(a_observer);
	}

	void SetDroppedAccumulator(const void* a_accumulator) noexcept
	{
		g_droppedAccumulator.store(a_accumulator, std::memory_order_release);
	}

	void SetCountedAccumulator(const void* a_accumulator) noexcept
	{
		g_countedAccumulator.store(a_accumulator, std::memory_order_release);
	}

	std::uint64_t ReadCountedRegistrations() noexcept
	{
		return g_countedRegistrations.load(std::memory_order_relaxed);
	}

	void SetFilteredAccumulator(const void* a_accumulator, CasterFilter a_filter) noexcept
	{
		// (the filter first: a reader that sees the accumulator sees its filter, or none)
		g_casterFilter.store(a_filter, std::memory_order_release);
		g_filteredAccumulator.store(a_accumulator, std::memory_order_release);
	}

	void SetFarCascade(const void* a_accumulator, CasterFilter a_filter) noexcept
	{
		g_farFilter.store(a_filter, std::memory_order_release);
		g_farAccumulator.store(a_accumulator, std::memory_order_release);
	}

	std::uint64_t ReadDroppedRegistrations() noexcept
	{
		std::uint64_t total = 0;
		for (const auto& slot : g_callCounts) {
			total += slot.dropped.load(std::memory_order_relaxed);
		}
		return total;
	}

	std::uint64_t TakeDroppedRegistrations() noexcept
	{
		static std::uint64_t reported = 0;
		const auto           total = ReadDroppedRegistrations();
		const auto           since = total - reported;
		reported = total;
		return since;
	}

	void SetCellNodeFilter(CellNodeFilter a_filter)
	{
		g_cellNodeFilter.store(a_filter);
	}

	std::uint64_t TakeGroupAddsConsidered() noexcept
	{
		return g_groupAddsConsidered.exchange(0);
	}

	void SetRegistrationMode(bool a_cbro) noexcept
	{
		g_registrationMode.store(a_cbro ? 1 : 0, std::memory_order_relaxed);
	}

	std::string TakeRegistrationSites(bool a_cbro, double a_frames)
	{
		std::vector<std::pair<std::uint64_t, std::uintptr_t>> sites;
		for (auto& slot : g_registrationSites[a_cbro ? 1 : 0]) {
			const auto site = slot.site.load(std::memory_order_relaxed);
			const auto count = slot.count.exchange(0, std::memory_order_relaxed);
			if (site && count) {
				sites.emplace_back(count, site);
			}
		}
		std::ranges::sort(sites, std::greater{});
		std::string text;
		for (std::size_t i = 0; i < sites.size() && i < 14; ++i) {
			const auto& [count, site] = sites[i];
			const auto  mode = ReadRenderMode(site);
			const auto  name = RenderModeName(mode);
			text += std::format(
				"{}{:#x} [mode {}{}{}]{} {:.0f}/frame", text.empty() ? "" : " | ", site, mode, name.empty() ? "" : " ", name,
				site == *reinterpret_cast<const std::uintptr_t*>(g_mainAccumulator) ? " (main)" : "", static_cast<double>(count) * 16.0 / std::max(1.0, a_frames));
		}
		return text.empty() ? "none" : text;
	}

	std::string TakeGroupAddSites(double a_frames)
	{
		std::vector<std::pair<std::uint64_t, std::uintptr_t>> sites;
		for (auto& slot : g_groupAddSites) {
			const auto site = slot.site.load(std::memory_order_relaxed);
			const auto count = slot.count.exchange(0, std::memory_order_relaxed);
			if (site && count) {
				sites.emplace_back(count, site);
			}
		}
		std::ranges::sort(sites, std::greater{});
		std::string text;
		for (std::size_t i = 0; i < sites.size() && i < 8; ++i) {
			const auto& [count, site] = sites[i];
			text += std::format("{}{} {:.0f}/frame", text.empty() ? "" : " | ", Util::DescribeCodeAddress(site), static_cast<double>(count) * 64.0 / std::max(1.0, a_frames));
		}
		return text.empty() ? "none" : text;
	}

	void SetBlockObserver(BlockObserver a_observer)
	{
		g_blockObserver.store(a_observer);
	}

	void SetGroupObserver(GroupObserver a_observer)
	{
		g_groupObserver.store(a_observer);
	}

	bool Install()
	{
		if (g_installed) {
			return g_ready;
		}
		g_installed = true;

		g_mainGroup = CBRO::Engine::OG(kMainGroupID).address();
		g_triShapeVtable = RE::VTABLE::BSTriShape[0].address();
		g_skyShaderVtable = RE::VTABLE::BSSkyShaderProperty[0].address();
		g_group1 = CBRO::Engine::OG(kGroup1ID).address();
		g_group2 = CBRO::Engine::OG(kGroup2ID).address();
		g_groupArray = CBRO::Engine::OG(kGroupArrayID).address();
		g_mainAccumulator = CBRO::Engine::OG(kMainAccumulatorID).address();
		g_groupsEnabled = CBRO::Engine::OG(kGroupsEnabledID).address();
		g_dirShadows = CBRO::Engine::OG(kDirShadowsID).address();
		g_shadowSceneNode = CBRO::Engine::OG(kShadowSceneNodeID).address();
		g_dirLightVtable = CBRO::Engine::OG(kDirLightVtableID).address();
		g_shadowRange = CBRO::Engine::OG(kShadowRangeID).address();
		g_godrayCascades = CBRO::Engine::OG(kGodrayCascadesID).address();
		ResolveSunPath();
		g_blockAddOriginal = Util::DetourSwitchable(g_blockAddHook, CBRO::Engine::OG(kBlockAddID).address(), Util::FnAddr(&BlockAddThunk), kBlockAddPrologue, "cullgroups:Block::Add");
		g_groupAddOriginal = Util::DetourSwitchable(g_groupAddHook, CBRO::Engine::OG(kGroupAddID).address(), Util::FnAddr(&GroupAddThunk), kGroupAddPrologue, "cullgroups:Group::Add");
		g_childPushOriginal = Util::DetourSwitchable(g_childPushHook, CBRO::Engine::OG(kChildPushID).address(), Util::FnAddr(&ChildPushThunk), kChildPushPrologue, "cullgroups:ChildPush");
		g_registerOriginal = Util::WriteVFuncSwitchable(g_registerHook, CBRO::Engine::OG(kAccumulatorVtableID).address(), kRegisterObjectSlot, Util::FnAddr(&RegisterObjectThunk), "cullgroups:BSShaderAccumulator::RegisterObject");
		// The walk's node loops carry two things: cell-node pruning and the AppCulled node-9 art (FileCulledCellArt).
		g_cellNodePruning = Settings::Get().cellNodePruning;
		g_cellArtNode9 = Settings::Get().cellArtNode9;
		if (g_cellNodePruning || g_cellArtNode9) {
			if (!InstallCellNodePrune()) {
				logger::warn("cullgroups: the scene walk's node loops couldn't be hooked: no cell-node pruning, and a cell's AppCulled node 9 (precombined art previs draws) stays undrawn in CBRO frames");
				g_cellNodePruning = false;
				g_cellArtNode9 = false;
			}
		}
		if (!Settings::Get().cellNodePruning) {
			logger::info("cullgroups: cell-node pruning off ([Occlusion] bCellNodePruning=0)");
		}
		if (!Settings::Get().cellArtNode9) {
			logger::info("cullgroups: a cell's AppCulled node 9 left to the engine ([Occlusion] bCellArtNode9=0): its precombined art is undrawn in CBRO frames");
		}
		g_interiorOutsideRooms = Settings::Get().interiorOutsideRooms && g_groupAddOriginal && ResolveRestrictedAdd();
		if (!Settings::Get().interiorOutsideRooms) {
			logger::info("cullgroups: an interior's objects outside its room bounds left to the engine ([Occlusion] bInteriorOutsideRooms=0): undrawn in CBRO frames while the camera is in a room");
		}

		// Without a group hook, Block::Add can't tell which group it serves: those entries count as shared
		// (dropped from the main view only), and nothing is left out early. Safe, but less culling.
		if (!g_groupAddOriginal || !g_childPushOriginal) {
			logger::warn(
				"cullgroups: Group::Add hooked {}, ChildPush hooked {}: entries without a known group are dropped from the main view only",
				g_groupAddOriginal != 0, g_childPushOriginal != 0);
		}
		// Without the accumulator hook a shared entry can't be dropped from the main view alone: cull nothing
		// rather than cut sun shadows (previs keeps the job).
		g_ready = g_blockAddOriginal != 0 && g_registerOriginal != 0;
		if (!g_ready) {
			logger::error(
				"cullgroups: Block::Add hooked {}, main-view registration hooked {}: occlusion culling can't run safely",
				g_blockAddOriginal != 0, g_registerOriginal != 0);
		}
		return g_ready;
	}

	void SetMainCullActive(bool a_active) noexcept
	{
		g_mainCullActive.store(a_active, std::memory_order_release);
	}

	bool SetHooksIn(bool a_in)
	{
		// The accumulator hook (RegisterObject) stays in either way: it filters only while DrawWorld culls in CBRO
		// mode and otherwise just counts, so the two modes' registrations (what each view draws) can be compared.
		bool ok = true;
		const std::array<Util::SwitchableHook*, 8> hooks{
			&g_blockAddHook, &g_groupAddHook, &g_childPushHook,
			&g_cellNodeHooks[0], &g_cellNodeHooks[1], &g_cellNodeHooks[2], &g_cellNodeHooks[3], &g_cellNodeHooks[4]
		};
		for (auto* hook : hooks) {
			if (!hook->address) {
				continue;  // never installed: nothing to switch
			}
			if (!Util::SetHook(*hook, a_in)) {
				ok = false;
				logger::warn("hook {}: couldn't be {} (another plugin changed that spot since); left {}", hook->name, a_in ? "put back" : "taken out", hook->in ? "in" : "out");
			}
		}
		return ok;
	}

	bool HooksIn() noexcept
	{
		// Culling needs Block::Add and the main-view registration hook in place (see Install).
		return g_blockAddHook.in && g_registerHook.in;
	}

	EngineFlags ReadEngineFlags() noexcept
	{
		if (!g_groupsEnabled || !g_dirShadows) {
			return {};
		}
		return { *reinterpret_cast<const std::uint8_t*>(g_groupsEnabled) != 0, *reinterpret_cast<const std::uint8_t*>(g_dirShadows) != 0 };
	}

	HookCalls ReadHookCalls() noexcept
	{
		HookCalls total{};
		for (const auto& slot : g_callCounts) {
			total.blockAdds += slot.blockAdds.load(std::memory_order_relaxed);
			total.groupAdds += slot.groupAdds.load(std::memory_order_relaxed);
			total.childPushes += slot.childPushes.load(std::memory_order_relaxed);
			total.registrations += slot.registrations.load(std::memory_order_relaxed);
		}
		return total;
	}

	HookCalls TakeHookCalls() noexcept
	{
		const HookCalls total = ReadHookCalls();
		const HookCalls since{
			total.blockAdds - g_callsReported.blockAdds,
			total.groupAdds - g_callsReported.groupAdds,
			total.childPushes - g_callsReported.childPushes,
			total.registrations - g_callsReported.registrations
		};
		g_callsReported = total;
		return since;
	}

	std::uint64_t TakeForcedCleared() noexcept
	{
		return g_forcedCleared.exchange(0);
	}

	SkyCounts TakeSkyCounts() noexcept
	{
		return { g_skyForced.exchange(0), g_skyKept.exchange(0) };
	}

	CellArtCounts TakeCellArtCounts() noexcept
	{
		return { g_culledArtNodes.exchange(0), g_culledArtFiled.exchange(0) };
	}

	OutsideRoomsCounts TakeOutsideRoomsCounts() noexcept
	{
		return { g_restrictedFrames.exchange(0), g_unboundFiled.exchange(0), g_artFiled.exchange(0), g_artCulled.exchange(0),
			g_roomlessFiled.exchange(0), g_roomsSeen.exchange(0), g_roomsTotal.exchange(0), g_hiddenRoomsFiled.exchange(0),
			g_hiddenRoomObjects.exchange(0), g_portalAddsSkipped.exchange(0) };
	}

	bool IsSky(const RE::NiAVObject* a_object) noexcept
	{
		return IsSkyMesh(a_object);
	}

	bool AddDirect(void* a_group, RE::NiAVObject* a_object, const RE::NiBound* a_bound, std::uint32_t a_flags) noexcept
	{
		if (!g_groupAddOriginal || !a_group || !a_object || !a_bound) {
			return false;
		}
		GroupScope scope(a_group);  // (so the Block::Add that follows knows the group, as after a hooked Group::Add)
		reinterpret_cast<GroupAddFn>(g_groupAddOriginal)(a_group, a_object, a_bound, a_flags);
		return true;
	}

	std::uintptr_t MainOwner() noexcept
	{
		return g_mainGroup ? ReadAt<std::uintptr_t>(reinterpret_cast<const void*>(g_mainGroup), kGroupOwnerOffset) : 0;
	}

	bool ReadSun(SunSource& a_out) noexcept
	{
		if (!g_installed || !g_shadowSceneNode) {
			return false;
		}
		__try {
			// The path this frame takes (bCullingBatch picks it) still calls where the engine put it, and the
			// light's update (which sets the direction the cascades use) is the engine's own.
			a_out.groupsEnabled = *reinterpret_cast<const std::uint8_t*>(g_groupsEnabled) != 0;
			a_out.dirShadows = *reinterpret_cast<const std::uint8_t*>(g_dirShadows) != 0;
			a_out.pathIntact = (a_out.groupsEnabled ? Intact(g_batchedPath) : UnbatchedIntact(a_out.dirShadows)) &&
			                   *reinterpret_cast<const std::uintptr_t*>(g_dirLightUpdateSlot) == g_dirLightUpdate;
			g_pathFailures = a_out.pathIntact ? 0 : g_pathFailures + 1;
			if (g_pathFailures == kPathWarnFrames && !g_pathWarned) {
				g_pathWarned = true;
				WarnPathChanged(a_out.groupsEnabled);
			}
			a_out.range = *reinterpret_cast<const float*>(g_shadowRange);

			const auto node = *reinterpret_cast<const std::byte* const*>(a_out.groupsEnabled ? g_shadowSceneNode : g_unbatchedSceneNode);
			const auto light = node ? *reinterpret_cast<const std::byte* const*>(node + kSunLightOffset) : nullptr;
			if (!light || *reinterpret_cast<const std::uintptr_t*>(light) != g_dirLightVtable) {
				return false;
			}
			const auto camera = *reinterpret_cast<const std::byte* const*>(light + kShadowCameraOffset);
			if (!camera) {
				return false;
			}
			const auto row0 = reinterpret_cast<const float*>(camera + kLocalRotateOffset);
			const auto translate = reinterpret_cast<const float*>(camera + kLocalTranslateOffset);
			for (int i = 0; i < 3; ++i) {
				a_out.dir[i] = row0[i];
				a_out.cameraPos[i] = translate[i];
			}
			a_out.cascades = *reinterpret_cast<const std::uint32_t*>(light + kCascadeCountOffset);
			const auto records = *reinterpret_cast<const std::byte* const*>(light + kCascadesOffset);
			if (records && a_out.cascades > 0 && a_out.cascades <= kMaxCascades) {
				const auto last = records + kCascadeStride * (a_out.cascades - 1);
				a_out.farAccumulator = *reinterpret_cast<const void* const*>(last + kCascadeAccumulatorOffset);
				for (int i = 0; i < 2; ++i) {
					std::copy_n(reinterpret_cast<const float*>(last + kCascadeSlabOffsets[i]), 4, a_out.farSlab[i]);
				}
			}
			a_out.godrayCascades = g_godrayCascades ? *reinterpret_cast<const std::uint32_t*>(g_godrayCascades) : kMaxCascades;
			return true;
		} __except (EXCEPTION_EXECUTE_HANDLER) {
			return false;
		}
	}

	RE::NiBound ReadEntryBound(const void* a_block, std::uint32_t a_index) noexcept
	{
		const auto  p = EntryBoundBase(a_block, a_index);
		RE::NiBound bound{};
		bound.center.x = p[0];
		bound.center.y = p[4];
		bound.center.z = p[8];
		bound.fRadius = p[12];
		return bound;
	}

	void WriteEntryBound(void* a_block, std::uint32_t a_index, const RE::NiBound& a_bound) noexcept
	{
		const auto p = EntryBoundBase(a_block, a_index);
		p[0] = a_bound.center.x;
		p[4] = a_bound.center.y;
		p[8] = a_bound.center.z;
		p[12] = a_bound.fRadius;
	}

	bool IsMergeInstanced(RE::NiAVObject* a_object) noexcept
	{
		const auto geometry = a_object ? a_object->IsGeometry() : nullptr;
		return geometry && ReadAt<std::uint8_t>(geometry, kGeometryTypeOffset) == kMergeInstancedType;
	}

	std::uint32_t InstanceCount(const RE::NiAVObject* a_object) noexcept
	{
		return ReadAt<std::uintptr_t>(a_object, kInstancesOffset) ? ReadAt<std::uint32_t>(a_object, kInstanceCountOffset) : 0u;
	}

	const RE::NiBound* InstanceBound(const RE::NiAVObject* a_object, std::uint32_t a_index) noexcept
	{
		const auto instances = ReadAt<std::uintptr_t>(a_object, kInstancesOffset);
		return reinterpret_cast<const RE::NiBound*>(instances + a_index * kInstanceStride + kInstanceBoundOffset);
	}
}
