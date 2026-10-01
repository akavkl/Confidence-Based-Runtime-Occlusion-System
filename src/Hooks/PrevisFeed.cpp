#include "Hooks/PrevisFeed.h"

#include "Util/Hooking.h"

#include <cstring>

namespace CBRO::Hooks::PrevisFeed
{
	namespace
	{
		// ---- ids (OG 1.10.163; PREVIS-FEED-PLAN.md §2) -----------------------------------------------------------------
		constexpr std::uint64_t kSceneWalkID = 1138818;      // the walk; its main-feed call is at +0x16A
		constexpr std::size_t   kMainFeedSiteOffset = 0x16A;
		constexpr std::uint64_t kMainFeedID = 997287;        // Feed(group 2, group 1)
		constexpr std::uint64_t kCascadeCullID = 1390075;    // the sun's cascade cull; its sun-feed call is at +0x1B4
		constexpr std::size_t   kSunFeedSiteOffset = 0x1B4;
		constexpr std::uint64_t kSunFeedID = 1142692;        // Feed(stack group)
		constexpr std::uint64_t kDrawWorldCullID = 718911;
		constexpr std::uint64_t kSceneRootID = 1327069;      // NiNode* (the ShadowSceneNode)
		constexpr std::uint64_t kCullerID = 865470;          // BSGeometryListCullingProcess* (the walk's first argument)
		constexpr std::uint64_t kOverrideRootID = 127974;    // NiAVObject*: set = the walk takes its override-root path
		constexpr std::uint64_t kSkipChild2ByteID = 485145;  // byte: set = a cell's child 2 is not walked
		constexpr std::uint64_t kEnabledByteID = 493183;
		constexpr std::uint64_t kIniByteID = 1472203;
		constexpr std::uint64_t kSuspendedByteID = 718924;
		constexpr std::uint64_t kSetSuspendedID = 1263609;   // void(bool suspend, bool flush)
		constexpr std::uint64_t kNiNodeRttiID = 191219;      // NiNode's NiRTTI (the walk expands exact NiNodes only)

		// ---- the suspension windows (FO4-ENGINE-NOTES 5.5c) ----------------------------------------------------------
		// Render_PreUI (984743): +0x80 the previs query (if IsActive()); +0x157 `call 322222`; +0x15C `if (!IsActive())
		// call 102390` (the per-frame pre-cull helper the walk itself calls when previs is active); +0x16A DrawWorld's
		// cull (RenderStages' cull stage). Suspending inside the +0x157 call's wrapper, after the engine's routine ran,
		// gives the engine's exact previs-off flow from +0x15C to the cull's end: the helper, the walk's expansion and
		// DrawWorld's own root loop and group processing, all as with previs disabled. The unbatched shadow stage
		// (1108521) calls the cascade cull (1390075) at +0xE49 with group 0: suspended around that call alone, the
		// cascades read group 0 (CBRO's sun verdicts apply) while the lamps in the same stage, the previs query, the
		// third view and cell loads see previs active, as vanilla.
		constexpr std::uint64_t kRenderPreUIID = 984743;
		constexpr std::size_t   kPreCullSiteOffset = 0x157;
		constexpr std::uint64_t kPreCullTargetID = 322222;
		constexpr std::uint64_t kPreCullHelperID = 102390;    // void(): what Render_PreUI+0x165 calls when previs is inactive
		constexpr std::uint64_t kUnbatchedStageID = 1108521;
		constexpr std::size_t   kCascadeCullSiteOffset = 0xE49;  // 1108521: call 1390075 (kCascadeCullID)

		// ---- layouts (OG; RE::NiAVObject: worldBound +0xB0, flags +0x108; RE::NiNode::children is a NiTObjectArray at
		// +0x120: data +8, u16 count +0x12; NiObject vtable: 2 GetRTTI, 4 IsNode, 6 IsFadeNode) -------------------------
		constexpr std::size_t kWorldBound = 0xB0;
		constexpr std::size_t kFlags = 0x108;
		constexpr std::size_t kChildren = 0x128;
		constexpr std::size_t kChildCount = 0x132;
		constexpr std::size_t kSlotGetRTTI = 2;
		constexpr std::size_t kSlotIsNode = 4;
		constexpr std::size_t kSlotIsFadeNode = 6;
		constexpr std::size_t kCullerState = 0x150;    // culler -> state block
		constexpr std::size_t kExteriorByte = 0x138;   // state block: exterior path byte
		constexpr std::size_t kCullerRoots = 0x10;     // state block -> registered-roots block
		constexpr std::size_t kRootsArray = 0x58;      // registered-roots block: NiAVObject* array
		constexpr std::size_t kRootsCount = 0x68;      // ... u32 count
		constexpr std::size_t kGateA = 0x14C;          // scene root: u32
		constexpr std::size_t kGateB = 0x180;          // scene root: u16

		// The Group::Add return addresses of the replicated sites (the plan's §2.6), as function offsets.
		struct SiteOffset
		{
			std::uint64_t functionID;
			std::size_t   offset;
			Site          site;
		};
		constexpr std::array<SiteOffset, 12> kSiteOffsets{ {
			{ kSceneWalkID, 0x29B, Site::kR1 },
			{ kSceneWalkID, 0x2D0, Site::kR1 },
			{ kSceneWalkID, 0x305, Site::kR1 },
			{ kSceneWalkID, 0x33F, Site::kR1 },
			{ kSceneWalkID, 0x388, Site::kR1 },
			{ kSceneWalkID, 0x4AC, Site::kR2 },
			{ kSceneWalkID, 0x681, Site::kR3b },
			{ kSceneWalkID, 0x631, Site::kR3c },
			{ kSceneWalkID, 0x64A, Site::kR3d },
			{ kSceneWalkID, 0x59D, Site::kR3e },
			{ kDrawWorldCullID, 0x3AE, Site::kR4Whole },
			{ kDrawWorldCullID, 0x395, Site::kR4Child },
		} };

		struct FeedSite
		{
			std::uintptr_t address{ 0 };
			std::uintptr_t previous{ 0 };  // what the site called before CBRO (the engine's feed or another plugin's thunk)
			bool           ok{ false };
		};
		FeedSite g_mainSite;
		FeedSite g_sunSite;
		bool     g_available{ false };
		FeedSite g_preCullSite;   // Render_PreUI+0x157: the cull window opens after its call
		FeedSite g_cascadeSite;   // 1108521+0xE49: the cascade window is that call
		bool     g_windowsAvailable{ false };

		// The windows (main thread only: Render_PreUI's thread). `g_window` is the policy's word for the frame ("a CBRO
		// frame: suspend inside the windows"), set at the cull begin and read by the cascade wrapper later that frame
		// and by the pre-cull wrapper of the next frame (before that frame's cull begin). `g_held` is the cull window
		// being open; `g_savedByte` what the engine had in the suspended byte when it opened (restored on close, so an
		// engine-side suspension is left as found).
		std::atomic<bool> g_window{ false };
		bool              g_held{ false };
		std::uint8_t      g_savedByte{ 0 };
		std::uintptr_t    g_preCullHelper{ 0 };
		WindowCounts      g_windowCounts{};

		std::atomic<Owner>         g_owner{ Owner::kPrevis };
		std::atomic<MainFeedFn>    g_mainFn{ nullptr };
		std::atomic<SunFeedFn>     g_sunFn{ nullptr };
		std::atomic<std::uint64_t> g_calls[4]{};  // main previous, main CBRO, sun previous, sun CBRO (main thread only)

		std::uintptr_t g_sceneRoot{ 0 };
		std::uintptr_t g_culler{ 0 };
		std::uintptr_t g_overrideRoot{ 0 };
		std::uintptr_t g_skipChild2{ 0 };
		std::uintptr_t g_enabled{ 0 };
		std::uintptr_t g_ini{ 0 };
		std::uintptr_t g_suspended{ 0 };
		std::uintptr_t g_setSuspended{ 0 };
		std::uintptr_t g_niNodeRtti{ 0 };
		std::array<std::pair<std::uintptr_t, Site>, kSiteOffsets.size()> g_siteAddresses{};

		// The engine's feeds take their arguments in rcx/rdx; forwarding all four integer registers keeps whatever
		// else a previous target expects, and RAX comes back unchanged.
		using PassFn = std::uintptr_t (*)(std::uintptr_t, std::uintptr_t, std::uintptr_t, std::uintptr_t);

		std::uintptr_t MainFeedThunk(std::uintptr_t a1, std::uintptr_t a2, std::uintptr_t a3, std::uintptr_t a4)
		{
			if (g_owner.load(std::memory_order_relaxed) == Owner::kCBRO) {
				if (const auto fn = g_mainFn.load(std::memory_order_relaxed)) {
					g_calls[1].fetch_add(1, std::memory_order_relaxed);
					fn(reinterpret_cast<void*>(a1), reinterpret_cast<void*>(a2));
					return 0;
				}
			}
			g_calls[0].fetch_add(1, std::memory_order_relaxed);
			return reinterpret_cast<PassFn>(g_mainSite.previous)(a1, a2, a3, a4);
		}

		std::uintptr_t SunFeedThunk(std::uintptr_t a1, std::uintptr_t a2, std::uintptr_t a3, std::uintptr_t a4)
		{
			if (g_owner.load(std::memory_order_relaxed) == Owner::kCBRO) {
				if (const auto fn = g_sunFn.load(std::memory_order_relaxed)) {
					g_calls[3].fetch_add(1, std::memory_order_relaxed);
					fn(reinterpret_cast<void*>(a1));
					return 0;
				}
			}
			g_calls[2].fetch_add(1, std::memory_order_relaxed);
			return reinterpret_cast<PassFn>(g_sunSite.previous)(a1, a2, a3, a4);
		}

		// ---- the windows' wrappers -----------------------------------------------------------------------------------
		void WriteSuspended(bool a_suspend) noexcept
		{
			if (g_setSuspended) {
				using Fn = void (*)(bool, bool);
				reinterpret_cast<Fn>(g_setSuspended)(a_suspend, false);  // (a byte write: 5.5c)
			}
		}

		std::uint8_t ReadSuspendedByte() noexcept
		{
			__try {
				return g_suspended ? *reinterpret_cast<const std::uint8_t*>(g_suspended) : std::uint8_t{ 0 };
			} __except (EXCEPTION_EXECUTE_HANDLER) {
				return 0;
			}
		}

		// Opens the cull window: previs suspended until ReleaseWindow (the cull stage's end), the engine's own state
		// remembered. No-op while open.
		void OpenWindow() noexcept
		{
			if (g_held) {
				return;
			}
			g_savedByte = ReadSuspendedByte();
			if (!g_savedByte) {
				WriteSuspended(true);
			}
			g_held = true;
		}

		// Render_PreUI+0x157: the engine's routine first (previs active for it, as vanilla), then, in a CBRO frame, the
		// cull window opens so that +0x15C's IsActive() is false and Render_PreUI itself runs the pre-cull helper.
		std::uintptr_t PreCullThunk(std::uintptr_t a1, std::uintptr_t a2, std::uintptr_t a3, std::uintptr_t a4)
		{
			const auto result = reinterpret_cast<PassFn>(g_preCullSite.previous)(a1, a2, a3, a4);
			if (g_window.load(std::memory_order_relaxed)) {
				OpenWindow();
				++g_windowCounts.preCull;
			}
			return result;
		}

		// 1108521+0xE49: the cascade cull with previs suspended around it (a CBRO frame), so it takes its group-0 path.
		std::uintptr_t CascadeCullThunk(std::uintptr_t a1, std::uintptr_t a2, std::uintptr_t a3, std::uintptr_t a4)
		{
			if (!g_window.load(std::memory_order_relaxed)) {
				return reinterpret_cast<PassFn>(g_cascadeSite.previous)(a1, a2, a3, a4);
			}
			const auto saved = ReadSuspendedByte();
			if (!saved) {
				WriteSuspended(true);
			}
			++g_windowCounts.cascade;
			const auto result = reinterpret_cast<PassFn>(g_cascadeSite.previous)(a1, a2, a3, a4);
			if (!saved) {
				WriteSuspended(false);
			}
			return result;
		}

		bool InstallSite(FeedSite& a_site, std::uint64_t a_functionID, std::size_t a_offset, std::uint64_t a_feedID, std::uintptr_t a_thunk, const char* a_name)
		{
			a_site.address = CBRO::Engine::OG(a_functionID).address() + a_offset;
			const auto feed = CBRO::Engine::OG(a_feedID).address();
			const auto current = Util::ReadCall5Target(a_site.address);
			if (!current) {
				logger::error("previs feed: {} site {} is not a call rel32 (another plugin patched it); feed mode unavailable", a_name, Util::DescribeCodeAddress(a_site.address));
				return false;
			}
			// (DescribeCodeAddress follows a detoured entry to where it lands, e.g. another plugin's replacement)
			logger::info(
				"previs feed: {} site {} calls {}{}; the engine's feed entry {} lands in {}",
				a_name, Util::DescribeCodeAddress(a_site.address), Util::DescribeCodeAddress(current), current == feed ? " (the engine's)" : " (NOT the engine's)",
				Util::DescribeCodeAddress(feed), Util::DescribeCodeAddress(Util::FollowJumps(feed)));
			a_site.previous = Util::WriteCall5(a_site.address, a_thunk, a_name);
			a_site.ok = a_site.previous != 0;
			return a_site.ok;
		}

		// ---- raw engine reads (SEH-guarded by the callers) ----------------------------------------------------------
		template <class T>
		T At(std::uintptr_t a_address) noexcept
		{
			return *reinterpret_cast<const T*>(a_address);
		}

		std::uintptr_t Ptr(std::uintptr_t a_address) noexcept
		{
			return At<std::uintptr_t>(a_address);
		}

		std::uintptr_t CallSlot(std::uintptr_t a_object, std::size_t a_slot) noexcept
		{
			using Fn = std::uintptr_t (*)(std::uintptr_t);
			const auto vtable = Ptr(a_object);
			return reinterpret_cast<Fn>(Ptr(vtable + a_slot * sizeof(std::uintptr_t)))(a_object);
		}

		bool Flag(std::uintptr_t a_object, unsigned a_bit) noexcept
		{
			return (At<std::uint64_t>(a_object + kFlags) >> a_bit) & 1;
		}

		std::uint32_t ChildCount(std::uintptr_t a_node) noexcept
		{
			return At<std::uint16_t>(a_node + kChildCount);
		}

		std::uintptr_t Child(std::uintptr_t a_node, std::uint32_t a_index) noexcept
		{
			const auto data = Ptr(a_node + kChildren);
			return data ? Ptr(data + a_index * sizeof(std::uintptr_t)) : 0;
		}

		// No C++ objects with destructors in here (MSVC C2712): the visitor is a plain function pointer.
		bool EnumerateImpl(Visitor a_visit, void* a_context) noexcept
		{
			__try {
				const auto root = Ptr(g_sceneRoot);
				const auto culler = Ptr(g_culler);
				if (!root || !culler) {
					return false;
				}
				const auto state = Ptr(culler + kCullerState);
				if (!state || At<std::uint8_t>(state + kExteriorByte) == 0 || Ptr(g_overrideRoot) != 0) {
					return false;  // interior, portal or override-root path: not replicated
				}
				const auto visit = [&](std::uintptr_t a_object, Route a_route, Site a_site) {
					a_visit(a_context, reinterpret_cast<RE::NiAVObject*>(a_object), a_route, a_site);
				};
				const auto add = [&](std::uintptr_t a_object, Route a_route, Site a_site) {  // the walk's usual gate: non-null, bit 0 clear
					if (a_object && !Flag(a_object, 0)) {
						visit(a_object, a_route, a_site);
					}
				};

				// R1: the scene root's child 2: S = IsNode(child 2), T = IsNode(S.child 0): T.0, T.1 -> G1, T.2 -> G0, S.1, S.2 -> G1.
				if (const auto child2 = Child(root, 2)) {
					if (const auto s = CallSlot(child2, kSlotIsNode)) {
						if (const auto child0 = Child(s, 0)) {
							if (const auto t = CallSlot(child0, kSlotIsNode)) {
								add(Child(t, 0), Route::kGroup1, Site::kR1);
								add(Child(t, 1), Route::kGroup1, Site::kR1);
								add(Child(t, 2), Route::kGroup0, Site::kR1);
							}
						}
						add(Child(s, 1), Route::kGroup1, Site::kR1);
						add(Child(s, 2), Route::kGroup1, Site::kR1);
					}
				}

				// R2, R3: the world node W = IsNode(root.child 3) and its cells.
				const auto child3 = Child(root, 3);
				const auto world = child3 ? CallSlot(child3, kSlotIsNode) : 0;
				if (world && !Flag(world, 0)) {
					add(Child(world, 0), Route::kGroup1, Site::kR2);
					const bool skipChild2 = At<std::uint8_t>(g_skipChild2) != 0;
					const auto cells = ChildCount(world);
					for (std::uint32_t i = 1; i < cells; ++i) {
						const auto cellChild = Child(world, i);
						const auto cell = cellChild ? CallSlot(cellChild, kSlotIsNode) : 0;
						if (!cell || Flag(cell, 0)) {
							continue;
						}
						// (the walk sets flag bit 42 on the cell node here: not replicated)
						const auto count = ChildCount(cell);
						for (std::uint32_t j = 0; j < count; ++j) {
							if (j == 0 || j == 1 || j == 4 || (j == 2 && skipChild2)) {
								continue;
							}
							const auto child = Child(cell, j);
							if (!child) {
								continue;
							}
							const auto node = CallSlot(child, kSlotIsNode);
							if (!node) {
								add(child, Route::kGroup0, Site::kR3b);  // not a node: whole
								continue;
							}
							if (Flag(node, 0)) {
								continue;
							}
							if (j == 3 || j == 9) {
								const auto grandchildren = ChildCount(node);
								for (std::uint32_t k = 0; k < grandchildren; ++k) {
									const auto grandchild = Child(node, k);
									if (!grandchild || Flag(grandchild, 0)) {
										continue;
									}
									if (j == 3 && CallSlot(grandchild, kSlotGetRTTI) == g_niNodeRtti) {
										// an exact-NiNode container: each of its children, with no bit-0 test (the engine reads the array directly)
										const auto members = ChildCount(grandchild);
										for (std::uint32_t m = 0; m < members; ++m) {
											if (const auto member = Child(grandchild, m)) {
												visit(member, Route::kGroup0, Site::kR3c);
											}
										}
									} else {
										visit(grandchild, Route::kGroup0, Site::kR3d);
									}
								}
							} else if (j == 2) {
								const auto grandchildren = ChildCount(node);
								for (std::uint32_t k = 0; k < grandchildren; ++k) {
									add(Child(node, k), Route::kGroup1, Site::kR3e);
								}
							}
						}
					}
				}

				// R4: DrawWorld's own loop over the roots registered with the culling camera.
				if (const auto roots = Ptr(state + kCullerRoots)) {
					const auto array = Ptr(roots + kRootsArray);
					const auto count = At<std::uint32_t>(roots + kRootsCount);
					for (std::uint32_t r = 0; array && r < count; ++r) {
						const auto rootObject = Ptr(array + r * sizeof(std::uintptr_t));
						if (!rootObject || Flag(rootObject, 0)) {
							continue;
						}
						const auto node = CallSlot(rootObject, kSlotIsNode);
						if (!node || (Flag(rootObject, 14) && CallSlot(rootObject, kSlotIsFadeNode))) {
							visit(rootObject, Route::kArray, Site::kR4Whole);
							continue;
						}
						const auto children = ChildCount(node);
						for (std::uint32_t k = 0; k < children; ++k) {
							const auto child = Child(node, k);
							if (child && !Flag(child, 0)) {
								visit(child, Flag(child, 40) ? Route::kGroup1 : Route::kGroup0, Site::kR4Child);
							}
						}
					}
				}
				return true;
			} __except (EXCEPTION_EXECUTE_HANDLER) {
				return false;
			}
		}

		bool ReadGatesImpl(Gates& a_out) noexcept
		{
			__try {
				a_out.enabled = At<std::uint8_t>(g_enabled) != 0;
				a_out.ini = At<std::uint8_t>(g_ini) != 0;
				a_out.suspended = At<std::uint8_t>(g_suspended) != 0;
				a_out.active = a_out.enabled && a_out.ini && !a_out.suspended;
				const auto root = Ptr(g_sceneRoot);
				if (root) {
					a_out.gateA = At<std::uint32_t>(root + kGateA);
					a_out.gateB = At<std::uint16_t>(root + kGateB);
				}
				const auto culler = Ptr(g_culler);
				const auto state = culler ? Ptr(culler + kCullerState) : 0;
				a_out.exterior = state != 0 && At<std::uint8_t>(state + kExteriorByte) != 0;
				a_out.overrideRoot = Ptr(g_overrideRoot) != 0;
				return true;
			} __except (EXCEPTION_EXECUTE_HANDLER) {
				return false;
			}
		}

		bool GroupPlanesImpl(const void* a_group, float a_planes[6][4]) noexcept
		{
			__try {
				std::memcpy(a_planes, a_group, 6 * 4 * sizeof(float));
				return true;
			} __except (EXCEPTION_EXECUTE_HANDLER) {
				return false;
			}
		}
	}

	bool Install()
	{
		g_sceneRoot = CBRO::Engine::OG(kSceneRootID).address();
		g_culler = CBRO::Engine::OG(kCullerID).address();
		g_overrideRoot = CBRO::Engine::OG(kOverrideRootID).address();
		g_skipChild2 = CBRO::Engine::OG(kSkipChild2ByteID).address();
		g_enabled = CBRO::Engine::OG(kEnabledByteID).address();
		g_ini = CBRO::Engine::OG(kIniByteID).address();
		g_suspended = CBRO::Engine::OG(kSuspendedByteID).address();
		g_setSuspended = CBRO::Engine::OG(kSetSuspendedID).address();
		g_niNodeRtti = CBRO::Engine::OG(kNiNodeRttiID).address();
		for (std::size_t i = 0; i < kSiteOffsets.size(); ++i) {
			g_siteAddresses[i] = { CBRO::Engine::OG(kSiteOffsets[i].functionID).address() + kSiteOffsets[i].offset, kSiteOffsets[i].site };
		}

		const bool main = InstallSite(g_mainSite, kSceneWalkID, kMainFeedSiteOffset, kMainFeedID, Util::FnAddr(&MainFeedThunk), "previsfeed:main feed");
		const bool sun = InstallSite(g_sunSite, kCascadeCullID, kSunFeedSiteOffset, kSunFeedID, Util::FnAddr(&SunFeedThunk), "previsfeed:sun feed");
		g_available = main && sun;
		logger::info("previs feed: wrappers {} (owner: previs; pass-through)", g_available ? "installed on both sites" : "incomplete: feed mode unavailable this session");

		// The windows: the pre-cull wrapper is what makes a CBRO frame a previs-off frame for the cull; without it CBRO
		// falls back to suspending at the cull begin (the cull-begin hold below, which skips the engine's pre-cull
		// helper for that frame). The cascade wrapper is the sun culling's window; without it the cascades take their
		// previs path and CBRO's sun verdicts have no reader (CullGroups then reports the path as changed).
		g_preCullHelper = CBRO::Engine::OG(kPreCullHelperID).address();
		const bool preCull = InstallSite(g_preCullSite, kRenderPreUIID, kPreCullSiteOffset, kPreCullTargetID, Util::FnAddr(&PreCullThunk), "previsfeed:pre-cull window");
		const bool cascade = InstallSite(g_cascadeSite, kUnbatchedStageID, kCascadeCullSiteOffset, kCascadeCullID, Util::FnAddr(&CascadeCullThunk), "previsfeed:cascade window");
		g_windowsAvailable = preCull && cascade;
		logger::info(
			"previs feed: suspension windows {} (pre-cull {}, cascade cull {}); previs is suspended only inside them in CBRO frames",
			g_windowsAvailable ? "installed" : "INCOMPLETE", preCull ? "wrapped" : "NOT wrapped: cull-begin hold instead", cascade ? "wrapped" : "NOT wrapped: no sun-shadow culling");
		return g_available;
	}

	bool WindowsAvailable() noexcept
	{
		return g_windowsAvailable;
	}

	void SetWindow(bool a_cbroFrame) noexcept
	{
		g_window.store(a_cbroFrame, std::memory_order_relaxed);
	}

	bool Held() noexcept
	{
		return g_held;
	}

	void HoldNow() noexcept
	{
		if (g_held) {
			return;
		}
		OpenWindow();
		++g_windowCounts.cullBegin;
		// Render_PreUI saw previs active at +0x15C and left the pre-cull helper to the walk, which will now run
		// previs-off and not call it either: run it here, as +0x165 would have (no arguments; globals only).
		if (g_preCullHelper) {
			reinterpret_cast<void (*)()>(g_preCullHelper)();
		}
	}

	void ReleaseWindow() noexcept
	{
		if (!g_held) {
			return;
		}
		if (!g_savedByte) {
			WriteSuspended(false);
		}
		g_held = false;
	}

	void ClearSuspension() noexcept
	{
		// A suspension CBRO itself set outside the windows (the loading screen, v1.45) is not the engine's state: a
		// window that opened over it must not keep it. Open window: the release writes the byte back; else now.
		g_savedByte = 0;
		if (!g_held) {
			WriteSuspended(false);
		}
	}

	bool BeginLampWindow() noexcept
	{
		// Only in a CBRO frame, and only over the engine's own state: an existing suspension is left as found.
		if (!g_window.load(std::memory_order_relaxed) || ReadSuspendedByte()) {
			return false;
		}
		WriteSuspended(true);
		++g_windowCounts.lamp;
		return true;
	}

	void EndLampWindow(bool a_opened) noexcept
	{
		if (a_opened) {
			WriteSuspended(false);
		}
	}

	bool ActiveNow() noexcept
	{
		if (!g_enabled || !g_ini || !g_suspended) {
			return false;
		}
		__try {
			return At<std::uint8_t>(g_enabled) != 0 && At<std::uint8_t>(g_ini) != 0 && At<std::uint8_t>(g_suspended) == 0;
		} __except (EXCEPTION_EXECUTE_HANDLER) {
			return false;
		}
	}

	WindowCounts TakeWindowCounts() noexcept
	{
		const auto counts = g_windowCounts;
		g_windowCounts = {};
		return counts;
	}

	std::uintptr_t CascadeCullThunkAddress() noexcept
	{
		return g_cascadeSite.ok ? Util::FnAddr(&CascadeCullThunk) : 0;
	}

	std::uintptr_t CascadeCullPrevious() noexcept
	{
		return g_cascadeSite.ok ? g_cascadeSite.previous : 0;
	}

	bool Available() noexcept
	{
		return g_available;
	}

	void SetOwner(Owner a_owner) noexcept
	{
		g_owner.store(a_owner, std::memory_order_relaxed);
	}

	void SetMainFeed(MainFeedFn a_fn) noexcept
	{
		g_mainFn.store(a_fn, std::memory_order_relaxed);
	}

	void SetSunFeed(SunFeedFn a_fn) noexcept
	{
		g_sunFn.store(a_fn, std::memory_order_relaxed);
	}

	FeedCalls TakeFeedCalls() noexcept
	{
		return {
			g_calls[0].exchange(0, std::memory_order_relaxed), g_calls[1].exchange(0, std::memory_order_relaxed),
			g_calls[2].exchange(0, std::memory_order_relaxed), g_calls[3].exchange(0, std::memory_order_relaxed)
		};
	}

	Gates ReadGates() noexcept
	{
		Gates gates{};
		if (g_sceneRoot && g_enabled) {
			gates.readable = ReadGatesImpl(gates);
		}
		return gates;
	}

	void SetSuspended(bool a_suspend) noexcept
	{
		WriteSuspended(a_suspend);
	}

	bool GroupPlanes(const void* a_group, float a_planes[6][4]) noexcept
	{
		return a_group && GroupPlanesImpl(a_group, a_planes);
	}

	bool EnumerateCandidates(Visitor a_visit, void* a_context) noexcept
	{
		return a_visit && g_sceneRoot && g_culler && EnumerateImpl(a_visit, a_context);
	}

	bool SiteOf(std::uintptr_t a_returnAddress, Site& a_out) noexcept
	{
		for (const auto& [address, site] : g_siteAddresses) {
			if (address == a_returnAddress) {
				a_out = site;
				return true;
			}
		}
		return false;
	}

	std::string_view SiteName(Site a_site) noexcept
	{
		switch (a_site) {
		case Site::kR1:
			return "R1 root child 2"sv;
		case Site::kR2:
			return "R2 world child 0"sv;
		case Site::kR3b:
			return "R3b cell child whole"sv;
		case Site::kR3c:
			return "R3c container members"sv;
		case Site::kR3d:
			return "R3d node 3/9 grandchild"sv;
		case Site::kR3e:
			return "R3e node 2 children"sv;
		case Site::kR4Whole:
			return "R4 root whole"sv;
		case Site::kR4Child:
			return "R4 root child"sv;
		default:
			return "?"sv;
		}
	}

	std::string_view RouteName(Route a_route) noexcept
	{
		switch (a_route) {
		case Route::kGroup0:
			return "group 0"sv;
		case Route::kGroup1:
			return "group 1"sv;
		case Route::kArray:
			return "array"sv;
		default:
			return "?"sv;
		}
	}
}
