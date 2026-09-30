#include "Core/ShadowLights.h"

#include "Core/Occlusion.h"
#include "Settings.h"
#include "Util/Hooking.h"

namespace CBRO::Core::ShadowLights
{
	namespace
	{
		// BSParabolicCullingProcess (OG vtable id 845854). Slot 0x1A = Process(const NiCamera*, NiAVObject*,
		// NiVisibleArray*): sets the camera and frustum, restarts the accumulator, then culls the scene.
		// Per-object Process reads cullMode (+0x158) first; kAllFail (2) rejects without appending.
		constexpr std::uint64_t kParabolicVtableID = 845854;
		constexpr std::size_t   kProcessCameraSlot = 0x1A;
		constexpr std::size_t   kCullModeOffset = 0x158;
		constexpr std::uint32_t kAllFail = 2;
		// The light's culling sphere, set on the culler right before Process by its setup (0x1429775C0,
		// called from BSShadowParabolicLight id 656208 with the NiLight's radius, light +0x138):
		// center (the shadow camera's world translate) at +0x1C0, radius at +0x1CC (also +0x1A0).
		// (The shadow camera's own frustum far is 1: the paraboloid projection is normalized.)
		constexpr std::size_t kSphereCenterOffset = 0x1C0;
		constexpr std::size_t kSphereRadiusOffset = 0x1CC;
		// Slot 0x19 = NiCullingProcess::Process(NiAVObject*) (0x1429772B0): the per-object test of the light's
		// traversal. An object that passes calls its OnVisible (vtable +0x1C8), which files geometry and walks a
		// node's children; one that fails is dropped with its subtree, and if the culler's updateAccumulateFlag
		// (+0x11D) is set its kFlagAccumulated (flags bit 42) is cleared. Objects with kFlagAlwaysDraw (bit 11)
		// pass without a test, kFlagPreProcessedNode (bit 26) ones are decided by kFlagNotVisible (bit 39), and
		// cullMode kAllPass/kAllFail (1/2) pass/fail everything; kNormal and kIgnoreMultiBounds (0/3) test
		// normally. CBRO leaves every untested case to the engine. (Names: CommonLibF4 BSCullingProcess /
		// NiCullingProcess, F4SE NiObjects.h.)
		constexpr std::size_t   kProcessObjectSlot = 0x19;
		constexpr std::size_t   kTrackSeenOffset = 0x11D;
		constexpr std::uint64_t kSeenFlag = std::uint64_t{ 1 } << 42;
		constexpr std::uint64_t kUntestedFlags = (std::uint64_t{ 1 } << 11) | (std::uint64_t{ 1 } << 26);
		// Shadow filtering reaches this far around a lookup (a generous 5 degrees seen from the light, plus a
		// little): a caster within it of a visible point's light segment may still darken that point.
		constexpr float kFilterSlope = 0.0875f;
		constexpr float kFilterBase = 16.0f;

		using ProcessFn = void (*)(void*, const RE::NiCamera*, RE::NiAVObject*, void*);
		using ProcessObjectFn = void (*)(void*, RE::NiAVObject*);
		std::uintptr_t       g_original{ 0 };
		std::uintptr_t       g_originalObject{ 0 };
		Util::SwitchableHook g_cameraHook;  // taken out while CBRO is off
		Util::SwitchableHook g_objectHook;

		// Spot lights: BSShadowFrustumLight (vtable id 67506) slot 9 (1559482) is its cull: it sets the shadow camera's
		// frustum (far = the NiLight's radius) and runs the group pass (FO4-ENGINE-NOTES 6.2a). The light's NiLight is at
		// +0xB8 (BSLight), as for the parabolic light; NiLight::spec.r (+0x138) is the radius.
		constexpr std::uint64_t kFrustumLightVtableID = 67506;
		constexpr std::size_t   kFrustumCullSlot = 9;
		constexpr std::size_t   kLightNiLightOffset = 0xB8;
		using PassFn = std::uintptr_t (*)(std::uintptr_t, std::uintptr_t, std::uintptr_t, std::uintptr_t);
		std::uintptr_t       g_originalFrustumCull{ 0 };
		Util::SwitchableHook g_frustumHook;

		// ---- the lamps of a frame (main thread: the shadow stage writes, the cull begin publishes) --------------------
		LampList                   g_lampLists[2];
		std::uint32_t              g_lampWrite{ 0 };
		std::atomic<const LampList*> g_lampsPublished{ &g_lampLists[1] };
		std::atomic<std::uint64_t> g_lampsPoint{ 0 };
		std::atomic<std::uint64_t> g_lampsSpot{ 0 };

		void RecordLamp(const RE::NiPoint3& a_position, float a_reach, bool a_spot) noexcept
		{
			auto& list = g_lampLists[g_lampWrite];
			if (!std::isfinite(a_reach) || a_reach < 16.0f || a_reach > 1.0e6f || !std::isfinite(a_position.x) || !std::isfinite(a_position.y) || !std::isfinite(a_position.z)) {
				return;
			}
			if (list.count >= LampList::kMax) {
				++list.overflow;
				return;
			}
			list.items[list.count++] = Lamp{ a_position, a_reach, a_spot };
			(a_spot ? g_lampsSpot : g_lampsPoint).fetch_add(1, std::memory_order_relaxed);
		}

		// The spot light's NiLight position and radius, read under SEH (the light must be one of the engine's).
		bool ReadSpotLight(std::uintptr_t a_light, RE::NiPoint3& a_position, float& a_reach) noexcept
		{
			__try {
				const auto niLight = *reinterpret_cast<const std::uintptr_t*>(a_light + kLightNiLightOffset);
				if (!niLight) {
					return false;
				}
				const auto* object = reinterpret_cast<const RE::NiAVObject*>(niLight);
				const auto* view = reinterpret_cast<const CBRO::Engine::NiLightView*>(niLight);
				const float scale = std::isfinite(object->world.scale) && object->world.scale > 0.0f ? object->world.scale : 1.0f;
				const auto  valid = [](float a_value) { return std::isfinite(a_value) ? a_value : 0.0f; };
				a_position = object->world.translate;
				a_reach = std::max({ valid(view->spec.r * scale), valid(view->modelBound.fRadius * scale), valid(object->worldBound.fRadius) });
				return true;
			} __except (EXCEPTION_EXECUTE_HANDLER) {
				return false;
			}
		}

		std::atomic<int> g_spotSamplesLogged{ 0 };

		std::uintptr_t FrustumCullThunk(std::uintptr_t a1, std::uintptr_t a2, std::uintptr_t a3, std::uintptr_t a4)
		{
			if (Occlusion::Active()) {
				RE::NiPoint3 position{};
				float        reach = 0.0f;
				if (ReadSpotLight(a1, position, reach)) {
					RecordLamp(position, reach, true);
					// (the first few, with the camera, so the read layout can be checked against the scene)
					if (g_spotSamplesLogged.load(std::memory_order_relaxed) < 6 && g_spotSamplesLogged.fetch_add(1) < 6) {
						const auto root = RE::Main::WorldRootCamera();
						logger::info(
							"shadow lights: spot light sample at ({:.0f},{:.0f},{:.0f}) reach {:.0f} | camera at ({:.0f},{:.0f},{:.0f})",
							position.x, position.y, position.z, reach,
							root ? root->world.translate.x : 0.0f, root ? root->world.translate.y : 0.0f, root ? root->world.translate.z : 0.0f);
					}
				}
			}
			return reinterpret_cast<PassFn>(g_originalFrustumCull)(a1, a2, a3, a4);
		}
		std::uint32_t  g_confirmFrames{ 2 };
		bool           g_casterCulling{ true };

		// The light whose casters the running traversal (this thread) collects, and where they may matter: the
		// view cone pushed out until it holds the light, plus the filter's reach (ShadowGeometry).
		struct Region
		{
			const void*          culler{ nullptr };
			ShadowGeometry::Cone cone{};
			float                push{ 0.0f };
			RE::NiPoint3         lamp{};         // the light's position (its culling sphere's center)
			float                reach{ 0.0f };  // ... and radius
			std::uint32_t        tests{ 0 };     // counted here (one thread), summed once per light
			std::uint32_t        culled{ 0 };    // outside the pushed view cone
			std::uint32_t        volumeCulled{ 0 };  // shadow volume outside the view or missing every visible surface
		};
		thread_local Region t_region{};
		bool                g_volumes{ true };  // bLampShadowVolumes

		std::atomic<std::uint64_t> g_regionLights{ 0 };
		std::atomic<std::uint64_t> g_regionInside{ 0 };  // lights inside the view cone
		std::atomic<std::uint64_t> g_casterTests{ 0 };
		std::atomic<std::uint64_t> g_casterCulled{ 0 };
		std::atomic<std::uint64_t> g_casterVolumeCulled{ 0 };

		// Confirmation streaks per light, keyed by its position and reach (shadow cameras can be
		// pooled across lights, so the camera pointer is not an identity).
		constexpr std::size_t kSlots = 256;
		struct Slot
		{
			std::atomic<std::uint64_t> key{ 0 };
			std::atomic<std::uint32_t> lastClock{ 0 };
			std::atomic<std::uint32_t> streak{ 0 };
		};
		std::array<Slot, kSlots> g_slots{};

		std::atomic<std::uint64_t> g_calls{ 0 };
		std::atomic<std::uint64_t> g_emptied{ 0 };
		std::atomic<std::uint64_t> g_wouldEmpty{ 0 };
		std::atomic<std::uint64_t> g_confirming{ 0 };
		std::atomic<std::uint64_t> g_visible{ 0 };
		std::atomic<std::uint64_t> g_unknown{ 0 };
		std::atomic<std::uint64_t> g_hiddenVerdicts{ 0 };
		std::atomic<std::uint64_t> g_outOfViewVerdicts{ 0 };
		std::atomic<int>           g_samplesLogged{ 0 };

		std::uint64_t LightKey(const RE::NiPoint3& a_position, float a_reach) noexcept
		{
			const auto q = [](float a_v) { return static_cast<std::uint64_t>(static_cast<std::int64_t>(std::floor(a_v / 8.0f)) & 0xFFFF); };
			const auto key = q(a_position.x) | (q(a_position.y) << 16) | (q(a_position.z) << 32) | (q(a_reach) << 48);
			return key ? key : 1;
		}

		Slot* FindSlot(std::uint64_t a_key) noexcept
		{
			const auto start = static_cast<std::size_t>((a_key * 0x9E3779B97F4A7C15ull) >> 56) % kSlots;
			for (std::size_t probe = 0; probe < 16; ++probe) {
				auto& slot = g_slots[(start + probe) % kSlots];
				auto  current = slot.key.load(std::memory_order_acquire);
				if (current == a_key) {
					return &slot;
				}
				if (current == 0 && slot.key.compare_exchange_strong(current, a_key, std::memory_order_acq_rel)) {
					return &slot;
				}
				if (current == a_key) {
					return &slot;
				}
			}
			// Full neighbourhood: recycle the first slot (its light gets a fresh, conservative streak).
			auto& slot = g_slots[start];
			slot.key.store(a_key);
			slot.streak.store(0);
			slot.lastClock.store(0);
			return &slot;
		}

		bool ShouldEmpty(const void* a_culler, const RE::NiCamera* a_camera) noexcept
		{
			const auto  base = static_cast<const std::byte*>(a_culler);
			const float reach = *reinterpret_cast<const float*>(base + kSphereRadiusOffset);
			const auto* center = reinterpret_cast<const float*>(base + kSphereCenterOffset);
			const RE::NiPoint3 position{ center[0], center[1], center[2] };
			const auto&        eye = a_camera->world.translate;
			const float        offset = std::sqrt(
                (position.x - eye.x) * (position.x - eye.x) + (position.y - eye.y) * (position.y - eye.y) + (position.z - eye.z) * (position.z - eye.z));

			if (g_samplesLogged.load(std::memory_order_relaxed) < 8 && g_samplesLogged.fetch_add(1) < 8) {
				logger::info(
					"shadow lights: sample light sphere at ({:.0f},{:.0f},{:.0f}) reach {:.0f} | shadow camera at ({:.0f},{:.0f},{:.0f})",
					position.x, position.y, position.z, reach, eye.x, eye.y, eye.z);
			}
			// The sphere must be this light's: centered on its shadow camera, with a real radius.
			if (!std::isfinite(reach) || reach < 16.0f || reach > 1.0e6f || !(offset < 8.0f)) {
				g_unknown.fetch_add(1, std::memory_order_relaxed);
				return false;
			}

			const auto verdict = Occlusion::TestSphere(position, reach);
			const auto clock = Occlusion::Clock();
			auto&      slot = *FindSlot(LightKey(position, reach));
			auto       streak = slot.streak.load(std::memory_order_relaxed);
			// Behind visible surfaces or out of view entirely: either way no visible pixel is in reach.
			const bool unseen = verdict == Occlusion::SphereVerdict::kHidden || verdict == Occlusion::SphereVerdict::kOutOfView;
			if (unseen) {
				(verdict == Occlusion::SphereVerdict::kHidden ? g_hiddenVerdicts : g_outOfViewVerdicts).fetch_add(1, std::memory_order_relaxed);
				if (slot.lastClock.exchange(clock, std::memory_order_relaxed) != clock) {
					streak = std::min(streak + 1, 255u);
					slot.streak.store(streak, std::memory_order_relaxed);
				}
			} else {
				slot.streak.store(0, std::memory_order_relaxed);
				slot.lastClock.store(clock, std::memory_order_relaxed);
				(verdict == Occlusion::SphereVerdict::kVisible ? g_visible : g_unknown).fetch_add(1, std::memory_order_relaxed);
				return false;
			}

			if (streak < g_confirmFrames) {
				g_confirming.fetch_add(1, std::memory_order_relaxed);
				return false;
			}
			if (Occlusion::Deciding()) {
				g_wouldEmpty.fetch_add(1, std::memory_order_relaxed);
				return false;
			}
			g_emptied.fetch_add(1, std::memory_order_relaxed);
			return true;
		}

		// Where this light's casters can shadow a visible point: the view cone (Occlusion, this frame) with
		// its planes pushed out until they hold the light, by the filter's reach, and by the camera's movement
		// since the cone was measured. False if the view or the light's sphere isn't known.
		bool MakeRegion(const void* a_culler, const RE::NiCamera* a_camera, Region& a_region) noexcept
		{
			if (!g_casterCulling) {
				return false;
			}
			const auto  base = static_cast<const std::byte*>(a_culler);
			const float reach = *reinterpret_cast<const float*>(base + kSphereRadiusOffset);
			const auto* center = reinterpret_cast<const float*>(base + kSphereCenterOffset);
			const auto& eye = a_camera->world.translate;
			const float offset = std::sqrt(
				(center[0] - eye.x) * (center[0] - eye.x) + (center[1] - eye.y) * (center[1] - eye.y) + (center[2] - eye.z) * (center[2] - eye.z));
			// The sphere must be this light's (as in ShouldEmpty).
			if (!std::isfinite(reach) || reach < 16.0f || reach > 1.0e6f || !(offset < 8.0f)) {
				return false;
			}
			float viewPush = 0.0f;
			if (!Occlusion::ViewCone(a_region.cone, viewPush) || Occlusion::Deciding()) {
				return false;
			}
			const float outside = ShadowGeometry::OutsideBy(a_region.cone, center);
			a_region.push = outside + viewPush + reach * kFilterSlope + kFilterBase;
			a_region.culler = a_culler;
			a_region.lamp = RE::NiPoint3{ center[0], center[1], center[2] };
			a_region.reach = reach;
			g_regionLights.fetch_add(1, std::memory_order_relaxed);
			if (outside <= 0.0f) {
				g_regionInside.fetch_add(1, std::memory_order_relaxed);
			}
			return true;
		}

		void ProcessCamera(void* a_self, const RE::NiCamera* a_camera, RE::NiAVObject* a_scene, void* a_visibleSet)
		{
			if (!Occlusion::Active()) {  // previs mode: the engine's own cull, untouched
				reinterpret_cast<ProcessFn>(g_original)(a_self, a_camera, a_scene, a_visibleSet);
				return;
			}
			g_calls.fetch_add(1, std::memory_order_relaxed);
			{
				// The lamp itself, for the group-0 trimming with the sun off (its culling sphere: center, radius).
				const auto  base = static_cast<const std::byte*>(a_self);
				const auto* center = reinterpret_cast<const float*>(base + kSphereCenterOffset);
				RecordLamp(RE::NiPoint3{ center[0], center[1], center[2] }, *reinterpret_cast<const float*>(base + kSphereRadiusOffset), false);
			}
			if (a_camera && a_scene && ShouldEmpty(a_self, a_camera)) {
				auto&      cullMode = *reinterpret_cast<std::uint32_t*>(static_cast<std::byte*>(a_self) + kCullModeOffset);
				const auto saved = cullMode;
				cullMode = kAllFail;
				reinterpret_cast<ProcessFn>(g_original)(a_self, a_camera, a_scene, a_visibleSet);
				cullMode = saved;
				return;
			}
			// The traversal below runs Process(object) for this light on this thread: casters outside the region
			// are dropped there (ProcessObject).
			const Region saved = t_region;
			Region       region{};
			const bool   active = a_camera && a_scene && MakeRegion(a_self, a_camera, region);
			if (active) {
				t_region = region;
			}
			reinterpret_cast<ProcessFn>(g_original)(a_self, a_camera, a_scene, a_visibleSet);
			if (active) {
				g_casterTests.fetch_add(t_region.tests, std::memory_order_relaxed);
				g_casterCulled.fetch_add(t_region.culled, std::memory_order_relaxed);
				g_casterVolumeCulled.fetch_add(t_region.volumeCulled, std::memory_order_relaxed);
			}
			t_region = saved;
		}

		void ProcessObject(void* a_self, RE::NiAVObject* a_object)
		{
			if (t_region.culler == a_self && a_object) {
				const auto base = static_cast<std::byte*>(a_self);
				const auto mode = *reinterpret_cast<const std::uint32_t*>(base + kCullModeOffset);
				const auto flags = a_object->GetFlags();
				if ((mode == 0 || mode == 3) && !(flags & kUntestedFlags)) {
					const auto& bound = a_object->worldBound;
					const float center[3]{ bound.center.x, bound.center.y, bound.center.z };
					++t_region.tests;
					const bool valid = bound.fRadius > 0.0f && bound.fRadius < 1.0e6f && std::isfinite(center[0]) && std::isfinite(center[1]) && std::isfinite(center[2]);
					bool       drop = false;
					if (valid && ShadowGeometry::CasterOutside(t_region.cone, t_region.push, center, bound.fRadius)) {
						++t_region.culled;
						drop = true;
					} else if (valid && g_volumes) {
						// Its shadow volume against the depth: outside the view, or missing every visible surface.
						const auto verdict = Occlusion::TestLampCaster(a_object, t_region.lamp, t_region.reach, bound);
						Occlusion::CountLampVerdict(verdict);
						if (verdict == Occlusion::LampVerdict::kOutside || verdict == Occlusion::LampVerdict::kMisses) {
							++t_region.volumeCulled;
							drop = true;
						}
					}
					if (drop) {
						// What the engine does for an object that fails its own test.
						if (*reinterpret_cast<const std::uint8_t*>(base + kTrackSeenOffset)) {
							a_object->flags.flags &= ~kSeenFlag;
						}
						return;
					}
				}
			}
			reinterpret_cast<ProcessObjectFn>(g_originalObject)(a_self, a_object);
		}
	}

	namespace
	{
		// ShadowSceneNode (global 879298 = 0x1467231B0, a pointer): the shadow lights at +0x170 (BSShadowLight* array,
		// count u16 at +0x180: what DeferredLightsImpl's lamp loop walks through 43862), and the light list at +0x158
		// (the list UpdateLightList 1479977 walks: data pointer at +0x158, entry count u32 at +0x168, as its loop reads
		// them: `rbx = [list]`, end = `rbx + [list+0x10]*8`). v1.42 read the count at +0x164 and cleared nothing there:
		// the non-shadow lamps, most lamps, kept previs's mark. BSLight::bOccluded is the byte at +0x17C.
		constexpr std::uint64_t kShadowSceneNodeID = 879298;
		constexpr std::size_t   kShadowLightsArray = 0x170;
		constexpr std::size_t   kShadowLightsCount = 0x180;
		constexpr std::size_t   kLightsArray = 0x158;
		constexpr std::size_t   kLightsCount = 0x168;
		constexpr std::size_t   kLightOccluded = 0x17C;
		constexpr std::uint32_t kMaxLights = 8192;
		std::uintptr_t             g_shadowSceneNode{ 0 };
		std::atomic<std::uint64_t> g_unoccludeFrames{ 0 };
		std::atomic<std::uint64_t> g_unoccluded{ 0 };       // shadow-list lights cleared
		std::atomic<std::uint64_t> g_unoccludedMain{ 0 };   // main-list lights cleared
		std::atomic<std::uint64_t> g_mainListLights{ 0 };   // main-list entries seen (per call)
		std::atomic<std::uint64_t> g_unoccludeCalls{ 0 };

		std::uint32_t ClearOccluded(std::uintptr_t a_array, std::uint32_t a_count) noexcept
		{
			std::uint32_t cleared = 0;
			if (!a_array || a_count > kMaxLights) {
				return cleared;
			}
			for (std::uint32_t i = 0; i < a_count; ++i) {
				const auto light = reinterpret_cast<const std::uintptr_t*>(a_array)[i];
				if (!light) {
					continue;
				}
				auto& occluded = *reinterpret_cast<std::uint8_t*>(light + kLightOccluded);
				if (occluded) {
					occluded = 0;
					++cleared;
				}
			}
			return cleared;
		}

		// Returns the shadow-list clears; the main list's clears and size through the out-parameters.
		std::uint32_t UnoccludeLightsGuarded(std::uint32_t& a_mainCleared, std::uint32_t& a_mainCount) noexcept
		{
			__try {
				const auto node = *reinterpret_cast<const std::uintptr_t*>(g_shadowSceneNode);
				if (!node) {
					return 0;
				}
				const std::uint32_t cleared = ClearOccluded(*reinterpret_cast<const std::uintptr_t*>(node + kShadowLightsArray), *reinterpret_cast<const std::uint16_t*>(node + kShadowLightsCount));
				const auto          count = *reinterpret_cast<const std::uint32_t*>(node + kLightsCount);
				a_mainCount = count <= kMaxLights ? count : 0;
				a_mainCleared = ClearOccluded(*reinterpret_cast<const std::uintptr_t*>(node + kLightsArray), a_mainCount);
				return cleared;
			} __except (EXCEPTION_EXECUTE_HANDLER) {
				return 0;
			}
		}
	}

	std::uint32_t UnoccludeLights(bool a_countFrame) noexcept
	{
		if (!g_shadowSceneNode) {
			g_shadowSceneNode = CBRO::Engine::OG(kShadowSceneNodeID).address();
		}
		std::uint32_t mainCleared = 0;
		std::uint32_t mainCount = 0;
		const auto    cleared = UnoccludeLightsGuarded(mainCleared, mainCount);
		if (a_countFrame) {
			g_unoccludeFrames.fetch_add(1, std::memory_order_relaxed);
		}
		g_unoccludeCalls.fetch_add(1, std::memory_order_relaxed);
		g_unoccluded.fetch_add(cleared, std::memory_order_relaxed);
		g_unoccludedMain.fetch_add(mainCleared, std::memory_order_relaxed);
		g_mainListLights.fetch_add(mainCount, std::memory_order_relaxed);
		return cleared + mainCleared;
	}

	// ---- the lamp loop's decisions (diagnostic, bLampDiagnostic) ------------------------------------------------------
	namespace
	{
		// DeferredLightsImpl's loop (0x142855590..0x142855CB0) skips shadow light i when: its NiLight (+0xB8) has flag bit
		// 0; BSLight::bOccluded (+0x17C); vtable slot 14 Update(camera) returns false; its face records (+0x198) have no
		// face-0 culler (+0xE0) or that culler has no state (+0x150); 1457421(main culler's state, light's state) is
		// false (both states "unrestricted" (+0x138) -> true; either room list (+0x18, count +0x28) empty -> false; else
		// true iff they share a room); after UpdateQueuedLight (222957) the light's queue slot (+0x18) is 0xFF (that
		// function drops a light whose NiLight fade (+0x144) is below 0.05, or when no slot is free); after slot 9 Cull
		// the face's slice (+0x54) is -1 (no shadow map). Otherwise the light renders its map and its light pass.
		constexpr std::uint64_t kMainCullerID = 865470;           // BSCullingProcess* (DrawWorld's main culler)
		constexpr std::uint64_t kParabolicLightVtableID = 585920; // BSShadowParabolicLight
		constexpr std::size_t   kUpdateSlot = 14;
		constexpr std::size_t   kCullerState = 0x150;
		constexpr std::size_t   kStateRooms = 0x18;
		constexpr std::size_t   kStateRoomCount = 0x28;
		constexpr std::size_t   kStateUnrestricted = 0x138;
		constexpr std::size_t   kLightQueued = 0x18;
		constexpr std::size_t   kLightFaces = 0x198;
		constexpr std::size_t   kFaceCuller = 0xE0;
		constexpr std::size_t   kFaceSlice = 0x54;
		constexpr std::size_t   kNiLightFade = 0x144;
		constexpr std::size_t   kNiObjectFlags = 0x108;
		constexpr float         kFadeThreshold = 0.05f;
		constexpr std::uint32_t kMaxRooms = 64;

		enum class LampFate : std::uint8_t
		{
			kHidden,
			kOccluded,
			kNoCuller,
			kRoomTest,
			kUnreached,    // passed the static tests, but Update(camera) was never called (a test this replica lacks)
			kUpdateFalse,
			kFaded,
			kNotQueued,
			kNoSlice,
			kRendered,
			kCount
		};
		constexpr const char* kFateNames[]{ "hidden", "occluded", "no culler state", "room test failed", "unreached", "Update(camera) false", "faded", "no queue slot", "no shadow-map slice", "rendered" };

		struct LampEntry
		{
			std::uintptr_t light{ 0 };
			LampFate       fate{ LampFate::kUnreached };
			bool           updated{ false };
			bool           updateResult{ false };
			bool           mainFree{ false };
			bool           lightFree{ false };
			std::uint32_t  mainRooms{ 0 };
			std::uint32_t  lightRooms{ 0 };
			float          x{ 0 }, y{ 0 }, z{ 0 };
			float          radius{ 0 };
			float          fade{ 0 };
		};
		constexpr std::size_t kMaxEntries = 256;
		LampEntry            g_entries[kMaxEntries];
		std::uint32_t        g_entryCount{ 0 };
		bool                 g_loopOpen{ false };
		std::uintptr_t       g_mainCuller{ 0 };
		std::uint64_t        g_fateCounts[static_cast<std::size_t>(LampFate::kCount)]{};
		std::uint64_t        g_loopFrames{ 0 };
		std::uint64_t        g_loopLights{ 0 };
		constexpr std::size_t kMaxDump = 8;
		LampEntry            g_dump[kMaxDump];
		std::uint32_t        g_dumpCount{ 0 };
		Util::SwitchableHook g_updatePointHook;
		Util::SwitchableHook g_updateSpotHook;
		std::uintptr_t       g_originalUpdatePoint{ 0 };
		std::uintptr_t       g_originalUpdateSpot{ 0 };
		using UpdateFn = bool (*)(std::uintptr_t, std::uintptr_t);

		void NoteUpdate(std::uintptr_t a_light, bool a_result) noexcept
		{
			if (!g_loopOpen) {
				return;
			}
			for (std::uint32_t i = 0; i < g_entryCount; ++i) {
				if (g_entries[i].light == a_light) {
					g_entries[i].updated = true;
					g_entries[i].updateResult = a_result;
					return;
				}
			}
		}

		bool UpdatePointThunk(std::uintptr_t a_light, std::uintptr_t a_camera)
		{
			const bool result = reinterpret_cast<UpdateFn>(g_originalUpdatePoint)(a_light, a_camera);
			NoteUpdate(a_light, result);
			return result;
		}

		bool UpdateSpotThunk(std::uintptr_t a_light, std::uintptr_t a_camera)
		{
			const bool result = reinterpret_cast<UpdateFn>(g_originalUpdateSpot)(a_light, a_camera);
			NoteUpdate(a_light, result);
			return result;
		}

		// 1457421 replicated.
		bool RoomTest(std::uintptr_t a_mainState, std::uintptr_t a_lightState, LampEntry& a_entry) noexcept
		{
			a_entry.mainFree = *reinterpret_cast<const std::uint8_t*>(a_mainState + kStateUnrestricted) != 0;
			a_entry.lightFree = *reinterpret_cast<const std::uint8_t*>(a_lightState + kStateUnrestricted) != 0;
			a_entry.mainRooms = *reinterpret_cast<const std::uint32_t*>(a_mainState + kStateRoomCount);
			a_entry.lightRooms = *reinterpret_cast<const std::uint32_t*>(a_lightState + kStateRoomCount);
			if (a_entry.mainFree && a_entry.lightFree) {
				return true;
			}
			if (a_entry.mainRooms == 0 || a_entry.lightRooms == 0) {
				return false;
			}
			const auto mainRooms = *reinterpret_cast<const std::uintptr_t* const*>(a_mainState + kStateRooms);
			const auto lightRooms = *reinterpret_cast<const std::uintptr_t* const*>(a_lightState + kStateRooms);
			if (!mainRooms || !lightRooms) {
				return false;
			}
			for (std::uint32_t i = 0; i < std::min(a_entry.mainRooms, kMaxRooms); ++i) {
				for (std::uint32_t j = 0; j < std::min(a_entry.lightRooms, kMaxRooms); ++j) {
					if (mainRooms[i] && mainRooms[i] == lightRooms[j]) {
						return true;
					}
				}
			}
			return false;
		}

		// The static tests of the loop for one light; false when nothing could be read.
		bool ReadLightStatic(std::uintptr_t a_light, std::uintptr_t a_mainState, LampEntry& a_entry) noexcept
		{
			__try {
				a_entry.light = a_light;
				const auto niLight = *reinterpret_cast<const std::uintptr_t*>(a_light + kLightNiLightOffset);
				if (niLight) {
					const auto* object = reinterpret_cast<const RE::NiAVObject*>(niLight);
					const auto* view = reinterpret_cast<const CBRO::Engine::NiLightView*>(niLight);
					a_entry.x = object->world.translate.x;
					a_entry.y = object->world.translate.y;
					a_entry.z = object->world.translate.z;
					a_entry.radius = view->spec.r;
					a_entry.fade = *reinterpret_cast<const float*>(niLight + kNiLightFade);
				}
				if (niLight && (*reinterpret_cast<const std::uint64_t*>(niLight + kNiObjectFlags) & 1)) {
					a_entry.fate = LampFate::kHidden;
					return true;
				}
				if (*reinterpret_cast<const std::uint8_t*>(a_light + kLightOccluded)) {
					a_entry.fate = LampFate::kOccluded;
					return true;
				}
				const auto faces = *reinterpret_cast<const std::uintptr_t*>(a_light + kLightFaces);
				const auto culler = faces ? *reinterpret_cast<const std::uintptr_t*>(faces + kFaceCuller) : 0;
				const auto state = culler ? *reinterpret_cast<const std::uintptr_t*>(culler + kCullerState) : 0;
				if (!state || !a_mainState) {
					a_entry.fate = LampFate::kNoCuller;
					return true;
				}
				if (!RoomTest(a_mainState, state, a_entry)) {
					a_entry.fate = LampFate::kRoomTest;
					return true;
				}
				a_entry.fate = LampFate::kUnreached;
				return true;
			} __except (EXCEPTION_EXECUTE_HANDLER) {
				return false;
			}
		}

		// After the loop: the queue slot and the shadow-map slice of a light that passed the static tests.
		void ReadLightOutcome(LampEntry& a_entry) noexcept
		{
			__try {
				if (!a_entry.updated) {
					return;  // stays kUnreached
				}
				if (!a_entry.updateResult) {
					a_entry.fate = LampFate::kUpdateFalse;
					return;
				}
				if (*reinterpret_cast<const std::uint32_t*>(a_entry.light + kLightQueued) == 0xFF) {
					a_entry.fate = a_entry.fade < kFadeThreshold ? LampFate::kFaded : LampFate::kNotQueued;
					return;
				}
				const auto faces = *reinterpret_cast<const std::uintptr_t*>(a_entry.light + kLightFaces);
				if (faces && *reinterpret_cast<const std::int32_t*>(faces + kFaceSlice) == -1) {
					a_entry.fate = LampFate::kNoSlice;
					return;
				}
				a_entry.fate = LampFate::kRendered;
			} __except (EXCEPTION_EXECUTE_HANDLER) {
			}
		}

		std::uint32_t ReadShadowLights(std::uintptr_t& a_array) noexcept
		{
			__try {
				const auto node = *reinterpret_cast<const std::uintptr_t*>(g_shadowSceneNode);
				if (!node) {
					return 0;
				}
				a_array = *reinterpret_cast<const std::uintptr_t*>(node + kShadowLightsArray);
				const auto count = *reinterpret_cast<const std::uint16_t*>(node + kShadowLightsCount);
				return a_array ? count : 0;
			} __except (EXCEPTION_EXECUTE_HANDLER) {
				return 0;
			}
		}

		std::uintptr_t ReadMainState() noexcept
		{
			__try {
				const auto culler = *reinterpret_cast<const std::uintptr_t*>(g_mainCuller);
				return culler ? *reinterpret_cast<const std::uintptr_t*>(culler + kCullerState) : 0;
			} __except (EXCEPTION_EXECUTE_HANDLER) {
				return 0;
			}
		}

		std::uintptr_t ReadLightAt(std::uintptr_t a_array, std::uint32_t a_index) noexcept
		{
			__try {
				return reinterpret_cast<const std::uintptr_t*>(a_array)[a_index];
			} __except (EXCEPTION_EXECUTE_HANDLER) {
				return 0;
			}
		}
	}

	void LampLoopBegin() noexcept
	{
		if (!g_shadowSceneNode) {
			g_shadowSceneNode = CBRO::Engine::OG(kShadowSceneNodeID).address();
		}
		if (!g_mainCuller) {
			g_mainCuller = CBRO::Engine::OG(kMainCullerID).address();
		}
		g_entryCount = 0;
		std::uintptr_t array = 0;
		const auto     count = std::min<std::uint32_t>(ReadShadowLights(array), kMaxEntries);
		const auto     mainState = ReadMainState();
		for (std::uint32_t i = 0; i < count; ++i) {
			const auto light = ReadLightAt(array, i);
			if (!light) {
				continue;
			}
			LampEntry entry{};
			if (ReadLightStatic(light, mainState, entry)) {
				g_entries[g_entryCount++] = entry;
			}
		}
		g_loopOpen = true;
	}

	void LampLoopEnd() noexcept
	{
		if (!g_loopOpen) {
			return;
		}
		g_loopOpen = false;
		++g_loopFrames;
		g_loopLights += g_entryCount;
		for (std::uint32_t i = 0; i < g_entryCount; ++i) {
			auto& entry = g_entries[i];
			if (entry.fate == LampFate::kUnreached) {
				ReadLightOutcome(entry);
			}
			++g_fateCounts[static_cast<std::size_t>(entry.fate)];
			if (entry.fate != LampFate::kRendered && g_dumpCount < kMaxDump) {
				bool seen = false;
				for (std::uint32_t j = 0; j < g_dumpCount; ++j) {
					seen = seen || g_dump[j].light == entry.light;
				}
				if (!seen) {
					g_dump[g_dumpCount++] = entry;
				}
			}
		}
	}

	void Install()
	{
		g_confirmFrames = Settings::Get().confirmFrames;
		g_casterCulling = Settings::Get().lampShadowCulling;
		g_volumes = Settings::Get().lampShadowVolumes;
		const auto vtable = CBRO::Engine::OG(kParabolicVtableID).address();
		g_original = Util::WriteVFuncSwitchable(g_cameraHook, vtable, kProcessCameraSlot, Util::FnAddr(&ProcessCamera), "shadowlights:BSParabolicCullingProcess::Process(cam)");
		if (!g_original) {
			logger::error("shadow lights: hook failed; point-light shadow maps are left to the engine");
			return;
		}
		// Only with the camera hook in place (it sets the region the object hook reads).
		g_originalObject = Util::WriteVFuncSwitchable(g_objectHook, vtable, kProcessObjectSlot, Util::FnAddr(&ProcessObject), "shadowlights:BSParabolicCullingProcess::Process(obj)");
		if (!g_originalObject) {
			logger::error("shadow lights: object hook failed; point-light casters are left to the engine");
			g_casterCulling = false;
		}
		// The spot-light hook (v1.37) is NOT installed: BSShadowFrustumLight's slot 9 is called at mode switches with
		// arguments a C++ thunk cannot forward (a thunk only passes the integer registers; any floating-point argument
		// or return value is clobbered by the recording work before the call). Recording spot lights needs a machine-code
		// stub that preserves the XMM registers, or another source; until then only point lights are recorded and the
		// lamp trimming stays off by default.
		(void)kFrustumLightVtableID;
		(void)kFrustumCullSlot;
		(void)&FrustumCullThunk;
		logger::info("shadow lights: spot-light hook not installed (see the comment: XMM arguments); spot lights are not recorded");

		// The lamp-loop diagnostic: Update(camera) takes only integer arguments and returns a bool, so a C++ thunk is safe.
		if (Settings::Get().lampDiagnostic) {
			g_originalUpdatePoint = Util::WriteVFuncSwitchable(g_updatePointHook, CBRO::Engine::OG(kParabolicLightVtableID).address(), kUpdateSlot, Util::FnAddr(&UpdatePointThunk), "shadowlights:BSShadowParabolicLight::Update(cam)");
			g_originalUpdateSpot = Util::WriteVFuncSwitchable(g_updateSpotHook, CBRO::Engine::OG(kFrustumLightVtableID).address(), kUpdateSlot, Util::FnAddr(&UpdateSpotThunk), "shadowlights:BSShadowFrustumLight::Update(cam)");
			logger::info("shadow lights: lamp-loop diagnostic on (Update(camera) hooks: point {}, spot {})", g_originalUpdatePoint ? "in" : "FAILED", g_originalUpdateSpot ? "in" : "FAILED");
		}
	}

	void PublishLamps() noexcept
	{
		g_lampsPublished.store(&g_lampLists[g_lampWrite], std::memory_order_release);
		g_lampWrite ^= 1u;
		g_lampLists[g_lampWrite].count = 0;
		g_lampLists[g_lampWrite].overflow = 0;
	}

	const LampList& Lamps() noexcept
	{
		return *g_lampsPublished.load(std::memory_order_acquire);
	}

	bool SetHooksIn(bool a_in)
	{
		bool ok = true;
		for (auto* hook : { &g_cameraHook, &g_objectHook, &g_frustumHook, &g_updatePointHook, &g_updateSpotHook }) {
			if (hook->address && !Util::SetHook(*hook, a_in)) {
				ok = false;
				logger::warn("hook {}: couldn't be {} (another plugin changed that slot since); left {}", hook->name, a_in ? "put back" : "taken out", hook->in ? "in" : "out");
			}
		}
		return ok;
	}

	void LogStats(std::uint32_t a_frames)
	{
		const double frames = std::max(1u, a_frames);
		const auto   take = [](std::atomic<std::uint64_t>& a_counter) { return static_cast<double>(a_counter.exchange(0)); };
		const auto   calls = take(g_calls);
		const auto   emptied = take(g_emptied);
		const auto   wouldEmpty = take(g_wouldEmpty);
		const auto   confirming = take(g_confirming);
		const auto   visible = take(g_visible);
		const auto   unknown = take(g_unknown);
		const auto   hidden = take(g_hiddenVerdicts);
		const auto   outOfView = take(g_outOfViewVerdicts);
		logger::info(
			"shadow lights per frame: shadow culls {:.1f} | emptied {:.1f}{} (verdicts: behind surfaces {:.1f}, out of view {:.1f}) | confirming {:.1f} | visible {:.1f} | kept (reaches camera/partly off-screen/no depth) {:.1f}",
			calls / frames, emptied / frames,
			wouldEmpty > 0 ? std::format(" (decide-only: would empty {:.1f})", wouldEmpty / frames) : std::string{},
			hidden / frames, outOfView / frames, confirming / frames, visible / frames, unknown / frames);
		const auto regions = take(g_regionLights);
		const auto inside = take(g_regionInside);
		const auto tests = take(g_casterTests);
		const auto culled = take(g_casterCulled);
		const auto volume = take(g_casterVolumeCulled);
		logger::info(
			"shadow light casters per frame: lights limited to the view {:.1f} (inside it {:.1f}) | objects tested {:.0f} | left out: outside the pushed view cone {:.0f}, shadow volume can't reach a visible surface {:.0f} (with their subtrees)",
			regions / frames, inside / frames, tests / frames, culled / frames, volume / frames);
		const auto point = take(g_lampsPoint);
		const auto spot = take(g_lampsSpot);
		logger::info("shadow lamps recorded per frame (for group 0 with the sun off): point {:.1f} | spot {:.1f} | list overflow {}", point / frames, spot / frames, Lamps().overflow);
		const auto unoccludeFrames = take(g_unoccludeFrames);
		const auto unoccludeCalls = take(g_unoccludeCalls);
		const auto unoccluded = take(g_unoccluded);
		const auto unoccludedMain = take(g_unoccludedMain);
		const auto mainLights = take(g_mainListLights);
		logger::info(
			"lights un-occluded (previs's bOccluded cleared at the cull begin and before the deferred-lights stage): {:.0f} CBRO frames | cleared per frame: shadow list {:.2f}, main list {:.2f} | main list holds {:.0f} lights",
			unoccludeFrames, unoccludeFrames > 0 ? unoccluded / unoccludeFrames : 0.0, unoccludeFrames > 0 ? unoccludedMain / unoccludeFrames : 0.0, unoccludeCalls > 0 ? mainLights / unoccludeCalls : 0.0);
		if (g_loopFrames > 0) {
			const double loopFrames = static_cast<double>(g_loopFrames);
			std::string  fates;
			for (std::size_t i = 0; i < static_cast<std::size_t>(LampFate::kCount); ++i) {
				fates += std::format("{}{} {:.2f}", i ? " | " : "", kFateNames[i], static_cast<double>(g_fateCounts[i]) / loopFrames);
				g_fateCounts[i] = 0;
			}
			logger::info("lamp loop per frame (the engine's shadow-light loop in CBRO frames, {:.0f} frames): shadow lights {:.2f} || {}", loopFrames, static_cast<double>(g_loopLights) / loopFrames, fates);
			for (std::uint32_t i = 0; i < g_dumpCount; ++i) {
				const auto& e = g_dump[i];
				logger::info(
					"lamp loop: light at ({:.0f},{:.0f},{:.0f}) radius {:.0f} fade {:.2f} -> {} (camera state: {} rooms{}; light state: {} rooms{})",
					e.x, e.y, e.z, e.radius, e.fade, kFateNames[static_cast<std::size_t>(e.fate)],
					e.mainRooms, e.mainFree ? ", unrestricted" : "", e.lightRooms, e.lightFree ? ", unrestricted" : "");
			}
			g_loopFrames = 0;
			g_loopLights = 0;
			g_dumpCount = 0;
		}
	}
}
