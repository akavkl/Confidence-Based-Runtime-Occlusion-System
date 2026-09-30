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
		const auto frustumVtable = CBRO::Engine::OG(kFrustumLightVtableID).address();
		g_originalFrustumCull = Util::WriteVFuncSwitchable(g_frustumHook, frustumVtable, kFrustumCullSlot, Util::FnAddr(&FrustumCullThunk), "shadowlights:BSShadowFrustumLight::cull");
		if (!g_originalFrustumCull) {
			logger::error("shadow lights: spot-light hook failed; spot lights are not recorded (group 0 keeps every caster with the sun off)");
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
		for (auto* hook : { &g_cameraHook, &g_objectHook, &g_frustumHook }) {
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
	}
}
