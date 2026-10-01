#include "Core/ShadowLights.h"

#include "Core/Occlusion.h"
#include "Hooks/CullGroups.h"
#include "Hooks/PrevisFeed.h"
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
		// traversal (every object reaches it: NiAVObject::Cull calls it through the vtable for any object without flag
		// bit 0, and NiNode::OnVisible culls each child). An object that passes calls its OnVisible (vtable +0x1C8),
		// which files geometry and walks a node's children; one that fails is dropped with its subtree, and if the
		// culler's updateAccumulateFlag (+0x11D) is set its kFlagAccumulated (flags bit 42) is cleared. Decompiled
		// (2026-10-01): cullMode kAllPass/kAllFail (1/2) pass/fail everything; in every other mode an object with
		// kFlagPreProcessedNode (bit 26) is decided by kFlagNotVisible (bit 39) unless the culler's byte +0x11E is set,
		// one with kFlagAlwaysDraw (bit 11) passes untested, and the rest get the sphere test (DoParabolicCulling, with
		// the culler's compound frustum +0x188 too unless the mode is 3). CBRO's test runs exactly where the engine's
		// sphere test runs (v1.52 tested modes 0 and 3 only: in the v1.52 interior run whole intervals of lights got no
		// test at all). (Names: CommonLibF4 BSCullingProcess / NiCullingProcess, F4SE NiObjects.h.)
		constexpr std::size_t   kProcessObjectSlot = 0x19;
		constexpr std::size_t   kTrackSeenOffset = 0x11D;
		constexpr std::size_t   kTestPreProcessedOffset = 0x11E;
		constexpr std::uint32_t kAllPass = 1;
		constexpr std::uint64_t kSeenFlag = std::uint64_t{ 1 } << 42;
		constexpr std::uint64_t kAlwaysDrawFlag = std::uint64_t{ 1 } << 11;
		constexpr std::uint64_t kPreProcessedFlag = std::uint64_t{ 1 } << 26;
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

		// Spot lights: BSShadowFrustumLight (vtable id 67506). v1.47 records spot lights from its slot 14 Update(camera)
		// (integer arguments, bool return; hooked by the lamp diagnostic since v1.43). Its slot 9 cull (1559482) takes
		// (light, group) in rcx/rdx only (the v1.37 reading of XMM arguments was wrong); v1.52 hooks it too (the lamp window).
		// The light's NiLight is at +0xB8 (BSLight), as for the parabolic light; NiLight::spec.r (+0x138) is the radius.
		constexpr std::uint64_t kFrustumLightVtableID = 67506;
		constexpr std::size_t   kLightNiLightOffset = 0xB8;
		constexpr std::size_t   kLightFaces = 0x198;  // the shadow faces record (one face for a spot light)

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

		// A spot light whose Update(camera) passed (it goes on to its shadow map): recorded for the lamp trimming.
		void RecordSpotLight(std::uintptr_t a_light) noexcept
		{
			RE::NiPoint3 position{};
			float        reach = 0.0f;
			if (!ReadSpotLight(a_light, position, reach)) {
				return;
			}
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
			std::uint32_t        seen{ 0 };      // objects offered to the light's per-object test (counted here, one thread)
			std::uint32_t        tests{ 0 };     // ... tested (summed once per light)
			std::uint32_t        culled{ 0 };    // outside the pushed view cone
			std::uint32_t        volumeCulled{ 0 };  // shadow volume outside the view or missing every visible surface
			std::uint32_t        skippedMode{ 0 };         // the culler passes or fails everything (cull mode 1/2): no test
			std::uint32_t        skippedAlwaysDraw{ 0 };   // flag bit 11: the engine passes it untested
			std::uint32_t        skippedPreProcessed{ 0 }; // flag bit 26 with the culler's +0x11E clear: bit 39 decides
			std::uint32_t        outsideReach{ 0 };        // outside the light's culling sphere: the engine's own test fails it
		};
		thread_local Region t_region{};
		bool                g_volumes{ true };  // bLampShadowVolumes

		std::atomic<std::uint64_t> g_regionLights{ 0 };
		std::atomic<std::uint64_t> g_regionInside{ 0 };  // lights inside the view cone
		std::atomic<std::uint64_t> g_regionModes[4]{};   // ... by the culler's mode: 0, 3, 4, any other
		std::atomic<std::uint64_t> g_casterSeen{ 0 };
		std::atomic<std::uint64_t> g_casterTests{ 0 };
		std::atomic<std::uint64_t> g_casterCulled{ 0 };
		std::atomic<std::uint64_t> g_casterVolumeCulled{ 0 };
		std::atomic<std::uint64_t> g_casterSkippedMode{ 0 };
		std::atomic<std::uint64_t> g_casterSkippedAlwaysDraw{ 0 };
		std::atomic<std::uint64_t> g_casterSkippedPreProcessed{ 0 };
		std::atomic<std::uint64_t> g_casterOutsideReach{ 0 };
		// The point lights' culls (Process(camera) with the engine's traversal inside), timed (v1.54): limited to the view
		// (the per-object test runs), emptied (all-fail), and the rest (left to the engine).
		std::atomic<std::int64_t>  g_cullTicks[3]{};
		std::atomic<std::uint64_t> g_cullCount[3]{};

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

		// Why lights were kept though not seen, and how many were judged hidden through a thin view overhang (for the log).
		struct ReasonCounts
		{
			std::atomic<std::uint64_t> overhang{ 0 };
			std::atomic<std::uint64_t> nearCamera{ 0 };
			std::atomic<std::uint64_t> edge{ 0 };
			std::atomic<std::uint64_t> noDepth{ 0 };
			std::atomic<std::uint64_t> invalid{ 0 };
			std::atomic<std::uint64_t> emptyReach{ 0 };

			void Count(Occlusion::SphereReason a_reason) noexcept
			{
				switch (a_reason) {
				case Occlusion::SphereReason::kOverhang:
					overhang.fetch_add(1, std::memory_order_relaxed);
					break;
				case Occlusion::SphereReason::kEmptyReach:
					emptyReach.fetch_add(1, std::memory_order_relaxed);
					break;
				case Occlusion::SphereReason::kNear:
					nearCamera.fetch_add(1, std::memory_order_relaxed);
					break;
				case Occlusion::SphereReason::kEdge:
					edge.fetch_add(1, std::memory_order_relaxed);
					break;
				case Occlusion::SphereReason::kNoDepth:
					noDepth.fetch_add(1, std::memory_order_relaxed);
					break;
				case Occlusion::SphereReason::kInvalid:
					invalid.fetch_add(1, std::memory_order_relaxed);
					break;
				default:
					break;
				}
			}

			std::string Take(double a_frames) noexcept
			{
				const auto per = [&](std::atomic<std::uint64_t>& a_counter) { return static_cast<double>(a_counter.exchange(0)) / a_frames; };
				const double nearCameraPer = per(nearCamera), edgePer = per(edge), noDepthPer = per(noDepth), invalidPer = per(invalid);
				const double overhangPer = per(overhang);
				return std::format(
					" [reaches the camera {:.2f}, view overhang too wide {:.2f}, no depth {:.2f}, bad bound {:.2f}] | hidden through a thin view overhang {:.2f} | hidden with no visible surface in reach (nothing drawn there, or surfaces beyond it) {:.2f}",
					nearCameraPer, edgePer, noDepthPer, invalidPer, overhangPer, per(emptyReach));
			}
		};
		ReasonCounts g_pointReasons;
		ReasonCounts g_spotReasons;

		std::uint64_t LightKey(const RE::NiPoint3& a_position, float a_reach) noexcept
		{
			const auto q = [](float a_v) { return static_cast<std::uint64_t>(static_cast<std::int64_t>(std::floor(a_v / 8.0f)) & 0xFFFF); };
			const auto key = q(a_position.x) | (q(a_position.y) << 16) | (q(a_position.z) << 32) | (q(a_reach) << 48);
			return key ? key : 1;
		}

		// Slots are never freed: over a session the table fills with lights left behind. A full neighbourhood
		// recycles the slot whose light went longest without a test (v1.45 and earlier took the first slot, which
		// two live lights sharing it took from each other every frame: neither was ever confirmed).
		Slot* FindSlot(std::uint64_t a_key, std::uint32_t a_clock) noexcept
		{
			const auto    start = static_cast<std::size_t>((a_key * 0x9E3779B97F4A7C15ull) >> 56) % kSlots;
			Slot*         stalest = nullptr;
			std::uint32_t stalestAge = 0;
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
				const auto age = a_clock - slot.lastClock.load(std::memory_order_relaxed);
				if (!stalest || age > stalestAge) {
					stalest = &slot;
					stalestAge = age;
				}
			}
			// A fresh, conservative streak, reset before the key names the new light (a reader finding the key
			// never sees the old light's streak).
			stalest->streak.store(0, std::memory_order_relaxed);
			stalest->lastClock.store(0, std::memory_order_relaxed);
			stalest->key.store(a_key, std::memory_order_release);
			return stalest;
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

			auto       reason = Occlusion::SphereReason::kNone;
			const auto verdict = Occlusion::TestSphere(position, reach, &reason);
			g_pointReasons.Count(reason);
			const auto clock = Occlusion::Clock();
			auto&      slot = *FindSlot(LightKey(position, reach), clock);
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
				if (streak >= g_confirmFrames) {
					Occlusion::NoteLightFlip(false, position, reach, verdict, reason);  // (diagnostic: an emptied map drawn again)
				}
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

		// ---- spot lights: an empty shadow map when the lit volume is hidden (v1.48) -------------------------------------
		// BSShadowFrustumLight::Update(camera) (slot 14, 0x1428D9E40) prepares the frame's shadow camera (faces +0x40: the
		// NiLight's world transform; frustum far = light +0x204 if positive, else NiLight::spec.r) and the shadow map's
		// accumulator (faces +0x48, render mode 15), then decides visibility (0x14285DA50) by overlapping the shadow
		// camera's pyramid (far plane at far along rotation row 0, half extents far x top along row 1 and far x right
		// along row 2) with the main camera's: the engine takes that pyramid as the lit volume. When its bounding sphere
		// is hidden behind the depth, no visible pixel is lit, so every caster the light's cull then files into the
		// accumulator is dropped (Hooks::CullGroups): the shadow map draws nothing; the light pass is left alone.
		constexpr std::size_t   kFaceShadowCamera = 0x40;
		constexpr std::size_t   kFaceAccumulator = 0x48;
		constexpr std::size_t   kLightShadowFar = 0x204;
		constexpr std::size_t   kCameraRotate = 0x70;     // NiAVObject world rotation (row 0: the view direction)
		constexpr std::size_t   kCameraTranslate = 0xA0;  // ... world translate
		constexpr std::size_t   kCameraFrustum = 0x160;   // NiFrustum: left, right, top, bottom, near, far, ortho (+0x18)
		constexpr std::uint64_t kSpotKeySalt = 0x8000'0000'0000'0000ull;  // spot and point streaks keep separate slots

		bool g_spotCulling{ true };  // bSpotShadowCulling
		bool g_lampStage{ false };   // the main frame's deferred-lights stage is running (CBRO frame)

		// The spot light whose Update(camera) ran last in the lamp loop (its cull, slot 9, comes next) and whether its
		// shadow map is emptied (main thread).
		std::uintptr_t g_lastSpotLight{ 0 };
		bool           g_lastSpotEmptied{ false };

		std::atomic<std::uint64_t> g_spotTests{ 0 };
		std::atomic<std::uint64_t> g_spotEmptied{ 0 };
		std::atomic<std::uint64_t> g_spotWouldEmpty{ 0 };
		std::atomic<std::uint64_t> g_spotConfirming{ 0 };
		std::atomic<std::uint64_t> g_spotVisible{ 0 };
		std::atomic<std::uint64_t> g_spotUnknown{ 0 };
		std::atomic<std::uint64_t> g_spotHidden{ 0 };
		std::atomic<std::uint64_t> g_spotOutOfView{ 0 };
		std::atomic<std::uint64_t> g_spotUnreadable{ 0 };
		std::atomic<std::uint64_t> g_spotByFrustum{ 0 };
		std::atomic<int>           g_spotVolumesLogged{ 0 };

		// The spot light's lit volume as a sphere (around the whole reach's part inside the shadow frustum) and its
		// shadow map's accumulator, read after the engine's Update(camera) passed (both are this frame's).
		bool ReadSpotVolume(std::uintptr_t a_light, const RE::NiPoint3& a_position, float a_reach, RE::NiBound& a_volume, bool& a_byFrustum, const void*& a_accumulator) noexcept
		{
			__try {
				const auto faces = *reinterpret_cast<const std::uintptr_t*>(a_light + kLightFaces);
				const auto camera = faces ? *reinterpret_cast<const std::uintptr_t*>(faces + kFaceShadowCamera) : 0;
				a_accumulator = faces ? *reinterpret_cast<const void* const*>(faces + kFaceAccumulator) : nullptr;
				if (!camera || !a_accumulator) {
					return false;
				}
				const auto  finite = [](float a_value) { return std::isfinite(a_value) ? a_value : 0.0f; };
				const auto* frustum = reinterpret_cast<const float*>(camera + kCameraFrustum);
				// Every lit point lies within this of the light: its radius, bounds, shadow far and frustum far.
				const float length = std::max({ a_reach, finite(*reinterpret_cast<const float*>(a_light + kLightShadowFar)), finite(frustum[5]) });
				a_volume.center = a_position;
				a_volume.fRadius = length;
				a_byFrustum = false;

				const bool  ortho = *reinterpret_cast<const std::uint8_t*>(camera + kCameraFrustum + 0x18) != 0;
				const auto* row0 = reinterpret_cast<const float*>(camera + kCameraRotate);
				const auto* apex = reinterpret_cast<const float*>(camera + kCameraTranslate);
				const float tanX = std::max(std::abs(frustum[0]), std::abs(frustum[1]));
				const float tanY = std::max(std::abs(frustum[2]), std::abs(frustum[3]));
				const float axis = row0[0] * row0[0] + row0[1] * row0[1] + row0[2] * row0[2];
				const float dx = apex[0] - a_position.x, dy = apex[1] - a_position.y, dz = apex[2] - a_position.z;
				// Only a perspective frustum with a unit axis, at the light (the camera follows the NiLight).
				if (ortho || !std::isfinite(tanX) || !std::isfinite(tanY) || !(tanX > 0.0f) || !(tanY > 0.0f) || !(std::abs(axis - 1.0f) < 0.01f) ||
					!(dx * dx + dy * dy + dz * dz < 64.0f)) {
					return true;
				}
				float along = 0.0f, radius = 0.0f;
				ShadowGeometry::SpotSphere(length, std::sqrt(tanX * tanX + tanY * tanY), along, radius);
				if (std::isfinite(radius) && radius < a_volume.fRadius) {
					a_volume.center = RE::NiPoint3{ apex[0] + row0[0] * along, apex[1] + row0[1] * along, apex[2] + row0[2] * along };
					a_volume.fRadius = radius;
					a_byFrustum = true;
				}
				return true;
			} __except (EXCEPTION_EXECUTE_HANDLER) {
				return false;
			}
		}

		// Whether the spot light's shadow map can stay empty this frame (its lit volume hidden for confirmFrames frames),
		// with the accumulator its casters go to. Main thread (the lamp loop).
		bool ShouldEmptySpot(std::uintptr_t a_light, const void*& a_accumulator) noexcept
		{
			g_spotTests.fetch_add(1, std::memory_order_relaxed);
			RE::NiPoint3 position{};
			float        reach = 0.0f;
			RE::NiBound  volume{};
			bool         byFrustum = false;
			if (!ReadSpotLight(a_light, position, reach) || !ReadSpotVolume(a_light, position, reach, volume, byFrustum, a_accumulator)) {
				g_spotUnreadable.fetch_add(1, std::memory_order_relaxed);
				return false;
			}
			if (byFrustum) {
				g_spotByFrustum.fetch_add(1, std::memory_order_relaxed);
			}
			auto       reason = Occlusion::SphereReason::kNone;
			const auto verdict = Occlusion::TestSphere(volume.center, volume.fRadius, &reason);
			g_spotReasons.Count(reason);
			if (g_spotVolumesLogged.load(std::memory_order_relaxed) < 8 && g_spotVolumesLogged.fetch_add(1) < 8) {
				constexpr const char* kVerdicts[]{ "unknown", "visible", "hidden", "out of view" };
				logger::info(
					"shadow lights: spot light at ({:.0f},{:.0f},{:.0f}) reach {:.0f} | lit volume ({}) sphere at ({:.0f},{:.0f},{:.0f}) radius {:.0f} -> {}",
					position.x, position.y, position.z, reach, byFrustum ? "shadow frustum" : "whole reach", volume.center.x, volume.center.y, volume.center.z,
					volume.fRadius, kVerdicts[static_cast<int>(verdict)]);
			}
			const auto clock = Occlusion::Clock();
			auto&      slot = *FindSlot(LightKey(position, reach) ^ kSpotKeySalt, clock);
			auto       streak = slot.streak.load(std::memory_order_relaxed);
			// Behind visible surfaces or out of view entirely: either way no visible pixel is lit.
			if (verdict == Occlusion::SphereVerdict::kHidden || verdict == Occlusion::SphereVerdict::kOutOfView) {
				(verdict == Occlusion::SphereVerdict::kHidden ? g_spotHidden : g_spotOutOfView).fetch_add(1, std::memory_order_relaxed);
				if (slot.lastClock.exchange(clock, std::memory_order_relaxed) != clock) {
					streak = std::min(streak + 1, 255u);
					slot.streak.store(streak, std::memory_order_relaxed);
				}
			} else {
				if (streak >= g_confirmFrames) {
					Occlusion::NoteLightFlip(true, volume.center, volume.fRadius, verdict, reason);  // (diagnostic)
				}
				slot.streak.store(0, std::memory_order_relaxed);
				slot.lastClock.store(clock, std::memory_order_relaxed);
				(verdict == Occlusion::SphereVerdict::kVisible ? g_spotVisible : g_spotUnknown).fetch_add(1, std::memory_order_relaxed);
				return false;
			}
			if (streak < g_confirmFrames) {
				g_spotConfirming.fetch_add(1, std::memory_order_relaxed);
				return false;
			}
			if (Occlusion::Deciding()) {
				g_spotWouldEmpty.fetch_add(1, std::memory_order_relaxed);
				return false;
			}
			g_spotEmptied.fetch_add(1, std::memory_order_relaxed);
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

		enum class CasterFate : std::uint8_t
		{
			kKept,
			kOutsideCone,  // outside the view cone pushed out to hold the light (plus the filter's reach)
			kVolume,       // its shadow volume lies outside the view or misses every visible surface (confirmed)
		};

		// The per-caster test, shared by the point lights' traversal (per object) and the spot lights' registrations (per
		// geometry): the light and every visible point lie in the pushed view cone, so each light-to-pixel segment does
		// too, and a caster outside it meets none of them; then the caster's shadow volume against the depth
		// (Occlusion::TestLampCaster). a_volumeRan/a_verdict: the volume test's verdict, when it ran.
		CasterFate JudgeCaster(const ShadowGeometry::Cone& a_cone, float a_push, const RE::NiPoint3& a_lamp, float a_reach, RE::NiAVObject* a_object, const RE::NiBound& a_bound,
			bool& a_volumeRan, Occlusion::LampVerdict& a_verdict) noexcept
		{
			const float center[3]{ a_bound.center.x, a_bound.center.y, a_bound.center.z };
			if (!(a_bound.fRadius > 0.0f) || !(a_bound.fRadius < 1.0e6f) || !std::isfinite(center[0]) || !std::isfinite(center[1]) || !std::isfinite(center[2])) {
				return CasterFate::kKept;
			}
			if (ShadowGeometry::CasterOutside(a_cone, a_push, center, a_bound.fRadius)) {
				return CasterFate::kOutsideCone;
			}
			if (!g_volumes) {
				return CasterFate::kKept;
			}
			a_volumeRan = true;
			a_verdict = Occlusion::TestLampCaster(a_object, a_lamp, a_reach, a_bound);
			return a_verdict == Occlusion::LampVerdict::kOutside || a_verdict == Occlusion::LampVerdict::kMisses ? CasterFate::kVolume : CasterFate::kKept;
		}

		// The engine's first test of a lamp caster (TestParabolicCulling 0x142977950, every branch of slot 0x19 runs it):
		// the object's sphere must meet the light's culling sphere (centre culler +0x1C0, radius +0x1CC), else it fails
		// whatever else holds (`sqrt(d^2) - r - reach < 0` passes; NaN fails there and here).
		bool SphereMeetsLight(const RE::NiBound& a_bound, const RE::NiPoint3& a_light, float a_reach) noexcept
		{
			const float dx = a_bound.center.x - a_light.x;
			const float dy = a_bound.center.y - a_light.y;
			const float dz = a_bound.center.z - a_light.z;
			const float limit = a_bound.fRadius + a_reach;
			return limit > 0.0f && dx * dx + dy * dy + dz * dz < limit * limit;
		}

		// ---- spot lights: casters that can't shadow a visible pixel are left out (v1.53, bSpotCasterTrim) -----------
		// A kept spot light's casters reach its shadow map through RegisterObject (its group pass files them), so Hooks::
		// CullGroups filters that one accumulator during the light's cull: each geometry gets JudgeCaster with the light
		// taken as a point light at its shadow camera's apex with the reach its lit-volume test uses (a spot light lights
		// a part of what that point light would: the test stays conservative). Only the lamp loop's own culls (the light
		// passed Update(camera) right before; never the focus shadows' direct lists). Main thread.
		struct SpotRegion
		{
			bool                 active{ false };
			ShadowGeometry::Cone cone{};
			float                push{ 0.0f };
			RE::NiPoint3         lamp{};
			float                reach{ 0.0f };
		};
		SpotRegion g_spotRegion;
		bool       g_spotTrim{ true };  // bSpotCasterTrim

		struct SpotCasterStats  // main thread
		{
			std::uint64_t lights{ 0 };            // kept culls whose casters were judged
			std::uint64_t noRegion{ 0 };          // kept culls left whole: no view cone, or the shadow camera isn't at the light
			std::uint64_t offered{ 0 };
			std::uint64_t outsideCone{ 0 };
			std::uint64_t volume{ 0 };
			std::uint64_t volumeConfirming{ 0 };  // kept: the volume misses, not yet for confirmFrames frames
			std::uint64_t alwaysDraw{ 0 };        // kept untested (flag bit 11: the bound may not be maintained)
			std::uint64_t unreadable{ 0 };
		};
		SpotCasterStats g_spotCasterStats;

		// The spot light's shadow camera must sit at the light (its apex is where the map is rendered from); the length
		// is what ReadSpotVolume takes for the light's reach (radius, shadow far, frustum far).
		bool ReadSpotOrigin(std::uintptr_t a_light, const RE::NiPoint3& a_position, float a_reach, float& a_length) noexcept
		{
			__try {
				const auto faces = *reinterpret_cast<const std::uintptr_t*>(a_light + kLightFaces);
				const auto camera = faces ? *reinterpret_cast<const std::uintptr_t*>(faces + kFaceShadowCamera) : 0;
				if (!camera || *reinterpret_cast<const std::uint8_t*>(camera + kCameraFrustum + 0x18) != 0) {
					return false;  // (no camera, or an orthographic one: not a point source)
				}
				const auto* apex = reinterpret_cast<const float*>(camera + kCameraTranslate);
				const float dx = apex[0] - a_position.x, dy = apex[1] - a_position.y, dz = apex[2] - a_position.z;
				if (!(dx * dx + dy * dy + dz * dz < 64.0f)) {
					return false;
				}
				const auto  finite = [](float a_value) { return std::isfinite(a_value) ? a_value : 0.0f; };
				const auto* frustum = reinterpret_cast<const float*>(camera + kCameraFrustum);
				a_length = std::max({ a_reach, finite(*reinterpret_cast<const float*>(a_light + kLightShadowFar)), finite(frustum[5]) });
				return a_length >= 16.0f && a_length <= 1.0e6f;
			} __except (EXCEPTION_EXECUTE_HANDLER) {
				return false;
			}
		}

		bool MakeSpotRegion(std::uintptr_t a_light, SpotRegion& a_region) noexcept
		{
			RE::NiPoint3 position{};
			float        reach = 0.0f;
			float        length = 0.0f;
			if (!ReadSpotLight(a_light, position, reach) || !ReadSpotOrigin(a_light, position, reach, length)) {
				return false;
			}
			float viewPush = 0.0f;
			if (!Occlusion::ViewCone(a_region.cone, viewPush) || Occlusion::Deciding()) {
				return false;
			}
			const float center[3]{ position.x, position.y, position.z };
			a_region.push = ShadowGeometry::OutsideBy(a_region.cone, center) + viewPush + length * kFilterSlope + kFilterBase;
			a_region.lamp = position;
			a_region.reach = length;
			return true;
		}

		bool ReadCasterBound(const RE::NiAVObject* a_object, RE::NiBound& a_bound, std::uint64_t& a_flags) noexcept
		{
			__try {
				a_bound = a_object->worldBound;
				a_flags = a_object->flags.flags;
				return true;
			} __except (EXCEPTION_EXECUTE_HANDLER) {
				return false;
			}
		}

		bool SpotCasterFilter(const RE::NiAVObject* a_object)
		{
			if (!g_spotRegion.active || !a_object) {
				return false;
			}
			auto& stats = g_spotCasterStats;
			++stats.offered;
			RE::NiBound   bound{};
			std::uint64_t flags = 0;
			if (!ReadCasterBound(a_object, bound, flags)) {
				++stats.unreadable;
				return false;
			}
			if (flags & kAlwaysDrawFlag) {
				++stats.alwaysDraw;
				return false;
			}
			bool       volumeRan = false;
			auto       verdict = Occlusion::LampVerdict::kUnknown;
			const auto fate = JudgeCaster(g_spotRegion.cone, g_spotRegion.push, g_spotRegion.lamp, g_spotRegion.reach, const_cast<RE::NiAVObject*>(a_object), bound, volumeRan, verdict);
			switch (fate) {
			case CasterFate::kOutsideCone:
				++stats.outsideCone;
				return true;
			case CasterFate::kVolume:
				++stats.volume;
				return true;
			default:
				if (volumeRan && verdict == Occlusion::LampVerdict::kConfirming) {
					++stats.volumeConfirming;
				}
				return false;
			}
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
			const auto timedCull = [&](std::size_t a_kind) {
				LARGE_INTEGER start{}, end{};
				QueryPerformanceCounter(&start);
				reinterpret_cast<ProcessFn>(g_original)(a_self, a_camera, a_scene, a_visibleSet);
				QueryPerformanceCounter(&end);
				g_cullTicks[a_kind].fetch_add(end.QuadPart - start.QuadPart, std::memory_order_relaxed);
				g_cullCount[a_kind].fetch_add(1, std::memory_order_relaxed);
			};
			if (a_camera && a_scene && ShouldEmpty(a_self, a_camera)) {
				auto&      cullMode = *reinterpret_cast<std::uint32_t*>(static_cast<std::byte*>(a_self) + kCullModeOffset);
				const auto saved = cullMode;
				cullMode = kAllFail;
				timedCull(1);
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
				const auto mode = *reinterpret_cast<const std::uint32_t*>(static_cast<const std::byte*>(a_self) + kCullModeOffset);
				g_regionModes[mode == 0 ? 0 : mode == 3 ? 1 : mode == 4 ? 2 : 3].fetch_add(1, std::memory_order_relaxed);
			}
			timedCull(active ? 0 : 2);
			if (active) {
				g_casterSeen.fetch_add(t_region.seen, std::memory_order_relaxed);
				g_casterTests.fetch_add(t_region.tests, std::memory_order_relaxed);
				g_casterCulled.fetch_add(t_region.culled, std::memory_order_relaxed);
				g_casterVolumeCulled.fetch_add(t_region.volumeCulled, std::memory_order_relaxed);
				g_casterSkippedMode.fetch_add(t_region.skippedMode, std::memory_order_relaxed);
				g_casterSkippedAlwaysDraw.fetch_add(t_region.skippedAlwaysDraw, std::memory_order_relaxed);
				g_casterSkippedPreProcessed.fetch_add(t_region.skippedPreProcessed, std::memory_order_relaxed);
				g_casterOutsideReach.fetch_add(t_region.outsideReach, std::memory_order_relaxed);
			}
			t_region = saved;
		}

		void ProcessObject(void* a_self, RE::NiAVObject* a_object)
		{
			if (t_region.culler == a_self && a_object) {
				const auto base = static_cast<std::byte*>(a_self);
				const auto mode = *reinterpret_cast<const std::uint32_t*>(base + kCullModeOffset);
				const auto flags = a_object->GetFlags();
				++t_region.seen;
				// Only where the engine's own sphere test runs (see kProcessObjectSlot): its bound is then one the engine
				// itself trusts for this light.
				if (mode == kAllPass || mode == kAllFail) {
					++t_region.skippedMode;
				} else if (flags & kAlwaysDrawFlag) {
					++t_region.skippedAlwaysDraw;
				} else if ((flags & kPreProcessedFlag) && !*reinterpret_cast<const std::uint8_t*>(base + kTestPreProcessedOffset)) {
					++t_region.skippedPreProcessed;
				} else if (const auto& bound = a_object->worldBound; !SphereMeetsLight(bound, t_region.lamp, t_region.reach)) {
					// The engine's own first test (TestParabolicCulling) fails it: left to the engine, uncounted as CBRO's
					// (v1.53 ran its own test first, on every object of the scene graph, and counted these as its drops).
					++t_region.outsideReach;
				} else {
					++t_region.tests;
					bool       volumeRan = false;
					auto       verdict = Occlusion::LampVerdict::kUnknown;
					const auto fate = JudgeCaster(t_region.cone, t_region.push, t_region.lamp, t_region.reach, a_object, a_object->worldBound, volumeRan, verdict);
					if (volumeRan) {
						Occlusion::CountLampVerdict(verdict);
					}
					if (fate != CasterFate::kKept) {
						++(fate == CasterFate::kOutsideCone ? t_region.culled : t_region.volumeCulled);
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
			const void* dropped = nullptr;
			if (result && Occlusion::Active()) {
				RecordSpotLight(a_light);  // (emptied or not: the lamp list stays every lamp the engine lights)
				// Only in the main frame's lamp loop: another view's render would be judged against the wrong depth.
				const void* accumulator = nullptr;
				if (g_spotCulling && g_lampStage && ShouldEmptySpot(a_light, accumulator)) {
					dropped = accumulator;
				}
			}
			// This light's cull comes next in the loop; the next light's Update (or the stage's end) resets it.
			Hooks::CullGroups::SetDroppedAccumulator(dropped);
			if (g_lampStage) {
				g_lastSpotLight = a_light;
				g_lastSpotEmptied = dropped != nullptr;
			}
			return result;
		}

		// ---- spot lights: the shadow-map cull and the lamp window (v1.52) -------------------------------------------
		// BSShadowFrustumLight slot 9 (1559482) takes (light, group) in rcx/rdx only (FO4-ENGINE-NOTES 6.2a; the v1.37
		// reading of XMM arguments was wrong). With the light's direct list (+0x208) set it files that list into the
		// shadow map's accumulator (faces +0x48); else 998671(light, faces, group) culls the light's own object list
		// (+0x1B8, count +0x1C8) in a stack group, or, with that list empty, re-aims the passed group (group 0 in the lamp
		// loop; focus shadows pass none) at the spot's camera, processes it (1147875) and registers it (626862). With
		// previs active the process skips the frustum test of every block whose +0x3A6F byte is clear, and group 0's
		// blocks have it clear (DrawWorld's cull filed them inside CBRO's cull window); Block::Add then marks the children
		// it pushes force-visible, and the register loop (962984, no camera re-test) files whatever the result bytes hold:
		// the last sun cascade's pass, or the main view's. So a kept spot light's group-0 cull in a CBRO frame runs inside
		// a lamp window (Hooks::PrevisFeed: previs suspended, as around the walk and the cascades). An emptied light needs
		// none: its registrations are dropped anyway. Every cull here is counted, timed and its casters counted.
		constexpr std::size_t kCullSlot = 9;
		constexpr std::size_t kLightDirectList = 0x208;
		constexpr std::size_t kLightObjectCount = 0x1C8;

		enum class SpotPath : std::uint8_t
		{
			kDirect,      // the light's direct list (+0x208): filed as is
			kOwnList,     // the light's own object list (+0x1C8 entries): culled in a stack group
			kGroup,       // the passed group (group 0): re-aimed at the spot's camera
			kNone,        // no list and no group: nothing is filed
			kUnreadable,
			kCount
		};
		constexpr const char* kSpotPathNames[]{ "direct list", "own list", "group 0", "none", "unreadable" };

		using CullFn = std::uintptr_t (*)(std::uintptr_t, std::uintptr_t, std::uintptr_t, std::uintptr_t);
		Util::SwitchableHook g_spotCullHook;
		std::uintptr_t       g_originalSpotCull{ 0 };
		bool                 g_spotWindow{ true };  // bSpotCullWindow
		double               g_qpcPerMs{ 0.0 };

		struct SpotCullStats  // main thread (the lamp loop)
		{
			std::uint64_t paths[static_cast<std::size_t>(SpotPath::kCount)]{};
			std::uint64_t groupActive{ 0 };     // group-0 culls with previs active at entry
			std::uint64_t windowed{ 0 };        // ... of them inside a lamp window
			std::uint64_t groupSunUp{ 0 };      // group-0 culls of kept lights with the sun's cascades on: group 0 then lacks what
			                                    // Occlusion rejected for the main view and the sun alone (spot shadows not judged)
			std::uint64_t filedWindowed{ 0 };   // casters offered to the shadow map by windowed culls
			std::uint64_t staleCulls{ 0 };      // kept lights' group-0 culls with previs active and no window (bSpotCullWindow=0)
			std::uint64_t filedStale{ 0 };
			std::uint64_t previsOffCulls{ 0 };  // kept lights' group-0 culls with previs inactive (v1.28's switch, interiors)
			std::uint64_t filedPrevisOff{ 0 };
			std::uint64_t listCulls{ 0 };       // kept lights' direct-list and own-list culls
			std::uint64_t filedLists{ 0 };
			std::uint64_t skippedPasses{ 0 };   // emptied lights' group-0 culls whose process and register were skipped
			std::uint64_t emptiedCulls{ 0 };
			std::int64_t  emptiedTicks{ 0 };
			std::uint64_t keptCulls{ 0 };
			std::int64_t  keptTicks{ 0 };
		};
		SpotCullStats g_spotCullStats;

		// An emptied light's group pass (v1.53): its registrations are all dropped (the shadow map draws nothing), so the
		// pass itself is skipped. 998671's group-0 path calls the process (1147875) at +0x249 and the register (626862) at
		// +0x255 (`E8 rel32`, FO4-ENGINE-NOTES 6.2a); both sites call these wrappers, which pass through unless the light
		// whose cull is running (SpotCullThunk, main thread) is emptied. The rest of 998671 (the accumulator's camera,
		// SetAccumulator, SetCamera on the group) runs as before; the next light's pass re-processes group 0 from scratch
		// (its +0x168 byte cleared), and nothing after the lamp loop reads group 0's results but its cleanup.
		constexpr std::uint64_t kSpotGroupPassID = 998671;
		constexpr std::uint64_t kGroupProcessID = 1147875;
		constexpr std::uint64_t kGroupRegisterID = 626862;
		constexpr std::size_t   kSpotProcessSiteOffset = 0x249;
		constexpr std::size_t   kSpotRegisterSiteOffset = 0x255;
		using GroupCallFn = std::uintptr_t (*)(std::uintptr_t, std::uintptr_t, std::uintptr_t, std::uintptr_t);
		std::uintptr_t g_spotProcessPrevious{ 0 };
		std::uintptr_t g_spotRegisterPrevious{ 0 };
		bool           g_skipSpotGroup{ false };  // the running cull is an emptied light's group-0 pass (main thread)

		std::uintptr_t SpotProcessThunk(std::uintptr_t a_1, std::uintptr_t a_2, std::uintptr_t a_3, std::uintptr_t a_4)
		{
			if (g_skipSpotGroup) {
				return 0;  // (the callers ignore the result)
			}
			return reinterpret_cast<GroupCallFn>(g_spotProcessPrevious)(a_1, a_2, a_3, a_4);
		}

		std::uintptr_t SpotRegisterThunk(std::uintptr_t a_1, std::uintptr_t a_2, std::uintptr_t a_3, std::uintptr_t a_4)
		{
			if (g_skipSpotGroup) {
				return 0;
			}
			return reinterpret_cast<GroupCallFn>(g_spotRegisterPrevious)(a_1, a_2, a_3, a_4);
		}

		// Wraps one of the two sites if it still calls the engine's function (another plugin's wrapper there is chained:
		// WriteCall5 returns whatever the site called).
		std::uintptr_t WrapSpotSite(std::size_t a_offset, std::uint64_t a_targetID, std::uintptr_t a_thunk, const char* a_name)
		{
			const auto site = CBRO::Engine::OG(kSpotGroupPassID).address() + a_offset;
			const auto current = Util::ReadCall5Target(site);
			if (!current) {
				logger::error("shadow lights: {} site {} is not a call rel32; emptied spot lights keep their group pass", a_name, Util::DescribeCodeAddress(site));
				return 0;
			}
			if (current != CBRO::Engine::OG(a_targetID).address()) {
				logger::info("shadow lights: {} site {} calls {} (not the engine's function directly); chained", a_name, Util::DescribeCodeAddress(site), Util::DescribeCodeAddress(current));
			}
			return Util::WriteCall5(site, a_thunk, a_name);
		}

		SpotPath ReadSpotPath(std::uintptr_t a_light, std::uintptr_t a_group, const void*& a_accumulator) noexcept
		{
			__try {
				const auto faces = *reinterpret_cast<const std::uintptr_t*>(a_light + kLightFaces);
				a_accumulator = faces ? *reinterpret_cast<const void* const*>(faces + kFaceAccumulator) : nullptr;
				if (*reinterpret_cast<const std::uintptr_t*>(a_light + kLightDirectList)) {
					return SpotPath::kDirect;
				}
				if (*reinterpret_cast<const std::uint32_t*>(a_light + kLightObjectCount)) {
					return SpotPath::kOwnList;
				}
				return a_group ? SpotPath::kGroup : SpotPath::kNone;
			} __except (EXCEPTION_EXECUTE_HANDLER) {
				return SpotPath::kUnreadable;
			}
		}

		std::uintptr_t SpotCullThunk(std::uintptr_t a_light, std::uintptr_t a_group, std::uintptr_t a_3, std::uintptr_t a_4)
		{
			const auto original = reinterpret_cast<CullFn>(g_originalSpotCull);
			if (!g_lampStage) {
				return original(a_light, a_group, a_3, a_4);  // (not the main frame's deferred-lights stage: untouched)
			}
			const void* accumulator = nullptr;
			const auto  path = ReadSpotPath(a_light, a_group, accumulator);
			const bool  ownCull = a_light == g_lastSpotLight;  // (the lamp loop's: Update(camera) ran right before)
			const bool  emptied = ownCull && g_lastSpotEmptied;
			const bool  active = path == SpotPath::kGroup && Hooks::PrevisFeed::ActiveNow();
			const bool  window = active && !emptied && g_spotWindow && Hooks::PrevisFeed::BeginLampWindow();
			// An emptied light's group pass files nothing anyway: skipped (wrappers at its two calls).
			g_skipSpotGroup = emptied && path == SpotPath::kGroup && g_spotProcessPrevious && g_spotRegisterPrevious;
			// A kept light's casters are judged one by one as they register.
			bool trimming = false;
			if (!emptied && ownCull && g_spotTrim && accumulator && (path == SpotPath::kGroup || path == SpotPath::kOwnList)) {
				if (MakeSpotRegion(a_light, g_spotRegion)) {
					g_spotRegion.active = true;
					trimming = true;
					++g_spotCasterStats.lights;
					Hooks::CullGroups::SetFilteredAccumulator(accumulator, &SpotCasterFilter);
				} else {
					++g_spotCasterStats.noRegion;
				}
			}
			Hooks::CullGroups::SetCountedAccumulator(accumulator);
			const auto    before = Hooks::CullGroups::ReadCountedRegistrations();
			LARGE_INTEGER start{}, end{};
			QueryPerformanceCounter(&start);
			const auto result = original(a_light, a_group, a_3, a_4);
			QueryPerformanceCounter(&end);
			Hooks::PrevisFeed::EndLampWindow(window);
			const auto filed = Hooks::CullGroups::ReadCountedRegistrations() - before;
			Hooks::CullGroups::SetCountedAccumulator(nullptr);
			if (trimming) {
				Hooks::CullGroups::SetFilteredAccumulator(nullptr, nullptr);
				g_spotRegion.active = false;
			}
			const bool skipped = g_skipSpotGroup;
			g_skipSpotGroup = false;

			auto& stats = g_spotCullStats;
			++stats.paths[static_cast<std::size_t>(path)];
			if (skipped) {
				++stats.skippedPasses;
			}
			if (path == SpotPath::kGroup) {
				if (active) {
					++stats.groupActive;
				}
				if (!emptied && Hooks::CullGroups::ReadEngineFlags().dirShadows) {
					++stats.groupSunUp;
				}
				if (window) {
					++stats.windowed;
					stats.filedWindowed += filed;
				} else if (!emptied && active) {
					++stats.staleCulls;
					stats.filedStale += filed;
				} else if (!emptied) {
					++stats.previsOffCulls;
					stats.filedPrevisOff += filed;
				}
			} else if (!emptied) {
				++stats.listCulls;
				stats.filedLists += filed;
			}
			const auto ticks = end.QuadPart - start.QuadPart;
			if (emptied) {
				++stats.emptiedCulls;
				stats.emptiedTicks += ticks;
			} else {
				++stats.keptCulls;
				stats.keptTicks += ticks;
			}
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
		// Spot lights are recorded from BSShadowFrustumLight's Update(camera) (integer arguments and a bool return, so a
		// C++ thunk is safe; the lamp diagnostic uses the same hook). Without it the lamp list stays incomplete and the
		// lamp trimming (bLampGroupTrim) does nothing.
		g_originalUpdateSpot = Util::WriteVFuncSwitchable(g_updateSpotHook, CBRO::Engine::OG(kFrustumLightVtableID).address(), kUpdateSlot, Util::FnAddr(&UpdateSpotThunk), "shadowlights:BSShadowFrustumLight::Update(cam)");
		logger::info("shadow lights: spot lights recorded from Update(camera): hook {}", g_originalUpdateSpot ? "in" : "FAILED (lamp trimming off)");
		g_spotCulling = Settings::Get().spotShadowCulling;
		logger::info("shadow lights: spot-light shadow maps emptied when the lit volume is hidden: {}", !g_spotCulling ? "off (bSpotShadowCulling=0)" : g_originalUpdateSpot ? "on" : "unavailable (no Update hook)");
		// The spot lights' shadow-map cull (slot 9): the lamp window and the per-cull counts. Pass-through outside the main
		// frame's deferred-lights stage.
		{
			LARGE_INTEGER frequency{};
			QueryPerformanceFrequency(&frequency);
			g_qpcPerMs = static_cast<double>(frequency.QuadPart) / 1000.0;
		}
		g_spotWindow = Settings::Get().spotCullWindow;
		g_originalSpotCull = Util::WriteVFuncSwitchable(g_spotCullHook, CBRO::Engine::OG(kFrustumLightVtableID).address(), kCullSlot, Util::FnAddr(&SpotCullThunk), "shadowlights:BSShadowFrustumLight::Cull(group)");
		logger::info(
			"shadow lights: spot-light culls {}; lamp window (previs suspended around a kept spot light's group-0 cull in CBRO frames): {}",
			g_originalSpotCull ? "hooked (counted and timed)" : "NOT hooked", !g_spotWindow ? "off (bSpotCullWindow=0)" : g_originalSpotCull ? "on" : "unavailable");
		// Both need the cull hook (it knows which light's cull runs). The skip needs both sites wrapped, or neither runs.
		if (g_originalSpotCull) {
			g_spotProcessPrevious = WrapSpotSite(kSpotProcessSiteOffset, kGroupProcessID, Util::FnAddr(&SpotProcessThunk), "shadowlights:spot group pass (process)");
			g_spotRegisterPrevious = WrapSpotSite(kSpotRegisterSiteOffset, kGroupRegisterID, Util::FnAddr(&SpotRegisterThunk), "shadowlights:spot group pass (register)");
		}
		g_spotTrim = Settings::Get().spotCasterTrim && g_originalSpotCull != 0;
		logger::info(
			"shadow lights: emptied spot lights skip their group pass: {} | kept spot lights' casters judged one by one: {}",
			g_spotProcessPrevious && g_spotRegisterPrevious ? "yes" : "NO (sites not wrapped; their registrations are still dropped)",
			g_spotTrim ? "on" : Settings::Get().spotCasterTrim ? "unavailable (no cull hook)" : "off (bSpotCasterTrim=0)");

		// The lamp-loop diagnostic: the point lights' Update(camera) too.
		if (Settings::Get().lampDiagnostic) {
			g_originalUpdatePoint = Util::WriteVFuncSwitchable(g_updatePointHook, CBRO::Engine::OG(kParabolicLightVtableID).address(), kUpdateSlot, Util::FnAddr(&UpdatePointThunk), "shadowlights:BSShadowParabolicLight::Update(cam)");
			logger::info("shadow lights: lamp-loop diagnostic on (Update(camera) hooks: point {}, spot {})", g_originalUpdatePoint ? "in" : "FAILED", g_originalUpdateSpot ? "in" : "FAILED");
		}
	}

	void PublishLamps() noexcept
	{
		g_lampLists[g_lampWrite].complete = g_originalUpdateSpot != 0 && g_updateSpotHook.in;
		g_lampsPublished.store(&g_lampLists[g_lampWrite], std::memory_order_release);
		g_lampWrite ^= 1u;
		g_lampLists[g_lampWrite].count = 0;
		g_lampLists[g_lampWrite].overflow = 0;
	}

	const LampList& Lamps() noexcept
	{
		return *g_lampsPublished.load(std::memory_order_acquire);
	}

	void SetLampStage(bool a_open) noexcept
	{
		g_lampStage = a_open;
		g_lastSpotLight = 0;
		g_lastSpotEmptied = false;
		if (!a_open) {
			Hooks::CullGroups::SetDroppedAccumulator(nullptr);
		}
	}

	bool SetHooksIn(bool a_in)
	{
		if (!a_in) {
			SetLampStage(false);  // (previs mode: nothing of the engine's is dropped)
		}
		bool ok = true;
		for (auto* hook : { &g_cameraHook, &g_objectHook, &g_updatePointHook, &g_updateSpotHook, &g_spotCullHook }) {
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
			"shadow lights per frame: shadow culls {:.1f} | emptied {:.1f}{} (verdicts: behind surfaces {:.1f}, out of view {:.1f}) | confirming {:.1f} | visible {:.1f} | kept (reaches camera/partly off-screen/no depth) {:.1f}{}",
			calls / frames, emptied / frames,
			wouldEmpty > 0 ? std::format(" (decide-only: would empty {:.1f})", wouldEmpty / frames) : std::string{},
			hidden / frames, outOfView / frames, confirming / frames, visible / frames, unknown / frames, g_pointReasons.Take(frames));
		const auto spotTests = take(g_spotTests);
		const auto spotWouldEmpty = take(g_spotWouldEmpty);
		const auto spotDropped = static_cast<double>(Hooks::CullGroups::TakeDroppedRegistrations());
		logger::info(
			"spot-light shadow maps per frame: tested {:.1f} | emptied {:.1f}{} (verdicts: behind surfaces {:.1f}, out of view {:.1f}) | confirming {:.1f} | visible {:.1f} | kept (reaches camera/partly off-screen/no depth) {:.1f}{} | unreadable {:.1f} | lit volume from the shadow frustum {:.1f} (else the whole reach) | casters dropped {:.0f}",
			spotTests / frames, take(g_spotEmptied) / frames,
			spotWouldEmpty > 0 ? std::format(" (decide-only: would empty {:.1f})", spotWouldEmpty / frames) : std::string{},
			take(g_spotHidden) / frames, take(g_spotOutOfView) / frames, take(g_spotConfirming) / frames, take(g_spotVisible) / frames,
			take(g_spotUnknown) / frames, g_spotReasons.Take(frames), take(g_spotUnreadable) / frames, take(g_spotByFrustum) / frames, spotDropped / frames);
		{
			auto&      s = g_spotCullStats;
			const auto ratio = [](double a_value, double a_count) { return a_count > 0.0 ? a_value / a_count : 0.0; };
			const auto ms = [](std::int64_t a_ticks) { return g_qpcPerMs > 0.0 ? static_cast<double>(a_ticks) / g_qpcPerMs : 0.0; };
			double     culls = 0.0;
			for (const auto count : s.paths) {
				culls += static_cast<double>(count);
			}
			const auto path = [&](SpotPath a_path) { return static_cast<double>(s.paths[static_cast<std::size_t>(a_path)]) / frames; };
			const auto emptiedMs = ms(s.emptiedTicks), keptMs = ms(s.keptTicks);
			logger::info(
				"spot-light culls per frame: {:.1f} | by path: {} {:.1f} (previs active at entry {:.1f}: in a lamp window {:.1f}; kept lights with the sun up {:.1f}), {} {:.1f}, {} {:.1f}, {} {:.1f}, {} {:.1f} | casters offered per kept light's cull: lamp window {:.0f}, previs active without a window {:.0f}, previs off {:.0f}, the light's lists {:.0f} | emptied lights' group passes skipped {:.1f} | CPU per frame: emptied {:.3f} ms ({:.1f} culls, {:.1f} us each), kept {:.3f} ms ({:.1f} culls, {:.1f} us each)",
				culls / frames, kSpotPathNames[2], path(SpotPath::kGroup), static_cast<double>(s.groupActive) / frames, static_cast<double>(s.windowed) / frames,
				static_cast<double>(s.groupSunUp) / frames,
				kSpotPathNames[1], path(SpotPath::kOwnList), kSpotPathNames[0], path(SpotPath::kDirect), kSpotPathNames[3], path(SpotPath::kNone), kSpotPathNames[4], path(SpotPath::kUnreadable),
				ratio(static_cast<double>(s.filedWindowed), static_cast<double>(s.windowed)), ratio(static_cast<double>(s.filedStale), static_cast<double>(s.staleCulls)),
				ratio(static_cast<double>(s.filedPrevisOff), static_cast<double>(s.previsOffCulls)), ratio(static_cast<double>(s.filedLists), static_cast<double>(s.listCulls)),
				static_cast<double>(s.skippedPasses) / frames,
				emptiedMs / frames, static_cast<double>(s.emptiedCulls) / frames, ratio(emptiedMs * 1000.0, static_cast<double>(s.emptiedCulls)),
				keptMs / frames, static_cast<double>(s.keptCulls) / frames, ratio(keptMs * 1000.0, static_cast<double>(s.keptCulls)));
			s = {};
		}
		{
			auto&      c = g_spotCasterStats;
			const auto per = [frames](std::uint64_t a_value) { return static_cast<double>(a_value) / frames; };
			logger::info(
				"spot-light casters per frame (kept lights' shadow maps{}): lights judged {:.1f}, left whole {:.1f} (no view cone, or the shadow camera off the light) | offered {:.0f} | left out: outside the pushed view cone {:.0f}, shadow volume can't reach a visible surface {:.0f} | kept untested: always-draw {:.0f}, unreadable {:.0f} | volume misses still confirming {:.0f}",
				g_spotTrim ? "" : ", bSpotCasterTrim=0: not judged", per(c.lights), per(c.noRegion), per(c.offered), per(c.outsideCone), per(c.volume), per(c.alwaysDraw),
				per(c.unreadable), per(c.volumeConfirming));
			c = {};
		}
		const auto regions = take(g_regionLights);
		const auto inside = take(g_regionInside);
		const auto seen = take(g_casterSeen);
		const auto tests = take(g_casterTests);
		const auto culled = take(g_casterCulled);
		const auto volume = take(g_casterVolumeCulled);
		const auto modes = [&](std::size_t a_index) { return take(g_regionModes[a_index]) / frames; };
		const auto mode0 = modes(0), mode3 = modes(1), mode4 = modes(2), modeOther = modes(3);
		const auto cullMs = [&](std::size_t a_kind, double& a_count) {
			a_count = static_cast<double>(g_cullCount[a_kind].exchange(0, std::memory_order_relaxed));
			const auto ticks = g_cullTicks[a_kind].exchange(0, std::memory_order_relaxed);
			return g_qpcPerMs > 0.0 ? static_cast<double>(ticks) / g_qpcPerMs : 0.0;
		};
		double     limitedCulls = 0.0, emptiedCulls = 0.0, otherCulls = 0.0;
		const auto limitedMs = cullMs(0, limitedCulls), emptiedMs = cullMs(1, emptiedCulls), otherMs = cullMs(2, otherCulls);
		const auto each = [](double a_ms, double a_count) { return a_count > 0.0 ? a_ms * 1000.0 / a_count : 0.0; };
		logger::info(
			"shadow light casters per frame: lights limited to the view {:.1f} (inside it {:.1f}; by cull mode: 0 {:.1f}, 3 {:.1f}, 4 {:.1f}, other {:.1f}) | objects seen {:.0f}: outside the light's reach (the engine's own test) {:.0f}, tested {:.0f}, not tested as by the engine (all-pass/all-fail mode {:.0f}, always-draw {:.0f}, pre-processed {:.0f}) | left out: outside the pushed view cone {:.0f}, shadow volume can't reach a visible surface {:.0f} (with their subtrees) | point-light culls CPU per frame: limited {:.3f} ms ({:.1f} us each), emptied {:.3f} ms ({:.1f} us each), left whole {:.3f} ms ({:.1f} us each)",
			regions / frames, inside / frames, mode0, mode3, mode4, modeOther, seen / frames, take(g_casterOutsideReach) / frames, tests / frames, take(g_casterSkippedMode) / frames,
			take(g_casterSkippedAlwaysDraw) / frames, take(g_casterSkippedPreProcessed) / frames, culled / frames, volume / frames,
			limitedMs / frames, each(limitedMs, limitedCulls), emptiedMs / frames, each(emptiedMs, emptiedCulls), otherMs / frames, each(otherMs, otherCulls));
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
