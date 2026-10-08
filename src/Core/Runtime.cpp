#include "Core/Runtime.h"

#include "Core/Async.h"
#include "Core/Feed.h"
#include "Core/HangWatch.h"
#include "Core/SetDiff.h"
#include "Core/HiZ.h"
#include "Core/Occlusion.h"
#include "Core/ShadowLights.h"
#include "Hooks/CullGroups.h"
#include "Hooks/FirstPersonOrder.h"
#include "Hooks/Precipitation.h"
#include "Hooks/PrevisFeed.h"
#include "Hooks/RenderStages.h"
#include "Settings.h"
#include "Util/D3D.h"

namespace CBRO::Core::Runtime
{
	namespace
	{
		using Stage = Hooks::RenderStages::Stage;

		// ---- previs (BSPreCulledObjects) through the engine's own switch --------------------
		// Same path as the console command `tpc`; ids verified in Phase 0 run 3.

		bool PrevisEnabled()
		{
			using func_t = bool (*)();
			static REL::Relocation<func_t> func{ CBRO::Engine::OG(652211) };
			return func();
		}

		void SetPrevisEnabled(bool a_enabled)
		{
			using func_t = void (*)(bool);
			static REL::Relocation<func_t> func{ CBRO::Engine::OG(1090712) };
			func(a_enabled);
		}

		// ---- projection convention --------------------------------------------------------------

		enum class Convention
		{
			kUnknown,
			kNone,  // self-check failed: never cull
			kRowRelative,
			kRowAbsolute,
			kColumnRelative,
			kColumnAbsolute,
		};

		std::string_view ConventionName(Convention a_convention)
		{
			switch (a_convention) {
			case Convention::kRowRelative:
				return "row-vector, camera-relative"sv;
			case Convention::kRowAbsolute:
				return "row-vector, absolute"sv;
			case Convention::kColumnRelative:
				return "column-vector, camera-relative"sv;
			case Convention::kColumnAbsolute:
				return "column-vector, absolute"sv;
			case Convention::kNone:
				return "none (self-check failed)"sv;
			default:
				return "unknown"sv;
			}
		}

		struct State
		{
			bool          installed{ false };  // hooks in place and the listener registered
			Convention    convention{ Convention::kUnknown };
			std::uint64_t renderFrame{ 0 };  // advances at every pre-pass end
			std::uint32_t clock{ 0 };        // advances at every cull stage
			bool          wantActive{ false };
			bool          interiorStandby{ false };   // the scene is an interior/override root and bInteriors=0: previs mode regardless of wantActive
			bool          interiorScene{ false };     // the scene is an interior/override root (the gates), whatever bInteriors says
			bool          toggleKeyDown{ false };
			bool          failureHandled{ false };
			std::atomic<int> previsRequest{ -1 };  // previs to switch at the next cull begin: 0 off, 1 back to the original (-1: none)
			bool          loadSwitch{ false };       // previs was switched off at a loading screen (v1.44); the first cull after decides what follows
			bool          loadSinkRegistered{ false };
			bool          loadSuspended{ false };    // ... by the suspension byte (the UI's thread): cleared after the switch lands
			bool          gameLoaded{ false };
			int           expectedPrevis{ -1 };  // previs state CBRO last set (-1: not yet)
			bool          previsThreadLogged{ false };
			int           viewSpaceLogged{ -1 };
			int           notifiedEffective{ -1 };
			bool          statusKeyDown{ false };
			bool          diagnosticKeyDown{ false };
			int           diagnostic{ 0 };  // 0 normal, 1 decide-only, 2 decide-only + no depth capture
			std::uint32_t selfCheckAttempts{ 0 };
			bool          originalPrevisKnown{ false };
			bool          originalPrevis{ true };
			std::uint32_t framesSinceLog{ 0 };
			bool          cullingThisFrame{ false };  // CBRO culls this frame: the culling-group hooks act (else pass-through)
			bool          capturedInPass{ false };    // this pre-pass's capture already ran at its world block's end (FirstPersonOrder)
			bool          hooksWanted{ true };        // what SyncHooks last asked for (the hooks go in at load)
			bool          hooksIn{ true };            // the hooks culling needs are in
		};
		State g_state;

		struct Raw
		{
			float m[4][4];
			float posAdjust[3];
		};

		// clip = (p - adjust) * M (row) or M * (p - adjust) (column)
		void Project(const Raw& a_raw, bool a_column, bool a_relative, const float a_p[3], float a_out[4])
		{
			const float p[4]{
				a_p[0] - (a_relative ? a_raw.posAdjust[0] : 0.0f),
				a_p[1] - (a_relative ? a_raw.posAdjust[1] : 0.0f),
				a_p[2] - (a_relative ? a_raw.posAdjust[2] : 0.0f),
				1.0f
			};
			for (int i = 0; i < 4; ++i) {
				a_out[i] = 0.0f;
				for (int k = 0; k < 4; ++k) {
					a_out[i] += p[k] * (a_column ? a_raw.m[i][k] : a_raw.m[k][i]);
				}
			}
		}

		// Picks the convention under which a point straight ahead of the camera lands at the
		// screen centre, and points to its right/up land right/up.
		Convention SelfCheck(const HiZ::Camera& a_camera, const Raw& a_raw, bool a_verbose)
		{
			const auto ahead = [&](float a_dist, const float* a_offsetDir, float a_offset, float a_out[3]) {
				for (int i = 0; i < 3; ++i) {
					a_out[i] = a_camera.eye[i] + a_camera.viewDir[i] * a_dist + (a_offsetDir ? a_offsetDir[i] * a_offset : 0.0f);
				}
			};
			float center[3], right[3], up[3];
			ahead(1000.0f, nullptr, 0.0f, center);
			ahead(1000.0f, a_camera.viewRight, 200.0f, right);
			ahead(1000.0f, a_camera.viewUp, 200.0f, up);

			const std::array candidates{
				std::tuple{ Convention::kRowRelative, false, true },
				std::tuple{ Convention::kRowAbsolute, false, false },
				std::tuple{ Convention::kColumnRelative, true, true },
				std::tuple{ Convention::kColumnAbsolute, true, false },
			};
			Convention chosen = Convention::kNone;
			for (const auto& [convention, column, relative] : candidates) {
				float c[4], r[4], u[4];
				Project(a_raw, column, relative, center, c);
				Project(a_raw, column, relative, right, r);
				Project(a_raw, column, relative, up, u);
				const bool ok =
					c[3] > 0.0f && std::abs(c[0] / c[3]) < 0.05f && std::abs(c[1] / c[3]) < 0.05f &&
					c[2] / c[3] > 0.0f && c[2] / c[3] < 1.0f &&
					r[3] > 0.0f && r[0] / r[3] > 0.02f && u[3] > 0.0f && u[1] / u[3] > 0.02f;
				if (a_verbose) {
					logger::info(
						"self-check {}: ahead ndc=({:.3f},{:.3f},{:.4f}) w={:.1f} | right x={:.3f} | up y={:.3f} -> {}",
						ConventionName(convention), c[0] / c[3], c[1] / c[3], c[2] / c[3], c[3], r[0] / r[3], u[1] / u[3], ok ? "PASS" : "fail");
				}
				if (ok && chosen == Convention::kNone) {
					chosen = convention;
				}
			}
			if (a_verbose || chosen != Convention::kNone) {
				logger::info(
					"self-check: eye=({:.0f},{:.0f},{:.0f}) posAdjust=({:.0f},{:.0f},{:.0f}) -> {}",
					a_camera.eye[0], a_camera.eye[1], a_camera.eye[2], a_raw.posAdjust[0], a_raw.posAdjust[1], a_raw.posAdjust[2],
					ConventionName(chosen));
			}
			return chosen;
		}

		void Normalize(float a_v[3])
		{
			const float length = std::sqrt(a_v[0] * a_v[0] + a_v[1] * a_v[1] + a_v[2] * a_v[2]);
			if (length > 0.0f) {
				a_v[0] /= length;
				a_v[1] /= length;
				a_v[2] /= length;
			}
		}

		// clip = (p - posAdjust) * viewProj with the baked (row-vector) convention.
		void ProjectBaked(const HiZ::Camera& a_camera, const float a_p[3], float a_out[4])
		{
			const float p[3]{ a_p[0] - a_camera.posAdjust[0], a_p[1] - a_camera.posAdjust[1], a_p[2] - a_camera.posAdjust[2] };
			for (int i = 0; i < 4; ++i) {
				a_out[i] = p[0] * a_camera.viewProj[0][i] + p[1] * a_camera.viewProj[1][i] + p[2] * a_camera.viewProj[2][i] + a_camera.viewProj[3][i];
			}
		}

		// Expresses viewProj as eye + orthonormal basis + per-axis scales + depth curve, then checks
		// that form against viewProj on an off-axis point. The occlusion test only uses it if it matches.
		void DeriveViewSpace(HiZ::Camera& a_camera)
		{
			a_camera.viewSpace = false;
			const bool relative = g_state.convention == Convention::kRowRelative || g_state.convention == Convention::kColumnRelative;
			for (int i = 0; i < 3; ++i) {
				a_camera.origin[i] = relative ? a_camera.posAdjust[i] : a_camera.eye[i];
			}
			Normalize(a_camera.viewDir);
			Normalize(a_camera.viewRight);
			Normalize(a_camera.viewUp);

			const auto at = [&](float a_forward, float a_right, float a_up, float a_out[4]) {
				float p[3];
				for (int i = 0; i < 3; ++i) {
					p[i] = a_camera.origin[i] + a_camera.viewDir[i] * a_forward + a_camera.viewRight[i] * a_right + a_camera.viewUp[i] * a_up;
				}
				ProjectBaked(a_camera, p, a_out);
			};

			float c[4];
			at(1000.0f, 1000.0f, 0.0f, c);
			a_camera.scaleX = c[0] / c[3];
			at(1000.0f, 0.0f, 1000.0f, c);
			a_camera.scaleY = c[1] / c[3];

			constexpr float kNear = 100.0f;
			constexpr float kFar = 10000.0f;
			float           n[4], f[4];
			at(kNear, 0.0f, 0.0f, n);
			at(kFar, 0.0f, 0.0f, f);
			const float zn = n[2] / n[3];
			const float zf = f[2] / f[3];
			a_camera.depthB = (zn - zf) / (1.0f / kNear - 1.0f / kFar);
			a_camera.depthA = zn - a_camera.depthB / kNear;

			// verify on an off-axis point
			constexpr float kZ = 2345.0f, kX = 333.0f, kY = -222.0f;
			float           t[4];
			at(kZ, kX, kY, t);
			const float dx = std::abs(t[0] / t[3] - a_camera.scaleX * kX / kZ);
			const float dy = std::abs(t[1] / t[3] - a_camera.scaleY * kY / kZ);
			const float dz = std::abs(t[2] / t[3] - (a_camera.depthA + a_camera.depthB / kZ));
			const float dw = std::abs(t[3] - kZ);
			a_camera.viewSpace = a_camera.scaleX > 0.0f && a_camera.scaleY > 0.0f && dx < 1.0e-3f && dy < 1.0e-3f && dz < 1.0e-5f && dw < 1.0f;

			if (static_cast<int>(a_camera.viewSpace) != g_state.viewSpaceLogged) {
				g_state.viewSpaceLogged = a_camera.viewSpace;
				logger::info(
					"view-space projection {}: scale=({:.4f},{:.4f}) depth=A {:.6f} + B {:.4f}/z | check dx={:.2e} dy={:.2e} dz={:.2e} dw={:.2e}",
					a_camera.viewSpace ? "verified (exact sphere test)" : "NOT verified (box-corner fallback)",
					a_camera.scaleX, a_camera.scaleY, a_camera.depthA, a_camera.depthB, dx, dy, dz, dw);
			}
		}

		// Reads the camera the pre-pass just rendered with and bakes the chosen convention into it,
		// so the per-object test is always `clip = (p - posAdjust) * viewProj`.
		bool ReadCamera(HiZ::Camera& a_camera, Raw& a_raw)
		{
			const auto state = CBRO::Engine::BSGraphics::State::GetSingleton();
			const auto root = RE::Main::WorldRootCamera();
			if (!state || !root) {
				return false;
			}

			// The pre-pass renders the WorldRoot camera, but other render-to-texture passes (scopes,
			// cubemaps) overwrite cameraState, so take the world camera's own cache entry (unjittered
			// preferred), as the Upscaling mod does. No entry: skip this capture.
			const CBRO::Engine::BSGraphics::CameraStateData* selected = nullptr;
			std::uint8_t                                     entries = 0;
			for (const auto& candidate : state->cameraDataCache) {
				if (candidate.referenceCamera == root) {
					entries = static_cast<std::uint8_t>(std::min(entries + 1, 255));
					if (!selected || (selected->useJitter && !candidate.useJitter)) {
						selected = std::addressof(candidate);
					}
				}
			}
			a_camera.cameraEntries = entries;
			a_camera.cameraSource = selected ? (selected->useJitter ? 2 : 1) : 0;
			if (!selected && state->cameraState.referenceCamera == root) {
				selected = std::addressof(state->cameraState);
				a_camera.cameraSource = 3;
			}
			if (!selected) {
				return false;
			}

			const auto& view = selected->camViewData;
			for (int i = 0; i < 4; ++i) {
				_mm_storeu_ps(a_raw.m[i], view.viewProjUnjittered[i]);
			}
			a_raw.posAdjust[0] = selected->posAdjust.x;
			a_raw.posAdjust[1] = selected->posAdjust.y;
			a_raw.posAdjust[2] = selected->posAdjust.z;

			const auto store3 = [](const __m128& a_v, float a_out[3]) {
				alignas(16) float tmp[4];
				_mm_store_ps(tmp, a_v);
				a_out[0] = tmp[0];
				a_out[1] = tmp[1];
				a_out[2] = tmp[2];
			};
			store3(view.viewDir, a_camera.viewDir);
			store3(view.viewRight, a_camera.viewRight);
			store3(view.viewUp, a_camera.viewUp);

			a_camera.eye[0] = root->world.translate.x;
			a_camera.eye[1] = root->world.translate.y;
			a_camera.eye[2] = root->world.translate.z;
			for (int r = 0; r < 3; ++r) {
				for (int c = 0; c < 3; ++c) {
					a_camera.rotate[r][c] = root->world.rotate.entry[r].pt[c];
				}
			}

			const bool column = g_state.convention == Convention::kColumnRelative || g_state.convention == Convention::kColumnAbsolute;
			const bool relative = g_state.convention == Convention::kRowRelative || g_state.convention == Convention::kColumnRelative;
			for (int r = 0; r < 4; ++r) {
				for (int c = 0; c < 4; ++c) {
					a_camera.viewProj[r][c] = column ? a_raw.m[c][r] : a_raw.m[r][c];
				}
			}
			for (int i = 0; i < 3; ++i) {
				a_camera.posAdjust[i] = relative ? a_raw.posAdjust[i] : 0.0f;
			}
			DeriveViewSpace(a_camera);
			return true;
		}

		// Camera change since the depth frame: translation and rotation angle (radians).
		void CameraDelta(const HiZ::Camera& a_then, float& a_move, float& a_angle)
		{
			a_move = 0.0f;
			a_angle = 0.0f;
			const auto root = RE::Main::WorldRootCamera();
			if (!root) {
				return;
			}
			const float dx = root->world.translate.x - a_then.eye[0];
			const float dy = root->world.translate.y - a_then.eye[1];
			const float dz = root->world.translate.z - a_then.eye[2];
			a_move = std::sqrt(dx * dx + dy * dy + dz * dz);

			// Rotation angle between the two orientations from ||R_now - R_then|| = 2*sqrt(2)*sin(theta/2):
			// exact 0 for an unchanged matrix and accurate for tiny turns (acos of the trace isn't).
			float squared = 0.0f;
			for (int r = 0; r < 3; ++r) {
				for (int c = 0; c < 3; ++c) {
					const float d = root->world.rotate.entry[r].pt[c] - a_then.rotate[r][c];
					squared += d * d;
				}
			}
			a_angle = 2.0f * std::asin(std::min(1.0f, std::sqrt(squared) / 2.8284271f));
		}

		// ---- verdict-cache epochs ---------------------------------------------------------------------------------
		// Occlusion reuses an object's last verdict while the camera stays within the cache tolerance of the
		// epoch's reference (the depth frame's camera when the epoch began: both it and the current NiCamera must
		// stay within half the tolerance, so any two cameras of an epoch are within the whole of it, which is how
		// far tested bounds are dilated). The sun has its own epoch (its direction and state).

		struct Epochs
		{
			bool          viewKnown{ false };
			float         eye[3]{};
			float         rotate[3][3]{};
			float         zoom[2]{};  // the frustum half-extents (zoom) of the depth frame's camera when the epoch began
			std::uint32_t viewEpoch{ 1 };
			int           sunState{ -1 };
			float         sunDir[3]{};
			std::uint8_t  sunEpoch{ 1 };
			std::uint32_t viewChanges{ 0 };  // this interval, for the log
			std::uint32_t sunChanges{ 0 };
		};
		Epochs g_epochs;

		float RotationAngle(const float a_a[3][3], const float a_b[3][3]) noexcept
		{
			float squared = 0.0f;
			for (int r = 0; r < 3; ++r) {
				for (int c = 0; c < 3; ++c) {
					const float d = a_a[r][c] - a_b[r][c];
					squared += d * d;
				}
			}
			return 2.0f * std::asin(std::min(1.0f, std::sqrt(squared) / 2.8284271f));
		}

		float Distance3(const float a_a[3], const float a_b[3]) noexcept
		{
			const float dx = a_a[0] - a_b[0];
			const float dy = a_a[1] - a_b[1];
			const float dz = a_a[2] - a_b[2];
			return std::sqrt(dx * dx + dy * dy + dz * dz);
		}

		// a_zoom: the NiCamera's frustum half-extents now. A zoom change ends the epoch too (v1.65): up to v1.64 the
		// verdicts judged in a scope's narrow view were reused after zooming out with the camera still, so objects
		// "out of view" there (left out of the walk, or out of group 0) could stay undrawn.
		void UpdateViewEpoch(const HiZ::Camera& a_camera, const float a_zoom[2], float a_tolMove, float a_tolAngle)
		{
			const auto sameZoom = [](float a_a, float a_b) { return std::abs(a_a - a_b) <= 0.002f * std::abs(a_b); };
			bool       same = g_epochs.viewKnown && Distance3(a_camera.eye, g_epochs.eye) <= a_tolMove && RotationAngle(a_camera.rotate, g_epochs.rotate) <= a_tolAngle &&
			            sameZoom(a_camera.frustumX, g_epochs.zoom[0]) && sameZoom(a_camera.frustumY, g_epochs.zoom[1]) && sameZoom(a_zoom[0], g_epochs.zoom[0]) &&
			            sameZoom(a_zoom[1], g_epochs.zoom[1]);
			if (same) {
				if (const auto root = RE::Main::WorldRootCamera()) {
					const float eye[3]{ root->world.translate.x, root->world.translate.y, root->world.translate.z };
					float       rotate[3][3];
					for (int r = 0; r < 3; ++r) {
						for (int c = 0; c < 3; ++c) {
							rotate[r][c] = root->world.rotate.entry[r].pt[c];
						}
					}
					same = Distance3(eye, g_epochs.eye) <= a_tolMove && RotationAngle(rotate, g_epochs.rotate) <= a_tolAngle;
				}
			}
			if (!same) {
				++g_epochs.viewEpoch;
				++g_epochs.viewChanges;
				std::copy_n(a_camera.eye, 3, g_epochs.eye);
				std::memcpy(g_epochs.rotate, a_camera.rotate, sizeof(g_epochs.rotate));
				g_epochs.zoom[0] = a_camera.frustumX;
				g_epochs.zoom[1] = a_camera.frustumY;
				g_epochs.viewKnown = true;
			}
		}

		void UpdateSunEpoch(int a_state, const float a_dir[3], float a_tolAngle)
		{
			bool same = a_state == g_epochs.sunState;
			if (same && a_state == static_cast<int>(Occlusion::FrameContext::Sun::State::kOn)) {
				const float dot = std::clamp(a_dir[0] * g_epochs.sunDir[0] + a_dir[1] * g_epochs.sunDir[1] + a_dir[2] * g_epochs.sunDir[2], -1.0f, 1.0f);
				same = std::acos(dot) <= a_tolAngle;
			}
			if (!same) {
				++g_epochs.sunEpoch;
				++g_epochs.sunChanges;
				g_epochs.sunState = a_state;
				std::copy_n(a_dir, 3, g_epochs.sunDir);
			}
		}

		// Consecutive frames (at cull time) in which the NiCamera's rotation didn't change.
		struct Stillness
		{
			float         rotate[3][3]{};
			bool          known{ false };
			std::uint32_t frames{ 0 };
			float         lastTurn{ 0.0f };  // radians the NiCamera turned since the previous frame's cull
		};
		Stillness g_stillness;

		void TrackStillness()
		{
			const auto root = RE::Main::WorldRootCamera();
			if (!root) {
				g_stillness = {};
				return;
			}
			float squared = 0.0f;
			for (int r = 0; r < 3; ++r) {
				for (int c = 0; c < 3; ++c) {
					const float value = root->world.rotate.entry[r].pt[c];
					const float d = value - g_stillness.rotate[r][c];
					squared += d * d;
					g_stillness.rotate[r][c] = value;
				}
			}
			// Same threshold as the still view: ||dR|| ~ sqrt(2) * angle for small turns.
			const bool still = g_stillness.known && squared < 2.0f * 0.0002f * 0.0002f;
			g_stillness.frames = still ? g_stillness.frames + 1 : 0;
			g_stillness.lastTurn = g_stillness.known ? 2.0f * std::asin(std::min(1.0f, std::sqrt(squared) / 2.8284271f)) : 0.0f;
			g_stillness.known = true;
		}

		// ---- the current view on the depth frame -------------------------------------------------
		// Objects crossing the depth frame's edge can still be judged on the part the current camera
		// sees, if that part was rendered. A camera that hasn't turned or zoomed sees exactly the depth
		// frame. A turned camera's frustum comes from the WorldRoot NiCamera (rotation + frustum) mapped
		// onto the render basis. At each capture CBRO checks which reading of the rotation matrix
		// (columns or rows) reproduces the render basis. Facing along a world axis both fit, and they
		// would turn the view in opposite directions, so then the union of both views is used. No fit
		// leaves edge objects unjudged.

		struct AxisMap
		{
			bool  valid{ false };
			bool  columns{ true };
			int   index[3]{};  // viewDir, viewRight, viewUp
			float sign[3]{};
			float fit{ 0.0f };  // the worst axis' |dot| with the render basis
		};

		enum class FrustumUnits
		{
			kUnknown,
			kTangents,  // left/right/top/bottom at unit distance
			kNearPlane  // at the near plane
		};

		// Footprint outcomes per summary interval: which path placed the view, or why none could.
		enum FootprintReason : std::size_t
		{
			kStill,              // used: camera still, same zoom
			kTurned,             // used: turned camera, one reading of its rotation fits
			kTurnedBoth,         // used: turned camera, union of both readings (facing along a world axis)
			kTrailing,           // capture: drawn from the previous frame's orientation (mid-turn), off by that turn
			kTrailingBeyond,     // capture: mid-turn, but off by more than the NiCamera's last turn (counted only)
			kAxesNoFit,          // capture: no reading of the rotation matches the render basis
			kOffsetStill,        // capture: the NiCamera still, yet the render basis off it (no footprint)
			kUnitsUnknown,       // capture: frustum values don't match the projection in any known unit
			kTangentMismatch,    // capture: frustum and projection disagree this frame
			kNoMapping,          // frame: turned, but the depth frame's capture had no usable mapping
			kCheckFailed,        // frame: the self-check refused the placed view
			kTurnedOff,          // frame: turned, and turned views were switched off by the self-check
			kSideways,           // frame: turned so far the view leaves the depth frame's hemisphere
			kImplausible,        // frame: frustum values out of range
			kReasonCount
		};

		struct Footprint
		{
			FrustumUnits  units{ FrustumUnits::kUnknown };
			bool          disabled{ false };
			bool          announced{ false };
			float         lastView[4]{};
			float         lastAngle{ 0.0f };
			bool          lastValid{ false };
			std::uint32_t strikes{ 0 };        // depth frames in a row whose view failed the self-check
			std::uint64_t strikeCapture{ 0 };  // the last of them
			bool          baselineLogged{ false };
			std::array<std::uint32_t, kReasonCount> reasons{};
		};
		Footprint g_footprint;

		// NiCamera::viewFrustum (+0x160): left, right, top, bottom, near, far (floats), ortho (bool).
		constexpr std::size_t kFrustumOffset = 0x160;

		struct Frustum
		{
			float left, right, top, bottom, nearPlane, farPlane;
		};

		Frustum ReadFrustum(const RE::NiCamera* a_camera)
		{
			Frustum frustum{};
			std::memcpy(&frustum, reinterpret_cast<const std::byte*>(a_camera) + kFrustumOffset, sizeof(frustum));
			return frustum;
		}

		float AxisComponent(const float a_rotate[3][3], bool a_columns, int a_axis, int a_component)
		{
			return a_columns ? a_rotate[a_component][a_axis] : a_rotate[a_axis][a_component];
		}

		// Rotation vs render basis agreement: same (~0.1 deg) means the depth was drawn from the NiCamera's
		// own orientation; strict (~0.8 deg) that the reading is right, though the render camera may be a
		// frame behind a turning NiCamera (FO4-ENGINE-NOTES); loose (~5.7 deg) only that they are close.
		constexpr float kSameFit = 0.9999985f;
		constexpr float kSameAngle = 0.0017453f;  // (the same ~0.1 deg, in radians)
		constexpr float kStrictFit = 0.9999f;
		constexpr float kLooseFit = 0.995f;

		bool MatchAxes(const HiZ::Camera& a_camera, bool a_columns, float a_threshold, AxisMap& a_out)
		{
			const float* basis[3]{ a_camera.viewDir, a_camera.viewRight, a_camera.viewUp };
			AxisMap      map{};
			map.columns = a_columns;
			map.fit = 1.0f;
			for (int b = 0; b < 3; ++b) {
				float best = 0.0f;
				for (int k = 0; k < 3; ++k) {
					float d = 0.0f;
					for (int c = 0; c < 3; ++c) {
						d += AxisComponent(a_camera.rotate, a_columns, k, c) * basis[b][c];
					}
					if (std::abs(d) > best) {
						best = std::abs(d);
						map.index[b] = k;
						map.sign[b] = d < 0.0f ? -1.0f : 1.0f;
					}
				}
				if (best < a_threshold) {
					return false;
				}
				map.fit = std::min(map.fit, best);
			}
			if (map.index[0] == map.index[1] || map.index[0] == map.index[2] || map.index[1] == map.index[2]) {
				return false;
			}
			map.valid = true;
			a_out = map;
			return true;
		}

		void FrustumTangents(const Frustum& a_frustum, float& a_x, float& a_y)
		{
			a_x = std::max(std::abs(a_frustum.left), std::abs(a_frustum.right));
			a_y = std::max(std::abs(a_frustum.top), std::abs(a_frustum.bottom));
			if (g_footprint.units == FrustumUnits::kNearPlane && a_frustum.nearPlane > 0.0f) {
				a_x /= a_frustum.nearPlane;
				a_y /= a_frustum.nearPlane;
			}
		}

		void FrustumExtents(const Frustum& a_frustum, float& a_x, float& a_y)
		{
			a_x = std::max(std::abs(a_frustum.left), std::abs(a_frustum.right));
			a_y = std::max(std::abs(a_frustum.top), std::abs(a_frustum.bottom));
		}

		// The player camera's state and FOVs, read raw under SEH (PlayerCamera: currentState +0x28 ->
		// TESCameraState::id +0x20; worldFOV +0x168, firstPersonFOV +0x16C, fovAdjustCurrent +0x170).
		struct PlayerCameraView
		{
			std::uint8_t state{ 0xFF };
			float        worldFOV{ 0.0f };
			float        firstPersonFOV{ 0.0f };
			float        fovAdjust{ 0.0f };

			static PlayerCameraView Read() noexcept
			{
				PlayerCameraView view{};
				const auto       camera = reinterpret_cast<const std::byte*>(RE::PlayerCamera::GetSingleton());
				if (!camera) {
					return view;
				}
				__try {
					const auto state = *reinterpret_cast<const std::byte* const*>(camera + 0x28);
					view.state = state ? static_cast<std::uint8_t>(*reinterpret_cast<const std::uint32_t*>(state + 0x20)) : 0xFF;
					view.worldFOV = *reinterpret_cast<const float*>(camera + 0x168);
					view.firstPersonFOV = *reinterpret_cast<const float*>(camera + 0x16C);
					view.fovAdjust = *reinterpret_cast<const float*>(camera + 0x170);
				} __except (EXCEPTION_EXECUTE_HANDLER) {
					view = {};
				}
				return view;
			}
		};

		std::string_view CameraStateName(std::uint8_t a_state) noexcept
		{
			static constexpr std::array kNames{
				"first person"sv, "auto vanity"sv, "VATS"sv, "free"sv, "iron sights"sv, "PC transition"sv, "tween"sv,
				"animated"sv, "third person"sv, "furniture"sv, "mount"sv, "bleedout"sv, "dialogue"sv
			};
			return a_state < kNames.size() ? kNames[a_state] : "unknown"sv;
		}

		// A NiCamera rotation's dir, right, up in the render basis' sense, under one reading of a capture's axis map.
		void MappedBasis(const HiZ::Camera& a_then, int a_reading, const float a_rotate[3][3], float a_basis[3][3]) noexcept
		{
			for (int b = 0; b < 3; ++b) {
				for (int c = 0; c < 3; ++c) {
					a_basis[b][c] = a_then.axisSign[a_reading][b] * AxisComponent(a_rotate, a_reading == 0, a_then.axisIndex[a_reading][b], c);
				}
			}
		}

		// Where a frustum (tangents a_tx, a_ty) of that basis lands on the depth frame: the box is grown to cover its
		// corner rays. False when a corner ray reaches sideways past the depth frame's image plane.
		bool PlaceView(const HiZ::Camera& a_then, const float a_basis[3][3], float a_tx, float a_ty, float& a_x0, float& a_x1, float& a_y0, float& a_y1) noexcept
		{
			for (int corner = 0; corner < 4; ++corner) {
				const float sx = (corner & 1) ? a_tx : -a_tx;
				const float sy = (corner & 2) ? a_ty : -a_ty;
				float       d[3];
				for (int c = 0; c < 3; ++c) {
					d[c] = a_basis[0][c] + a_basis[1][c] * sx + a_basis[2][c] * sy;
				}
				const float length = std::sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2]);
				const float z = d[0] * a_then.viewDir[0] + d[1] * a_then.viewDir[1] + d[2] * a_then.viewDir[2];
				if (z < 0.05f * length) {
					return false;
				}
				const float x = d[0] * a_then.viewRight[0] + d[1] * a_then.viewRight[1] + d[2] * a_then.viewRight[2];
				const float y = d[0] * a_then.viewUp[0] + d[1] * a_then.viewUp[1] + d[2] * a_then.viewUp[2];
				a_x0 = std::min(a_x0, a_then.scaleX * x / z);
				a_x1 = std::max(a_x1, a_then.scaleX * x / z);
				a_y0 = std::min(a_y0, a_then.scaleY * y / z);
				a_y1 = std::max(a_y1, a_then.scaleY * y / z);
			}
			return true;
		}

		// Where a NiCamera rotation's view (frustum tangents a_tx, a_ty) lands on a depth frame, raw: the box over every
		// reading of the rotation that fit at capture. False without one, or when the view reaches sideways past the
		// frame's image plane.
		bool PlaceRotation(const HiZ::Camera& a_then, const float a_rotate[3][3], float a_tx, float a_ty, float a_out[4]) noexcept
		{
			if (!a_then.footprint || !a_then.axisReadings) {
				return false;
			}
			float x0 = std::numeric_limits<float>::infinity(), x1 = -x0, y0 = x0, y1 = -x0;
			for (int reading = 0; reading < 2; ++reading) {
				if (!(a_then.axisReadings & (1u << reading))) {
					continue;
				}
				float basis[3][3];
				MappedBasis(a_then, reading, a_rotate, basis);
				if (!PlaceView(a_then, basis, a_tx, a_ty, x0, x1, y0, y1)) {
					return false;
				}
			}
			a_out[0] = x0;
			a_out[1] = x1;
			a_out[2] = y0;
			a_out[3] = y1;
			return true;
		}

		// Yaw, pitch, roll (degrees) of basis a_b against basis a_a, both {dir, right, up}.
		void TurnBetween(const float a_a[3][3], const float a_b[3][3], float a_out[3]) noexcept
		{
			const auto dot = [](const float* a_u, const float* a_v) { return a_u[0] * a_v[0] + a_u[1] * a_v[1] + a_u[2] * a_v[2]; };
			constexpr float toDegrees = 180.0f / 3.14159265f;
			a_out[0] = std::atan2(dot(a_b[0], a_a[1]), dot(a_b[0], a_a[0])) * toDegrees;
			a_out[1] = std::atan2(dot(a_b[0], a_a[2]), dot(a_b[0], a_a[0])) * toDegrees;
			a_out[2] = std::atan2(dot(a_b[1], a_a[2]), dot(a_b[1], a_a[1])) * toDegrees;
		}

		// What a self-check saw (its failures, and the first pass of the session as a baseline): enough to tell a
		// wrong axis map, an off-centre or asymmetric frustum, a camera state, or a stale camera entry apart.
		void LogCheckDetail(const char* a_what, const HiZ::Camera& a_then, const float a_rotate[3][3], const Frustum& a_frustum, float a_tx, float a_ty,
			std::uint32_t a_stillFrames, std::uint64_t a_age, std::uint64_t a_capture) noexcept
		{
			const int reading = (a_then.axisReadings & 1) ? 0 : 1;
			float     now[3][3], then[3][3];
			MappedBasis(a_then, reading, a_rotate, now);
			MappedBasis(a_then, reading, a_then.rotate, then);
			const float render[3][3]{
				{ a_then.viewDir[0], a_then.viewDir[1], a_then.viewDir[2] },
				{ a_then.viewRight[0], a_then.viewRight[1], a_then.viewRight[2] },
				{ a_then.viewUp[0], a_then.viewUp[1], a_then.viewUp[2] }
			};
			float sinceCapture[3], captureVsRender[3], nowVsRender[3];
			TurnBetween(then, now, sinceCapture);
			TurnBetween(render, then, captureVsRender);
			TurnBetween(render, now, nowVsRender);

			// The capture's own NiCamera placed the same way: with this frame's tangents, and with the capture's.
			constexpr float inf = std::numeric_limits<float>::infinity();
			float           own[4]{ inf, -inf, inf, -inf }, ownThen[4]{ inf, -inf, inf, -inf };
			const bool      ownOk = PlaceView(a_then, then, a_tx, a_ty, own[0], own[1], own[2], own[3]);
			float           ttx = std::max(std::abs(a_then.frustumRaw[0]), std::abs(a_then.frustumRaw[1]));
			float           tty = std::max(std::abs(a_then.frustumRaw[2]), std::abs(a_then.frustumRaw[3]));
			if (g_footprint.units == FrustumUnits::kNearPlane && a_frustum.nearPlane > 0.0f) {
				ttx /= a_frustum.nearPlane;  // (the capture's near plane isn't kept; it doesn't change with the FOV)
				tty /= a_frustum.nearPlane;
			}
			const bool ownThenOk = PlaceView(a_then, then, ttx, tty, ownThen[0], ownThen[1], ownThen[2], ownThen[3]);
			const auto player = PlayerCameraView::Read();
			logger::info(
				"view footprint self-check {} on depth frame {} (age {}, NiCamera still {} frames, last turn {:.3f} deg): capture fit {:.3f} deg ({} reading{}, map dir {}{} right {}{} up {}{}), "
				"camera data {} ({} entries for the world camera), view axis at NDC ({:.4f},{:.4f}), scale ({:.4f},{:.4f}) | frustum l/r/t/b then ({:.4f},{:.4f},{:.4f},{:.4f}) now ({:.4f},{:.4f},{:.4f},{:.4f}) near {:.2f} | "
				"capture's own NiCamera on its depth frame: [{:.3f},{:.3f}]x[{:.3f},{:.3f}]{} with today's frustum, [{:.3f},{:.3f}]x[{:.3f},{:.3f}]{} with its own | "
				"turn yaw/pitch/roll (deg): NiCamera since capture ({:.3f},{:.3f},{:.3f}), capture NiCamera vs render ({:.3f},{:.3f},{:.3f}), NiCamera now vs render ({:.3f},{:.3f},{:.3f}) | "
				"player camera: {} then, {} now, FOV world {:.2f} first person {:.2f} adjust {:.3f}",
				a_what, a_capture, a_age, a_stillFrames, g_stillness.lastTurn * 180.0f / 3.14159265f, a_then.fitAngle,
				reading == 0 ? "columns" : "rows", a_then.axisReadings == 3 ? " (both fit)" : "",
				a_then.axisSign[reading][0] < 0 ? "-" : "+", a_then.axisIndex[reading][0], a_then.axisSign[reading][1] < 0 ? "-" : "+", a_then.axisIndex[reading][1],
				a_then.axisSign[reading][2] < 0 ? "-" : "+", a_then.axisIndex[reading][2],
				a_then.cameraSource == 1 ? "cache entry (unjittered)" : a_then.cameraSource == 2 ? "cache entry (jittered)" : a_then.cameraSource == 3 ? "cameraState" : "?",
				a_then.cameraEntries, a_then.axisNdc[0], a_then.axisNdc[1], a_then.scaleX, a_then.scaleY,
				a_then.frustumRaw[0], a_then.frustumRaw[1], a_then.frustumRaw[2], a_then.frustumRaw[3], a_frustum.left, a_frustum.right, a_frustum.top, a_frustum.bottom, a_frustum.nearPlane,
				own[0], own[1], own[2], own[3], ownOk ? "" : " (sideways)", ownThen[0], ownThen[1], ownThen[2], ownThen[3], ownThenOk ? "" : " (sideways)",
				sinceCapture[0], sinceCapture[1], sinceCapture[2], captureVsRender[0], captureVsRender[1], captureVsRender[2], nowVsRender[0], nowVsRender[1], nowVsRender[2],
				CameraStateName(a_then.cameraState), CameraStateName(player.state), player.worldFOV, player.firstPersonFOV, player.fovAdjust);
		}

		// At capture: record the zoom, and check whether the WorldRoot camera's rotation can be mapped
		// onto the render camera this depth was drawn with.
		void CheckFootprint(HiZ::Camera& a_camera)
		{
			a_camera.footprint = false;
			a_camera.looseFit = false;
			a_camera.sameOrientation = false;
			a_camera.axisReadings = 0;
			const auto root = RE::Main::WorldRootCamera();
			if (!root) {
				return;
			}
			const auto frustum = ReadFrustum(root);
			FrustumExtents(frustum, a_camera.frustumX, a_camera.frustumY);
			a_camera.frustumRaw[0] = frustum.left;
			a_camera.frustumRaw[1] = frustum.right;
			a_camera.frustumRaw[2] = frustum.top;
			a_camera.frustumRaw[3] = frustum.bottom;
			a_camera.cameraState = PlayerCameraView::Read().state;
			if (!a_camera.viewSpace) {
				return;
			}
			{
				const float ahead[3]{ a_camera.origin[0] + a_camera.viewDir[0] * 1000.0f, a_camera.origin[1] + a_camera.viewDir[1] * 1000.0f,
					a_camera.origin[2] + a_camera.viewDir[2] * 1000.0f };
				float clip[4];
				ProjectBaked(a_camera, ahead, clip);
				a_camera.axisNdc[0] = clip[3] != 0.0f ? clip[0] / clip[3] : 0.0f;
				a_camera.axisNdc[1] = clip[3] != 0.0f ? clip[1] / clip[3] : 0.0f;
			}
			AxisMap loose{};
			a_camera.looseFit = MatchAxes(a_camera, true, kLooseFit, loose) || MatchAxes(a_camera, false, kLooseFit, loose);

			// Which readings of the rotation reproduce the render basis this frame (both when facing
			// along a world axis); each fitting one is kept with the depth frame.
			AxisMap first{};
			float   fit = 0.0f;
			for (int reading = 0; reading < 2; ++reading) {
				AxisMap map{};
				if (!MatchAxes(a_camera, reading == 0, kStrictFit, map)) {
					continue;
				}
				a_camera.axisReadings |= static_cast<std::uint8_t>(1u << reading);
				for (int b = 0; b < 3; ++b) {
					a_camera.axisIndex[reading][b] = static_cast<std::int8_t>(map.index[b]);
					a_camera.axisSign[reading][b] = static_cast<std::int8_t>(map.sign[b]);
				}
				if (!first.valid) {
					first = map;
				}
				fit = std::max(fit, map.fit);
			}
			if (!a_camera.axisReadings) {
				++g_footprint.reasons[kAxesNoFit];
				return;
			}
			// Caught mid-turn, the depth was drawn from the previous frame's orientation (the render camera
			// trails the NiCamera by a frame): the footprint, which maps the NiCamera, then places a still view
			// off the frame by that turn, rightly. An offset with the NiCamera still for frames can't be that
			// lag, and would misplace every view placed from this capture.
			a_camera.sameOrientation = fit >= kSameFit;
			a_camera.fitAngle = std::acos(std::min(fit, 1.0f)) * 180.0f / 3.14159265f;
			if (!a_camera.sameOrientation) {
				if (g_stillness.frames >= 3) {
					++g_footprint.reasons[kOffsetStill];
					return;
				}
				++g_footprint.reasons[std::acos(fit) <= g_stillness.lastTurn + kSameAngle ? kTrailing : kTrailingBeyond];
			}

			const auto close = [](float a_value) { return std::abs(a_value - 1.0f) < 0.03f; };
			if (g_footprint.units == FrustumUnits::kUnknown) {
				if (close(a_camera.frustumX * a_camera.scaleX) && close(a_camera.frustumY * a_camera.scaleY)) {
					g_footprint.units = FrustumUnits::kTangents;
				} else if (frustum.nearPlane > 0.0f && close(a_camera.frustumX / frustum.nearPlane * a_camera.scaleX) &&
						   close(a_camera.frustumY / frustum.nearPlane * a_camera.scaleY)) {
					g_footprint.units = FrustumUnits::kNearPlane;
				} else {
					++g_footprint.reasons[kUnitsUnknown];
					return;
				}
			}
			float tx = 0.0f, ty = 0.0f;
			FrustumTangents(frustum, tx, ty);
			if (!close(tx * a_camera.scaleX) || !close(ty * a_camera.scaleY)) {
				++g_footprint.reasons[kTangentMismatch];
				return;
			}

			if (!g_footprint.announced) {
				g_footprint.announced = true;
				logger::info(
					"view footprint: first capture fits {} (e.g. {} {}{} {}{} {}{} for dir, right, up), frustum in {} (tan {:.3f} x {:.3f} vs projection {:.3f} x {:.3f})",
					a_camera.axisReadings == 3 ? "both readings (facing along a world axis)" : a_camera.axisReadings == 1 ? "columns" : "rows",
					first.columns ? "columns" : "rows",
					first.sign[0] < 0 ? "-" : "+", first.index[0],
					first.sign[1] < 0 ? "-" : "+", first.index[1],
					first.sign[2] < 0 ? "-" : "+", first.index[2],
					g_footprint.units == FrustumUnits::kTangents ? "tangents" : "near-plane units",
					tx, ty, 1.0f / a_camera.scaleX, 1.0f / a_camera.scaleY);
			}
			a_camera.footprint = true;
		}

		// At cull time: where the current view lands on the depth frame (NDC box), or false if unknown.
		// a_stillFrames: consecutive frames the NiCamera hasn't turned; a_age: frames since the capture
		// (a_capture: its frame).
		bool CurrentView(const HiZ::Camera& a_then, float a_angle, std::uint32_t a_stillFrames, std::uint64_t a_age, std::uint64_t a_capture, float a_out[4])
		{
			const auto root = RE::Main::WorldRootCamera();
			if (!root) {
				return false;
			}
			const auto frustum = ReadFrustum(root);
			float      fx = 0.0f, fy = 0.0f;
			FrustumExtents(frustum, fx, fy);
			if (!(a_then.frustumX > 0.0f && a_then.frustumY > 0.0f && fx > 0.0f && fy > 0.0f)) {
				++g_footprint.reasons[kImplausible];
				return false;
			}

			// Not turned (well under a pixel) and not zoomed: the view is the depth frame itself, provided
			// the depth frame was rendered from this orientation too. That holds if the render camera
			// matched the NiCamera at capture (same orientation, not merely the strict fit: caught mid-turn
			// it trails by up to the fit's ~0.8 deg, and its view is placed below), or if the camera had
			// already been still for longer than the capture is old. Turns count by how far they move the view
			// (NDC) at this zoom: a scope's ~5-degree view (scale ~40) moves 0.04 NDC for a 0.1-degree turn.
			constexpr float kStillShift = 0.0005f;  // NDC: ~0.013 degrees at the default FOV
			const float     maxScale = std::max(a_then.scaleX, a_then.scaleY);
			const float     shift = a_angle * maxScale;
			const bool      sameZoom = std::abs(fx / a_then.frustumX - 1.0f) < 0.002f && std::abs(fy / a_then.frustumY - 1.0f) < 0.002f;
			const bool      renderedHere = a_then.sameOrientation || (a_then.looseFit && a_stillFrames >= a_age + 2);
			if (shift < kStillShift && sameZoom && renderedHere) {
				a_out[0] = -1.0f;
				a_out[1] = 1.0f;
				a_out[2] = -1.0f;
				a_out[3] = 1.0f;
				++g_footprint.reasons[kStill];
				return true;
			}
			if (g_footprint.disabled) {
				++g_footprint.reasons[kTurnedOff];
				return false;
			}
			if (!a_then.footprint || !a_then.axisReadings) {
				++g_footprint.reasons[kNoMapping];
				return false;
			}
			float rotate[3][3];
			for (int r = 0; r < 3; ++r) {
				for (int c = 0; c < 3; ++c) {
					rotate[r][c] = root->world.rotate.entry[r].pt[c];
				}
			}
			float tx = 0.0f, ty = 0.0f;
			FrustumTangents(frustum, tx, ty);
			if (!(tx > 0.0f && tx < 20.0f && ty > 0.0f && ty < 20.0f)) {
				++g_footprint.reasons[kImplausible];
				return false;
			}

			// The current frustum's corner rays under each reading that fit at capture, projected into
			// the depth frame; the box covers all of them.
			float x0 = std::numeric_limits<float>::infinity(), x1 = -x0, y0 = x0, y1 = -x0;
			for (int reading = 0; reading < 2; ++reading) {
				if (!(a_then.axisReadings & (1u << reading))) {
					continue;
				}
				float basis[3][3];  // dir, right, up of the current camera, in the render basis' sense
				MappedBasis(a_then, reading, rotate, basis);
				if (!PlaceView(a_then, basis, tx, ty, x0, x1, y0, y1)) {
					++g_footprint.reasons[kSideways];
					return false;  // the view reaches sideways past the depth frame's image plane
				}
			}

			// Self-check: barely turned at the same zoom, a single-reading view must sit on the depth frame.
			// (With both readings the box is the union of two mirrored guesses and is wide on purpose.) Only
			// from a capture drawn from the NiCamera's own orientation: one caught mid-turn is off the frame by
			// that turn, rightly. v1.58 checked those too: one failed (0.049 deg turned, view [-1.008,0.996] x
			// [-1.034,0.969]) and switched turned views off for the session, which left the async worker no
			// view at all (every object crossing the screen edge kept, nothing out of view: exteriors drew up
			// to 10x previs's main view, and their sun shadows). A failure now refuses this frame's view; only
			// depth frames failing in a row switch turned views off. "Barely turned" and the tolerance are in
			// NDC at this zoom (the turn and the capture's fit, times the projection scale; 1.5 covers the
			// edges' stretch): v1.61-v1.63 measured them in degrees, and a zoomed third-person view (world FOV
			// 5, scale 25 x 40) that turned 0.1 deg moved the view 0.043 NDC, rightly, but failed three depth
			// frames in a row and switched turned views off.
			constexpr std::uint32_t kStrikes = 3;
			if (g_footprint.strikes > 0 && a_capture == g_footprint.strikeCapture) {
				++g_footprint.reasons[kCheckFailed];
				return false;  // (this depth frame failed on an earlier frame)
			}
			const bool  single = a_then.axisReadings == 1 || a_then.axisReadings == 2;
			const float tolerance = 0.03f + 1.5f * (a_angle + a_then.fitAngle * 3.14159265f / 180.0f) * maxScale;
			if (single && a_then.sameOrientation && shift < 0.004f && sameZoom) {
				if (std::abs(x0 + 1.0f) > tolerance || std::abs(x1 - 1.0f) > tolerance || std::abs(y0 + 1.0f) > tolerance || std::abs(y1 - 1.0f) > tolerance) {
					++g_footprint.reasons[kCheckFailed];
					g_footprint.strikeCapture = a_capture;
					g_footprint.disabled = ++g_footprint.strikes >= kStrikes;
					logger::warn(
						"view footprint: self-check failed on depth frame {} ({}/{} in a row: camera turned {:.3f} deg, view [{:.3f},{:.3f}]x[{:.3f},{:.3f}] instead of ~[-1,1]); {}",
						a_capture, g_footprint.strikes, kStrikes, a_angle * 180.0f / 3.14159265f, x0, x1, y0, y1,
						g_footprint.disabled ? "turned views are no longer placed (a still camera still judges edge objects)" : "no view placed from it");
					LogCheckDetail("failed", a_then, rotate, frustum, tx, ty, a_stillFrames, a_age, a_capture);
					return false;
				}
				g_footprint.strikes = 0;
				if (!g_footprint.baselineLogged) {
					g_footprint.baselineLogged = true;
					LogCheckDetail("passed (first of the session, for comparison)", a_then, rotate, frustum, tx, ty, a_stillFrames, a_age, a_capture);
				}
			}

			// A small numeric margin, but never past a frame edge the view doesn't really cross (a margin
			// there would make every object crossing that edge unjudgeable). A side within kSnap of
			// the edge counts as on it: at most a 2-pixel sliver goes unchecked for a frame.
			constexpr float kMargin = 0.002f;
			constexpr float kSnap = 0.004f;
			a_out[0] = x0 >= -1.0f - kSnap ? std::max(x0 - kMargin, -1.0f) : x0 - kMargin;
			a_out[1] = x1 <= 1.0f + kSnap ? std::min(x1 + kMargin, 1.0f) : x1 + kMargin;
			a_out[2] = y0 >= -1.0f - kSnap ? std::max(y0 - kMargin, -1.0f) : y0 - kMargin;
			a_out[3] = y1 <= 1.0f + kSnap ? std::min(y1 + kMargin, 1.0f) : y1 + kMargin;
			++g_footprint.reasons[a_then.axisReadings == 3 ? kTurnedBoth : kTurned];
			return true;
		}

		// ---- the sun's shadow cascades --------------------------------------------------------------------
		// With previs off they read DrawWorld group 0 (Hooks/CullGroups). An object neither the main view nor
		// its sun shadow needs can then be left out of group 0 entirely; that needs this frame's light direction,
		// verified against how the engine placed its shadow camera. Anything unverified keeps every shadow.

		struct SunStats
		{
			std::uint32_t on{ 0 };
			std::uint32_t off{ 0 };            // directional shadows off: no cascade reads group 0
			std::uint32_t disabled{ 0 };       // bSunShadowCulling=0
			std::uint32_t unreadable{ 0 };     // the engine's structures couldn't be read
			std::uint32_t pathChanged{ 0 };    // a call on the active cascade path, or the light's update, isn't the engine's
			std::uint32_t badDirection{ 0 };   // not unit length, or not pointing down (sun at the horizon)
			std::uint32_t badPlacement{ 0 };   // the shadow camera isn't on the sun's side of the view
			std::uint32_t batched{ 0 };        // frames with bCullingBatch set (the other path)
			// The last cascade judged alone (v1.80), per frame with the sun on: on, or why not.
			std::uint32_t farOn{ 0 };
			std::uint32_t farDisabled{ 0 };    // bFarCascadeTrim=0
			std::uint32_t farGodrays{ 0 };     // a single cascade, or godrays read the last one too
			std::uint32_t farSlabBad{ 0 };     // its slab planes didn't validate (normal, order, range)
			float         farNear{ 0.0f };     // last frame on: the slab's view depths, from the planes
			float         farFar{ 0.0f };
			float         farFrom{ 0.0f };     // ... and the depth judged from (corners, margin)
			float         dir[3]{};
			float         range{ 0.0f };
			float         placement{ 0.0f };   // (view - shadow camera) . direction: 15000 when as the engine builds it
		};
		SunStats g_sun;

		// The sun's last cascade (v1.80): judged alone when godrays don't read it (FO4-ENGINE-NOTES 6.1). Its slab planes
		// are last frame's (the light's update runs after the cull), taken along the depth frame's view axis: n . x = c at
		// view depth (c - n . origin) / (n . viewDir). A cascade picked by distance from the eye rather than by view depth
		// starts nearer at the view's corners, so receivers count from that corner depth; both ends get a margin.
		void ReadFarCascade(const HiZ::Camera& a_camera, const Hooks::CullGroups::SunSource& a_source, Occlusion::FrameContext::Sun& a_out)
		{
			if (!Settings::Get().farCascadeTrim) {
				++g_sun.farDisabled;
				return;
			}
			if (a_source.cascades < 2 || !a_source.farAccumulator || a_source.cascades <= std::min(a_source.cascades, a_source.godrayCascades)) {
				++g_sun.farGodrays;
				return;
			}
			const auto dot = [](const float* a_a, const float* a_b) { return a_a[0] * a_b[0] + a_a[1] * a_b[1] + a_a[2] * a_b[2]; };
			const auto& nearPlane = a_source.farSlab[0];
			const auto& farPlane = a_source.farSlab[1];
			const float nearFacing = dot(nearPlane, a_camera.viewDir);
			const float farFacing = dot(farPlane, a_camera.viewDir);
			const float zNear = (nearPlane[3] - dot(nearPlane, a_camera.origin)) / nearFacing;
			const float zFar = (farPlane[3] - dot(farPlane, a_camera.origin)) / farFacing;
			const float range = a_source.range;
			if (!(nearFacing > 0.98f) || !(farFacing < -0.98f) || !(zNear > 0.0f) || !(zFar > zNear) || !(range > 0.0f) ||
				!(zFar > 0.9f * range) || !(zFar < 1.5f * range) || !(a_camera.scaleX > 0.0f) || !(a_camera.scaleY > 0.0f)) {
				++g_sun.farSlabBad;
				return;
			}
			const float tx = 1.1f / a_camera.scaleX;  // (the view's edge, with room for a turn's overhang)
			const float ty = 1.1f / a_camera.scaleY;
			const float corner = std::sqrt(1.0f + tx * tx + ty * ty);
			a_out.farNear = zNear / corner - 0.02f * zNear - 64.0f;
			a_out.farFar = zFar * 1.02f + 64.0f;
			a_out.farOn = a_out.farNear > 0.0f;
			if (!a_out.farOn) {
				++g_sun.farSlabBad;
				return;
			}
			++g_sun.farOn;
			g_sun.farNear = zNear;
			g_sun.farFar = zFar;
			g_sun.farFrom = a_out.farNear;
			Occlusion::SetFarCascade(a_source.farAccumulator);
		}

		void ReadSunState(const HiZ::Camera& a_camera, Occlusion::FrameContext::Sun& a_out)
		{
			using State = Occlusion::FrameContext::Sun::State;
			a_out = {};
			Occlusion::SetFarCascade(nullptr);
			if (!Settings::Get().sunShadowCulling) {
				++g_sun.disabled;
				return;
			}
			Hooks::CullGroups::SunSource source{};
			if (!Hooks::CullGroups::ReadSun(source)) {
				++g_sun.unreadable;
				return;
			}
			g_sun.batched += source.groupsEnabled;
			if (!source.pathIntact) {
				++g_sun.pathChanged;
				return;
			}
			// Both paths check this byte themselves (fixed for the frame before the cull stage): off, neither
			// culls the cascades against group 0.
			if (!source.dirShadows) {
				++g_sun.off;
				a_out.state = State::kOff;
				return;
			}
			const auto  root = RE::Main::WorldRootCamera();
			const float length = std::sqrt(source.dir[0] * source.dir[0] + source.dir[1] * source.dir[1] + source.dir[2] * source.dir[2]);
			if (!root || !(length > 0.99f && length < 1.01f) || !(source.dir[2] < -0.02f)) {
				++g_sun.badDirection;
				return;
			}
			float dir[3];
			for (int i = 0; i < 3; ++i) {
				dir[i] = source.dir[i] / length;
			}
			const auto& eye = root->world.translate;
			const float placement = (eye.x - source.cameraPos[0]) * dir[0] + (eye.y - source.cameraPos[1]) * dir[1] + (eye.z - source.cameraPos[2]) * dir[2];
			g_sun.placement = placement;
			if (!(placement > 1000.0f)) {
				++g_sun.badPlacement;
				return;
			}
			std::copy_n(dir, 3, g_sun.dir);
			g_sun.range = source.range;
			++g_sun.on;

			// Into the depth frame's view space (right, up, forward).
			const auto dot = [&](const float a_axis[3]) { return dir[0] * a_axis[0] + dir[1] * a_axis[1] + dir[2] * a_axis[2]; };
			a_out.dir[0] = dot(a_camera.viewRight);
			a_out.dir[1] = dot(a_camera.viewUp);
			a_out.dir[2] = dot(a_camera.viewDir);
			// Receivers end at the cascade range (padded: splits run along view depth, and the range can change);
			// the engine eases the direction over time when the sun moves; shadow maps filter over a few texels.
			a_out.reach = std::clamp(source.range, 100.0f, 1.0e6f) * 1.25f + 256.0f;
			a_out.spread = 0.0175f;  // sin(1 deg)
			a_out.margin = 64.0f;
			a_out.state = State::kOn;
			ReadFarCascade(a_camera, source, a_out);
		}

		void LogSun()
		{
			logger::info(
				"sun shadows per interval: on {} | off {} | not used: disabled {}, unreadable {}, path changed {}, direction {}, placement {} || batched frames {} || last direction ({:.3f},{:.3f},{:.3f}) range {:.0f} placement {:.0f} || last cascade judged alone: {} frames | not: off {}, read by godrays or single {}, slab unverified {} | its slab (view depth) {:.0f}-{:.0f}, receivers counted from {:.0f}",
				g_sun.on, g_sun.off, g_sun.disabled, g_sun.unreadable, g_sun.pathChanged, g_sun.badDirection, g_sun.badPlacement, g_sun.batched,
				g_sun.dir[0], g_sun.dir[1], g_sun.dir[2], g_sun.range, g_sun.placement,
				g_sun.farOn, g_sun.farDisabled, g_sun.farGodrays, g_sun.farSlabBad, g_sun.farNear, g_sun.farFar, g_sun.farFrom);
			SunStats next{};
			std::copy_n(g_sun.dir, 3, next.dir);
			next.range = g_sun.range;
			next.placement = g_sun.placement;
			g_sun = next;
		}

		// ---- timing -------------------------------------------------------------------------------

		double QpcMs(std::int64_t a_ticks)
		{
			static const double frequency = [] {
				LARGE_INTEGER value{};
				QueryPerformanceFrequency(&value);
				return static_cast<double>(value.QuadPart);
			}();
			return static_cast<double>(a_ticks) * 1000.0 / frequency;
		}

		std::int64_t Qpc()
		{
			LARGE_INTEGER value{};
			QueryPerformanceCounter(&value);
			return value.QuadPart;
		}

		// ---- the frame in buckets, per mode ----------------------------------------------------------------------
		// The v1.27 run showed CBRO's own cost falling (tests 0.9 ms a frame, the cull stage ~1.7 ms) while the same-
		// session A/B gap to previs stayed at +2 ms: the gap is not in the two stages CBRO times. So every frame is
		// split at the render-stage hooks (all on the main thread) into marks 0-8: cull begin, cull end, pre-pass
		// begin, pre-pass end, sun-cascades begin, sun-cascades end (Render_PreUI+0x1BF: the light's update, the
		// cascade cull over group 0 and, with previs off, the shadow-map draws of every caster in range), forward
		// begin, forward end, and the next frame's cull begin. Buckets: the four stages, "between" (the gaps inside
		// Render_PreUI: HBAO and whatever else runs between the stages) and "rest" (from the forward pass to the next
		// cull: deferred composite, post, UI, present and the game's update), kept per mode so a 600-frame interval
		// with F8 presses inside still compares like with like.
		enum Bucket : std::size_t
		{
			kBucketCull,
			kBucketPrePass,
			kBucketHiZ,  // CBRO's depth capture and hi-z build, right after the pre-pass (marks 3-4; previs mode: nothing)
			kBucketSun,
			kBucketForward,
			kBucketBetween,
			kBucketRest,
			kBucketCount
		};
		// (the "shadow maps" stage, Render_PreUI+0x1BF, renders the lamps' shadow maps as well as the sun's cascades: the
		// v1.31 interior run had 10 ms of lamp shadow maps in it with no sun at all)
		constexpr std::array   kBucketNames{ "cull"sv, "pre-pass"sv, "hi-z"sv, "shadow maps"sv, "forward"sv, "between"sv, "rest"sv };
		// Marks: 0 cull begin, 1 cull end, 2 pre-pass begin, 3 pre-pass end (the engine's), 4 hi-z capture end, 5 shadow
		// maps begin, 6 shadow maps end, 7 forward begin, 8 forward end, 9 the next frame's cull begin.
		constexpr std::size_t  kMarkCount = 10;

		struct ModeBuckets
		{
			std::array<double, kBucketCount> ms{};
			std::uint32_t                    frames{ 0 };

			std::string Describe() const
			{
				if (!frames) {
					return "-";
				}
				std::string text;
				double      total = 0.0;
				for (std::size_t i = 0; i < kBucketCount; ++i) {
					text += std::format("{}{} {:.2f}", i ? " | " : "", kBucketNames[i], ms[i] / frames);
					total += ms[i];
				}
				return std::format("{} = {:.2f} ({} frames)", text, total / frames, frames);
			}
		};

		// Adds the spans between consecutive present marks (0 = absent, in ms) to the buckets: a stage's own span when
		// both its marks are present and adjacent, "rest" for a span ending at the next cull begin, else "between".
		void FoldMarks(const double a_marks[kMarkCount], ModeBuckets& a_out) noexcept
		{
			std::size_t last = 0;
			for (std::size_t j = 1; j < kMarkCount; ++j) {
				if (!(a_marks[j] > 0.0)) {
					continue;
				}
				Bucket bucket = kBucketBetween;
				if (j == last + 1) {
					switch (last) {
					case 0:
						bucket = kBucketCull;
						break;
					case 2:
						bucket = kBucketPrePass;
						break;
					case 3:
						bucket = kBucketHiZ;
						break;
					case 5:
						bucket = kBucketSun;
						break;
					case 7:
						bucket = kBucketForward;
						break;
					default:
						break;
					}
				}
				if (j == kMarkCount - 1 && bucket == kBucketBetween) {
					bucket = kBucketRest;
				}
				a_out.ms[bucket] += std::max(0.0, a_marks[j] - a_marks[last]);
				last = j;
			}
			++a_out.frames;
		}

		// Frame modes for the per-mode figures: 0 previs (CBRO off), 1 CBRO feed, 2 CBRO classic (Core/Feed's path).
		constexpr std::size_t kModes = 3;
		constexpr std::array  kModeNames{ "previs"sv, "CBRO feed"sv, "CBRO classic"sv };

		int CurrentMode() noexcept
		{
			if (!Occlusion::Active()) {
				return 0;
			}
			return Feed::Current() == Feed::Path::kFeed ? 1 : 2;
		}

		struct Timing
		{
			std::array<std::int64_t, kMarkCount> marks{};  // this frame's QPC marks (0 = not seen)
			int                                  mode{ -1 };  // the frame's mode once its cull begin ran (see kModeNames)
			std::int64_t                         setupTicks{ 0 };
			std::int64_t                         captureTicks{ 0 };
			std::uint32_t                        cullFrames{ 0 };
			std::uint32_t                        prepassFrames{ 0 };
			std::array<ModeBuckets, kModes>      cpu{};
		};
		Timing g_timing;

		// The previous frame's buckets, closed by this frame's cull begin.
		void CloseFrameTiming(std::int64_t a_now)
		{
			if (g_timing.marks[0] == 0 || g_timing.mode < 0) {
				return;
			}
			double marks[kMarkCount];
			for (std::size_t i = 0; i + 1 < kMarkCount; ++i) {
				marks[i] = g_timing.marks[i] ? QpcMs(g_timing.marks[i]) : 0.0;
			}
			marks[kMarkCount - 1] = QpcMs(a_now);
			FoldMarks(marks, g_timing.cpu[static_cast<std::size_t>(g_timing.mode)]);
		}

		// The same buckets on the GPU: D3D11 timestamp queries on the engine's immediate context, a disjoint bracket
		// per frame from its cull begin to the next frame's cull begin, timestamps at the marks, read back without
		// waiting once the GPU is done (4+ frames later). A span includes the GPU's idle time between its marks (the
		// GPU waiting for the CPU), so a span that grows with a mode while its CPU bucket doesn't is GPU work, and a
		// frame whose GPU total equals the CPU total in both modes is paced by whichever the buckets say.
		class GpuTimeline
		{
		public:
			void FrameBegin(int a_mode)
			{
				const auto context = Util::GetContext();
				const auto device = Util::GetDevice();
				if (!context || !device || m_failed) {
					return;
				}
				if (!m_ready && !Create(device)) {
					return;
				}
				Poll(context);
				auto& previous = m_ring[m_current];
				if (previous.open) {
					context->End(previous.marks[kMarkCount - 1]);  // this cull begin closes the previous frame's "rest"
					previous.written |= 1u << (kMarkCount - 1);
					context->End(previous.disjoint);
					previous.open = false;
					previous.pending = true;
				}
				m_current = (m_current + 1) % kRing;
				auto& frame = m_ring[m_current];
				if (frame.pending) {
					++m_dropped;  // never read: the GPU is more than a ring behind
					frame.pending = false;
				}
				frame.open = true;
				frame.mode = a_mode;
				frame.written = 0;
				context->Begin(frame.disjoint);
				Mark(0);
			}

			void Mark(std::size_t a_index)
			{
				auto& frame = m_ring[m_current];
				if (!frame.open || m_failed || a_index + 1 >= kMarkCount) {
					return;
				}
				if (const auto context = Util::GetContext()) {
					context->End(frame.marks[a_index]);
					frame.written |= 1u << a_index;
				}
			}

			std::array<ModeBuckets, kModes> Take() noexcept
			{
				auto out = m_gpu;
				m_gpu = {};
				return out;
			}

			std::string Status() const
			{
				if (m_failed) {
					return "unavailable (query creation failed)";
				}
				return std::format("read {} frames, dropped {}, disjoint {}", m_read, m_dropped, m_disjoint);
			}

		private:
			static constexpr std::uint32_t kRing = 8;

			struct Frame
			{
				ID3D11Query*  disjoint{ nullptr };
				ID3D11Query*  marks[kMarkCount]{};
				bool          open{ false };
				bool          pending{ false };
				int           mode{ 0 };
				std::uint32_t written{ 0 };
			};

			bool Create(ID3D11Device* a_device)
			{
				for (auto& frame : m_ring) {
					D3D11_QUERY_DESC desc{};
					desc.Query = D3D11_QUERY_TIMESTAMP_DISJOINT;
					if (FAILED(a_device->CreateQuery(&desc, &frame.disjoint))) {
						m_failed = true;
					}
					desc.Query = D3D11_QUERY_TIMESTAMP;
					for (auto& mark : frame.marks) {
						if (FAILED(a_device->CreateQuery(&desc, &mark))) {
							m_failed = true;
						}
					}
				}
				if (m_failed) {
					logger::error("gpu timeline: timestamp queries couldn't be created; no GPU times this session");
					return false;
				}
				m_ready = true;
				return true;
			}

			void Poll(ID3D11DeviceContext* a_context)
			{
				// Oldest first; once one isn't finished, the newer ones aren't either.
				for (std::uint32_t k = 1; k < kRing; ++k) {
					auto& frame = m_ring[(m_current + k) % kRing];
					if (!frame.pending) {
						continue;
					}
					D3D11_QUERY_DATA_TIMESTAMP_DISJOINT disjoint{};
					const auto hr = a_context->GetData(frame.disjoint, &disjoint, sizeof(disjoint), D3D11_ASYNC_GETDATA_DONOTFLUSH);
					if (hr == S_FALSE) {
						break;
					}
					frame.pending = false;
					if (FAILED(hr) || disjoint.Disjoint || disjoint.Frequency == 0) {
						++m_disjoint;
						continue;
					}
					double marks[kMarkCount]{};
					bool   ok = true;
					for (std::size_t i = 0; i < kMarkCount && ok; ++i) {
						if (!(frame.written & (1u << i))) {
							continue;
						}
						std::uint64_t stamp = 0;
						if (a_context->GetData(frame.marks[i], &stamp, sizeof(stamp), D3D11_ASYNC_GETDATA_DONOTFLUSH) != S_OK) {
							ok = false;
							break;
						}
						marks[i] = static_cast<double>(stamp) * 1000.0 / static_cast<double>(disjoint.Frequency);
					}
					if (ok && marks[0] > 0.0) {
						FoldMarks(marks, m_gpu[static_cast<std::size_t>(std::clamp(frame.mode, 0, static_cast<int>(kModes) - 1))]);
						++m_read;
					} else {
						++m_disjoint;
					}
				}
			}

			std::array<Frame, kRing>   m_ring{};
			std::uint32_t              m_current{ 0 };
			bool                       m_ready{ false };
			bool                       m_failed{ false };
			std::array<ModeBuckets, kModes> m_gpu{};
			std::uint64_t              m_read{ 0 };
			std::uint64_t              m_dropped{ 0 };
			std::uint64_t              m_disjoint{ 0 };
		};
		GpuTimeline g_gpu;

		// ---- A/B frame time -----------------------------------------------------------------------
		// Base frame time (pre-pass to pre-pass) per mode. A switch starts a new segment; its first
		// kSettleFrames frames (previs flush, streaks, first depth) and any hitch or menu frame are left
		// out, so standing still and pressing the toggle gives a like-for-like comparison.

		constexpr std::uint32_t kSettleFrames = 120;
		constexpr std::uint32_t kLoadSettleFrames = 600;  // after a load: streaming, shader compiles and LOD builds belong to neither mode
		constexpr double        kHitchMs = 250.0;
		constexpr double        kLoadMs = 1000.0;        // a frame this long is a load or fast travel: new location
		constexpr float         kLocationRadius = 2048.0f;  // moving farther than this starts a new comparison

		struct ModeTime
		{
			double        sum{ 0.0 };
			std::uint32_t count{ 0 };
		};

		struct FrameClock
		{
			std::int64_t        last{ 0 };
			int                 mode{ -1 };  // 0 previs, 1 CBRO
			std::uint32_t       sinceSwitch{ 0 };
			std::vector<float>  segment;       // settled frame times of the current segment
			std::int64_t        segmentStart{ 0 };
			std::array<ModeTime, kModes> session{};  // settled frames at the current location
			std::array<ModeTime, kModes> still{};    // ... of which the camera was exactly where it was the frame before
			RE::NiPoint3        anchor{};       // where the current location's comparison started
			bool                anchored{ false };
			std::uint32_t       location{ 0 };
			bool                dumpedHere{ false };  // the kept-object dump ran at this location
			double              intervalSum{ 0.0 };
			double              intervalMax{ 0.0 };
			std::uint32_t       intervalCount{ 0 };
			ModeTime            intervalStill{};  // this interval's frames with the camera still (what a standing-still overlay shows)
			std::array<ModeTime, kModes> intervalMode{};  // this interval's frames by mode (hitches and menus left out)
			std::uint32_t       sinceLoad{ 0 };
			RE::NiPoint3        lastPosition{};
			float               lastRotate[3][3]{};
			bool                lastCameraKnown{ false };
		};
		FrameClock g_frames;

		// ---- asynchronous verdicts (Core/Async): what the worker's job needs from the frame -------------------------
		// The worker judges this frame's candidates for the NEXT frame's walk, so its context is this frame's with the
		// dilation grown by the movement the next frame may bring (twice the last frame's, plus a floor) and the view box
		// widened toward the turn it may bring (JobView). The next frame uses the map only if its camera stayed within
		// the margins and its view inside the box.

		// The view a job judges with (v1.70). Up to v1.69 it was the frame's own, and the map was used after any turn
		// within the turn margin (1-8 deg): an object in the strip the turn brought into view had been judged out of view
		// (left out of the walk, or out of group 0) or by its part inside the old view, and was missing for a frame at the
		// leading edge of every turn. Now the box reaches where the frame's turn, repeated once and twice, places the
		// view: only the sides the camera turns toward grow, by the turn's speed. Every side also gets a floor within the
		// view overhang (fViewOverhang, and the lights'), for turns the last one doesn't predict: a still view's side that
		// far past the depth frame's edge judges every object crossing it as the edge itself would, so a still camera
		// loses nothing but objects (and sun shadows) wholly inside the floor's strip beyond the screen, which stay filed
		// (the engine's frustum test still drops them from the main view).
		struct JobView
		{
			bool          valid{ false };   // a job was submitted with it
			float         box[4]{};         // NDC on the depth frame {x0, x1, y0, y1} (infinite: no view, nothing judged by one)
			float         widen[4]{};       // how far each side lies past the frame's own view
			bool          placed{ false };  // a later view can be placed on the depth frame (else only its epoch holds)
			std::uint32_t epoch{ 0 };       // the view epoch its records carry
			HiZ::Camera   camera{};         // the depth frame's camera
		};

		struct AsyncFrame
		{
			Occlusion::FrameContext context{};   // this frame's (as given to Occlusion::BeginFrame)
			Async::Pose             pose{};      // the camera at this frame's cull begin
			bool                    lastKnown{ false };
			Async::Pose             last{};      // last frame's camera (the frame's own motion)
			float                   moveMargin{ 0.0f };
			float                   frameMove{ 0.0f };   // the camera's movement since last frame's cull begin (the log)
			JobView                 next{};      // for the job this frame submits
			JobView                 judged{};    // the last submitted job's (what the map this frame reads was judged with)
		};
		AsyncFrame g_asyncFrame;

		float PoseDistance(const Async::Pose& a_a, const Async::Pose& a_b) noexcept
		{
			const float dx = a_a.eye[0] - a_b.eye[0], dy = a_a.eye[1] - a_b.eye[1], dz = a_a.eye[2] - a_b.eye[2];
			return std::sqrt(dx * dx + dy * dy + dz * dz);
		}

		// The current camera as a pose; false without one.
		bool ReadPose(Async::Pose& a_out) noexcept
		{
			const auto root = RE::Main::WorldRootCamera();
			if (!root) {
				return false;
			}
			a_out.eye[0] = root->world.translate.x;
			a_out.eye[1] = root->world.translate.y;
			a_out.eye[2] = root->world.translate.z;
			for (int r = 0; r < 3; ++r) {
				for (int c = 0; c < 3; ++c) {
					a_out.rotate[r][c] = root->world.rotate.entry[r].pt[c];
				}
			}
			FrustumExtents(ReadFrustum(root), a_out.zoom[0], a_out.zoom[1]);
			return true;
		}

		// The view this frame's job judges with (JobView), from the frame's context and its turn since a_last.
		void PrepareJobView(const Occlusion::FrameContext& a_context, const Async::Pose& a_pose, bool a_turnKnown, const Async::Pose& a_last, JobView& a_out)
		{
			a_out = {};
			std::copy_n(a_context.view, 4, a_out.box);
			a_out.epoch = a_context.viewEpoch;
			const auto root = RE::Main::WorldRootCamera();
			if (!a_context.snapshot || !std::isfinite(a_context.view[0]) || !root) {
				return;
			}
			const auto& camera = a_context.snapshot->camera;
			if (g_footprint.disabled || !camera.footprint || !camera.axisReadings) {
				return;  // (a still view on a depth frame with no mapping: held within its epoch only)
			}
			a_out.camera = camera;
			a_out.placed = true;
			float box[4]{ a_context.view[0], a_context.view[1], a_context.view[2], a_context.view[3] };
			if (a_turnKnown) {
				// The frame's turn E = R_last^T * R, applied again once and twice: R * E^k extrapolates under either
				// reading of the matrices (a world-frame turn, or a local one).
				float step[3][3]{};
				for (int r = 0; r < 3; ++r) {
					for (int c = 0; c < 3; ++c) {
						for (int k = 0; k < 3; ++k) {
							step[r][c] += a_last.rotate[k][r] * a_pose.rotate[k][c];
						}
					}
				}
				float tx = 0.0f, ty = 0.0f;
				FrustumTangents(ReadFrustum(root), tx, ty);
				float turned[3][3];
				std::memcpy(turned, a_pose.rotate, sizeof(turned));
				for (int times = 0; times < 2; ++times) {
					float next[3][3]{};
					for (int r = 0; r < 3; ++r) {
						for (int c = 0; c < 3; ++c) {
							for (int k = 0; k < 3; ++k) {
								next[r][c] += turned[r][k] * step[k][c];
							}
						}
					}
					std::memcpy(turned, next, sizeof(turned));
					float ahead[4];
					if (PlaceRotation(camera, turned, tx, ty, ahead)) {  // (a view placed sideways: the next frame's check fails it)
						box[0] = std::min(box[0], ahead[0]);
						box[1] = std::max(box[1], ahead[1]);
						box[2] = std::min(box[2], ahead[2]);
						box[3] = std::max(box[3], ahead[3]);
					}
				}
			}
			// The floor never takes a side past the overhang (v1.74): a side the view already reaches past the depth frame by
			// that much is where the turn goes, and the prediction covers it; up to v1.73 the floor took the leading side of
			// every slow pan past it, so each object and sun shadow crossing it was edge in the map and judged again in the walk.
			const float overhang = std::min(Settings::Get().viewOverhang, Occlusion::kLightOverhang);
			const float least = 0.9f * overhang;
			const float past[4]{ -1.0f - a_context.view[0], a_context.view[1] - 1.0f, -1.0f - a_context.view[2], a_context.view[3] - 1.0f };
			const float ahead[4]{ a_context.view[0] - box[0], box[1] - a_context.view[1], a_context.view[2] - box[2], box[3] - a_context.view[3] };
			for (int i = 0; i < 4; ++i) {
				a_out.widen[i] = std::max(ahead[i], std::clamp(overhang - past[i], 0.0f, least));
			}
			a_out.box[0] = a_context.view[0] - a_out.widen[0];
			a_out.box[1] = a_context.view[1] + a_out.widen[1];
			a_out.box[2] = a_context.view[2] - a_out.widen[2];
			a_out.box[3] = a_context.view[3] + a_out.widen[3];
		}

		// Whether this frame's view (camera at a_pose, view epoch a_epoch) lies inside the view a job judged with: placed on
		// the job's depth frame, raw, to within CurrentView's still shift. An unplaced job holds within its own epoch only
		// (the tested bounds' angular slack covers any two cameras of one; with the verdict cache off, none).
		bool ViewHeld(const JobView& a_job, const Async::Pose& a_pose, std::uint32_t a_epoch, bool a_cacheEnabled) noexcept
		{
			if (!a_job.valid || !std::isfinite(a_job.box[0])) {
				return true;  // (no job: no map; no view: no verdict rests on one)
			}
			if (!a_job.placed || g_footprint.disabled) {
				return a_cacheEnabled && a_epoch == a_job.epoch;
			}
			const auto root = RE::Main::WorldRootCamera();
			if (!root) {
				return false;
			}
			float tx = 0.0f, ty = 0.0f;
			FrustumTangents(ReadFrustum(root), tx, ty);
			float view[4];
			if (!PlaceRotation(a_job.camera, a_pose.rotate, tx, ty, view)) {
				return false;
			}
			constexpr float kSlack = 0.0005f;  // (CurrentView's kStillShift)
			return view[0] >= a_job.box[0] - kSlack && view[1] <= a_job.box[1] + kSlack && view[2] >= a_job.box[2] - kSlack && view[3] <= a_job.box[3] + kSlack;
		}

		// ---- what the culled frames draw by how fast the camera turns and moves (diagnostic, v1.71) ----------------
		// Per interval, the frames binned by the NiCamera's turn since the previous frame (degrees), and by its movement
		// (units, v1.73): how often the worker's map was used, and the main view's and the other views' registrations,
		// confirming and edge verdicts (walk and worker together). The aggregate spreads can't tell a turn's leading-edge
		// strip from a load or a flick.
		struct MotionTable
		{
			const char*          what{ nullptr };
			std::array<float, 7> limits{};
			struct Bin
			{
				std::uint32_t frames{ 0 };
				std::uint32_t mapUsed{ 0 };
				double        kept{ 0.0 };
				float         keptMax{ 0.0f };
				double        other{ 0.0 };
				float         otherMax{ 0.0f };
				double        confirming{ 0.0 };
				double        edge{ 0.0 };
				double        finalEdge{ 0.0 };
				double        finalConfirming{ 0.0 };
			};
			std::array<Bin, 8> bins{};
		};
		MotionTable g_turnTable{ "camera turn (deg per frame)", { 0.1f, 0.5f, 1.0f, 2.0f, 4.0f, 8.0f, 16.0f } };
		MotionTable g_moveTable{ "camera movement (units per frame)", { 0.25f, 1.0f, 2.0f, 4.0f, 8.0f, 16.0f, 32.0f } };

		void NoteMotionSample(MotionTable& a_table, float a_value, const Occlusion::FrameSample& a_sample, bool a_mapUsed)
		{
			std::size_t index = 0;
			while (index < a_table.limits.size() && a_value >= a_table.limits[index]) {
				++index;
			}
			auto&       bin = a_table.bins[index];
			const auto& sample = a_sample;
			++bin.frames;
			bin.mapUsed += a_mapUsed ? 1u : 0u;
			bin.kept += sample.kept;
			bin.keptMax = std::max(bin.keptMax, sample.kept);
			bin.other += sample.other;
			bin.otherMax = std::max(bin.otherMax, sample.other);
			bin.confirming += sample.confirming;
			bin.edge += sample.edge;
			bin.finalEdge += sample.finalEdge;
			bin.finalConfirming += sample.finalConfirming;
		}

		// How old the depth a culled frame judges by is (frames since its capture, at the cull begin), and whether a newer
		// capture had been read back by the cull's end, when the worker's job starts (diagnostic, v1.73): what a later
		// pick of the depth could gain. A turn's leading strip with no depth is the turn times that age.
		struct DepthAge
		{
			std::array<std::uint32_t, 6> atBegin{};    // [age], 5 = 5 or more
			std::array<std::uint32_t, 6> newerAtEnd{}; // [its age then]
			std::uint32_t                frames{ 0 };
		};
		DepthAge g_depthAge;

		void LogDepthAge()
		{
			const auto row = [](const std::array<std::uint32_t, 6>& a_counts) {
				std::string text;
				for (std::size_t i = 1; i < a_counts.size(); ++i) {
					if (a_counts[i]) {
						text += std::format(" {}{}:{}", i, i + 1 == a_counts.size() ? "+" : "", a_counts[i]);
					}
				}
				return text.empty() ? std::string(" none") : text;
			};
			if (g_depthAge.frames) {
				std::uint32_t newer = 0;
				for (const auto count : g_depthAge.newerAtEnd) {
					newer += count;
				}
				logger::info(
					"depth age this interval (culled frames, frames since the capture judged by):{} | a newer capture read back by the cull's end in {} of {} frames (its age then:{})",
					row(g_depthAge.atBegin), newer, g_depthAge.frames, row(g_depthAge.newerAtEnd));
			}
			g_depthAge = {};
		}

		void NoteMotion(float a_turn, float a_move, bool a_mapUsed)
		{
			const auto sample = Occlusion::LastFrameSample();
			if (!sample.valid) {
				return;
			}
			NoteMotionSample(g_turnTable, a_turn * 180.0f / 3.14159265f, sample, a_mapUsed);
			NoteMotionSample(g_moveTable, a_move, sample, a_mapUsed);
		}

		void LogMotionTable(MotionTable& a_table)
		{
			std::string line;
			for (std::size_t i = 0; i < a_table.bins.size(); ++i) {
				const auto& bin = a_table.bins[i];
				if (!bin.frames) {
					continue;
				}
				const double n = bin.frames;
				const auto   range = i == 0 ? std::format("<{}", a_table.limits[0]) :
				                     i == a_table.limits.size() ? std::format("{}+", a_table.limits[i - 1]) :
				                                                  std::format("{}-{}", a_table.limits[i - 1], a_table.limits[i]);
				line += std::format(
					"{}{}: {} frames, map used {:.0f}%, main view kept {:.0f} (max {:.0f}), other views {:.0f} (max {:.0f}), confirming {:.0f}, edge {:.0f}, walk decided edge {:.0f} confirming {:.0f}",
					line.empty() ? "" : " | ", range, bin.frames, 100.0 * bin.mapUsed / n, bin.kept / n, bin.keptMax, bin.other / n, bin.otherMax,
					bin.confirming / n, bin.edge / n, bin.finalEdge / n, bin.finalConfirming / n);
			}
			if (!line.empty()) {
				logger::info("culled frames by {} this interval: {}", a_table.what, line);
			}
			a_table.bins = {};
		}

		void CloseSegment(std::int64_t a_now)
		{
			if (g_frames.mode < 0) {
				return;
			}
			auto& samples = g_frames.segment;
			if (samples.size() >= 60) {
				double sum = 0.0;
				for (const auto ms : samples) {
					sum += ms;
				}
				const auto middle = samples.begin() + static_cast<std::ptrdiff_t>(samples.size() / 2);
				std::nth_element(samples.begin(), middle, samples.end());
				const double avg = sum / static_cast<double>(samples.size());
				logger::info(
					"A/B segment: {} for {:.1f} s | {} settled frames | avg {:.2f} ms ({:.0f} fps) | median {:.2f} ms",
					kModeNames[g_frames.mode], QpcMs(a_now - g_frames.segmentStart) / 1000.0, samples.size(), avg, 1000.0 / avg, *middle);
			}
			samples.clear();
		}

		// Previs against each CBRO mode that has enough frames at this location.
		void LogSessionAB()
		{
			const auto report = [](const std::array<ModeTime, kModes>& a_times, std::string_view a_what) {
				const auto& previs = a_times[0];
				if (previs.count < 60) {
					return;
				}
				const double previsMs = previs.sum / previs.count;
				for (std::size_t mode = 1; mode < kModes; ++mode) {
					const auto& cbro = a_times[mode];
					if (cbro.count < 60) {
						continue;
					}
					const double cbroMs = cbro.sum / cbro.count;
					logger::info(
						"A/B at location {} ({}): previs {:.2f} ms ({:.0f} fps, {} frames) | {} {:.2f} ms ({:.0f} fps, {} frames) | {} is {:.1f}% {}",
						g_frames.location, a_what, previsMs, 1000.0 / previsMs, previs.count, kModeNames[mode], cbroMs, 1000.0 / cbroMs, cbro.count,
						kModeNames[mode], std::abs(previsMs / cbroMs - 1.0) * 100.0, cbroMs <= previsMs ? "faster" : "slower");
				}
			};
			report(g_frames.session, "settled frames");
			// Standing still only: the comparison the user makes with the overlay (movement changes both modes' load).
			report(g_frames.still, "camera still");
		}

		// A new place (load, fast travel, or walked away): close the comparison and start another.
		void NewLocation(std::string_view a_reason, std::int64_t a_now)
		{
			CloseSegment(a_now);
			LogSessionAB();
			g_frames.session = {};
			g_frames.still = {};
			g_frames.sinceSwitch = 0;
			g_frames.segmentStart = a_now;
			g_frames.anchored = false;
			g_frames.dumpedHere = false;
			++g_frames.location;
			SetDiff::Reset();
			logger::info("A/B: location {} starts ({})", g_frames.location, a_reason);
		}

		void OnFrameBoundary()
		{
			const auto now = Qpc();
			const int  mode = CurrentMode();
			if (mode != g_frames.mode) {
				CloseSegment(now);
				g_frames.mode = mode;
				g_frames.sinceSwitch = 0;
				g_frames.segmentStart = now;
			}
			bool cameraStill = false;
			if (const auto root = RE::Main::WorldRootCamera()) {
				const auto& position = root->world.translate;
				if (!g_frames.anchored) {
					g_frames.anchor = position;
					g_frames.anchored = true;
				} else if ((position - g_frames.anchor).Length() > kLocationRadius) {
					NewLocation("moved away", now);
					g_frames.anchor = position;
					g_frames.anchored = true;
				}
				// The same camera as the frame before, within the idle sway of a standing player (the v1.16 run
				// showed the camera is never bit-identical two frames running): what "standing still" means for
				// an overlay reading. Limits: 2 units of movement and ~0.06 degrees of turn per frame.
				const auto& rotate = root->world.rotate.entry;
				float       squared = 0.0f;
				for (int r = 0; r < 3; ++r) {
					for (int c = 0; c < 3; ++c) {
						const float d = rotate[r].pt[c] - g_frames.lastRotate[r][c];
						squared += d * d;
					}
				}
				cameraStill = g_frames.lastCameraKnown && (position - g_frames.lastPosition).Length() <= 2.0f && squared <= 2.0f * 0.001f * 0.001f;
				g_frames.lastPosition = position;
				std::memcpy(g_frames.lastRotate, rotate, sizeof(g_frames.lastRotate));
				g_frames.lastCameraKnown = true;
			} else {
				g_frames.lastCameraKnown = false;
			}
			if (g_frames.last) {
				const double ms = QpcMs(now - g_frames.last);
				if (ms > kLoadMs) {
					NewLocation("loading pause", now);
				}
				g_frames.intervalSum += ms;
				g_frames.intervalMax = std::max(g_frames.intervalMax, ms);
				++g_frames.intervalCount;

				const auto ui = RE::UI::GetSingleton();
				const bool menu = ui && ui->menuMode != 0;
				if (ms < kHitchMs && !menu) {
					g_frames.intervalMode[mode].sum += ms;
					++g_frames.intervalMode[mode].count;
				}
				if (cameraStill && ms < kHitchMs && !menu) {
					g_frames.intervalStill.sum += ms;
					++g_frames.intervalStill.count;
				}
				const bool warm = ++g_frames.sinceLoad > kLoadSettleFrames;
				if (++g_frames.sinceSwitch > kSettleFrames && warm && ms < kHitchMs && !menu) {
					g_frames.segment.push_back(static_cast<float>(ms));
					g_frames.session[mode].sum += ms;
					++g_frames.session[mode].count;
					if (cameraStill) {
						g_frames.still[mode].sum += ms;
						++g_frames.still[mode].count;
					}
				}
			}
			g_frames.last = now;
		}

		void ResetAB()
		{
			if (g_frames.mode >= 0) {
				NewLocation("game loaded", Qpc());
			}
			g_frames.segment.clear();
			g_frames.session = {};
			g_frames.still = {};
			g_frames.mode = -1;
			g_frames.last = 0;
			g_frames.sinceLoad = 0;
			g_frames.anchored = false;
			g_frames.lastCameraKnown = false;
		}

		void LogTiming()
		{
			const double intervalFrames = std::max(1u, g_frames.intervalCount);
			const double frameMs = g_frames.intervalSum / intervalFrames;
			const auto&  still = g_frames.intervalStill;
			const double stillMs = still.count ? still.sum / still.count : 0.0;
			const auto byMode = [](const ModeTime& a_time) {
				return a_time.count ? std::format("{:.2f} ms ({:.0f} fps) over {} frames", a_time.sum / a_time.count, 1000.0 * a_time.count / a_time.sum, a_time.count) : std::string("-");
			};
			logger::info(
				"frame time (base, pre-pass to pre-pass): avg {:.2f} ms ({:.0f} fps), max {:.2f} ms over {} frames | by mode: previs {} / CBRO feed {} / CBRO classic {} | camera still: {} | previs {}",
				frameMs, 1000.0 / std::max(0.001, frameMs), g_frames.intervalMax, g_frames.intervalCount,
				byMode(g_frames.intervalMode[0]), byMode(g_frames.intervalMode[1]), byMode(g_frames.intervalMode[2]),
				still.count ? std::format("{:.2f} ms ({:.0f} fps) over {} frames", stillMs, 1000.0 / std::max(0.001, stillMs), still.count) : std::string("no still frames"),
				Feed::PrevisState());
			g_frames.intervalStill = {};
			g_frames.intervalMode = {};
			logger::info(
				"CPU per frame by mode (ms, main thread, between the render-stage hooks): previs: {} || CBRO feed: {} || CBRO classic: {}",
				g_timing.cpu[0].Describe(), g_timing.cpu[1].Describe(), g_timing.cpu[2].Describe());
			const auto gpu = g_gpu.Take();
			logger::info(
				"GPU per frame by mode (ms, timestamp spans, idle between marks included): previs: {} || CBRO feed: {} || CBRO classic: {} || {}",
				gpu[0].Describe(), gpu[1].Describe(), gpu[2].Describe(), g_gpu.Status());
			const auto calls = Hooks::CullGroups::TakeHookCalls();
			const auto flags = Hooks::CullGroups::ReadEngineFlags();
			logger::info(
				"hooks {} | calls per frame: Block::Add {:.0f}, Group::Add {:.0f}, ChildPush {:.0f}, registrations {:.0f} || engine: bCullingBatch {}, directional shadows {}",
				g_state.hooksIn ? "in" : "out (the engine runs its own code; the registration hook stays in, counting only)",
				calls.blockAdds / intervalFrames, calls.groupAdds / intervalFrames, calls.childPushes / intervalFrames, calls.registrations / intervalFrames,
				flags.cullingBatch ? 1 : 0, flags.dirShadows ? "on" : "off");
			logger::info(
				"registrations per frame by accumulator (which views register how much): previs: {} || CBRO (feed and classic together): {}",
				Hooks::CullGroups::TakeRegistrationSites(false, std::max(1u, g_timing.cpu[0].frames)),
				Hooks::CullGroups::TakeRegistrationSites(true, std::max(1u, g_timing.cpu[1].frames + g_timing.cpu[2].frames)));
			g_frames.intervalSum = 0.0;
			g_frames.intervalMax = 0.0;
			g_frames.intervalCount = 0;
			LogSessionAB();

			const double cullFrames = std::max(1u, g_timing.cullFrames);
			const double prepassFrames = std::max(1u, g_timing.prepassFrames);
			const double cbroFrames = std::max(1u, g_timing.cpu[1].frames + g_timing.cpu[2].frames);  // (the tests only run in CBRO mode)
			const auto tests = Occlusion::TakeTestMilliseconds();
			logger::info(
				"cost per frame (ms): CBRO setup {:.3f} | CBRO depth capture {:.3f} | CBRO object tests per CBRO-mode frame {:.3f} CPU on all threads, {:.3f} of it on the main thread: view evaluations {:.3f} (of which mesh shapes {:.3f}), sun evaluations {:.3f}, cache reuse and bookkeeping {:.3f} (record lookups {:.3f}, reuse checks {:.3f} with depth re-checks {:.3f}, record writes {:.3f}, cell-node scans {:.3f}, decisions and counters {:.3f})",
				QpcMs(g_timing.setupTicks) / cullFrames, QpcMs(g_timing.captureTicks) / prepassFrames,
				tests.all / cbroFrames, tests.mainThread / cbroFrames,
				tests.evaluate / cbroFrames, tests.shape / cbroFrames, tests.sun / cbroFrames, std::max(0.0, tests.all - tests.evaluate - tests.sun) / cbroFrames,
				tests.lookup / cbroFrames, tests.reuse / cbroFrames, tests.recheck / cbroFrames, tests.record / cbroFrames, tests.nodeScan / cbroFrames,
				std::max(0.0, tests.all - tests.evaluate - tests.sun - tests.lookup - tests.reuse - tests.record - tests.nodeScan) / cbroFrames);
			{
				const double readbacks = std::max(1.0, static_cast<double>(g_timing.prepassFrames));
				const auto   kinds = HiZ::TakeBlockChangeKinds();
				logger::info(
					"verdict cache this interval: camera epochs {} | sun epochs {} | hi-z blocks changed {:.1f} per readback (holding first-person pixels {:.1f}, else the far plane {:.1f}; in the lower third of the view {:.1f})",
					g_epochs.viewChanges, g_epochs.sunChanges, static_cast<double>(HiZ::TakeBlocksChanged()) / readbacks,
					static_cast<double>(kinds.firstPerson) / readbacks, static_cast<double>(kinds.farPlane) / readbacks, static_cast<double>(kinds.lowerThird) / readbacks);
			}
			g_epochs.viewChanges = 0;
			g_epochs.sunChanges = 0;
			if (g_footprint.lastValid) {
				logger::info(
					"view footprint (last frame): [{:.3f},{:.3f}] x [{:.3f},{:.3f}] of the depth frame, camera turned {:.2f} deg since",
					g_footprint.lastView[0], g_footprint.lastView[1], g_footprint.lastView[2], g_footprint.lastView[3],
					g_footprint.lastAngle * 180.0f / 3.14159265f);
			} else if (!g_footprint.disabled) {
				logger::info("view footprint (last frame): unavailable, objects on the screen edge stay unjudged");
			}
			const auto& r = g_footprint.reasons;
			logger::info(
				"view footprint per interval: still {} | turned {} | turned (both readings) {} || captures drawn a frame behind the NiCamera (mid-turn) {}, off by more than its last turn {} || unusable: capture no axis fit {} / offset with the camera still {} / frustum units {} / frustum mismatch {} | frame no mapping {} / self-check failed {} / turned views off {} / sideways {} / implausible {}",
				r[kStill], r[kTurned], r[kTurnedBoth], r[kTrailing], r[kTrailingBeyond], r[kAxesNoFit], r[kOffsetStill], r[kUnitsUnknown], r[kTangentMismatch], r[kNoMapping], r[kCheckFailed], r[kTurnedOff], r[kSideways], r[kImplausible]);
			g_footprint.reasons = {};
			Timing next{};
			next.marks = g_timing.marks;
			next.mode = g_timing.mode;
			g_timing = next;
		}

		// On-screen message (HUD corner), queued to the main thread outside rendering.
		void Notify(std::string a_text)
		{
			if (!Settings::Get().notify) {
				return;
			}
			if (const auto tasks = F4SE::GetTaskInterface()) {
				tasks->AddTask([text = std::move(a_text)]() {
					// No sound: nullptr, as the engine's own callers pass. "" is not "no sound": the HUD plays
					// UIUtils::PlayMenuSound(""), which looks the sound up by the CRC of its editor ID, and a sound
					// form keyed by the empty name (a Fallout London looping water sound) played and never stopped.
					RE::SendHUDMessage::ShowHUDMessage(text.c_str(), nullptr, false, false);
				});
			}
		}

		void NotifyStatus()
		{
			if (!Occlusion::Active()) {
				Notify(!g_state.wantActive ? "CBRO off - previs on" : g_state.interiorStandby ? "CBRO standing by (interior) - previs is handling visibility" : "CBRO unavailable - previs is handling visibility");
				return;
			}
			const auto tested = Occlusion::TestedPerFrame();
			const auto rejected = Occlusion::RejectedPerFrame();
			static constexpr std::array kDiagnostic{ ""sv, " [decide-only]"sv, " [decide-only, no depth capture]"sv };
			Notify(tested > 0.0f ?
			           std::format(
						   "CBRO on{} - hiding {:.0f} of {:.0f} objects, {:.0f} of {:.0f} lights per frame",
						   kDiagnostic[g_state.diagnostic], rejected, tested, Occlusion::LightsRejectedPerFrame(), Occlusion::LightsPerFrame()) :
			           std::string("CBRO on (previs off) - warming up"));
		}

		// The effective mode from the wish (wantActive), the interior standby and CBRO's availability.
		void ApplyEffective(std::string_view a_reason)
		{
			const auto& settings = Settings::Get();
			const bool  wanted = g_state.wantActive;
			const bool  standby = g_state.interiorStandby;
			// Previs only goes off when CBRO can actually take over, and never in an interior CBRO stands by in.
			const bool effective = wanted && !standby && g_state.convention != Convention::kNone && !HiZ::Failed();
			Occlusion::SetActive(effective);
			Occlusion::ResetHistory();
			Async::Reset();
			logger::info(
				"mode: {} ({})",
				effective ? "CBRO occlusion" : !wanted ? "previs (CBRO off)" : standby ? "previs (CBRO standing by: interior, bInteriors=0)" : "previs (CBRO requested but unavailable)", a_reason);
			if (static_cast<int>(effective) != g_state.notifiedEffective) {
				g_state.notifiedEffective = effective;
				const char* off = !wanted ? "CBRO off - previs on" : standby ? "CBRO standing by (interior) - previs on" : "CBRO unavailable - previs on";
				if (Feed::ManagesPrevis()) {
					Notify(effective ? "CBRO on - previs suspended only inside CBRO's cull windows" : off);
				} else if (settings.disablePrevis) {
					Notify(effective ? "CBRO on - previs off" : off);
				} else {
					Notify(effective ? "CBRO on - working with previs" : !wanted ? "CBRO off - previs only" : standby ? "CBRO standing by (interior) - previs only" : "CBRO unavailable - previs only");
				}
				if (effective && Occlusion::Deciding()) {
					Notify(g_state.diagnostic == 2 ? "CBRO DIAGNOSTIC: decide-only, no depth capture (nothing hidden)" : "CBRO DIAGNOSTIC: decide-only (nothing hidden)");
				}
			}

			// With bPrevisFeed=1 previs is never switched off (Core/Feed suspends it without a flush at the cull begin).
			if (!settings.disablePrevis || Feed::ManagesPrevis()) {
				return;
			}
			// The switch itself runs at the next cull begin (ApplyPrevisRequest): main thread, before DrawWorld's
			// cull. Up to v1.15 it ran from F4SE's task queue, which on OG F4SE is drained inside its hook on the
			// engine's message-queue processing (f4se/Hooks_Threads.cpp), i.e. on worker threads: the v1.15 log
			// shows the switch on five different thread ids. Its disable path runs the engine's flush callbacks,
			// which must not race the main thread's cull.
			g_state.previsRequest.store(effective ? 0 : 1);
		}

		void ApplyMode(bool a_active, std::string_view a_reason)
		{
			g_state.wantActive = a_active;
			ApplyEffective(a_reason);
		}

		// Interiors and override-root scenes, decided at the cull begin from the engine's own gates (Hooks/PrevisFeed):
		//   bInteriors=0: CBRO stands by there (previs mode, nothing touched) and resumes in exteriors;
		//   bInteriors=1 with bInteriorLegacyPrevis=1: CBRO culls there the way v1.28 did (previs switched off and
		//     flushed while CBRO is on, judged inside the walk unless bAsyncInteriors), the exterior keeps the windows.
		// Like the hotkey, the frame's window may still be the previous mode's (Core/Feed handles that switch frame).
		void SyncInteriorMode()
		{
			const auto& settings = Settings::Get();
			if (!settings.previsFeed) {
				return;  // (bPrevisFeed=0: v1.28's switch everywhere already)
			}
			const auto gates = Hooks::PrevisFeed::ReadGates();
			const bool interior = gates.readable && (!gates.exterior || gates.overrideRoot);
			if (g_state.loadSwitch) {
				// The loading screen made previs inactive for the attach (v1.44/v1.45: switched off on the main thread,
				// or suspended by the byte from the UI's thread). The suspension byte goes back before the windows
				// logic (Core/Feed) reads it; then: interior -> the engine's switch (off and flushed) as the legacy path
				// wants, whether or not the scene kind changed; exterior -> previs enabled for the windows.
				// The request lands at the next cull begin (ApplyPrevisRequest runs before this); the suspension byte,
				// if CBRO set it, stays for this frame's walk and is cleared there, after the switch.
				g_state.loadSwitch = false;
				g_state.previsRequest.store(interior ? 0 : 1);
				logger::info("previs feed: after the load: {}", interior ? "interior: previs switched off (legacy path)" : "exterior: previs enabled (the windows)");
			}
			if (interior == g_state.interiorScene) {
				return;
			}
			g_state.interiorScene = interior;
			if (!settings.interiors) {
				g_state.interiorStandby = interior;
				ApplyEffective(interior ? "interior/override root: the engine's previs and rooms cull it (bInteriors=0)" : "exterior: CBRO resumes");
				return;
			}
			if (!settings.interiorLegacyPrevis) {
				return;  // (the windows indoors too)
			}
			Feed::SetLegacyScene(interior);
			if (!interior) {
				// Back outside: previs was switched off for the interior; restore it (the windows take over).
				g_state.previsRequest.store(1);
			}
			ApplyEffective(interior ? "interior: v1.28's previs switch (bInteriorLegacyPrevis=1)" : "exterior: the windows again");
		}

		// Switches previs where the console's `tpc` does: on the main thread, with no cull reading previs data
		// (this runs right before DrawWorld's cull of the frame).
		void ApplyPrevisRequest()
		{
			const int request = g_state.previsRequest.exchange(-1);
			if (request < 0) {
				return;
			}
			if (!g_state.previsThreadLogged) {
				g_state.previsThreadLogged = true;
				const auto main = RE::Main::GetSingleton();
				const auto thread = GetCurrentThreadId();
				logger::info("previs switch runs on thread {} ({})", thread, main && main->threadID == thread ? "the main thread" : "NOT the main thread");
			}
			if (!g_state.originalPrevisKnown) {
				g_state.originalPrevis = PrevisEnabled();
				g_state.originalPrevisKnown = true;
			}
			const bool want = request == 0 ? false : g_state.originalPrevis;
			if (PrevisEnabled() != want) {
				SetPrevisEnabled(want);
				logger::info("previs {}", want ? "re-enabled" : "disabled (CBRO is the visibility authority)");
			}
			g_state.expectedPrevis = want ? 1 : 0;
			if (g_state.loadSuspended) {
				// The loading screen's suspension (v1.45) has done its job: the switch above is the real state now.
				g_state.loadSuspended = false;
				Hooks::PrevisFeed::ClearSuspension();
			}
		}

		// v1.44: a loading screen while CBRO is on, with bPrevisFeed=1 and bInteriorLegacyPrevis=1. v1.28 had previs off
		// from the game load on, so every cell attached with previs inactive; with the windows previs is active between
		// culls, so an interior attached with it active and was switched only at its first cull, after the engine had
		// already done its per-cell setup for the previs-active case (the previs-off setup of rooms and lights is gated
		// by IsActive(): FO4-ENGINE-NOTES 5.5c, 6.7, 6.8). The lamps of such an interior stayed dark (v1.31-v1.43).
		// So the switch is made when the loading screen opens, before the cells attach, as v1.28 had it; the first
		// cull after the load re-enables previs for an exterior (SyncInteriorMode). Main thread (the UI's event source).
		void OnLoadingScreen()
		{
			const auto& settings = Settings::Get();
			// (Not before a game is loaded: the very first load then behaves as v1.28, previs switched after it; a save
			// that starts in an interior needs one more load for its lamps.)
			if (!g_state.installed || !g_state.gameLoaded || !g_state.wantActive || !settings.previsFeed || !settings.interiors || !settings.interiorLegacyPrevis || !settings.disablePrevis) {
				return;
			}
			const auto main = RE::Main::GetSingleton();
			const bool mainThread = main && main->threadID == GetCurrentThreadId();
			g_state.loadSwitch = true;
			if (mainThread) {
				g_state.previsRequest.store(0);
				ApplyPrevisRequest();
				logger::info("loading screen: previs switched off before the cells attach (as v1.28; the first cull after the load decides what follows)");
			} else {
				// The UI's event source runs on another thread (seen: the v1.44 run). The engine's switch (a flush) is
				// not for that thread; its flush-free suspension is a byte write and makes IsActive() false for the
				// attach just the same. The first cull after the load takes it back and makes the proper switch.
				Hooks::PrevisFeed::SetSuspended(true);
				g_state.loadSuspended = true;
				logger::info("loading screen: previs suspended (flush-free byte) before the cells attach, from thread {}; the first cull after the load decides what follows", GetCurrentThreadId());
			}
		}

		class LoadSink final : public RE::BSTEventSink<RE::MenuOpenCloseEvent>
		{
		public:
			RE::BSEventNotifyControl ProcessEvent(const RE::MenuOpenCloseEvent& a_event, RE::BSTEventSource<RE::MenuOpenCloseEvent>*) override
			{
				if (a_event.opening && a_event.menuName == "LoadingMenu") {
					OnLoadingScreen();
				}
				return RE::BSEventNotifyControl::kContinue;
			}
		};
		LoadSink g_loadSink;

		void RegisterLoadSink()
		{
			if (g_state.loadSinkRegistered) {
				return;
			}
			const auto ui = RE::UI::GetSingleton();
			if (!ui) {
				return;
			}
			ui->RegisterSink<RE::MenuOpenCloseEvent>(&g_loadSink);
			g_state.loadSinkRegistered = true;
			logger::info("loading-screen listener registered (previs off before cells attach while CBRO is on: bPrevisFeed=1 with bInteriorLegacyPrevis=1)");
		}

		// The hooks follow the mode, switched on the main thread before DrawWorld's cull: out while previs has the
		// job (the engine then runs exactly its own code, as without CBRO), in while CBRO culls.
		void SyncHooks()
		{
			const bool want = Occlusion::Active();
			if (want == g_state.hooksWanted) {
				return;
			}
			g_state.hooksWanted = want;
			const bool groups = Hooks::CullGroups::SetHooksIn(want);
			const bool lights = ShadowLights::SetHooksIn(want);
			g_state.hooksIn = Hooks::CullGroups::HooksIn();
			logger::info(
				"hooks: {}{}", want ? "put back in (CBRO culls)" : "taken out (previs mode: the engine runs its own code)",
				groups && lights ? "" : " - some were left as they were (see above)");
			if (want && !g_state.hooksIn) {
				logger::error("hooks: the culling hooks couldn't be put back; previs keeps the job");
				ApplyMode(false, "hooks unavailable");
			}
		}

		bool KeyPressed(std::uint32_t a_vk, bool& a_wasDown)
		{
			if (a_vk == 0) {
				return false;
			}
			const auto main = RE::Main::GetSingleton();
			if (!main || GetForegroundWindow() != reinterpret_cast<HWND>(main->hwnd)) {
				a_wasDown = false;
				return false;
			}
			const bool down = (GetAsyncKeyState(static_cast<int>(a_vk)) & 0x8000) != 0;
			const bool pressed = down && !a_wasDown;
			a_wasDown = down;
			return pressed;
		}

		void OnCullBegin()
		{
			const auto& settings = Settings::Get();

			if (KeyPressed(settings.toggleHotkey, g_state.toggleKeyDown)) {
				ApplyMode(!g_state.wantActive, "hotkey");
			}
			if (KeyPressed(settings.statusHotkey, g_state.statusKeyDown)) {
				NotifyStatus();
			}
			if (KeyPressed(settings.diagnosticHotkey, g_state.diagnosticKeyDown)) {
				g_state.diagnostic = (g_state.diagnostic + 1) % 3;
				Occlusion::SetObserveOnly(g_state.diagnostic != 0);
				if (g_state.diagnostic == 2) {
					HiZ::Reset();
				}
				static constexpr std::array kNames{
					"CBRO diagnostic: normal (hiding on)"sv,
					"CBRO diagnostic: decide-only (nothing hidden, previs still off)"sv,
					"CBRO diagnostic: decide-only, depth capture off (no CBRO GPU work)"sv
				};
				logger::info("{}", kNames[g_state.diagnostic]);
				Notify(std::string(kNames[g_state.diagnostic]));
			}
			ApplyPrevisRequest();
			// Previs switched outside CBRO (console `tpc`): follow it, so the two stay inverse. (Not with bPrevisFeed=1:
			// there CBRO never owns the enabled byte; Core/Feed only notes that previs is inactive.)
			if (settings.disablePrevis && !Feed::ManagesPrevis() && g_state.expectedPrevis >= 0 && g_state.previsRequest.load() < 0) {
				const int actual = PrevisEnabled() ? 1 : 0;
				if (actual != g_state.expectedPrevis) {
					g_state.expectedPrevis = actual;
					ApplyMode(actual == 0, actual ? "previs re-enabled outside CBRO (console tpc?)" : "previs disabled outside CBRO (console tpc?)");
				}
			}
			if (HiZ::Failed() && !g_state.failureHandled) {
				g_state.failureHandled = true;
				ApplyMode(g_state.wantActive, "hi-z unavailable");
			}
			// The frame's path (previs / CBRO classic / CBRO feed) and the previs suspension, before the hooks follow.
			SyncInteriorMode();
			Feed::BeginFrame(Occlusion::Active(), g_state.clock + 1);
			SyncHooks();

			TrackStillness();
			// Previs mode: no depth is captured, so there is nothing to poll or prepare (the hooks pass through).
			const bool active = Occlusion::Active();
			if (active) {
				HiZ::Poll();
			}
			const auto snapshot = active ? HiZ::Latest() : nullptr;

			Occlusion::FrameContext context{};
			context.clock = ++g_state.clock;
			context.snapshot = snapshot;
			g_footprint.lastValid = false;

			if (snapshot && g_state.convention != Convention::kNone && g_state.convention != Convention::kUnknown && !HiZ::Failed()) {
				const auto age = g_state.renderFrame + 1 - snapshot->frame;
				float      move = 0.0f;
				float      angle = 0.0f;
				CameraDelta(snapshot->camera, move, angle);
				const float maxAngle = settings.maxCameraAngle * 3.14159265f / 180.0f;
				context.cull = Occlusion::Active() && g_state.hooksIn && age <= settings.maxSnapshotAge && move <= settings.maxCameraMove && angle <= maxAngle;
				context.dilateMove = move + snapshot->mergeMove + 1.0f;  // (older merged frames' cameras lie within mergeMove of this one)
				if (settings.verdictCache) {
					const float toRadians = 3.14159265f / 180.0f;
					float zoom[2]{};
					if (const auto root = RE::Main::WorldRootCamera()) {
						FrustumExtents(ReadFrustum(root), zoom[0], zoom[1]);
					}
					UpdateViewEpoch(snapshot->camera, zoom, settings.cacheMove * 0.5f, settings.cacheAngle * 0.5f * toRadians);
					context.cacheEnabled = true;
					context.viewEpoch = g_epochs.viewEpoch;
					// Bounds grow by the whole tolerance (and by depth x the turn tolerance in the tests) so a verdict
					// holds for any camera of its epoch.
					context.dilateMove = std::max(move, settings.cacheMove) + snapshot->mergeMove + 1.0f;
					context.angularSlack = std::sin(settings.cacheAngle * toRadians);
					context.readback = snapshot->readbackIndex;
					const auto blocks = HiZ::Blocks();
					context.blockChangedAt = blocks.changedAt;
					context.blocksW = blocks.width;
					context.blocksH = blocks.height;
				}
				g_footprint.lastValid = CurrentView(snapshot->camera, angle, g_stillness.frames, age, snapshot->frame, context.view);
				if (g_footprint.lastValid) {
					std::copy_n(context.view, 4, g_footprint.lastView);
					g_footprint.lastAngle = angle;
				}

				// A point far behind the depth-frame camera. The camera has turned less than 90 degrees
				// since (maxCameraAngle is clamped below that), so it is behind the current near plane
				// too: the engine's frustum test rejects any entry carrying this bound.
				const auto& camera = snapshot->camera;
				context.reject.center.x = camera.eye[0] - camera.viewDir[0] * 1.0e7f;
				context.reject.center.y = camera.eye[1] - camera.viewDir[1] * 1.0e7f;
				context.reject.center.z = camera.eye[2] - camera.viewDir[2] * 1.0e7f;
				context.reject.fRadius = 1.0f;

				ReadSunState(camera, context.sun);
				if (settings.verdictCache) {
					UpdateSunEpoch(static_cast<int>(context.sun.state), g_sun.dir, settings.cacheAngle * 0.5f * 3.14159265f / 180.0f);
					context.sunEpoch = g_epochs.sunEpoch;
				}

				// This frame's view for point-light shadow casters: the depth frame's cone, widened by the turn
				// since (plus half a degree) and by any zoom-out, pushed out by the movement since.
				const auto root = RE::Main::WorldRootCamera();
				if (settings.lampShadowCulling && camera.viewSpace && root && camera.frustumX > 0.0f && camera.frustumY > 0.0f) {
					float fx = 0.0f, fy = 0.0f;
					FrustumExtents(ReadFrustum(root), fx, fy);
					const float zoom = std::max({ 1.0f, fx / camera.frustumX, fy / camera.frustumY }) * 1.01f;
					context.cone = ShadowGeometry::MakeCone(
						camera.eye, camera.viewDir, camera.viewRight, camera.viewUp, zoom / camera.scaleX, zoom / camera.scaleY, angle + 0.0087f);
					context.conePush = move + 4.0f;
					context.coneValid = true;
				}
			}
			// Asynchronous verdicts: this frame hides by the worker's map only if the camera stayed within the margins the
			// worker judged with; the margins for the job this frame submits follow the frame's own motion.
			if (Async::Enabled()) {
				Async::Pose pose{};
				const bool  known = ReadPose(pose);
				float       frameMove = 0.0f;
				if (known && g_asyncFrame.lastKnown) {
					frameMove = PoseDistance(pose, g_asyncFrame.last);
				}
				// Interiors judge inside the walk (as v1.28) unless bAsyncInteriors.
				context.asyncFrame = !g_state.interiorScene || settings.asyncInteriors;
				const bool viewHeld = !known || ViewHeld(g_asyncFrame.judged, pose, context.viewEpoch, context.cacheEnabled);
				const bool mapValid = Async::BeginFrame(pose, viewHeld);
				context.asyncValid = context.asyncFrame && known && context.cull && mapValid;
				PrepareJobView(context, pose, known && g_asyncFrame.lastKnown, g_asyncFrame.last, g_asyncFrame.next);
				g_asyncFrame.pose = pose;
				g_asyncFrame.last = pose;
				g_asyncFrame.lastKnown = known;
				g_asyncFrame.moveMargin = std::clamp(2.0f * frameMove + 4.0f, 6.0f, 48.0f);
				g_asyncFrame.frameMove = frameMove;
				// (No turn margin since v1.71: up to v1.70 a map was refused after any turn past 2x the last one plus
				// 0.5 deg, at most 8 deg, shrunk by the zoom. ViewHeld places the turned view itself, in NDC at the zoom.)
			}
			ShadowLights::PublishLamps();  // last frame's shadow-casting lamps, for group 0 with the sun off
			if (settings.unoccludeLights && Occlusion::Active() && g_state.hooksIn) {
				// Before the engine walks its light lists this frame (UpdateLightList reads BSLight::bOccluded for the
				// non-shadow lights; the lamp loop for the shadow lights): previs's marks go, as with previs off (6.7).
				ShadowLights::UnoccludeLights(true);
			}
			Occlusion::BeginFrame(context);
			g_asyncFrame.context = context;
			g_state.cullingThisFrame = context.cull;

			// Once per location, after ~5 s standing still with CBRO culling: list what is still drawn.
			Occlusion::FlushKeptDump();
			constexpr std::uint32_t kDumpAfterStillFrames = 300;
			if (context.cull && g_footprint.lastValid && g_stillness.frames >= kDumpAfterStillFrames && !g_frames.dumpedHere) {
				g_frames.dumpedHere = true;
				Occlusion::RequestKeptDump();
			}

			if (++g_state.framesSinceLog >= settings.summaryIntervalFrames) {
				logger::info(
					"==== CBRO: mode {} | convention {} | {} ====",
					g_state.wantActive ? "occlusion" : "previs", ConventionName(g_state.convention), HiZ::Describe());
				Occlusion::LogStats(g_state.framesSinceLog);
				LogSun();
				ShadowLights::LogStats(g_state.framesSinceLog);
				Feed::LogStats(g_state.framesSinceLog);
				if (const auto rain = Hooks::Precipitation::Take(); rain.runs > 0) {
					const double frames = std::max(1u, g_state.framesSinceLog);
					const auto   inactive = rain.runs - rain.previsActive;
					logger::info(
						"rain occlusion map per frame: runs {:.2f} (previs active at entry {:.2f}: its list; previs off {:.2f}: the whole world node walked) | CPU per run: {:.3f} ms with previs's list, {:.3f} ms walking the world node",
						static_cast<double>(rain.runs) / frames, static_cast<double>(rain.previsActive) / frames, static_cast<double>(inactive) / frames,
						rain.previsActive ? rain.msActive / static_cast<double>(rain.previsActive) : 0.0, inactive ? rain.msInactive / static_cast<double>(inactive) : 0.0);
				}
				Hooks::FirstPersonOrder::LogStats(g_state.framesSinceLog);
				SetDiff::LogStats();
				Async::LogStats(g_state.framesSinceLog);
				LogMotionTable(g_turnTable);
				LogMotionTable(g_moveTable);
				LogDepthAge();
				LogTiming();
				g_state.framesSinceLog = 0;
			}
		}

		// The world's depth is complete: at the pre-pass stage's end, or (first person held back, Hooks/FirstPersonOrder) at its
		// world block's end, before the first-person block.
		void OnPrePassEnd(bool a_firstPersonAfter)
		{
			HiZ::Camera camera{};
			Raw         raw{};
			if (!ReadCamera(camera, raw)) {
				return;
			}

			// The first frames after a load can have a camera that isn't placed yet: retry for a while.
			if (g_state.convention == Convention::kUnknown) {
				constexpr std::uint32_t kMaxAttempts = 300;
				const auto              attempt = ++g_state.selfCheckAttempts;
				const auto              result = SelfCheck(camera, raw, attempt == 1 || attempt == kMaxAttempts);
				if (result != Convention::kNone) {
					g_state.convention = result;
				} else if (attempt >= kMaxAttempts) {
					g_state.convention = Convention::kNone;
					logger::error("occlusion disabled: no projection convention passed the self-check in {} frames", attempt);
				}
				if (g_state.convention != Convention::kUnknown) {
					ApplyMode(g_state.wantActive, "self-check done");
					ReadCamera(camera, raw);  // re-bake with the chosen convention
				} else {
					return;
				}
			}
			if (g_state.convention == Convention::kNone) {
				return;
			}

			// The frame counter always advances, so a snapshot kept from before a previs spell reads as stale.
			++g_state.renderFrame;
			if (g_state.diagnostic == 2 || !Occlusion::Active()) {
				return;  // no GPU work while previs has the job
			}
			CheckFootprint(camera);
			HiZ::Capture(g_state.renderFrame, camera, a_firstPersonAfter);
		}

		class Listener final :
			public Hooks::RenderStages::Listener
		{
		public:
			void OnStageBegin(Stage a_stage) override
			{
				switch (a_stage) {
				case Stage::kCull: {
					HangWatch::Beat();
					const auto now = Qpc();
					CloseFrameTiming(now);  // the previous frame's buckets go to its mode
					g_timing.marks = {};
					g_timing.marks[0] = now;
					OnCullBegin();
					g_timing.setupTicks += Qpc() - now;
					++g_timing.cullFrames;
					g_timing.mode = CurrentMode();  // (OnCullBegin may have switched it)
					Hooks::CullGroups::SetRegistrationMode(g_timing.mode != 0);
					g_gpu.FrameBegin(g_timing.mode);
					// DrawWorld's cull registers the main view's objects (and waits for its jobs) inside this
					// stage: only then may the main accumulator's registrations be filtered. On a frame CBRO
					// doesn't cull (previs mode, stale depth, camera jump) every hook stays a pass-through.
					Hooks::CullGroups::SetMainCullActive(g_state.cullingThisFrame);
					break;
				}
				case Stage::kPrePass:
					OnFrameBoundary();
					g_timing.marks[2] = Qpc();
					g_gpu.Mark(2);
					break;
				case Stage::kSunCascades:
					// The deferred-lights stage follows. In a CBRO frame the walk ran with previs inactive, so a light previs
					// had marked occluded stays marked and dark (FO4-ENGINE-NOTES 6.7); clear the mark, as previs-off vanilla.
					if (Settings::Get().unoccludeLights && Occlusion::Active() && g_state.hooksIn) {
						ShadowLights::UnoccludeLights(false);  // (again: the cull begin's clear is the frame's main one)
					}
					if (Settings::Get().lampDiagnostic && Occlusion::Active() && g_state.hooksIn) {
						ShadowLights::LampLoopBegin();
					}
					ShadowLights::SetLampStage(Occlusion::Active() && g_state.hooksIn);  // (spot-light shadow maps may be emptied)
					g_timing.marks[5] = Qpc();
					g_gpu.Mark(5);
					break;
				case Stage::kForward:
					g_timing.marks[7] = Qpc();
					g_gpu.Mark(7);
					break;
				default:
					break;
				}
			}

			void OnStageEnd(Stage a_stage) override
			{
				switch (a_stage) {
				case Stage::kCull:
					Hooks::CullGroups::SetMainCullActive(false);
					Occlusion::EndFrameSample(Hooks::CullGroups::ReadHookCalls().registrations, Hooks::CullGroups::ReadDroppedRegistrations());
					if (g_state.cullingThisFrame) {
						NoteMotion(g_stillness.lastTurn, g_asyncFrame.frameMove, g_asyncFrame.context.asyncValid);
						if (const auto judged = g_asyncFrame.context.snapshot) {
							++g_depthAge.frames;
							++g_depthAge.atBegin[std::min<std::uint64_t>(g_state.renderFrame + 1 - judged->frame, 5)];
							if (const auto ready = HiZ::NewestReady(); ready > judged->frame) {
								++g_depthAge.newerAtEnd[std::min<std::uint64_t>(g_state.renderFrame + 1 - ready, 5)];
							}
						}
					}
					Feed::EndCull();  // (closes an audit frame: DrawWorld's cull and its jobs are done)
					{
						// The set-diff diagnostic files this frame's main-view registrations (settled frames only).
						const auto ui = RE::UI::GetSingleton();
						const bool settled = g_frames.sinceSwitch > kSettleFrames && g_frames.sinceLoad > kLoadSettleFrames && !(ui && ui->menuMode != 0);
						const auto root = RE::Main::WorldRootCamera();
						SetDiff::EndCull(CurrentMode(), settled, root ? root->world.translate : RE::NiPoint3{});
					}
					// Asynchronous verdicts: this frame's candidates go to the worker with this frame's context, dilated for
					// the next frame's camera (the walk is done: every candidate is recorded). The view is this frame's,
					// placed once at the cull begin, still or turned, and widened toward the frame's turn (JobView). (Up to
					// v1.58 the worker placed its own, always as a turned view: it never took the still path, so with turned
					// views switched off it had none, and every object crossing the screen edge stayed drawn. Nor is the
					// view widened by a turn margin on every side: that would leave every such object unjudged with the
					// camera still.)
					if (Async::Enabled() && g_state.cullingThisFrame && g_asyncFrame.context.asyncFrame && g_asyncFrame.context.snapshot) {
						auto  worker = g_asyncFrame.context;
						auto& job = g_asyncFrame.next;
						worker.dilateMove += g_asyncFrame.moveMargin;
						worker.asyncValid = false;
						std::copy_n(job.box, 4, worker.view);
						// The epoch's slack covers the camera's turn within it, not a box grown past the last job's: a side
						// widened further in the same epoch (or a view lost: an infinite box, used at any turn) starts a new
						// one, so the worker re-judges what it judged with the narrower box (a slow turn while zoomed; at the
						// default FOV the floor exceeds such turns).
						const auto& last = g_asyncFrame.judged;
						const bool  grew = !std::isfinite(job.box[0]) ?
						                       std::isfinite(last.box[0]) :
						                       job.widen[0] > last.widen[0] + 1.0e-4f || job.widen[1] > last.widen[1] + 1.0e-4f ||
						                           job.widen[2] > last.widen[2] + 1.0e-4f || job.widen[3] > last.widen[3] + 1.0e-4f;
						if (last.valid && worker.cacheEnabled && worker.viewEpoch == last.epoch && grew) {
							worker.viewEpoch = ++g_epochs.viewEpoch;
							++g_epochs.viewChanges;
						}
						job.epoch = worker.viewEpoch;
						job.valid = true;
						g_asyncFrame.judged = job;
						Occlusion::PrepareContext(worker);
						Async::Submit(worker, g_asyncFrame.pose, g_asyncFrame.moveMargin, std::max({ job.widen[0], job.widen[1], job.widen[2], job.widen[3] }));
					} else {
						g_asyncFrame.judged.valid = false;
					}
					g_timing.marks[1] = Qpc();
					g_gpu.Mark(1);
					break;
				case Stage::kPrePass: {
					const auto end = Qpc();
					++g_timing.prepassFrames;
					g_timing.marks[3] = end;  // the engine's pre-pass ends here; CBRO's depth capture follows (the "hi-z" bucket)
					g_gpu.Mark(3);
					// (with first person held back the capture ran at the world block's end, inside the pre-pass bucket; a
					// block still held runs now, after its capture)
					Hooks::FirstPersonOrder::Flush();
					if (!std::exchange(g_state.capturedInPass, false)) {
						OnPrePassEnd(false);
					}
					const auto captured = Qpc();
					g_timing.captureTicks += captured - end;
					g_timing.marks[4] = captured;
					g_gpu.Mark(4);
					break;
				}
				case Stage::kSunCascades:
					// (the last cascade's registration filter lives from this frame's cull begin to here: never across a
					// menu, a loading screen or a frame CBRO doesn't cull, whose context it would read)
					Occlusion::SetFarCascade(nullptr);
					ShadowLights::LampLoopEnd();  // (nothing unless LampLoopBegin ran this frame)
					ShadowLights::SetLampStage(false);
					g_timing.marks[6] = Qpc();
					g_gpu.Mark(6);
					break;
				case Stage::kForward:
					Async::Wait();  // the worker's frame ends with the render: it never touches objects across the game's update
					g_timing.marks[8] = Qpc();
					g_gpu.Mark(8);
					break;
				default:
					break;
				}
			}
		};
		Listener g_listener;

		// The pre-pass's first-person block waits past the world block whenever this pre-pass's depth is captured.
		class FirstPersonListener final :
			public Hooks::FirstPersonOrder::Listener
		{
		public:
			bool WantWorldDepth() override
			{
				return Occlusion::Active() && g_state.diagnostic != 2 && g_state.convention != Convention::kUnknown && g_state.convention != Convention::kNone;
			}

			void OnWorldDepth() override
			{
				const auto start = Qpc();
				OnPrePassEnd(true);
				g_timing.captureTicks += Qpc() - start;
				g_state.capturedInPass = true;
			}
		};
		FirstPersonListener g_firstPersonListener;
	}

	void Install()
	{
		const auto& settings = Settings::Get();
		if (!settings.occlusion) {
			logger::info("occlusion: disabled in CBRO.ini ([Occlusion] bEnabled=0)");
			return;
		}
		if (!REL::Module::get().is_og()) {
			logger::warn("occlusion: engine offsets are verified for 1.10.163 only; not installing");
			return;
		}
		// Without the culling-group and main-view hooks CBRO can't hide anything without also cutting
		// shadows: leave previs in charge rather than switch it off for nothing.
		if (!Hooks::CullGroups::Install()) {
			logger::error("occlusion: unavailable (culling-group hooks missing); previs stays in charge");
			return;
		}

		Occlusion::Install();
		ShadowLights::Install();
		Hooks::PrevisFeed::Install();  // (pass-through wrappers on the engine's two previs feed sites, plus the accessors)
		Hooks::Precipitation::Install();  // (diagnostic: the rain occlusion pass's runs and cost)
		Feed::Install();
		SetDiff::Install(settings.setDiff);
		Async::Install(settings.async);
		HangWatch::Install();
		Hooks::RenderStages::AddListener(&g_listener);
		if (settings.firstPersonAfterWorld) {
			Hooks::FirstPersonOrder::Install(&g_firstPersonListener);
		} else {
			logger::info("first person after the world: off ([Occlusion] bFirstPersonAfterWorld=0): the pre-pass draws first person first, and CBRO keeps everything behind it drawn");
		}
		g_state.installed = true;
		g_state.wantActive = settings.startActive;
		g_state.diagnostic = static_cast<int>(settings.startDiagnostic);
		if (g_state.diagnostic != 0) {
			Occlusion::SetObserveOnly(true);
			logger::info("diagnostic state at load (iStartDiagnostic={}): {}", g_state.diagnostic, g_state.diagnostic == 2 ? "decide-only, no depth capture" : "decide-only (nothing hidden)");
		}
		logger::info(
			"occlusion: installed ({}); starts in {} mode; {} toggles CBRO <-> previs",
			settings.observeOnly ? "observe-only" : "culling", settings.startActive ? "CBRO" : "previs (previs untouched until the toggle)",
			std::format("VK 0x{:X}", settings.toggleHotkey));
	}

	void OnGameDataReady()
	{
		if (g_state.installed) {
			RegisterLoadSink();
		}
	}

	void OnGameLoaded()
	{
		if (!g_state.installed) {
			return;
		}
		HiZ::Reset();
		ResetAB();
		Feed::OnGameLoaded();
		RegisterLoadSink();
		g_state.gameLoaded = true;
		ApplyMode(g_state.wantActive, "game loaded");
	}
}
