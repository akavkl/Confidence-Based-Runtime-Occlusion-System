#pragma once

namespace CBRO
{
	struct Settings
	{
		// [General]
		bool enabled{ true };

		// [Occlusion]
		bool          occlusion{ true };           // CBRO culls the main camera pass
		bool          disablePrevis{ true };       // switch previs off while CBRO is active (one visibility authority)
		bool          startActive{ true };         // start in CBRO mode (0: start in previs mode, previs untouched until the toggle)
		bool          observeOnly{ false };        // decide but never drop (stats only)
		std::uint32_t toggleHotkey{ VK_F8 };       // flips CBRO <-> previs at runtime for A/B measurement
		std::uint32_t statusHotkey{ 0 };           // shows the current mode and cull rate on screen (unbound by default)
		std::uint32_t diagnosticHotkey{ 0 };       // cycles normal / decide-only / decide-only without depth capture (unbound)
		std::uint32_t startDiagnostic{ 0 };        // the diagnostic state at load: 0 normal, 1 decide-only (hides nothing), 2 decide-only without the depth capture
		bool          notify{ true };              // on-screen message when the mode changes
		std::uint32_t hiZDownsample{ 4 };          // depth pixels per Hi-Z texel edge (1280x800 -> 320x200)
		std::uint32_t confirmFrames{ 2 };          // consecutive hidden verdicts before an object is rejected
		std::uint32_t maxSnapshotAge{ 6 };         // frames; older depth is not trusted
		float         maxCameraMove{ 256.0f };     // game units since the depth frame; beyond this, cull nothing
		float         maxCameraAngle{ 60.0f };     // degrees since the depth frame; beyond this, cull nothing (must stay < 90)
		float         nearDistance{ 8.0f };        // objects with any point closer than this (view depth) stay visible
		bool          cullActors{ true };          // NPC/creature parts may be hidden too (their shadows stay: main view only)
		bool          meshShapes{ true };          // big static meshes are judged by their actual geometry (box + occupancy grid)
		bool          sunShadowCulling{ true };    // skip hidden objects whose sun shadow can't fall on anything visible
		bool          farCascadeTrim{ true };      // ... and leave a needed shadow out of the sun's last cascade when nothing visible lies in its slab (v1.80)
		bool          lampShadowCulling{ true };   // leave casters out of point-light shadow maps where they can't shadow the view
		bool          lampShadowVolumes{ true };   // ... also when their shadow volume misses every visible surface (nearest/farthest depth test)
		bool          spotShadowCulling{ true };   // spot lights: the shadow map draws nothing while the lit volume (shadow frustum) is hidden
		bool          spotCullWindow{ true };      // CBRO frames: previs suspended around a spot light's group-0 cull, so its frustum test runs (else stale results)
		bool          spotCasterTrim{ true };      // kept spot lights: casters that can't shadow a visible pixel are left out of the shadow map (as point lights' are)
		bool          emptiedSpotTrim{ true };     // sun up: group 0 keeps no entry for a spot lamp whose shadow map stays empty this frame (v1.81)
		bool          spotOnlyCascadeTrim{ true }; // sun up: a group-0 entry kept for a spot lamp alone is left out of the sun's cascades (v1.82)
		bool          scolRootBounds{ true };      // a static collection's pieces are judged (and filed) with the collection's bound, not their own (v1.83)
		bool          cellNodePruning{ true };     // a cell's static-object node outside the view (and its sun shadow) is never walked
		bool          cellArtNode9{ true };        // CBRO frames: a cell's node 9 (precombined art) the engine has AppCulled is filed anyway, as previs draws it by id
		bool          interiorOutsideRooms{ true };  // CBRO frames, camera in an interior room bound: what lies outside every room bound is filed anyway (previs ignores rooms)
		bool          firstPersonAfterWorld{ true };  // CBRO frames: the pre-pass draws first person after the world, whose depth CBRO captures in between
		bool          previsFeed{ true };          // previs is never switched off: suspended (flush-free) while CBRO is on (Core/Feed)
		bool          skipPrevisQuery{ true };     // CBRO frames: previs's per-frame query is neither launched nor collected (Hooks/PrevisFeed)
		bool          interiors{ true };           // CBRO culls interiors/override-root scenes too (0: stands by there, the engine's previs and rooms untouched)
		bool          interiorLegacyPrevis{ true }; // interiors: v1.28's previs switch (off and flushed while CBRO is on) instead of the windows
		bool          asyncInteriors{ false };     // interiors: judge through the worker too (0: inside the walk, as v1.28)
		bool          setDiff{ true };             // diagnostic: main-view registrations compared between the modes at each A/B location (Core/SetDiff)
		bool          async{ true };               // verdicts judged by a worker after the cull, looked up by the walk a frame later (Core/Async); 0 = inside the walk
		bool          lampGroupTrim{ true };       // sun off: group-0 entries the main view doesn't need are left out when no lamp's shadow of them can reach a visible surface
		bool          unoccludeLights{ true };     // CBRO frames: clear the previs-driven BSLight::bOccluded mark before the deferred-lights stage (dark lamps otherwise)
		bool          lampDiagnostic{ true };      // diagnostic: the engine's shadow-light loop decisions per light in CBRO frames (Core/ShadowLights)
		bool          leftOutDump{ false };        // diagnostic: with each still-location dump, what the main view leaves out, kept mid-screen entries' fade state, a scene scan and the depth picture
		std::uint32_t feedAuditInterval{ 600 };    // frames between audits of CBRO's scene-walk replica (0 = off)
		bool          verdictCache{ true };        // reuse last frame's verdict while nothing it depended on changed
		float         cacheMove{ 4.0f };           // camera movement (units) a reused verdict tolerates; bounds grow by it
		float         cacheAngle{ 0.1f };          // camera turn (degrees) a reused verdict tolerates; bounds grow by depth x it
		float         viewOverhang{ 0.03f };       // NDC the view may reach past the (older) depth frame and objects/sun shadows still be judged, by its edge (0 = kept)
		float         viewOverhangTurn{ 0.08f };    // ... while turning, up to this much: the strip the turn brought into view, for objects mostly inside the frame (0 = off)
		std::uint32_t hiZTemporalFrames{ 4 };      // Hi-Z = farthest depth over this many consecutive frames of a still camera (1 = off)
		float         depthTolerance{ 0.002f };    // relative margin on linear view depth before an object counts as hidden
		float         depthSlack{ 4.0f };          // plus this many game units
		float         worldDepthMin{ 0.01f };      // viewport depth range used for world geometry (first person uses [0, 0.01])
		float         worldDepthMax{ 1.0f };

		// [Probe] (Phase 0 instrumentation; logging only). Off by default: it hooks D3D11 context
		// methods (wrapped by ENB and other mods) and four extra render stages shared with Upscaling.
		bool          probe{ false };
		bool          renderStageHooks{ true };
		bool          cullingHooks{ false };
		bool          cullingGroupHooks{ false };
		bool          drawCallCounter{ true };
		bool          sceneSurvey{ false };
		std::uint32_t summaryIntervalFrames{ 600 };
		std::uint32_t traceFrames{ 3 };
		std::uint32_t traceDelayFramesAfterLoad{ 300 };
		std::uint32_t traceHotkey{ VK_F3 };
		std::uint32_t surveyHotkey{ VK_F12 };

		void Load();

		[[nodiscard]] static Settings& Get() noexcept
		{
			static Settings instance;
			return instance;
		}
	};
}
