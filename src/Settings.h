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
		std::uint32_t hiZDownsample{ 4 };          // depth texture pixels per Hi-Z texel edge (1280x800 -> 320x200; not the drawn region, which dynamic resolution changes)
		std::uint32_t confirmFrames{ 2 };          // consecutive hidden verdicts before an object is rejected
		std::uint32_t maxSnapshotAge{ 6 };         // frames; older depth is not trusted
		float         maxCameraMove{ 256.0f };     // game units since the depth frame; beyond this, cull nothing
		float         maxCameraAngle{ 60.0f };     // degrees since the depth frame; beyond this, cull nothing (must stay < 90)
		float         nearDistance{ 8.0f };        // objects with any point closer than this (view depth) stay visible
		bool          cullActors{ true };          // NPC/creature parts may be hidden too (their shadows stay: main view only)
		bool          cullPlayer{ false };         // the player's own 3D may be culled too; off: never left out of any view (v1.89)
		bool          traceFeed{ true };           // a path tracer that asks (Renderer Replacer) gets every refused registration, and what CBRO leaves out filed for it on a turn (v1.90)
		bool          traceLink{ true };           // a tracer that keeps what it is handed gets each object once, through the direct link (v2.0)
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
		bool          entryWalk{ true };           // a node pruning can't leave out whole has its entries walked and filed by CBRO itself (v1.94)
		bool          childCheapTests{ true };     // a static mesh child out of the view that no view needs is left out by the walk's cheap tests alone, with no verdict record (v1.96)
		bool          childInherit{ true };        // a child with its kept top entry's very bound takes that decision, no judgement of its own (v1.98)
		bool          walkLeases{ true };          // runs of walked entries out of the view are left untouched between walks (v1.98)
		std::uint32_t walkLeaseFrames{ 4 };        // ... a run is read again at most this many frames after its last walk
		float         walkLeaseMargin{ 64.0f };    // ... units a run's sphere must clear the view by on top of the verdict's margins, for what moves meanwhile
		bool          stableIndex{ true };         // children find their decision parent's record by a cursor in filing order, not by table lookups (v1.99.2)
		bool          traceDirect{ true };         // an entry filed for the trace alone is handed to the tracer directly, not filed (v1.99.3)
		bool          deferSunOnly{ true };        // a sun-only drop is filed into group 0 in the cascade window, after the main view (v1.99.3)
		bool          cellArtNode9{ true };        // CBRO frames: a cell's node 9 (precombined art) the engine has AppCulled is filed anyway, as previs draws it by id
		bool          interiorOutsideRooms{ true };  // CBRO frames, camera in an interior room bound: what lies outside every room bound is filed anyway (previs ignores rooms)
		bool          disableRoomBounds{ true };     // CBRO frames, interiors: every room is filed whole with the camera's own frustum, never through portals, and so is every portal's shared geometry (v1.85-v1.86)
		bool          ignoreOccluders{ true };       // CBRO frames: the main view's frustum is the camera's alone, without the cells' occlusion planes/boxes or the portals onto the outside (v1.87)
		bool          parallaxMargin{ true };        // while the camera moves: an object's screen box widens by a near occluder's parallax since the depth frame (v1.88)
		bool          keepFadeState{ true };         // fade nodes left out of the main view keep their fade, so a revealed object doesn't fade in again from 0 (v1.88.2)
		bool          firstPersonAfterWorld{ true };  // CBRO frames: the pre-pass draws first person after the world, whose depth CBRO captures in between
		bool          previsFeed{ true };          // previs is never switched off: suspended (flush-free) while CBRO is on (Core/Feed)
		bool          skipPrevisQuery{ true };     // CBRO frames: previs's per-frame query is neither launched nor collected (Hooks/PrevisFeed)
		bool          interiors{ true };           // CBRO culls interiors/override-root scenes too (0: stands by there, the engine's previs and rooms untouched)
		bool          interiorLegacyPrevis{ true }; // interiors: v1.28's previs switch (off and flushed while CBRO is on) instead of the windows
		bool          asyncInteriors{ false };     // interiors: judge through the worker too (0: inside the walk, as v1.28)
		bool          setDiff{ false };             // diagnostic: main-view registrations compared between the modes at each A/B location (Core/SetDiff)
		bool          sampler{ false };            // diagnostic: the main thread's instruction pointer sampled by mode and stage (Core/Sampler)
		bool          async{ true };               // verdicts judged by a worker after the cull, looked up by the walk a frame later (Core/Async); 0 = inside the walk
		bool          lampGroupTrim{ true };       // sun off: group-0 entries the main view doesn't need are left out when no lamp's shadow of them can reach a visible surface
		bool          unoccludeLights{ true };     // CBRO frames: clear the previs-driven BSLight::bOccluded mark before the deferred-lights stage (dark lamps otherwise)
		bool          lampDiagnostic{ false };      // diagnostic: the engine's shadow-light loop decisions per light in CBRO frames (Core/ShadowLights)
		bool          leftOutDump{ false };        // diagnostic: with each still-location dump, what the main view leaves out, kept mid-screen entries' fade state, a scene scan and the depth picture
		float         rayWindow[4]{ -1.0f, -0.45f, -0.85f, 0.05f };  // diagnostic (bLeftOutDump, bSetDiff): the screen window the dumps list everything in (NDC x from, to; y from, to)
		float         rayLength{ 4000.0f };        // how far along that window's rays a bound still counts
		std::uint32_t feedAuditInterval{ 0 };    // frames between audits of CBRO's scene-walk replica (0 = off)
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

		// [Occlusion] the decision recorder (Core/Recorder, v1.95)
		bool          recorder{ false };          // the recorder is available (its hotkey starts and stops it); off: no cost
		std::uint32_t recorderFrames{ 1800 };     // the frame limit (the recording saves itself there)
		std::uint32_t recorderHotkey{ 0 };        // starts a recording; pressed again, stops and saves it (unbound by default)
		std::int32_t  recorderHiZLevel{ 2 };      // the Hi-Z pyramid level recorded with each frame (0 = full, -1 = none)

		void Load();

		[[nodiscard]] static Settings& Get() noexcept
		{
			static Settings instance;
			return instance;
		}
	};
}
