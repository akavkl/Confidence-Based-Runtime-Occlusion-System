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
		bool          lampShadowCulling{ true };   // leave casters out of point-light shadow maps where they can't shadow the view
		bool          lampShadowVolumes{ true };   // ... also when their shadow volume misses every visible surface (nearest/farthest depth test)
		bool          cellNodePruning{ true };     // a cell's static-object node outside the view (and its sun shadow) is never walked
		bool          previsFeed{ true };          // previs is never switched off: suspended (flush-free) while CBRO is on (Core/Feed)
		bool          interiors{ false };          // CBRO culls interiors/override-root scenes too (0: stands by there, the engine's previs and rooms untouched)
		std::uint32_t feedAuditInterval{ 600 };    // frames between audits of CBRO's scene-walk replica (0 = off)
		bool          verdictCache{ true };        // reuse last frame's verdict while nothing it depended on changed
		float         cacheMove{ 4.0f };           // camera movement (units) a reused verdict tolerates; bounds grow by it
		float         cacheAngle{ 0.1f };          // camera turn (degrees) a reused verdict tolerates; bounds grow by depth x it
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
