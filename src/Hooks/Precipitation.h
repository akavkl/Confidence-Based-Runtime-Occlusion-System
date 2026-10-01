#pragma once

// The precipitation (rain) occlusion map (FO4-ENGINE-NOTES 4.6), a diagnostic (v1.52): Render_PreUI+0x16F calls the
// callback 1114882 through a global; with rain occlusion on and precipitation active it tail-calls 856638(precipitation
// object, extra), which renders a top-down camera's depth into render target 86. With previs active it files previs's
// third list into a stack culling group; with previs inactive it runs BSShaderUtil::AccumulateScene over the whole world
// node (the scene root's child 3) every frame it rains. In a CBRO frame the pass runs after the cull window has closed,
// so it should see previs active, as vanilla. The detour on 856638 only counts the runs, the previs state at entry and
// the CPU time; it changes nothing and stays in in both modes.
namespace CBRO::Hooks::Precipitation
{
	void Install();

	struct Stats
	{
		std::uint64_t runs{ 0 };
		std::uint64_t previsActive{ 0 };  // runs with previs active at entry (previs's list)
		double        msActive{ 0.0 };    // CPU time of those runs
		double        msInactive{ 0.0 };  // ... of the others (the whole world node walked)
	};
	[[nodiscard]] Stats Take() noexcept;  // since the last call (main thread)
}
