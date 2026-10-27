#pragma once

// Diagnostic (bSampler, v1.93.4): a sampling profiler of the main thread. The per-bucket CPU timing (Core/Runtime's
// render-stage marks) says how long each stage takes in each mode, and CBRO's own rdtsc buckets say what its hooks
// cost, but neither says where the rest of a stage goes: the engine's own walk, another plugin, the driver, the
// runtime. A thread of its own reads the main thread's instruction pointer about 2,500 times a second (the thread
// suspended a few microseconds each time: 1-3 % of its time) with the top of its stack, and files the sample under
// the frame's mode and the stage the main thread is in (SetPhase at every mark). The summary gives, per mode and
// stage, the share by module and the busiest addresses as module+offset: CBRO's named from the linker map (CBRO.map
// in the build folder), the engine's from the address database. A sample inside a system library (the CRT, D3D, the
// driver) is charged to the first return address on its stack that isn't, so memcpy and the runtime's work land on
// the code that asked for them; the module shares are by the sampled address itself. Nothing that can wait on a lock
// the main thread may hold runs while it is suspended (no allocation, no logger, no loader calls).
namespace CBRO::Core::Sampler
{
	void Install(bool a_enabled, std::span<const std::string_view> a_modeNames, std::span<const std::string_view> a_bucketNames);
	// Main thread, at each render-stage mark: the mode and bucket the samples until the next mark count under.
	void SetPhase(int a_mode, std::size_t a_bucket) noexcept;
	// Main thread, with the summary: the interval's samples by mode, stage, module and address, then cleared.
	void LogStats(std::uint32_t a_frames);
}
