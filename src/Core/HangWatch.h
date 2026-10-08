#pragma once

// Diagnostic (v1.80.2): a watchdog for freezes. The 2026-10-09 runs hung within half a second of an in-game loading
// screen opening (a door into an interior), twice, with no crash log and no dump. A background thread counts the
// main thread's world frames (Beat, at the cull stage); when none has run for 15 s it samples every thread of the
// process once (each suspended in turn: its instruction pointer and the return addresses on its stack, read into a
// buffer allocated beforehand) and, with every thread resumed, writes CBRO-hang.log next to CBRO.log: one line per
// thread, and the full frames of the main thread and of every thread with a CBRO frame, as module+offset. Nothing that
// can wait on a lock a sampled thread may hold runs while one is suspended (no allocation, no logger, no loader
// calls). A second sample follows 10 s later (a loop moves between samples, a wait stays). A legitimate loading screen
// longer than 15 s writes one too.
namespace CBRO::Core::HangWatch
{
	void Install();
	void Beat() noexcept;  // main thread, once per world frame
}
