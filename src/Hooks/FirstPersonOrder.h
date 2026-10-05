#pragma once

// The deferred pre-pass (56596, FO4-ENGINE-NOTES 4.2) draws first person before the world twice over, in its depth-only
// z-prepass (1491502's first-person lists) and in the first-person accumulator's block, so no world depth ever lies
// behind a drawn weapon: CBRO's capture held "anything could be there" over it, every object and every sun-shadow sweep
// touching the weapon stayed drawn, and outdoors that cost ~6 ms (v1.68 logs). Since v1.69, in a pre-pass whose depth
// CBRO captures, both are held back: the z-prepass draws only the world's lists in its place, the world block runs, the
// listener captures the world's complete depth, then first person runs in the engine's own order (its z-prepass lists,
// then its block) under its own camera. The image is the same (first person's depth range [0, 0.01] lies in front of
// the world's [0.01, 1]); the cost is the world's pixels behind the weapon, shaded once in the g-buffer. Any other
// pre-pass (previs mode, other callers) runs untouched.
namespace CBRO::Hooks::FirstPersonOrder
{
	class Listener
	{
	public:
		virtual ~Listener() = default;
		[[nodiscard]] virtual bool WantWorldDepth() = 0;  // this pre-pass captures depth: hold the first-person block back
		virtual void               OnWorldDepth() = 0;    // the world's depth is complete, first person not drawn yet
	};

	// Patches the four call sites in 56596 (verified first: an engine target each, else nothing is written).
	bool Install(Listener* a_listener);
	[[nodiscard]] bool Installed() noexcept;

	// At the pre-pass stage's end: a first-person block still held back (the world block's end never came) runs now,
	// after the listener's capture, so first person is never lost.
	void Flush();

	// Where the main pre-pass is (probe diagnostics): 0 before its first-person block (or nothing held), 1 the world
	// block with first person held back, 2 the held block running, 3 after it.
	[[nodiscard]] std::uint32_t Phase() noexcept;

	void LogStats(std::uint32_t a_frames);
}
