#pragma once

namespace CBRO::Util
{
	// "Fallout4.exe+0x1234", "Upscaling.dll+0x10", or "0x7FF6... (no module)". Follows up to
	// a few E9/FF25 jumps so a hook trampoline reports the module it finally lands in.
	[[nodiscard]] std::string DescribeCodeAddress(std::uintptr_t a_address);

	// Replaces the 5-byte `call rel32` at a_src so it calls a_dst, and returns the previous
	// call target (which may be another plugin's thunk; calling it keeps their hook in the chain).
	// Returns 0 and leaves the code untouched if a_src is not a `call rel32`.
	[[nodiscard]] std::uintptr_t WriteCall5(std::uintptr_t a_src, std::uintptr_t a_dst, std::string_view a_name);

	// Reads the target of the `call rel32` at a_src, or 0 if it isn't one.
	[[nodiscard]] std::uintptr_t ReadCall5Target(std::uintptr_t a_src) noexcept;

	// Follows `jmp rel32` / `jmp [rip+disp32]` chains from a_address (at most a_hops of them) and returns where they
	// land; a_address itself when it isn't a jump. A `write_call<5>` makes a site call a jump stub in the trampoline
	// that lands on the thunk: this resolves such a stub to the thunk.
	[[nodiscard]] std::uintptr_t FollowJumps(std::uintptr_t a_address, int a_hops = 4) noexcept;

	// Function-entry detour. a_prologue must be the exact leading bytes of a_src, made of whole,
	// position-independent instructions totalling >= 5 bytes. They are verified (so an entry some
	// other plugin already patched is left alone), relocated into a gateway, and replaced by a
	// jmp to a_dst. Returns the gateway (call it to run the original), or 0 on mismatch.
	[[nodiscard]] std::uintptr_t DetourEntry(std::uintptr_t a_src, std::uintptr_t a_dst, std::span<const std::uint8_t> a_prologue, std::string_view a_name);

	// Swaps one vtable slot and returns the previous function pointer.
	[[nodiscard]] std::uintptr_t WriteVFunc(std::uintptr_t a_vtable, std::size_t a_index, std::uintptr_t a_dst, std::string_view a_name);

	// A hook that can be taken out and put back at run time, so the engine runs its own code untouched
	// while CBRO is off. A detour swaps the function's first 8 bytes (the entry must be 8-byte aligned)
	// between the original instructions and the jump to the thunk; a vtable hook swaps the slot. Each
	// swap is one atomic compare-exchange, so a thread entering the function at that moment sees one
	// complete version or the other, and a hook another plugin has since stacked on top is never undone
	// (the swap then fails and the hook stays in).
	struct SwitchableHook
	{
		std::uintptr_t address{ 0 };  // the function entry, or the vtable slot
		std::uint64_t  original{ 0 };  // the 8 bytes there without CBRO
		std::uint64_t  patched{ 0 };   // ... with CBRO's hook
		bool           in{ false };
		const char*    name{ "" };
	};

	// DetourEntry that also records what SetHook needs. Returns the gateway (0 on failure, a_out untouched).
	[[nodiscard]] std::uintptr_t DetourSwitchable(SwitchableHook& a_out, std::uintptr_t a_src, std::uintptr_t a_dst, std::span<const std::uint8_t> a_prologue, const char* a_name);

	// WriteVFunc that also records what SetHook needs. Returns the previous function pointer (0 on failure).
	[[nodiscard]] std::uintptr_t WriteVFuncSwitchable(SwitchableHook& a_out, std::uintptr_t a_vtable, std::size_t a_index, std::uintptr_t a_dst, const char* a_name);

	// A switchable `jmp rel32` in the middle of a function: the 5-byte jump plus int3 padding replaces the
	// a_verify.size() bytes at a_src (whole instructions, which the caller's stub must replicate; a_verify
	// holds their expected bytes). The swap is the 8-byte word holding a_src, so the patch (7 bytes: jmp +
	// 2 int3) must lie inside it: a_src % 8 must be 0 or 1; a byte of the word before a_src is kept as it
	// is. Bytes of the replaced instructions past the 7 patched ones stay too (nothing jumps to them). The
	// patch goes in through SetHook. Returns false (nothing written) on any mismatch.
	[[nodiscard]] bool PatchJumpSwitchable(SwitchableHook& a_out, std::uintptr_t a_src, std::span<const std::uint8_t> a_verify, std::uintptr_t a_dst, const char* a_name);

	// Copies a_bytes into the executable trampoline (code only: never data that gets written at run time).
	[[nodiscard]] std::uintptr_t WriteStub(std::span<const std::uint8_t> a_bytes);

	// Puts the hook in or takes it out. False if the memory no longer holds what CBRO left there (another
	// plugin patched it since): then nothing is written.
	bool SetHook(SwitchableHook& a_hook, bool a_in) noexcept;

	template <class F>
	[[nodiscard]] std::uintptr_t FnAddr(F a_fn) noexcept
	{
		return reinterpret_cast<std::uintptr_t>(a_fn);
	}
}
