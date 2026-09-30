#include "Util/Hooking.h"

namespace CBRO::Util
{
	namespace
	{
		bool IsReadable(std::uintptr_t a_address, std::size_t a_size) noexcept
		{
			MEMORY_BASIC_INFORMATION mbi{};
			if (!VirtualQuery(reinterpret_cast<LPCVOID>(a_address), &mbi, sizeof(mbi))) {
				return false;
			}
			if (mbi.State != MEM_COMMIT || (mbi.Protect & (PAGE_NOACCESS | PAGE_GUARD))) {
				return false;
			}
			const auto end = reinterpret_cast<std::uintptr_t>(mbi.BaseAddress) + mbi.RegionSize;
			return a_address + a_size <= end;
		}

		std::string DescribeModuleOffset(std::uintptr_t a_address)
		{
			HMODULE module = nullptr;
			if (!GetModuleHandleExW(
					GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
					reinterpret_cast<LPCWSTR>(a_address),
					&module) ||
				!module) {
				return {};
			}

			wchar_t buf[MAX_PATH]{};
			const auto len = GetModuleFileNameW(module, buf, MAX_PATH);
			const auto name = std::filesystem::path(std::wstring_view(buf, len)).filename().string();
			return std::format("{}+0x{:X}", name, a_address - reinterpret_cast<std::uintptr_t>(module));
		}

		// Resolves one level of `jmp rel32` / `jmp [rip+disp32]`, or returns 0.
		std::uintptr_t FollowJump(std::uintptr_t a_address) noexcept
		{
			if (!IsReadable(a_address, 14)) {
				return 0;
			}
			const auto* code = reinterpret_cast<const std::uint8_t*>(a_address);
			if (code[0] == 0xE9) {
				const auto disp = *reinterpret_cast<const std::int32_t*>(code + 1);
				return a_address + 5 + disp;
			}
			if (code[0] == 0xFF && code[1] == 0x25) {
				const auto disp = *reinterpret_cast<const std::int32_t*>(code + 2);
				const auto slot = a_address + 6 + disp;
				if (!IsReadable(slot, sizeof(std::uintptr_t))) {
					return 0;
				}
				return *reinterpret_cast<const std::uintptr_t*>(slot);
			}
			return 0;
		}
	}

	std::string DescribeCodeAddress(std::uintptr_t a_address)
	{
		if (!a_address) {
			return "null";
		}

		std::string result;
		auto        current = a_address;
		for (int hop = 0; hop < 4; ++hop) {
			auto described = DescribeModuleOffset(current);
			if (described.empty()) {
				described = std::format("0x{:X} (no module)", current);
			}
			result += hop == 0 ? described : " -> " + described;

			const auto next = FollowJump(current);
			if (!next || next == current) {
				break;
			}
			current = next;
		}
		return result;
	}

	std::uintptr_t ReadCall5Target(std::uintptr_t a_src) noexcept
	{
		if (!IsReadable(a_src, 5)) {
			return 0;
		}
		const auto* code = reinterpret_cast<const std::uint8_t*>(a_src);
		if (code[0] != 0xE8) {
			return 0;
		}
		const auto disp = *reinterpret_cast<const std::int32_t*>(code + 1);
		return a_src + 5 + disp;
	}

	std::uintptr_t WriteCall5(std::uintptr_t a_src, std::uintptr_t a_dst, std::string_view a_name)
	{
		const auto original = ReadCall5Target(a_src);
		if (!original) {
			const auto byte = IsReadable(a_src, 1) ? *reinterpret_cast<const std::uint8_t*>(a_src) : 0;
			logger::error("hook {}: {} is not a call rel32 (first byte 0x{:02X}); skipped", a_name, DescribeCodeAddress(a_src), byte);
			return 0;
		}

		logger::info("hook {}: call site {} currently calls {}", a_name, DescribeCodeAddress(a_src), DescribeCodeAddress(original));

		auto& trampoline = F4SE::GetTrampoline();
		const auto previous = trampoline.write_call<5>(a_src, a_dst);
		return previous;
	}

	std::uintptr_t DetourEntry(std::uintptr_t a_src, std::uintptr_t a_dst, std::span<const std::uint8_t> a_prologue, std::string_view a_name)
	{
		const auto size = a_prologue.size();
		if (size < 5 || !IsReadable(a_src, size)) {
			logger::error("hook {}: bad prologue spec for {}; skipped", a_name, DescribeCodeAddress(a_src));
			return 0;
		}

		const auto* code = reinterpret_cast<const std::uint8_t*>(a_src);
		if (std::memcmp(code, a_prologue.data(), size) != 0) {
			std::string actual;
			for (std::size_t i = 0; i < size; ++i) {
				actual += std::format("{:02X} ", code[i]);
			}
			logger::error("hook {}: {} starts with {}(expected prologue differs; already hooked?); skipped", a_name, DescribeCodeAddress(a_src), actual);
			return 0;
		}

		// gateway = relocated prologue + jmp [rip+0] -> a_src + size
		auto&      trampoline = F4SE::GetTrampoline();
		auto*      gateway = static_cast<std::uint8_t*>(trampoline.allocate(size + 14));
		std::memcpy(gateway, a_prologue.data(), size);
		gateway[size + 0] = 0xFF;
		gateway[size + 1] = 0x25;
		std::memset(gateway + size + 2, 0, 4);
		const std::uintptr_t resume = a_src + size;
		std::memcpy(gateway + size + 6, &resume, sizeof(resume));

		trampoline.write_branch<5>(a_src, a_dst);
		if (size > 5) {
			REL::safe_fill(a_src + 5, REL::INT3, size - 5);
		}

		logger::info("hook {}: detoured {} (gateway {})", a_name, DescribeCodeAddress(a_src), fmt_ptr(gateway));
		return reinterpret_cast<std::uintptr_t>(gateway);
	}

	std::uintptr_t WriteVFunc(std::uintptr_t a_vtable, std::size_t a_index, std::uintptr_t a_dst, std::string_view a_name)
	{
		const auto slot = a_vtable + sizeof(std::uintptr_t) * a_index;
		if (!IsReadable(slot, sizeof(std::uintptr_t))) {
			logger::error("hook {}: vtable slot {:#x} unreadable; skipped", a_name, slot);
			return 0;
		}

		const auto original = *reinterpret_cast<const std::uintptr_t*>(slot);
		logger::info("hook {}: vtable {} [0x{:X}] currently {}", a_name, DescribeCodeAddress(a_vtable), a_index, DescribeCodeAddress(original));

		REL::safe_write(slot, a_dst);
		return original;
	}

	std::uintptr_t DetourSwitchable(SwitchableHook& a_out, std::uintptr_t a_src, std::uintptr_t a_dst, std::span<const std::uint8_t> a_prologue, const char* a_name)
	{
		if (a_src % 8 != 0 || a_prologue.size() > 8 || !IsReadable(a_src, 8)) {
			logger::error("hook {}: {} can't be switched (needs an 8-byte aligned entry and a prologue of at most 8 bytes); skipped", a_name, DescribeCodeAddress(a_src));
			return 0;
		}
		const auto original = *reinterpret_cast<const std::uint64_t*>(a_src);
		const auto gateway = DetourEntry(a_src, a_dst, a_prologue, a_name);
		if (!gateway) {
			return 0;
		}
		a_out.address = a_src;
		a_out.original = original;
		a_out.patched = *reinterpret_cast<const std::uint64_t*>(a_src);
		a_out.in = true;
		a_out.name = a_name;
		return gateway;
	}

	std::uintptr_t WriteVFuncSwitchable(SwitchableHook& a_out, std::uintptr_t a_vtable, std::size_t a_index, std::uintptr_t a_dst, const char* a_name)
	{
		const auto original = WriteVFunc(a_vtable, a_index, a_dst, a_name);
		if (!original) {
			return 0;
		}
		a_out.address = a_vtable + sizeof(std::uintptr_t) * a_index;
		a_out.original = original;
		a_out.patched = a_dst;
		a_out.in = true;
		a_out.name = a_name;
		return original;
	}

	bool PatchJumpSwitchable(SwitchableHook& a_out, std::uintptr_t a_src, std::span<const std::uint8_t> a_verify, std::uintptr_t a_dst, const char* a_name)
	{
		constexpr std::size_t kPatchSize = 7;  // jmp rel32 + 2 x int3
		const auto            offset = a_src % 8;
		if (offset + kPatchSize > 8 || a_verify.size() < 5 || !IsReadable(a_src, a_verify.size())) {
			logger::error("hook {}: {} can't take a switchable jump (needs 7 bytes inside one 8-byte word); skipped", a_name, DescribeCodeAddress(a_src));
			return false;
		}
		const auto* code = reinterpret_cast<const std::uint8_t*>(a_src);
		if (std::memcmp(code, a_verify.data(), a_verify.size()) != 0) {
			std::string actual;
			for (std::size_t i = 0; i < a_verify.size(); ++i) {
				actual += std::format("{:02X} ", code[i]);
			}
			logger::error("hook {}: {} holds {}(expected bytes differ; another plugin patched it, or a different exe); skipped", a_name, DescribeCodeAddress(a_src), actual);
			return false;
		}
		const auto     displacement = static_cast<std::int64_t>(a_dst) - static_cast<std::int64_t>(a_src + 5);
		if (displacement < INT32_MIN || displacement > INT32_MAX) {
			logger::error("hook {}: stub too far from {} for a rel32 jump; skipped", a_name, DescribeCodeAddress(a_src));
			return false;
		}
		const auto    aligned = a_src - offset;
		std::uint64_t original = *reinterpret_cast<const std::uint64_t*>(aligned);
		std::uint8_t  patchedBytes[8];
		std::memcpy(patchedBytes, &original, 8);
		patchedBytes[offset] = 0xE9;
		const auto rel32 = static_cast<std::int32_t>(displacement);
		std::memcpy(patchedBytes + offset + 1, &rel32, 4);
		patchedBytes[offset + 5] = REL::INT3;
		patchedBytes[offset + 6] = REL::INT3;
		a_out.address = aligned;
		a_out.original = original;
		std::memcpy(&a_out.patched, patchedBytes, 8);
		a_out.in = false;
		a_out.name = a_name;
		if (!SetHook(a_out, true)) {
			a_out.address = 0;
			logger::error("hook {}: {} changed while patching; skipped", a_name, DescribeCodeAddress(a_src));
			return false;
		}
		logger::info("hook {}: switchable jump at {} -> {}", a_name, DescribeCodeAddress(a_src), DescribeCodeAddress(a_dst));
		return true;
	}

	std::uintptr_t WriteStub(std::span<const std::uint8_t> a_bytes)
	{
		auto& trampoline = F4SE::GetTrampoline();
		auto* stub = static_cast<std::uint8_t*>(trampoline.allocate(a_bytes.size()));
		if (!stub) {
			return 0;
		}
		std::memcpy(stub, a_bytes.data(), a_bytes.size());
		return reinterpret_cast<std::uintptr_t>(stub);
	}

	bool SetHook(SwitchableHook& a_hook, bool a_in) noexcept
	{
		if (!a_hook.address) {
			return false;
		}
		if (a_hook.in == a_in) {
			return true;
		}
		const auto from = a_in ? a_hook.original : a_hook.patched;
		const auto to = a_in ? a_hook.patched : a_hook.original;
		auto* const target = reinterpret_cast<void*>(a_hook.address);
		// Code pages must stay executable throughout (other threads may be running them).
		DWORD old = 0;
		if (!VirtualProtect(target, sizeof(std::uint64_t), PAGE_EXECUTE_READWRITE, &old)) {
			return false;
		}
		const auto seen = static_cast<std::uint64_t>(InterlockedCompareExchange64(
			static_cast<volatile LONG64*>(target), static_cast<LONG64>(to), static_cast<LONG64>(from)));
		DWORD unused = 0;
		VirtualProtect(target, sizeof(std::uint64_t), old, &unused);
		FlushInstructionCache(GetCurrentProcess(), target, sizeof(std::uint64_t));
		if (seen != from) {
			return false;
		}
		a_hook.in = a_in;
		return true;
	}
}
