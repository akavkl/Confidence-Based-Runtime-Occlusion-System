#include "Util/Gamebryo.h"

namespace CBRO::Util
{
	std::uintptr_t TryReadVtable(const void* a_object) noexcept
	{
		if (reinterpret_cast<std::uintptr_t>(a_object) < 0x10000) {
			return 0;
		}
		__try {
			return *static_cast<const std::uintptr_t*>(a_object);
		} __except (EXCEPTION_EXECUTE_HANDLER) {
			return 0;
		}
	}

	bool TryGetRTTIName(const void* a_object, char* a_out, std::size_t a_size) noexcept
	{
		if (!a_object || !a_out || a_size == 0) {
			return false;
		}
		a_out[0] = '\0';
		// NiObject::GetRTTI (vtable slot 2) is always `lea rax, [rip+ms_RTTI]; ret`. Decode it
		// instead of calling it, so an object that isn't an NiObject can't run arbitrary code.
		__try {
			const auto vtable = *static_cast<const std::uintptr_t* const*>(a_object);
			const auto getRTTI = reinterpret_cast<const std::uint8_t*>(vtable[2]);
			if (getRTTI[0] != 0x48 || getRTTI[1] != 0x8D || getRTTI[2] != 0x05 || getRTTI[7] != 0xC3) {
				return false;
			}
			const auto disp = *reinterpret_cast<const std::int32_t*>(getRTTI + 3);
			const auto rtti = reinterpret_cast<const RE::NiRTTI*>(getRTTI + 7 + disp);
			const auto name = rtti->GetName();
			if (!name) {
				return false;
			}
			strncpy_s(a_out, a_size, name, _TRUNCATE);
			return true;
		} __except (EXCEPTION_EXECUTE_HANDLER) {
			a_out[0] = '\0';
			return false;
		}
	}

	bool TryGetObjectName(const void* a_object, char* a_out, std::size_t a_size) noexcept
	{
		if (!a_object || !a_out || a_size == 0) {
			return false;
		}
		a_out[0] = '\0';
		__try {
			const auto name = static_cast<const RE::NiObjectNET*>(a_object)->name.c_str();
			if (!name) {
				return false;
			}
			strncpy_s(a_out, a_size, name, _TRUNCATE);
			return true;
		} __except (EXCEPTION_EXECUTE_HANDLER) {
			a_out[0] = '\0';
			return false;
		}
	}
}
