#pragma once

// SEH-guarded readers for Gamebryo objects whose type isn't known yet (hook arguments, userData).

namespace CBRO::Util
{
	// First qword of a_object (its vtable), or 0 if it isn't readable.
	[[nodiscard]] std::uintptr_t TryReadVtable(const void* a_object) noexcept;

	// Copies the NiRTTI name of a Gamebryo object into a_out. The name is decoded from the
	// GetRTTI body (`lea rax, [rip+ms_RTTI]; ret`), never by calling it.
	bool TryGetRTTIName(const void* a_object, char* a_out, std::size_t a_size) noexcept;

	// Copies the NiObjectNET name of a Gamebryo object into a_out.
	bool TryGetObjectName(const void* a_object, char* a_out, std::size_t a_size) noexcept;
}