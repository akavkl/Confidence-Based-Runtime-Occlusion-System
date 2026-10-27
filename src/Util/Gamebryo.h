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

	// The engine's loop bound over a NiTArray such as NiNode::children: the slot past the last used one (the array's
	// +0x12, a NiNode's +0x132: the scene walk's `cmp dx, word [rax+132h]`). CommonLib's size() reads +0x14, the count
	// of filled slots, which is lower once a slot was emptied (a detached child leaves its slot null) and a later
	// attach appended behind it. Runtime Combiner parks a cell's originals that way and appends its chunk container
	// last (2026-10-26: a cell's node 9 with the container at slot 53), so a loop bounded by size() never reached it.
	template <class T>
	[[nodiscard]] std::uint16_t UsedSlots(const T& a_array) noexcept
	{
		static_assert(sizeof(T) == 0x18);  // vtable, data, capacity, used, filled, growth
		return *reinterpret_cast<const std::uint16_t*>(reinterpret_cast<const std::byte*>(&a_array) + 0x12);
	}
}
