#pragma once

// Objects left out of the main view this frame (Occlusion: entries of groups the sun's shadow cascades
// read too stay in their group; only the main accumulator's registration of them is skipped).
// Engine-free, so the offline brute-force test compiles it as is.
//
// Lock-free open addressing, one word per slot: (address >> 4) << 20 | frame tag. A slot tagged with
// another frame is free, so within a frame slots only ever go from free to taken: a lookup may stop at
// the first free slot, and nothing needs clearing, except once when the 20-bit tag wraps (Clear, at a
// frame start with no culling running).

namespace CBRO::Core
{
	class DropSet
	{
	public:
		static constexpr std::uint32_t kTagBits = 20;
		static constexpr std::uint64_t kTagMask = (std::uint64_t{ 1 } << kTagBits) - 1;
		static constexpr std::uint32_t kTagCycle = static_cast<std::uint32_t>(kTagMask);  // live tags 1..kTagMask; 0 = never used

		static std::uint32_t Tag(std::uint32_t a_clock) noexcept { return a_clock % kTagCycle + 1; }

		void Allocate() { m_slots = std::make_unique<std::atomic<std::uint64_t>[]>(kSize); }

		void Clear() noexcept
		{
			for (std::size_t i = 0; i < kSize; ++i) {
				m_slots[i].store(0, std::memory_order_relaxed);
			}
		}

		// False if the neighbourhood is full (then the object stays drawn).
		bool Insert(const void* a_object, std::uint32_t a_tag) noexcept
		{
			const auto word = Pack(a_object, a_tag);
			auto       index = Hash(a_object);
			for (std::size_t probe = 0; probe < kMaxProbe; ++probe, index = (index + 1) & (kSize - 1)) {
				auto& slot = m_slots[index];
				auto  current = slot.load(std::memory_order_acquire);
				for (;;) {
					if (current == word) {
						return true;
					}
					if ((current & kTagMask) == a_tag) {
						break;  // taken this frame by another object
					}
					if (slot.compare_exchange_weak(current, word, std::memory_order_acq_rel, std::memory_order_acquire)) {
						return true;
					}
				}
			}
			return false;
		}

		[[nodiscard]] bool Contains(const void* a_object, std::uint32_t a_tag) const noexcept
		{
			const auto word = Pack(a_object, a_tag);
			auto       index = Hash(a_object);
			for (std::size_t probe = 0; probe < kMaxProbe; ++probe, index = (index + 1) & (kSize - 1)) {
				const auto current = m_slots[index].load(std::memory_order_acquire);
				if (current == word) {
					return true;
				}
				if ((current & kTagMask) != a_tag) {
					return false;  // a free slot ends the chain
				}
			}
			return false;
		}

		static constexpr std::size_t kBits = 16;  // 64k slots; a frame drops at most ~12k entries
		static constexpr std::size_t kSize = std::size_t{ 1 } << kBits;
		static constexpr std::size_t kMaxProbe = 32;

	private:
		// User-space addresses stay below 2^47, and engine objects are 16-byte aligned: 43 bits of key.
		static std::uint64_t Pack(const void* a_object, std::uint32_t a_tag) noexcept
		{
			return ((reinterpret_cast<std::uintptr_t>(a_object) >> 4) << kTagBits) | a_tag;
		}

		static std::size_t Hash(const void* a_object) noexcept
		{
			return static_cast<std::size_t>(((reinterpret_cast<std::uintptr_t>(a_object) >> 4) * 0x9E3779B97F4A7C15ull) >> (64 - kBits));
		}

		std::unique_ptr<std::atomic<std::uint64_t>[]> m_slots;
	};
}
