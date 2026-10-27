#pragma once

// Objects left out of the main view this frame (Occlusion: entries of groups the sun's shadow cascades
// read too stay in their group; only the main accumulator's registration of them is skipped).
// Engine-free, so the offline brute-force test compiles it as is.
//
// Lock-free open addressing, one word per slot: (address >> 4) << 20 | frame tag. A slot tagged with
// another frame is free, so within a frame slots only ever go from free to taken: a lookup may stop at
// the first free slot, and nothing needs clearing, except once when the 20-bit tag wraps (Clear, at a
// frame start with no culling running).
//
// Presence bits (v1.92.2): one bit per hashed address, 4 KB per tag parity. Most lookups are for objects
// not in the set (the parent of a child that was not forced, the last cascade's parents), and each read a
// random slot of the 512 KB table: a cache miss, ~0.5 ms a frame on the main thread across the sets in
// the 2026-10-26 profile. A clear bit answers those from a block that stays in the core's caches; a set
// bit (the object's, or another's hashing to it) goes on to the table. A tag's bits are used only when
// BeginTag cleared them before the tag's first insert.

#include <xmmintrin.h>

namespace CBRO::Core
{
	class DropSet
	{
	public:
		static constexpr std::uint32_t kTagBits = 20;
		static constexpr std::uint64_t kTagMask = (std::uint64_t{ 1 } << kTagBits) - 1;
		static constexpr std::uint32_t kTagCycle = static_cast<std::uint32_t>(kTagMask);  // live tags 1..kTagMask; 0 = never used

		static std::uint32_t Tag(std::uint32_t a_clock) noexcept { return a_clock % kTagCycle + 1; }

		void Allocate()
		{
			m_slots = std::make_unique<std::atomic<std::uint64_t>[]>(kSize);
			m_presence = std::make_unique<std::atomic<std::uint64_t>[]>(2 * kPresenceWords);
		}

		void Clear() noexcept
		{
			for (std::size_t i = 0; i < kSize; ++i) {
				m_slots[i].store(0, std::memory_order_relaxed);
			}
			for (std::uint32_t parity = 0; parity < 2; ++parity) {
				m_presenceTag[parity].store(0, std::memory_order_relaxed);
				m_presenceLate[parity].store(0, std::memory_order_relaxed);
			}
		}

		// The tag's presence bits, cleared (at a frame start, before the frame's clock is published; a second call
		// for the same tag changes nothing).
		void BeginTag(std::uint32_t a_tag) noexcept
		{
			const auto parity = a_tag & 1;
			if (!m_presence || m_presenceTag[parity].load(std::memory_order_relaxed) == a_tag) {
				return;
			}
			auto* const bits = &m_presence[parity * kPresenceWords];
			for (std::size_t i = 0; i < kPresenceWords; ++i) {
				bits[i].store(0, std::memory_order_relaxed);
			}
			m_presenceTag[parity].store(a_tag, std::memory_order_release);
		}

		// The slot an Insert or Contains of the object reads first, fetched into L2 ahead of time: the judge between the
		// prefetch and the insert hides the table's miss (Insert's slot read was 0.15 ms a frame on the main thread in
		// the 2026-10-26 profile; v1.92.3).
		void Prefetch(const void* a_object) const noexcept
		{
			if (m_slots) {
				_mm_prefetch(reinterpret_cast<const char*>(&m_slots[Hash(a_object)]), _MM_HINT_T1);
			}
		}

		// False if the neighbourhood is full (then the object stays drawn).
		bool Insert(const void* a_object, std::uint32_t a_tag) noexcept
		{
			const auto parity = a_tag & 1;
			if (m_presence && m_presenceTag[parity].load(std::memory_order_acquire) == a_tag) {
				const auto bit = PresenceBit(a_object);
				auto&      bits = m_presence[parity * kPresenceWords + bit / 64];
				const auto mask = std::uint64_t{ 1 } << (bit % 64);
				if (!(bits.load(std::memory_order_relaxed) & mask)) {
					bits.fetch_or(mask, std::memory_order_acq_rel);
				}
			} else if (m_presenceLate[parity].load(std::memory_order_relaxed) != a_tag) {
				m_presenceLate[parity].store(a_tag, std::memory_order_relaxed);  // (inserted before BeginTag: the bits can't answer for this tag)
			}
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
			const auto parity = a_tag & 1;
			if (m_presence && m_presenceTag[parity].load(std::memory_order_acquire) == a_tag && m_presenceLate[parity].load(std::memory_order_relaxed) != a_tag) {
				const auto bit = PresenceBit(a_object);
				if (!(m_presence[parity * kPresenceWords + bit / 64].load(std::memory_order_acquire) & (std::uint64_t{ 1 } << (bit % 64)))) {
					return false;
				}
			}
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
		static constexpr std::size_t kPresenceBits = 15;  // 32k bits a tag: ~3% of absent lookups reach the table at 1,000 entries
		static constexpr std::size_t kPresenceWords = (std::size_t{ 1 } << kPresenceBits) / 64;

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

		// (bits 33-47 of the same product: the table's index takes the top 16)
		static std::size_t PresenceBit(const void* a_object) noexcept
		{
			return static_cast<std::size_t>(((reinterpret_cast<std::uintptr_t>(a_object) >> 4) * 0x9E3779B97F4A7C15ull) >> (64 - kBits - kPresenceBits)) &
			       ((std::size_t{ 1 } << kPresenceBits) - 1);
		}

		std::unique_ptr<std::atomic<std::uint64_t>[]> m_slots;
		std::unique_ptr<std::atomic<std::uint64_t>[]> m_presence;  // two tag parities of kPresenceWords
		std::atomic<std::uint32_t>                    m_presenceTag[2]{};   // the tag each parity's bits hold
		std::atomic<std::uint32_t>                    m_presenceLate[2]{};  // a tag inserted before its BeginTag
	};
}
