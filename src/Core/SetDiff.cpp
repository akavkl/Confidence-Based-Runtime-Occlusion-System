#include "Core/SetDiff.h"

#include "Hooks/CullGroups.h"
#include "Util/Gamebryo.h"

#include <atomic>
#include <cmath>
#include <cstring>

namespace CBRO::Core::SetDiff
{
	namespace
	{
		constexpr std::size_t kRing = 65536;      // registrations kept per frame (the main view registers ~1-6k)
		constexpr std::size_t kSetBits = 17;      // 131,072 slots per set
		constexpr std::size_t kSetSlots = std::size_t{ 1 } << kSetBits;
		constexpr std::size_t kTypeSlots = 24;
		constexpr std::size_t kSamples = 10;
		constexpr std::uint32_t kMinOtherFrames = 30;  // the other mode's set must hold this many settled frames before a miss counts

		bool g_enabled{ false };

		// ---- the per-frame ring (any thread) ---------------------------------------------------------------------
		std::atomic<std::uint32_t> g_count{ 0 };
		RE::NiAVObject*            g_ring[kRing]{};

		// ---- open-addressing pointer sets (main thread) ----------------------------------------------------------
		struct PointerSet
		{
			std::uintptr_t* slots{ nullptr };
			std::size_t     count{ 0 };

			void Ensure()
			{
				if (!slots) {
					slots = new std::uintptr_t[kSetSlots]();
				}
			}

			static std::size_t Hash(std::uintptr_t a_key) noexcept
			{
				return static_cast<std::size_t>((a_key * 0x9E3779B97F4A7C15ull) >> (64 - kSetBits));
			}

			bool Contains(std::uintptr_t a_key) const noexcept
			{
				if (!slots) {
					return false;
				}
				for (std::size_t i = Hash(a_key), n = 0; n < kSetSlots; i = (i + 1) & (kSetSlots - 1), ++n) {
					if (slots[i] == a_key) {
						return true;
					}
					if (slots[i] == 0) {
						return false;
					}
				}
				return false;
			}

			// True if newly inserted; false if present or the set is full (a full set counts nothing more).
			bool Insert(std::uintptr_t a_key)
			{
				Ensure();
				if (count >= kSetSlots / 2) {
					return false;
				}
				for (std::size_t i = Hash(a_key);; i = (i + 1) & (kSetSlots - 1)) {
					if (slots[i] == a_key) {
						return false;
					}
					if (slots[i] == 0) {
						slots[i] = a_key;
						++count;
						return true;
					}
				}
			}

			void Clear() noexcept
			{
				if (slots && count) {
					std::memset(slots, 0, kSetSlots * sizeof(std::uintptr_t));
				}
				count = 0;
			}
		};

		// ---- what an object looked like when it was registered -------------------------------------------------
		struct Info
		{
			float         radius{ 0.0f };
			float         distance{ 0.0f };
			std::uint64_t flags{ 0 };
			char          type[48]{};
			char          name[64]{};
		};

		bool DescribeImpl(const RE::NiAVObject* a_object, const RE::NiPoint3& a_eye, Info& a_out) noexcept
		{
			__try {
				const auto& bound = a_object->worldBound;
				a_out.radius = bound.fRadius;
				const float dx = bound.center.x - a_eye.x, dy = bound.center.y - a_eye.y, dz = bound.center.z - a_eye.z;
				a_out.distance = std::sqrt(dx * dx + dy * dy + dz * dz);
				a_out.flags = a_object->GetFlags();
				Util::TryGetRTTIName(a_object, a_out.type, sizeof(a_out.type));
				Util::TryGetObjectName(a_object, a_out.name, sizeof(a_out.name));
				return std::isfinite(a_out.radius) && std::isfinite(a_out.distance);
			} __except (EXCEPTION_EXECUTE_HANDLER) {
				return false;
			}
		}

		// ---- one direction's histogram ---------------------------------------------------------------------------
		struct Histogram
		{
			std::uint32_t total{ 0 };
			std::uint32_t unreadable{ 0 };
			std::uint32_t alwaysDraw{ 0 };
			std::uint32_t previsHidden{ 0 };   // flags bit 26 (pre-processed) with bit 39 (not visible): previs' own mark
			std::uint32_t previsShown{ 0 };    // bit 26 without bit 39
			std::uint32_t size[4]{};           // radius / distance: < 0.002, < 0.01, < 0.05, >= 0.05
			std::uint32_t distance[4]{};       // < 500, < 2000, < 8000, >= 8000
			char          typeNames[kTypeSlots][48]{};
			std::uint32_t typeCounts[kTypeSlots]{};
			std::uint32_t typeOther{ 0 };
			char          samples[kSamples][160]{};
			std::uint32_t sampleCount{ 0 };

			void Add(const Info& a_info, bool a_readable)
			{
				++total;
				if (!a_readable) {
					++unreadable;
					return;
				}
				if ((a_info.flags >> 11) & 1) {
					++alwaysDraw;
				}
				if ((a_info.flags >> 26) & 1) {
					++(((a_info.flags >> 39) & 1) ? previsHidden : previsShown);
				}
				const float ratio = a_info.distance > 1.0f ? a_info.radius / a_info.distance : 1.0f;
				++size[ratio < 0.002f ? 0 : ratio < 0.01f ? 1 : ratio < 0.05f ? 2 : 3];
				++distance[a_info.distance < 500.0f ? 0 : a_info.distance < 2000.0f ? 1 : a_info.distance < 8000.0f ? 2 : 3];
				bool typed = false;
				for (std::size_t i = 0; i < kTypeSlots; ++i) {
					if (typeNames[i][0] == 0) {
						std::snprintf(typeNames[i], sizeof(typeNames[i]), "%s", a_info.type[0] ? a_info.type : "(unknown)");
						typeCounts[i] = 1;
						typed = true;
						break;
					}
					if (std::strncmp(typeNames[i], a_info.type[0] ? a_info.type : "(unknown)", sizeof(typeNames[i]) - 1) == 0) {
						++typeCounts[i];
						typed = true;
						break;
					}
				}
				if (!typed) {
					++typeOther;
				}
				if (sampleCount < kSamples) {
					std::snprintf(
						samples[sampleCount], sizeof(samples[sampleCount]), "%s \"%s\" r=%.0f d=%.0f%s%s",
						a_info.type[0] ? a_info.type : "?", a_info.name[0] ? a_info.name : "", a_info.radius, a_info.distance,
						((a_info.flags >> 11) & 1) ? " always-draw" : "", ((a_info.flags >> 26) & 1) ? (((a_info.flags >> 39) & 1) ? " previs:hidden" : " previs:shown") : "");
					++sampleCount;
				}
			}

			std::string Describe() const
			{
				std::string text = std::format(
					"{} (unreadable {}, always-draw {}, previs-marked hidden {} / shown {}) | by size (radius/distance): <0.002 {}, <0.01 {}, <0.05 {}, larger {} | by distance: <500 {}, <2000 {}, <8000 {}, farther {} | by type:",
					total, unreadable, alwaysDraw, previsHidden, previsShown, size[0], size[1], size[2], size[3], distance[0], distance[1], distance[2], distance[3]);
				// Types by count, descending (a small table: selection sort in place of a copy).
				std::size_t order[kTypeSlots];
				std::size_t n = 0;
				for (std::size_t i = 0; i < kTypeSlots; ++i) {
					if (typeNames[i][0]) {
						order[n++] = i;
					}
				}
				for (std::size_t a = 0; a < n; ++a) {
					for (std::size_t b = a + 1; b < n; ++b) {
						if (typeCounts[order[b]] > typeCounts[order[a]]) {
							std::swap(order[a], order[b]);
						}
					}
				}
				for (std::size_t k = 0; k < n; ++k) {
					text += std::format(" {} {}{}", typeNames[order[k]], typeCounts[order[k]], k + 1 < n ? "," : "");
				}
				if (typeOther) {
					text += std::format(", other {}", typeOther);
				}
				return text;
			}

			std::string Samples() const
			{
				std::string text;
				for (std::uint32_t i = 0; i < sampleCount; ++i) {
					text += std::format("{}{}", i ? " | " : "", samples[i]);
				}
				return text;
			}
		};

		struct Side
		{
			PointerSet    set;       // every object this mode registered in a settled frame at this location
			PointerSet    counted;   // objects already counted as this mode's exclusive
			Histogram     onlyHere;  // registered by this mode while the other mode's set (populated) lacks them
			std::uint32_t frames{ 0 };
			std::uint64_t registrations{ 0 };
		};
		Side g_side[2];  // 0 previs, 1 CBRO
		std::uint32_t g_unsettled{ 0 };
		std::uint32_t g_overflow{ 0 };
	}

	void Install(bool a_enabled)
	{
		g_enabled = a_enabled;
		Hooks::CullGroups::SetMainRegistrationObserver(a_enabled ? &Observe : nullptr);
		logger::info("set diff: {}", a_enabled ? "on (bSetDiff=1): main-view registrations compared between the modes at each A/B location" : "off");
	}

	void Observe(RE::NiAVObject* a_object) noexcept
	{
		const auto index = g_count.fetch_add(1, std::memory_order_relaxed);
		if (index < kRing) {
			g_ring[index] = a_object;
		}
	}

	void EndCull(int a_mode, bool a_settled, const RE::NiPoint3& a_eye)
	{
		const auto count = std::min<std::size_t>(g_count.exchange(0, std::memory_order_acq_rel), kRing);
		if (!g_enabled) {
			return;
		}
		if (count == kRing) {
			++g_overflow;
		}
		if (!a_settled || count == 0) {
			++g_unsettled;
			return;
		}
		auto& mine = g_side[a_mode == 0 ? 0 : 1];
		auto& other = g_side[a_mode == 0 ? 1 : 0];
		++mine.frames;
		mine.registrations += count;
		const bool compare = other.frames >= kMinOtherFrames;
		for (std::size_t i = 0; i < count; ++i) {
			auto* object = g_ring[i];
			if (!object) {
				continue;
			}
			const auto key = reinterpret_cast<std::uintptr_t>(object);
			mine.set.Insert(key);
			if (compare && !other.set.Contains(key) && mine.counted.Insert(key)) {
				Info       info{};
				const bool readable = DescribeImpl(object, a_eye, info);
				mine.onlyHere.Add(info, readable);
			}
		}
	}

	void Reset()
	{
		for (auto& side : g_side) {
			side.set.Clear();
			side.counted.Clear();
			side.onlyHere = {};
			side.frames = 0;
			side.registrations = 0;
		}
		g_unsettled = 0;
		g_overflow = 0;
	}

	void LogStats()
	{
		if (!g_enabled) {
			return;
		}
		const auto& previs = g_side[0];
		const auto& cbro = g_side[1];
		if (previs.frames == 0 && cbro.frames == 0) {
			return;
		}
		logger::info(
			"set diff at this location (main view, settled frames): previs {} objects over {} frames ({:.0f} registrations/frame) | CBRO {} objects over {} frames ({:.0f}/frame) | unsettled frames {} | ring overflow {}",
			previs.set.count, previs.frames, previs.frames ? static_cast<double>(previs.registrations) / previs.frames : 0.0,
			cbro.set.count, cbro.frames, cbro.frames ? static_cast<double>(cbro.registrations) / cbro.frames : 0.0, g_unsettled, g_overflow);
		if (cbro.onlyHere.total || previs.frames >= kMinOtherFrames) {
			logger::info("set diff: drawn by CBRO, never by previs here: {}", cbro.onlyHere.Describe());
			if (cbro.onlyHere.sampleCount) {
				logger::info("set diff: CBRO-only samples: {}", cbro.onlyHere.Samples());
			}
		}
		if (previs.onlyHere.total || cbro.frames >= kMinOtherFrames) {
			logger::info("set diff: drawn by previs, never by CBRO here: {}", previs.onlyHere.Describe());
			if (previs.onlyHere.sampleCount) {
				logger::info("set diff: previs-only samples: {}", previs.onlyHere.Samples());
			}
		}
	}
}
