#include "Core/Feed.h"

#include "Hooks/CullGroups.h"
#include "Hooks/Precipitation.h"
#include "Hooks/PrevisFeed.h"
#include "Settings.h"
#include "Util/Gamebryo.h"

#include <intrin.h>

namespace CBRO::Core::Feed
{
	namespace
	{
		using Hooks::PrevisFeed::Gates;
		using Hooks::PrevisFeed::Route;
		using Hooks::PrevisFeed::Site;

		constexpr std::uint64_t kGroup0ID = 1117782;
		constexpr std::uint64_t kGroup1ID = 133326;
		constexpr std::uint64_t kGroupArrayID = 459440;
		constexpr std::size_t   kGroupSize = 0x170;
		constexpr std::size_t   kGroupArrayCountOffset = 0x10;

		bool          g_enabled{ false };   // bPrevisFeed
		bool          g_legacyScene{ false };  // an interior with bInteriorLegacyPrevis: v1.28's switch (Runtime disables and flushes previs) instead of the windows
		std::uint32_t g_auditInterval{ 0 };
		Path          g_path{ Path::kPrevis };
		bool          g_cbroFrame{ false };    // the last cull begin was a CBRO frame (previs suspended inside the windows)
		bool          g_windowLogged{ false };
		std::uintptr_t g_group0{ 0 };
		std::uintptr_t g_group1{ 0 };
		std::uintptr_t g_groupArray{ 0 };

		// ---- per-interval statistics (main thread) ---------------------------------------------------------------
		struct Stats
		{
			std::uint32_t previs{ 0 };
			std::uint32_t classicSuspended{ 0 };   // classic frames with previs suspended by CBRO inside the windows
			std::uint32_t classicInactive{ 0 };    // classic frames with previs already inactive (INI off, tpc, workshop)
			std::uint32_t classicLegacy{ 0 };      // classic frames with bPrevisFeed=0 (Runtime's switch)
			std::uint32_t feed{ 0 };
			std::uint32_t switchFrames{ 0 };       // frames whose cull window was still the previous mode's (a switch)
			std::uint32_t gateA{ 0 };
			std::uint32_t gateB{ 0 };
			std::uint32_t interior{ 0 };           // not the exterior path, or an override root
			std::uint32_t unreadable{ 0 };
			std::uint32_t audits{ 0 };
			std::uint32_t auditsNotReplicable{ 0 };
			std::uint64_t auditMissing{ 0 };
			std::uint64_t auditMismatch{ 0 };
			std::uint64_t auditExtra{ 0 };
		};
		Stats g_stats;

		// ---- the audit (plan §4.5, first part) -----------------------------------------------------------------------
		struct Candidate
		{
			Route        route;
			Site         site;
			bool         seen{ false };
		};
		struct Offered
		{
			const void*    group;
			RE::NiAVObject* object;
			std::uintptr_t returnAddress;
		};
		std::unordered_map<const void*, Candidate> g_replica;  // set A: what the replica enumerates
		std::vector<Offered>                       g_offered;  // set E: what the engine offered to Group::Add
		std::atomic<bool>                          g_auditing{ false };
		std::atomic<bool>                          g_offeredLock{ false };
		std::uint32_t                              g_auditClock{ 0 };

		void LockOffered() noexcept
		{
			while (g_offeredLock.exchange(true, std::memory_order_acquire)) {
				_mm_pause();
			}
		}

		void UnlockOffered() noexcept
		{
			g_offeredLock.store(false, std::memory_order_release);
		}

		void ReplicaVisitor(void* a_context, RE::NiAVObject* a_object, Route a_route, Site a_site)
		{
			auto& replica = *static_cast<std::unordered_map<const void*, Candidate>*>(a_context);
			replica.try_emplace(a_object, Candidate{ a_route, a_site });
		}

		// The group observer (Hooks/CullGroups) runs for every Group::Add, before CBRO's filters: only during an audit
		// frame, only for DrawWorld's own groups.
		void Observer(void* a_group, RE::NiAVObject* a_object, std::uintptr_t a_owner, std::uintptr_t a_returnAddress)
		{
			if (!g_auditing.load(std::memory_order_relaxed) || a_owner == 0 || a_owner != Hooks::CullGroups::MainOwner()) {
				return;
			}
			LockOffered();
			if (g_offered.size() < 65536) {
				g_offered.push_back({ a_group, a_object, a_returnAddress });
			}
			UnlockOffered();
		}

		bool RouteOfGroup(const void* a_group, Route& a_out) noexcept
		{
			const auto group = reinterpret_cast<std::uintptr_t>(a_group);
			if (group == g_group0) {
				a_out = Route::kGroup0;
				return true;
			}
			if (group == g_group1) {
				a_out = Route::kGroup1;
				return true;
			}
			if (g_groupArray) {
				const auto data = *reinterpret_cast<const std::uintptr_t*>(g_groupArray);
				const auto count = *reinterpret_cast<const std::uint32_t*>(g_groupArray + kGroupArrayCountOffset);
				if (data && group >= data && group < data + static_cast<std::uintptr_t>(count) * kGroupSize) {
					a_out = Route::kArray;
					return true;
				}
			}
			return false;
		}

		RE::NiAVObject* ParentOf(RE::NiAVObject* a_object) noexcept
		{
			__try {
				return a_object->parent;
			} __except (EXCEPTION_EXECUTE_HANDLER) {
				return nullptr;
			}
		}

		std::string Describe(RE::NiAVObject* a_object)
		{
			char type[48]{};
			char name[48]{};
			char parent[48]{};
			Util::TryGetRTTIName(a_object, type, sizeof(type));
			Util::TryGetObjectName(a_object, name, sizeof(name));
			if (const auto parentObject = ParentOf(a_object)) {
				Util::TryGetObjectName(parentObject, parent, sizeof(parent));
			}
			return std::format("{} '{}' (parent '{}')", type[0] ? type : "?", name, parent);
		}

		void StartAudit(std::uint32_t a_clock)
		{
			g_replica.clear();
			if (!Hooks::PrevisFeed::EnumerateCandidates(&ReplicaVisitor, &g_replica)) {
				++g_stats.auditsNotReplicable;
				return;
			}
			LockOffered();
			g_offered.clear();
			UnlockOffered();
			g_auditClock = a_clock;
			g_auditing.store(true, std::memory_order_release);
			Hooks::CullGroups::SetGroupObserver(&Observer);
		}

		void FinishAudit()
		{
			Hooks::CullGroups::SetGroupObserver(nullptr);
			g_auditing.store(false, std::memory_order_release);
			++g_stats.audits;

			std::uint32_t                              fromSites = 0, other = 0, missing = 0, mismatch = 0;
			std::array<std::uint32_t, static_cast<std::size_t>(Site::kCount)> extraBySite{};
			std::vector<std::string>                   samples;
			LockOffered();
			for (const auto& offered : g_offered) {
				Site site{};
				if (!Hooks::PrevisFeed::SiteOf(offered.returnAddress, site)) {
					++other;  // an add from a site the replica doesn't cover (the walk's fixed root adds, the feeds, ...)
					continue;
				}
				++fromSites;
				const auto found = g_replica.find(offered.object);
				if (found == g_replica.end()) {
					++missing;
					if (samples.size() < 10) {
						samples.push_back(std::format("MISSING {} at {}", Describe(offered.object), Hooks::PrevisFeed::SiteName(site)));
					}
					continue;
				}
				found->second.seen = true;
				Route engineRoute{};
				if (RouteOfGroup(offered.group, engineRoute) && engineRoute != found->second.route) {
					++mismatch;
					if (samples.size() < 10) {
						samples.push_back(std::format(
							"ROUTE {} engine {} replica {} at {}", Describe(offered.object), Hooks::PrevisFeed::RouteName(engineRoute),
							Hooks::PrevisFeed::RouteName(found->second.route), Hooks::PrevisFeed::SiteName(site)));
					}
				}
			}
			UnlockOffered();
			std::uint32_t extra = 0;
			for (const auto& [object, candidate] : g_replica) {
				if (!candidate.seen) {
					++extra;
					++extraBySite[static_cast<std::size_t>(candidate.site)];
				}
			}
			g_stats.auditMissing += missing;
			g_stats.auditMismatch += mismatch;
			g_stats.auditExtra += extra;

			std::string extraText;
			for (std::size_t i = 0; i < extraBySite.size(); ++i) {
				if (extraBySite[i]) {
					extraText += std::format(" {} {}", Hooks::PrevisFeed::SiteName(static_cast<Site>(i)), extraBySite[i]);
				}
			}
			logger::info(
				"feed audit (frame {}): engine adds from the replicated sites {} (other sites {}) | replica {} | missing {} | route mismatch {} | extra {} (replica only; includes the entries of nodes CBRO pruned whole):{}",
				g_auditClock, fromSites, other, g_replica.size(), missing, mismatch, extra, extraText.empty() ? " none" : extraText);
			for (const auto& sample : samples) {
				logger::info("  feed audit sample: {}", sample);
			}
			g_replica.clear();
		}

		const char* Reason(const Gates& a_gates) noexcept
		{
			if (!a_gates.readable) {
				return "engine state unreadable";
			}
			if (!a_gates.enabled) {
				return "previs disabled (console tpc?)";
			}
			if (!a_gates.ini) {
				return "bUsePreCulledObjects=0";
			}
			return "previs suspended by the engine";
		}
	}

	void Install()
	{
		const auto& settings = Settings::Get();
		g_enabled = settings.previsFeed;
		g_auditInterval = settings.feedAuditInterval;
		g_group0 = CBRO::Engine::OG(kGroup0ID).address();
		g_group1 = CBRO::Engine::OG(kGroup1ID).address();
		g_groupArray = CBRO::Engine::OG(kGroupArrayID).address();
		g_offered.reserve(16384);
		logger::info(
			"previs feed: {}; audit every {} frames",
			g_enabled ? "previs is never switched off: in CBRO frames it is suspended (without flush) only inside the cull and cascade windows" : "off (bPrevisFeed=0: v1.28 previs switching)",
			g_auditInterval);
		// The query skip needs both windows (the walk and the cascades must not read previs's lists) and the rain pass's
		// detour (its window), else the query keeps running.
		if (g_enabled && settings.skipPrevisQuery) {
			if (Hooks::PrevisFeed::WindowsAvailable() && Hooks::Precipitation::Hooked()) {
				Hooks::PrevisFeed::InstallQuerySkip();
			} else {
				logger::warn("previs feed: previs query skip not installed (a suspension window or the rain pass hook is missing); the query keeps running");
			}
		}
	}

	void OnGameLoaded()
	{
		// The engine's bytes survive a load; whether CBRO still holds the suspension is re-read at the next cull begin.
	}

	Path BeginFrame(bool a_cbroMode, std::uint32_t a_clock)
	{
		if (!g_enabled || g_legacyScene) {
			// v1.28's way (bPrevisFeed=0, or an interior with bInteriorLegacyPrevis): Runtime switched previs off (and the
			// engine flushed it) when CBRO went on, so the walk is previs-off without any window.
			Hooks::PrevisFeed::SetWindow(false);
			Hooks::PrevisFeed::ReleaseWindow();
			g_cbroFrame = false;
			g_path = a_cbroMode ? Path::kClassic : Path::kPrevis;
			++(a_cbroMode ? g_stats.classicLegacy : g_stats.previs);
			Hooks::PrevisFeed::SetOwner(Hooks::PrevisFeed::Owner::kPrevis);
			return g_path;
		}

		const auto gates = Hooks::PrevisFeed::ReadGates();
		if (!gates.readable) {
			++g_stats.unreadable;
		}
		if (gates.gateA) {
			++g_stats.gateA;
		}
		if (gates.gateB) {
			++g_stats.gateB;
		}
		if (gates.readable && (!gates.exterior || gates.overrideRoot)) {
			++g_stats.interior;
		}
		Hooks::PrevisFeed::SetOwner(Hooks::PrevisFeed::Owner::kPrevis);  // (feed mode: plan phase 3)

		if (!a_cbroMode) {
			// CBRO off: nothing of previs is touched from the next pre-cull on. A cull window the pre-cull wrapper already
			// opened this frame (a switch frame) stays open until the cull's end: this one frame's walk is previs-off with
			// CBRO's hooks out (frustum-only, nothing hidden), rather than releasing mid-frame with the engine's pre-cull
			// helper already run.
			Hooks::PrevisFeed::SetWindow(false);
			if (Hooks::PrevisFeed::Held()) {
				++g_stats.switchFrames;
			}
			if (g_cbroFrame) {
				g_cbroFrame = false;
				logger::info("previs feed: CBRO off: previs runs untouched from the next frame on (it was never switched off or flushed; full-strength previs)");
			}
			g_path = Path::kPrevis;
			++g_stats.previs;
			return g_path;
		}

		// CBRO on: the classic path. Previs is suspended (never switched off) only inside the two windows: from
		// Render_PreUI+0x157 to the cull stage's end, and around the cascade cull. The pre-cull wrapper opened this
		// frame's window if the previous frame was a CBRO frame; on a switch frame (or without the wrapper) it opens here.
		if (gates.readable && gates.enabled && gates.ini) {
			Hooks::PrevisFeed::SetWindow(true);
			if (!Hooks::PrevisFeed::Held()) {
				Hooks::PrevisFeed::HoldNow();
				if (g_cbroFrame) {
					++g_stats.switchFrames;  // (held state lost mid-run: counted so it shows)
				}
			}
			if (!g_windowLogged) {
				g_windowLogged = true;
				logger::info(
					"previs feed: CBRO on: previs suspended without flush inside the cull window (Render_PreUI+0x157 to the cull's end) and the cascade window (1108521+0xE49) of every CBRO frame; active for the rest of the frame{}",
					Hooks::PrevisFeed::WindowsAvailable() ? "" : " (a window wrapper is missing: see the install lines)");
			}
			g_cbroFrame = true;
			++g_stats.classicSuspended;
		} else {
			// Already inactive for another reason (INI, tpc, the engine's own suspension) or unreadable: no window needed.
			Hooks::PrevisFeed::SetWindow(false);
			g_cbroFrame = false;
			++g_stats.classicInactive;
		}
		g_path = Path::kClassic;

		if (g_auditInterval && a_clock % g_auditInterval == 0 && Hooks::PrevisFeed::Available()) {
			StartAudit(a_clock);
		}
		return g_path;
	}

	void EndCull()
	{
		Hooks::PrevisFeed::ReleaseWindow();  // the cull window closes: previs active again for the rest of the frame
		if (g_auditing.load(std::memory_order_acquire)) {
			FinishAudit();
		}
	}

	Path Current() noexcept
	{
		return g_path;
	}

	std::string_view PathName(Path a_path) noexcept
	{
		switch (a_path) {
		case Path::kPrevis:
			return "previs"sv;
		case Path::kClassic:
			return "CBRO classic"sv;
		case Path::kFeed:
			return "CBRO feed"sv;
		default:
			return "?"sv;
		}
	}

	std::string_view PrevisState() noexcept
	{
		const auto gates = Hooks::PrevisFeed::ReadGates();
		if (!gates.readable) {
			return "unknown"sv;
		}
		if (gates.active) {
			return g_cbroFrame ? "ACTIVE outside CBRO's cull windows"sv : "ACTIVE"sv;
		}
		if (gates.suspended && Hooks::PrevisFeed::Held()) {
			return "suspended by CBRO (cull window open)"sv;
		}
		return gates.enabled && gates.ini ? "suspended by the engine"sv : "OFF"sv;
	}

	bool ManagesPrevis() noexcept
	{
		return g_enabled && !g_legacyScene;
	}

	void SetLegacyScene(bool a_legacy) noexcept
	{
		if (g_legacyScene != a_legacy) {
			g_legacyScene = a_legacy;
			logger::info("previs feed: {}", a_legacy ? "interior with bInteriorLegacyPrevis=1: v1.28's previs switch (off and flushed while CBRO is on), no windows" : "back to the windows (previs never switched off)");
		}
	}

	void LogStats(std::uint32_t a_frames)
	{
		(void)a_frames;
		const auto calls = Hooks::PrevisFeed::TakeFeedCalls();
		const auto windows = Hooks::PrevisFeed::TakeWindowCounts();
		const auto gates = Hooks::PrevisFeed::ReadGates();
		logger::info(
			"previs feed paths this interval: previs {} | CBRO classic {} (previs suspended inside CBRO's windows {}, already inactive {}, legacy switch {}) | CBRO feed {} | windows: cull {} (opened at the cull begin {}), cascade {}, lamp {}, switch frames {} || gates: A non-zero {} frames, B non-zero {}, interior/override {}, unreadable {} | now: enabled {} ini {} suspended {} A {} B {} exterior {} || feed sites: main previous {} / CBRO {}, sun previous {} / CBRO {}",
			g_stats.previs, g_stats.classicSuspended + g_stats.classicInactive + g_stats.classicLegacy, g_stats.classicSuspended, g_stats.classicInactive, g_stats.classicLegacy,
			g_stats.feed, windows.preCull + windows.cullBegin, windows.cullBegin, windows.cascade, windows.lamp, g_stats.switchFrames,
			g_stats.gateA, g_stats.gateB, g_stats.interior, g_stats.unreadable,
			gates.enabled, gates.ini, gates.suspended, gates.gateA, gates.gateB, gates.exterior,
			calls.mainPrevious, calls.mainCBRO, calls.sunPrevious, calls.sunCBRO);
		logger::info(
			"previs query this interval: skipped {} frames (rain windows {}), run on in CBRO frames while it rains {} | collected {} times with previs active, {:.3f} ms per collect on the main thread",
			windows.querySkips, windows.rain, windows.queryRain, windows.queryRuns, windows.queryRuns ? windows.queryMs / windows.queryRuns : 0.0);
		if (g_auditInterval) {
			logger::info(
				"feed audits this interval: {} run, {} not replicable (interior/override) | totals: missing {} | route mismatch {} | extra {}",
				g_stats.audits, g_stats.auditsNotReplicable, g_stats.auditMissing, g_stats.auditMismatch, g_stats.auditExtra);
		}
		g_stats = {};
	}
}
