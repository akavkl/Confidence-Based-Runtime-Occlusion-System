#pragma once

// The engine's previs fast path (OG 1.10.163; PREVIS-FEED-PLAN.md §2, FO4-ENGINE-NOTES §5.5b). With previs active,
// DrawWorld's scene walk (1138818) skips the cell expansion and files the previs main list through the feed it calls
// at walk+0x16A (997287: `Feed(group 2, group 1)`), and the sun's cascade cull (1390075) files the previs sun list
// through the feed at +0x1B4 (1142692: `Feed(stack group)`), with previs suspended around it. CBRO wraps both call
// sites: in frames CBRO owns, its own feed functions run instead; in every other frame the previous target runs
// (the engine's feed, or another plugin's such as Addictol's). Also here: read-only accessors for the engine state the
// policy (Core/Feed) decides by, the flush-free suspend switch, and EnumerateCandidates, the replica of what the
// previs-off walk would file (the input a CBRO-owned feed judges). Nothing here decides anything.

namespace CBRO::Hooks::PrevisFeed
{
	enum class Owner : std::uint8_t
	{
		kPrevis,  // the previous targets run (the engine's lists)
		kCBRO,    // CBRO's feed functions run
	};
	using MainFeedFn = void (*)(void* a_group2, void* a_group1);
	using SunFeedFn = void (*)(void* a_stackGroup);

	// Wraps both sites (pass-through until an owner and feed functions are set). False, and Available() false, if a
	// site isn't a `call rel32` (another plugin patched it): feed mode is then off for the session.
	bool Install();
	[[nodiscard]] bool Available() noexcept;
	void SetOwner(Owner a_owner) noexcept;  // main thread, at the cull begin (both thunks run on the main thread)
	void SetMainFeed(MainFeedFn a_fn) noexcept;
	void SetSunFeed(SunFeedFn a_fn) noexcept;

	struct FeedCalls
	{
		std::uint64_t mainPrevious{ 0 };  // main-site calls passed to the previous target
		std::uint64_t mainCBRO{ 0 };      // ... served by CBRO's feed
		std::uint64_t sunPrevious{ 0 };
		std::uint64_t sunCBRO{ 0 };
	};
	[[nodiscard]] FeedCalls TakeFeedCalls() noexcept;

	// The engine's previs state and the gates the fast path depends on (read under SEH; `readable` false = unknown).
	struct Gates
	{
		bool          readable{ false };
		bool          enabled{ false };       // BSPreCulledObjects enabled byte (493183)
		bool          ini{ false };           // bUsePreCulledObjects (1472203)
		bool          suspended{ false };     // suspended byte (718924)
		bool          active{ false };        // enabled && ini && !suspended: what IsActive() returns
		std::uint32_t gateA{ 0 };             // u32 scene root +0x14C: non-zero = the walk expands the cells even with previs active
		std::uint16_t gateB{ 0 };             // u16 scene root +0x180: non-zero = DrawWorld's root loop runs even with previs active
		bool          exterior{ false };      // the culler's exterior byte ([[culler+0x150]+0x138])
		bool          overrideRoot{ false };  // the override-root global (127974) is set
	};
	[[nodiscard]] Gates ReadGates() noexcept;

	// The engine's BSPreCulledObjects::SetSuspended(a_suspend, flush = false): a byte write, never a flush (the flush
	// is what breaks previs until the cells reload; see PLAN.md §7 "v1.28 runs"). Main thread.
	void SetSuspended(bool a_suspend) noexcept;

	// The suspension windows (FO4-ENGINE-NOTES 5.5c): in a CBRO frame previs is suspended only from Render_PreUI+0x157's
	// return to the cull stage's end (the walk, DrawWorld's cull and its jobs: the engine's previs-off flow, helper
	// included) and around the cascade cull (1108521+0xE49: the group-0 path). Everything else in the frame (the
	// previs query, the lamps, the third view, cell loads, the game's update) sees previs active, as vanilla. Both
	// wrappers are pass-through when the frame isn't a CBRO frame. Main thread.
	[[nodiscard]] bool WindowsAvailable() noexcept;  // both sites wrapped
	void SetWindow(bool a_cbroFrame) noexcept;       // at the cull begin: this frame's cascade window and the next frame's pre-cull window
	[[nodiscard]] bool Held() noexcept;              // the cull window is open (the pre-cull wrapper or HoldNow opened it)
	void HoldNow() noexcept;                         // at the cull begin, when the pre-cull wrapper didn't open it (a switch frame): opens it and runs the engine's pre-cull helper
	void ReleaseWindow() noexcept;                   // at the cull stage's end: closes it (the engine's own byte restored)
	void ClearSuspension() noexcept;                 // a suspension CBRO set with SetSuspended outside the windows ends (now, or at the open window's release)
	// The lamp window (v1.52): a spot light's group-0 cull in the deferred-lights stage (BSShadowFrustumLight slot 9,
	// Core/ShadowLights), previs suspended around it in a CBRO frame. With previs active the culling-group pass (1147875)
	// skips the frustum test of every block whose +0x3A6F byte is clear, and group 0's blocks have it clear (DrawWorld's
	// cull filed them inside the cull window); the register loop (962984) then files whatever the last pass left in the
	// result bytes (the last sun cascade's, or the main view's), not the spot's frustum (FO4-ENGINE-NOTES 6.2a). True =
	// previs was suspended now: pass it to EndLampWindow. Main thread.
	[[nodiscard]] bool BeginLampWindow() noexcept;
	void EndLampWindow(bool a_opened) noexcept;
	[[nodiscard]] bool ActiveNow() noexcept;  // what the engine's IsActive() returns now (enabled, INI, not suspended)
	struct WindowCounts
	{
		std::uint32_t preCull{ 0 };    // cull windows opened by the pre-cull wrapper
		std::uint32_t cullBegin{ 0 };  // ... opened at the cull begin instead (switch frames, or no pre-cull wrapper)
		std::uint32_t cascade{ 0 };    // cascade windows
		std::uint32_t lamp{ 0 };       // lamp windows (spot-light group-0 culls)
	};
	[[nodiscard]] WindowCounts TakeWindowCounts() noexcept;
	// For CullGroups' sun-path check: the cascade site calls CBRO's wrapper, which chains to this previous target.
	[[nodiscard]] std::uintptr_t CascadeCullThunkAddress() noexcept;
	[[nodiscard]] std::uintptr_t CascadeCullPrevious() noexcept;

	// A culling group's six frustum planes (NiPlane {n, d}, 16 bytes each, at group+0). False if unreadable.
	[[nodiscard]] bool GroupPlanes(const void* a_group, float a_planes[6][4]) noexcept;

	// Where the previs-off walk would have filed a candidate.
	enum class Route : std::uint8_t
	{
		kGroup0,  // shadow casters (the sun's cascades read it with previs off)
		kGroup1,  // non-casters (main view only)
		kArray,   // a group-array slot (a root filed whole; main view only)
	};
	// Which rule of the walk (PREVIS-FEED-PLAN.md §2.6) produced it, for the audit.
	enum class Site : std::uint8_t
	{
		kR1,       // scene root child 2's subtree (five fixed adds)
		kR2,       // the world node's child 0
		kR3b,      // a cell child that isn't a node, whole
		kR3c,      // a cell's node 3: an exact-NiNode container's children
		kR3d,      // a cell's node 3 or 9: any other grandchild, whole
		kR3e,      // a cell's node 2: its children
		kR4Whole,  // DrawWorld's root loop: a root filed whole
		kR4Child,  // DrawWorld's root loop: a root's children
		kCount
	};
	using Visitor = void (*)(void* a_context, RE::NiAVObject* a_object, Route a_route, Site a_site);

	// Replica of the previs-off scene walk's exterior path (rules R1-R3) plus DrawWorld's root loop (R4), in the
	// engine's order: every object the engine would offer to Group::Add, with its route. Enumeration only, nothing is
	// filed and no engine flag is written. False when the walk isn't replicable (interior or override-root path, no
	// world node, a faulting read): the caller falls back to the engine's own walk. Main thread.
	[[nodiscard]] bool EnumerateCandidates(Visitor a_visit, void* a_context) noexcept;

	// The replicated site a main-pass Group::Add return address belongs to, if any (for the audit).
	[[nodiscard]] bool SiteOf(std::uintptr_t a_returnAddress, Site& a_out) noexcept;
	[[nodiscard]] std::string_view SiteName(Site a_site) noexcept;
	[[nodiscard]] std::string_view RouteName(Route a_route) noexcept;
}
