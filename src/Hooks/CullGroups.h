#pragma once

// Detours on the engine's culling-group entry points (OG 1.10.163, verified in Phase 0 runs 2-3):
//   Group::Add(group, obj, bound, flags)             id 1175493 - DrawWorld's top-level adds (scene walk, previs list)
//   ChildPush(group, obj, bound, flag, b5, b6, ctx)  id 357475  - BSCullingGroup::AddP1: the culling-group pass
//                                                                 adding ONE object (a child of a visible node)
//                                                                 to a group
// The pass itself (1147875) runs inline on the calling thread with bCullingBatch 0 (the London default); BSJobs
// exist only on the batched path (FO4-ENGINE-NOTES §3). 626862 registers a processed group into an accumulator.
//   Block::Add(block, obj, bound, startIndex)        id 1143206 - every entry into a cull block; only ever called
//                                                                 from the two above
// and the main camera's registration: BSShaderAccumulator::RegisterObject (vtable id 357329, slot 45,
// only ever called through the vtable) files a visible geometry into an accumulator's render lists.
// Every view has its own accumulator; DrawWorld's main one is the global id 1211381.
//
// A block carries its group's owner (group+0x158) at +0x3A60; DrawWorld's groups share one owner, so
// blocks can be attributed to DrawWorld's cull ("main pass"). Which group a Block::Add belongs to is
// known from the Group::Add / ChildPush call running on the same thread.
//
// Shared groups: with previs off, DrawWorld group 0 (id 1117782: the shadow casters, NiAVObject flags
// bit 40 clear) is also a view source for the sun's shadow cascades. Every frame, with the directional-
// shadows byte (id 241042) set, the directional light's update (BSShadowDirectionalLight vtable 97945 slot
// 14, id 1242204: direction, shadow camera) runs and then its cascade cull (id 1390075) with group 0, which
// (previs off) registers every cascade as another view of group 0. Which function does it depends on the INI
// setting bCullingBatch:General ("GroupsEnabled", 0x141CCB850 = `movzx eax, byte [id 938585]`, default 0):
//   batched (1):   Render_PreUI+0x175 -> 339369, +0x185 -> 432406 (light from ShadowSceneNode id 1327069)
//   unbatched (0): Render_PreUI+0x1BF -> 1108521, +0xE49 -> 1390075 (light from ShadowSceneNode id 879298)
// Each checks both bytes itself (1108521 skips its block when batched). With the directional-shadows byte
// clear neither runs the cascade cull: nothing but the main camera reads group 0. (259940 is 432406's code
// with no caller; DrawWorld's callback slot, global id 473367, holds a no-op in this build.)
// Order inside Render_PreUI (984743): +0x0 the directional-shadows byte is fixed for the frame (cleared when
// byte 0x146721F58 is set; its only setter, 0x142857980, runs from the engine's one-time render setup),
// +0x16A DrawWorld's cull (CBRO's cull stage), +0x175 the batched setup, +0x17F the pre-pass, +0x1BF the
// unbatched setup. So the bytes CBRO reads when its cull stage begins are the ones the cascades see later that
// frame. CBRO checks the active path's call bytes each frame; nothing on it is patched.
//
// v1.14 put counting stubs (writable counters inside the executable trampoline page, next to the hot
// Block::Add / Group::Add / ChildPush gateways) on that path: frames took 2-3x longer in both modes. Never
// write data into code pages again.
// An entry rejected in, or left out of,
// group 0 therefore loses its sun shadow as well (v1.11: the shadows of objects behind walls or outside
// the view vanished). So a group-0 entry is rejected or left out only when neither the main camera nor its
// sun shadow needs it (Occlusion tests the shadow); otherwise CBRO drops it from the main view alone, where
// the main accumulator registers it. Entries of unknown groups are only ever dropped from the main view.
// Main-only groups (rejection and the early skip are allowed): group 1 (id 133326, non-casters), group 2
// (id 1328670, the previs list) and the group array (id 459440). DrawWorld registers each of them with
// the main accumulator only.
//
// The sun's shadow direction (BSShadowDirectionalLight update, vtable 97945 slot 14): the engine normalizes
// row 0 of its NiLight's world rotation, smooths it over time, and builds its shadow camera (light +0x2C0)
// with local rotation row 0 = that direction, placed 15000 units back along it from the view camera. So the
// camera's row 0 is the direction the light travels, and (view - camera) . row0 = +15000 checks the sign.
// The cascade range is the global id 777729 (fDirShadowDistance as applied; 1000 if unset).
//
// Merge-instanced type 0x0F (not BSMergeInstancedTriShape, which is a different type): Block::Add writes
// the object's own entry, then one entry per instance until the block is full; the caller continues in a
// new block with startIndex = the next instance. Never seen in the main pass so far.
//
// The scene walk (id 1138818; DrawWorld's cull calls it with (culler, group 0, group 1, group 2, &group array,
// previs active)). Exterior path, previs off: the world node is the scene root's child 3; each of its children
// from index 1 on is a cell node (the walk sets flag bit 42 on it), of which only children 2, 3 and 9 are looked
// at: 2 -> its children whole into group 1; 3 and 9 -> each child whose GetRTTI() is exactly NiNode's (NiRTTI id
// 191219) has every one of its own children added to group 0 with flags 0, anything else goes to group 0 whole.
// The v1.17 run showed no exact-NiNode child there at all ("containers seen 0"): the precombined chunks (BSFadeNodes)
// and static refs are direct children of the cell's node 3, each added whole. So cell-node pruning (v1.18) acts one
// level up: the instruction that starts a child node's loop (+0x5B8: cmp dx,word [rax+132h], rax = r14 = the node,
// r15d = its index) is replaced by a switchable jump (8-byte compare-exchange, offset 0, the jae's first byte kept)
// to a stub that asks the cell-node filter with (node, index): a skipped node continues at +0x681 (the cell's next
// child), otherwise the stub restores rax and dx, repeats the compare and resumes at +0x5BF (the jae). The loop
// keeps its state in callee-saved registers (r12 the cell, r14 the node, r15d the index, rbp/rbx/rdi/r13).

namespace CBRO::Hooks::CullGroups
{
	// Who reads a culling group's entries.
	enum class GroupKind : std::uint8_t
	{
		kMainOnly,   // groups 1, 2 and the group array: the main camera only
		kSunShared,  // group 0: the main camera and (with previs off) the sun's shadow cascades
		kUnknown,    // any other group (or none known): other readers are possible
	};

	struct BlockAdd
	{
		void*               block;
		RE::NiAVObject*     object;
		const RE::NiBound*  bound;
		std::int32_t        startIndex;     // -1 for a fresh add; >= 0 continues a multi-instance mesh
		std::uintptr_t      owner;          // block + 0x3A60
		bool                mainPass;       // owner belongs to DrawWorld's culling groups
		GroupKind           kind;           // the group's readers (meaningful for main-pass entries)
		std::uintptr_t      returnAddress;
	};

	// Consulted for main-pass fresh adds while DrawWorld culls. Return a replacement bound to make the engine
	// reject the entry in every view of its group (and with it the object's subtree), or nullptr to add it
	// unchanged. The entry is still added, so the block's group markers and bookkeeping stay intact; the
	// culling-group pass's frustum tests (fed from the copied bound) reject it, and no view registers it. Honored for
	// main-only groups, and for group 0 when neither the main camera nor the sun's shadow needs the object;
	// ignored for unknown groups (a filter records a main-view drop instead, see MainViewFilter). May run on
	// any thread; must be lock-free. The returned bound must outlive the call.
	using BlockFilter = const RE::NiBound* (*)(const BlockAdd& a_add);

	// Called after a main-pass add into a main-only group that wrote instance entries [a_first, a_end) into
	// a_add.block (merge-instanced meshes, fresh or continued). Instance k of the mesh is entry
	// a_first + (k - max(startIndex, 0)). Use Read/WriteEntryBound to inspect or reject them.
	using InstanceFilter = void (*)(const BlockAdd& a_add, std::uint32_t a_first, std::uint32_t a_end);

	// Consulted by Group::Add for main-pass objects added to a main-only group or group 0 with no flags while
	// the target block has no group markers pending (so leaving the object out can't shift the engine's group
	// structure). Return true to leave the object out of the culling groups entirely (as if culled): for group
	// 0 only when neither the main camera nor the sun's shadow needs it.
	using GroupFilter = bool (*)(RE::NiAVObject* a_object, const RE::NiBound* a_bound, GroupKind a_kind);

	// While DrawWorld culls (SetMainCullActive): the main accumulator is registering a_object. Return true
	// to leave it out of the main view; no other view is affected. a_object is whatever the engine passes:
	// never dereference it.
	using MainViewFilter = bool (*)(const RE::NiAVObject* a_object);

	// Consulted right before a node's children are added to the main groups one by one, while DrawWorld culls: by
	// the scene walk (id 1138818, both its cell paths) for a cell's child node 3 or 9 (a_index), and by DrawWorld's
	// cull (id 718911) for each root registered with it (a_index 0xFFFFFFFF; its children go to group 0 or 1 by
	// their shadow-caster flag). Return true to leave the whole node out: none of its objects is then filed with
	// any view (main camera or sun cascades), so only for a node entirely outside the view whose sun shadow can't
	// reach the view. Main thread.
	using CellNodeFilter = bool (*)(RE::NiAVObject* a_node, std::uint32_t a_index);
	inline constexpr std::uint32_t kRootIndex = 0xFFFFFFFFu;       // a_index: a root registered with DrawWorld's cull (its children are the entries)
	inline constexpr std::uint32_t kContainerIndex = 0xFFFFFFFEu;  // a_index: an exact-NiNode container under a cell's node 3 (its children are the entries)

	using BlockObserver = void (*)(const BlockAdd& a_add, bool a_skipped);
	using GroupObserver = void (*)(void* a_group, RE::NiAVObject* a_object, std::uintptr_t a_owner, std::uintptr_t a_returnAddress);

	void SetFilter(BlockFilter a_filter);
	void SetInstanceFilter(InstanceFilter a_filter);
	void SetGroupFilter(GroupFilter a_filter);
	void SetMainViewFilter(MainViewFilter a_filter);
	// Every main-accumulator registration, in both modes (a diagnostic observer, Core/SetDiff): the engine's object
	// pointer, on whichever thread registers it. Null = none.
	using MainRegistrationObserver = void (*)(RE::NiAVObject* a_object);
	void SetMainRegistrationObserver(MainRegistrationObserver a_observer);
	void SetCellNodeFilter(CellNodeFilter a_filter);
	[[nodiscard]] std::uint64_t TakeGroupAddsConsidered() noexcept;  // main-only Group::Adds offered to the group filter
	[[nodiscard]] std::string   TakeGroupAddSites(double a_frames);  // their callers (return addresses) since the last call, most frequent first
	// RegisterObject calls per accumulator (with its render mode) since the last call, kept apart by mode (the
	// accumulator hook stays in during previs mode, counting only, so both modes' draw sets can be compared).
	void SetRegistrationMode(bool a_cbro) noexcept;  // main thread, at each frame's cull begin
	[[nodiscard]] std::string TakeRegistrationSites(bool a_cbro, double a_frames);
	void SetBlockObserver(BlockObserver a_observer);
	void SetGroupObserver(GroupObserver a_observer);

	// A lamp shadow map's accumulator that files nothing (Core/ShadowLights: a spot light whose lit volume is hidden):
	// every registration into it is dropped. Null = none. Set on the main thread, read on any thread.
	void SetDroppedAccumulator(const void* a_accumulator) noexcept;
	[[nodiscard]] std::uint64_t TakeDroppedRegistrations() noexcept;  // registrations dropped there since the last call (main thread)
	[[nodiscard]] std::uint64_t ReadDroppedRegistrations() noexcept;  // ... running total since load
	// Registrations offered to one accumulator, dropped ones included (diagnostic: a spot light's shadow map during its
	// cull, Core/ShadowLights). Null = none. Set on the main thread.
	void SetCountedAccumulator(const void* a_accumulator) noexcept;
	[[nodiscard]] std::uint64_t ReadCountedRegistrations() noexcept;  // running total since load
	// A shadow map's accumulator whose registrations a filter judges one by one (Core/ShadowLights: a kept spot light's
	// casters that can't shadow a visible pixel). The filter returns true to leave the geometry out; it runs on the
	// registering thread (the lamp loop's, inline). Null = none. Set on the main thread around the light's cull.
	using CasterFilter = bool (*)(const RE::NiAVObject* a_object);
	void SetFilteredAccumulator(const void* a_accumulator, CasterFilter a_filter) noexcept;

	// Installs the detours and the accumulator hook (idempotent). Returns false unless Block::Add and the
	// main-view registration are both hooked: without them nothing can be culled without also cutting
	// shadows. A missing Group::Add or ChildPush hook only costs culling (their entries count as shared).
	bool Install();

	// True while the main-view filter may drop registrations: the render stage that runs DrawWorld's cull
	// (the main groups' culling-group passes finish inside it). Main thread.
	void SetMainCullActive(bool a_active) noexcept;

	// Takes every hook out (the engine then runs its own code, as without CBRO) or puts them back. Call on
	// the main thread before DrawWorld's cull. False if a hook couldn't be switched (another plugin patched
	// the same spot since; it is left as it is).
	bool SetHooksIn(bool a_in);
	[[nodiscard]] bool HooksIn() noexcept;  // the hooks culling needs are in

	// Hook calls on all threads since the last call.
	struct HookCalls
	{
		std::uint64_t blockAdds{ 0 };
		std::uint64_t groupAdds{ 0 };
		std::uint64_t childPushes{ 0 };
		std::uint64_t registrations{ 0 };
	};
	[[nodiscard]] HookCalls TakeHookCalls() noexcept;
	[[nodiscard]] HookCalls ReadHookCalls() noexcept;  // running totals since load

	// The two engine bytes the sun's cascades depend on, as they are now (for the log).
	struct EngineFlags
	{
		bool cullingBatch{ false };
		bool dirShadows{ false };
	};
	[[nodiscard]] EngineFlags ReadEngineFlags() noexcept;

	// Owner value shared by DrawWorld's culling groups (0 before DrawWorld initialized).
	[[nodiscard]] std::uintptr_t MainOwner() noexcept;

	// Files a_object into a_group through the engine's Group::Add (its gateway, inside the group scope so the
	// Block::Add that follows knows the group), bypassing the group filter: for a caller that has already decided
	// (Core/Feed). Main thread, while DrawWorld culls. False if Group::Add isn't hooked.
	bool AddDirect(void* a_group, RE::NiAVObject* a_object, const RE::NiBound* a_bound, std::uint32_t a_flags) noexcept;

	// The sun's shadow cascades as the engine has them set up (main thread, once per culled frame: what ran is
	// counted since the previous call). False if the engine's structures couldn't be read (then nothing about
	// the sun is known).
	struct SunSource
	{
		bool  pathIntact{ false };     // this frame's cascade path (batched or not) and the light's update are the engine's
		bool  groupsEnabled{ false };  // bCullingBatch: which of the two paths runs
		bool  dirShadows{ false };     // without it neither path culls the cascades against group 0
		float dir[3]{};                // row 0 of the shadow camera's rotation: the light's travel direction
		float cameraPos[3]{};          // the shadow camera's position
		float range{ 0.0f };           // cascade range
	};
	[[nodiscard]] bool ReadSun(SunSource& a_out) noexcept;

	// Rejected entries that had been marked force-visible (previs active) since the last call.
	[[nodiscard]] std::uint64_t TakeForcedCleared() noexcept;

	// Block entry bounds (SoA: groups of 4 entries, x[4] y[4] z[4] radius[4], at block + 0x60).
	[[nodiscard]] RE::NiBound ReadEntryBound(const void* a_block, std::uint32_t a_index) noexcept;
	void WriteEntryBound(void* a_block, std::uint32_t a_index, const RE::NiBound& a_bound) noexcept;

	// Merge-instanced type 0x0F data: instance array at +0x1C0 (0xF0 bytes each, world NiBound at +0xC0),
	// count at +0x1D0. Null / 0 if the object isn't one.
	[[nodiscard]] bool IsMergeInstanced(RE::NiAVObject* a_object) noexcept;
	[[nodiscard]] std::uint32_t InstanceCount(const RE::NiAVObject* a_object) noexcept;
	[[nodiscard]] const RE::NiBound* InstanceBound(const RE::NiAVObject* a_object, std::uint32_t a_index) noexcept;
}
