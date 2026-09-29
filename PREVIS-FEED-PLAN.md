# CBRO previs-feed mode: implementation plan (for the implementing model)

Written 2026-09-30 after a read-only disassembly session. Target: CBRO v1.29+ on Fallout4.exe OG 1.10.163 (GOG).
Read this file completely before touching code. Then read `FO4-ENGINE-NOTES.md` §4-§6 and the headers
`src/Hooks/CullGroups.h`, `src/Hooks/RenderStages.h`, `src/Core/Occlusion.h`, `src/Util/Hooking.h`.
Everything below that is not in those files or in `PLAN.md` was established in the 2026-09-30 session and is
recorded here (section 2). Copy section 2 into `FO4-ENGINE-NOTES.md` as a new §5.5b when you implement phase 1,
and correct §5.5's claim that previs sets flag bits 26/39 (see 2.9).

---

## 1. Goal in one paragraph

Today, CBRO mode switches previs off. The engine then walks every object of the 25 loaded cells each frame
(~5,000-7,000 top-level `Group::Add`s), and CBRO judges each one inside its hooks: the cull stage costs 2.7-2.9 ms.
With previs on, the engine skips that walk and only files the short list previs hands it (cull stage ~0.4 ms).
**Previs-feed mode keeps previs switched on, so the engine keeps taking its fast path, but CBRO supplies the lists
the engine files: the main camera's list and the sun cascades' list.** The `.uvd` data then decides nothing for the
main camera or the sun. CBRO stays the only authority. Previs is never switched off or flushed, so F8 becomes a
clean switch between "previs fills the lists" and "CBRO fills the lists", and the A/B baseline is always
full-strength previs.

Out of scope: interiors (they fall back to today's behaviour, see 4.6), the third previs list (a separate view,
see 2.4), lamp shadows (unchanged), precombines (unchanged).

---

## 2. Engine facts established on 2026-09-30 (disassembly, OG 1.10.163, image base 0x140000000)

Tools used: full `dumpbin /disasm:nobytes` of the exe (~65 s, 667 MB; regenerate into a scratch folder if
needed), `tools/re/xref_scan.ps1` (`Xref.Scan`), `tools/re/rip_refs.ps1`, and an 8-byte pointer scan of the file.
Offsets like `walk+0x16A` are relative to the function start.

### 2.1 Previs is a per-frame query, not a per-cell "apply"
- `Render_PreUI` (984743 = 0x142857480) **+0x80** (0x142857500): `E8` call to **0x14284F1D0 (id 1264353)**. If
  `IsActive()`, it calls **0x1427AE4B0** (a TES-side query; rcx = `[0x1458D0AA8]`, rdx = `[0x146723240]` = the camera)
  which fills **three lists** for this frame from the `.uvd` data. Each list = an ID array plus an object pool:

  | List | ID array (`BSTArray<u32>`; count at +0x10) | Pool pointer (`BSTObjectArena<BSPreCulledObjects::ObjectRecord, ..., 512>*`) | Fed by |
  |---|---|---|---|
  | main camera | 0x1467222A8 (count 0x1467222B8) | `[0x146722290]` | 0x142809EE0 (997287) |
  | sun cascades | 0x1467222C0 (count 0x1467222D0) | `[0x146722298]` | 0x14280A160 (1142692) |
  | third view | 0x1467222D8 (count 0x1467222E8) | `[0x146722280]` | 0x14280A3C0 (410310) |

  Getters: 0x142809C80 (main IDs), C90 (main pool), CA0/CB0 (sun), CC0/CD0 (third). Lock for all: 0x1467222A0.
- Pool pages are 0x2000 bytes of 16-byte `ObjectRecord {NiAVObject* object; u32 flags}`; `flags` is passed to
  `Group::Add` as its 4th argument. The Addictol PDB confirms the type name.
- The pools are created each frame by 0x142809910 (1332111), called from 502840 (0x14284D9D0) +0x34 only when
  `IsActive()`. All lists are cleared at the end of the frame: 200894 (0x1428573F0) +0x77 tail-jumps to
  0x142809A90 → 0x142809B10 (1131946).

### 2.2 How an ID becomes an object
- Lookup 0x14280A5E0 (1087700): a hash map at 0x1438C78A8 (buckets `[0x1438C78C8]`, size 0x1438C78AC, sentinel
  `[0x1438C78B8]`), 24-byte entries `{u32 formID; NiPointer<NiAVObject> object; entry* next}`.
- Registration 0x142809160 (1162647): `(formID, NiAVObject*)` inserts; a null object removes and calls 136788.
  Callers are cell/reference code (0x14039CF00, 0x1403A1974, 0x1403A2042, 0x1403B46F8, 0x1403B4867, 0x1403CC930,
  0x1403CCA00, 0x1404016EC, 0x140405A70, 0x140410AB1/B51, 0x1407D19C0, 0x140D241C0). Byte 0x14672228A = the
  highest load-order index seen.
- The vanilla feeds try an ID whose top byte is neither 0 nor 0xFD under **every** load-order index 1..max
  (up to ~250 map lookups per ID). That is presumably why Addictol replaces this module (2.8).

### 2.3 The main feed (997287 = 0x142809EE0) and the scene walk
- Signature: `void Feed(group* casters, group* nonCasters)`. The scene walk (1138818 = 0x14284DF50) calls it at
  **walk+0x16A (0x14284E0BA, `E8 rel32` → 0x142809EE0)** with `rcx = group 2`, `rdx = group 1`.
- For each ID (under the lock), and then for each pool record, it skips the object if flags bit 0 is set. Otherwise
  it calls `Group::Add(group, obj, &obj->worldBound, flags)`: **flags bit 40 clear (shadow caster) → group 2,
  set → group 1**. After each ID it calls 0x14280A690(formID). Tail: jmp 0x14280A7E0.
- The walk, in order:
  1. Scene root = `[0x146721B70]`. Root children 0 and 1 → group 1 (both modes).
  2. `if IsActive()`:
     - call the feed (above);
     - call 0x14284DDD0 (a helper; `Render_PreUI`+0x15C calls it instead when previs is inactive);
     - `Group::Add(group 2, [0x14609A030])` (a global node; identity unverified).
  3. **Gate A:** `r14b = IsActive() && u32 [[0x146721B70]+0x14C] > 0`. **If `IsActive()` and not r14b, jump to the
     tail at walk+0xEB4 (0x14284EE04): the whole exterior/interior cell expansion is skipped.** The writer of
     +0x14C was not found. The field was 0 in every measured previs frame (300-1,000 `Block::Add` entries per frame).
  4. Tail (both modes): root children 6, 7 and 10 → group 1.

### 2.4 Sun cascades and the third view with previs active
- Cascade cull 1390075 (0x1428CA7D0) +0xBD: `IsActive()` picks the path. On the previs path it builds a
  **stack group** at `rsp+0x30`, then:
  - calls `SetSuspended(true, false)` at +0x1AA;
  - **calls the sun feed at +0x1B4 (0x1428CA984, `E8` → 0x14280A160)** with `rcx = &stack group`. The sun feed adds
    only flags-bit-40-clear objects from the sun IDs and pool;
  - for each cascade: set planes (0x141CCC7E0), `Group::Process` (0x141CCCE70), register (0x141CCD520);
  - calls `SetSuspended(false, false)` at +0x2C2.

  Previs is suspended while the stack group is processed, so the cascades' normal frustum tests apply. The
  cascades do **not** read DrawWorld group 0 on this path.
- The third list is fed at 0x140649F61 inside 0x140649BE0 (856638), reached from 0x140648DF0 (1114882). It is a
  separate view with its own stack group, probably water reflections (not verified). Leave it alone.

### 2.5 DrawWorld's cull (718911 = 0x14284F350) with previs active
- Group globals: **G0 0x1467233C0, G1 0x146723530, G2 0x146723810, group array 0x1467233A8 (count 0x1467233B8)**.
  These are the ids 1117782 / 133326 / 1328670 / 459440 that CullGroups already resolves. The main accumulator
  is `[0x146723350]` (1211381).
- **Gate B:** `bpl = IsActive() && u16 [[0x146721B70]+0x180] > 0` (at +0xBF). `byte G0+0x16A = bpl`; G2, G0 and G1
  get `+0x169 = 1`. The +0x16A flags of G1 and G2 are not written here (expect 0).
- The walk is called with `previsActive = IsActive()`. Then, **if `IsActive()` and not bpl, DrawWorld's own loop
  over the roots registered with the culling camera is skipped** (0x14284F57D-0x14284F70B, notes §5.3 last bullet).
- Group processing, unbatched (`bCullingBatch` 0, this setup; 0x14284F845):
  - `IsActive()`: **Process + Register G2** with the main accumulator.
  - Otherwise: Process + Register G0.
  - Then G1 always.
  - Then the group array only when previs is inactive.

  **So with previs active the main camera draws only G1 and G2. G0 and the group array are never processed.**
  Batched (0x14284F725): the view manager registers G2 (active) or G0, plus G1, plus the array when inactive.
- `Block::Add` with previs active marks entries force-visible when the block's +0x3A6F (= group+0x16A) is 0. So
  **the engine applies no frustum test to anything filed in G1/G2 on this path.** This includes the children that
  jobs push under a filed node. CullGroups' `BlockAddThunk` already clears the force-visible byte of entries CBRO
  rejects (`g_forcedCleared`).

### 2.6 What the previs-off walk adds that the fast path skips (the "replica" rules)
Exterior path only: the byte `[[culler+0x150]+0x138]` is set, with `culler = [0x1467232D0]` (the walk's first
argument), and the override root `[0x146723208]` is null. Every add below is `Group::Add(group, obj,
&obj->worldBound, 0)`. Every one first requires `obj != nullptr` and flags bit 0 clear, except where marked.
`IsNode()` = vtable +0x20 (CommonLib slot 04), `IsFadeNode()` = +0x30 (slot 06), `GetRTTI()` = +0x10. A node's
children array is at +0x128 with a u16 count at +0x132. NiNode's NiRTTI is 0x145C08EC0 (191219).

| # | Walk site (Group::Add return address) | Rule | Previs-off group |
|---|---|---|---|
| R1 | 0x14284E146-0x14284E2D8 (ret E1EB, E220, E255, E28F, E2D8) | S = `IsNode(root.child[2])`; T = `IsNode(S.child[0])`. T.child[0], T.child[1] → G1; T.child[2] → **G0**. S.child[1], S.child[2] → G1. | G1 / G0 |
| R2 | ret 0x14284E3FC | W = `IsNode(root.child[3])` (the world node; W must exist with bit 0 clear, otherwise skip everything below). `W.child[0]` → G1. | G1 |
| R3a | cell loop 0x14284E390 | For i ≥ 1: `c = IsNode(W.child[i])`, skip if null or bit 0. The walk also sets flags bit 42 on c: **do not replicate this write.** For each j over c's children: skip j ∈ {0, 1, 4}; skip j == 2 when byte `[0x14672327A]` != 0. | |
| R3b | ret 0x14284E5D1 | Child at j is **not** a node (`IsNode` null) and bit 0 clear → the child whole. | G0 |
| R3c | ret 0x14284E581 | j == 3 and a grandchild's `GetRTTI()` == NiNode's exactly → each of **its** children (no bit-0 test on these). | G0 |
| R3d | ret 0x14284E59A | j == 3 or 9: every other grandchild (bit 0 clear) whole. | G0 |
| R3e | ret 0x14284E4ED | j == 2 (a node): each of its children (bit 0 clear). | G1 |
| R4 | DrawWorld cull 0x14284F61C-0x14284F705 | Roots in `[[culler+0x150]+0x10]+0x58` (count u32 at +0x68, see notes §5.3). Skip bit 0. Non-node root, or bit 14 set and `IsFadeNode()` != null → the root whole (ret 0x14284F6FE). Otherwise each child (bit 0 clear): bit 40 clear → G0; bit 40 set → G1 (ret 0x14284F6E5). | array slot / G0 / G1 |

j values other than 2, 3 and 9 add nothing when the child is a node.

These rules come from one reading of the disassembly. Phase 2's audit is what proves them, so do not ship phase 3
without it.

### 2.7 Every caller of `IsActive()` (0x142809E30)
In feed mode previs really is active, so every reader below behaves exactly as in vanilla previs mode. Feed mode
differs from vanilla previs mode only in the contents of the main feed and the sun feed.

0x140418605 (356257), 0x14052CB0B (189985, console `tpc`), 0x140649E1D (third view), 0x1407A7D4F (407994),
0x1407C2323 (1140265), 0x140D3C167 (891949), 0x140E9E63B (workshop), 0x141CCB967 (`Block::Add`, force-visible),
0x141CCBC09 (1022362), 0x141CCD109 (`Group::Process`), 0x141CCD219/30C/479, 0x141CCED38, 0x141CCF0F8,
0x141CD103E (1175715), 0x141CD119F, 0x141CD1328 (block jobs / finish loops), 0x1427AE326 (1276848), 0x1427D0EAE,
0x14284D9FB (pools), walk 0x14284E0AB/0DD/0FE/EA78, 0x14284F1D7 (query), DrawWorld cull 0x14284F40F/52F/56B/72A/
77E/845/901, `Render_PreUI` 0x1428575DC, cascade cull 0x1428CA88D.

### 2.8 Addictol
Addictol 1.7's `bBSPreCulledObjects = true` (`D:\MO2\Fallout 4\mods\Addictol\f4se\plugins\Addictol.toml`, line
409) replaces the previs ID→3D map with its own (`IDTo3DMap`) and calls `Group::Add` itself. How it hooks is not
known: function entries or call sites. CBRO must detect at run time what the two feed call sites call (4.1) and
log it with `Util::DescribeCodeAddress`, which follows jmp chains. In CBRO-owner frames CBRO never calls the
previous target, so Addictol's feed simply doesn't run. In previs-owner frames CBRO calls the previous target, so
Addictol keeps working. If Addictol patched something CBRO needs (a site is not `E8`), feed mode stays off with a
log line. The user can then set `bBSPreCulledObjects = false`. That is harmless, because it is a performance
module.

### 2.9 Corrections to FO4-ENGINE-NOTES
- §5.5 says previs marks hidden objects with flags bits 26/39. The only writers of bit 39 found are the fade-node
  culler (vtable method 0x1428844F0, which also sets bit 26) and a fade/LOD routine (0x140E92F80). None of them is
  in the previs module. How previs trims lamp shadows (PLAN §7 pier run) remains unexplained.
- §5.3 "with previs on none of this expansion runs" holds only while gate A is 0 (2.3).
- `SetSuspended(true/false, false)` is flush-free. The cascade cull uses exactly that pair every frame on the previs
  path.

### 2.10 A CBRO bug found in the same session (fix first: phase 0)
In v1.28 the sun-shadow culling never ran: every log interval shows `sun shadows per interval: ... path changed
600`. The cause is in `Hooks::CullGroups::CallsTo` (`src/Hooks/CullGroups.cpp`, ~line 580). It compares the
sun-cascade site's call target with `RenderStages::ThunkOf(kSunCascades)` (CBRO's function in CBRO.dll). But
`Util::WriteCall5` uses `trampoline.write_call<5>`, so the site calls a **jump stub in the trampoline**. The log
shows `stage:sunCascades site now calls 0x7FF638C30697 (no module) -> CBRO.dll+0x42BD0`. The two never match.
Every v1.28 number for the sun cascades was measured with all casters kept.

---

## 3. Rules that bind this work (from the user; do not relax)

- **Never hide visible geometry or its shadows.** Anything uncertain is drawn. An object CBRO has not judged yet is
  fed. When a phase can't prove completeness, it doesn't feed.
- **Modular:**
  - Keep the subsystem interfaces: HiZ, Occlusion, CullGroups, RenderStages, Runtime.
  - Put the new engine-facing code in a new `Hooks/PrevisFeed` and the policy in a new `Core/Feed`.
  - `bPrevisFeed=0` must give exactly v1.28 behaviour (plus the phase-0 fix).
  - Name every cross-module change in the reply.
  - Verify the untouched paths in the next log: previs mode registrations, hook counts, A/B lines.
- **No new hotkeys.** An optional key ships unbound (0) and the user is asked. F8 stays the toggle.
- **Never write data into code pages.** Counters live in normal memory, per thread, summed at log time.
- **Engine state switches happen on the main thread,** before DrawWorld's cull (the cull stage begin) or at a
  render-stage boundary.
- **Never call previs `SetEnabled(false)` in feed mode.** It flushes, and that is what broke every earlier previs
  baseline.
- **Don't write engine flags** (e.g. the walk's bit 42 on cell nodes). Don't `SetAppCulled`.
- **Unknown memory is read under SEH.** Everything that runs on job threads is lock-free.
- **A measured A/B at the same spot and view** is the only evidence of speed. Report numbers from the log. The user's
  screenshots/FPS reports are trusted.
- After each build, say plainly what changed, what could break, and which log line shows it.

---

## 4. Design

### 4.1 `Hooks/PrevisFeed` (new; engine-facing)
Installs two call-site wrappers with `Util::WriteCall5`. Both are always installed and pass through by default:

| Site | Address (from ids) | Expected | Thunk behaviour |
|---|---|---|---|
| main feed | `OG(1138818)+0x16A` = 0x14284E0BA | `E8` → `OG(997287)` = 0x142809EE0 (or another module's thunk: chain it) | `owner == CBRO` this frame: call the installed `MainFeedFn(group2, group1)`; else call the previous target |
| sun feed | `OG(1390075)+0x1B4` = 0x1428CA984 | `E8` → `OG(1142692)` = 0x14280A160 | `owner == CBRO`: call `SunFeedFn(stackGroup)`; else call the previous target |

- Verify at install: the first byte is `E8`, and read the current target. Log `"previs feed: main site
  Fallout4.exe+0x284E0BA calls <DescribeCodeAddress>"`. Do the same for the sun site, and also describe the first
  bytes of 0x142809EE0 / 0x14280A160 (an E9/FF25 there means another plugin detoured the entry). Any site that isn't
  `E8` → feed mode unavailable (`Available() == false`) plus an error line.
- `owner` is one `std::atomic<std::uint8_t>`, set on the main thread at the cull begin. Both thunks run on the main
  thread (inside the walk and inside `Render_PreUI`+0x1BF / +0x175).
- Also exports read-only engine accessors:
  - `ReadGates()` → `{ isActive (0x142809E30 or the three bytes), gateA (u32 root+0x14C), gateB (u16 root+0x180),
    exterior byte, overrideRoot != null, suspended byte 0x146722289 }`;
  - `GroupPlanes(group)` → the six planes at group+0x00 (`NiPlane {NiPoint3 n; float d}`, 16 bytes each). G2's are
    set before the walk (0x142859B60 at DrawWorld cull +0x10E) and are valid inside the feed.
- Exports `bool EnumerateCandidates(Visitor&)`, which implements R1-R4 of 2.6 exactly. The visitor receives
  `(NiAVObject* obj, Route route, std::uint8_t siteClass)`:
  - `Route` = where the previs-off path would have filed it: G0, G1, or array slot;
  - `siteClass` = which rule (R1…R4) for the audit.

  It returns false (the caller falls back) for the interior/override paths or when the world node is missing. Read
  engine pointers under SEH; a fault → false. Do not call `Group::Add` from here: enumeration only.
- `AddDirect(group, obj, bound, flags)` lives in **CullGroups** (it owns the gateway), not here: it must call
  `g_groupAddOriginal` inside a `GroupScope` so `Block::Add` still knows the group, and it must bypass the group
  filter (the decision was already made).

### 4.2 `Core/Feed` (new; policy)
One per-frame object, main thread only. At the cull begin (`Runtime::OnCullBegin`, after `ApplyPrevisRequest`,
before `SyncHooks`), it decides this frame's **path**:

| Path | When | What happens |
|---|---|---|
| `kPrevis` | mode = previs (F8), or `bPrevisFeed=0` with previs owner | owner = previs; hooks out (as today). Real previs. |
| `kFeed` | CBRO mode, `bPrevisFeed=1`, `PrevisFeed::Available()`, `IsActive()`, exterior, override root null, gate A == 0, gate B == 0 | owner = CBRO; CBRO's hooks in; the main and sun feeds run CBRO's lists. |
| `kClassic` | CBRO mode, and any `kFeed` condition fails (interior, a gate, previs made inactive by `tpc`/workshop, a site unavailable) | Today's behaviour without the flush: if `IsActive()`, call `SetSuspended(true, false)` (1263609) at the cull begin and `SetSuspended(false, false)` at `OnStageEnd(kSunCascades)`, **only if CBRO set it** (the byte was 0 before and CBRO's write is still there). If previs is already inactive (`tpc`, workshop), do nothing: the engine walks anyway. |

Log each path change with its reason (counted per interval, not per frame).

**Main feed** (`MainFeedFn(g2, g1)`, called from inside the walk):
1. `EnumerateCandidates`. For each candidate:
   - a. Frustum test against G2's planes, sphere vs 6 planes with the sign fixed by the self-check in 4.4. Outside
     → skip. (This replaces the frustum test the engine no longer applies.)
   - b. The CBRO decision: add `bool Occlusion::FeedKeep(obj, const NiBound&)` = `!SkipTopLevel(obj, &bound,
     GroupKind::kMainOnly)`.
     - It must run with the frame context (the feed runs inside the cull stage, so it does).
     - Keep `SkipTopLevel`'s always-draw (bit 11) and light rules unchanged.
     - On frames without a usable context (no depth, camera jump), `FeedKeep` returns true (frustum-only feed).
   - c. Kept → `CullGroups::AddDirect(flags bit 40 clear ? g2 : g1, obj, &obj->worldBound, 0)`. `SkipTopLevel`
     already set `t_top`, so `Block::Add`'s filter reuses the decision.
   - d. If `Route == G0`, also push `obj` onto this frame's **sun list** (a `std::vector<NiAVObject*>` reused
     across frames), **before** the frustum/CBRO filter: sun casters are needed off-screen too.
2. Nothing else. The engine's own adds (root children 0/1/6/7/10, `[0x14609A030]`) still run around the feed.

**Sun feed** (`SunFeedFn(stackGroup)`, called by the cascade cull on the previs path, previs suspended by the engine):
- Phase 3 (conservative): for every object on this frame's sun list, re-check it (non-null, bit 0 clear) and call
  `CullGroups::AddDirect(stackGroup, obj, &obj->worldBound, 0)`. This equals the previs-off group-0 input the
  cascades read today, so the shadows are the same as CBRO classic mode.
- Phase 4: drop casters whose sun shadow can't reach the view, using `Occlusion`'s existing sun test, which needs
  the phase-0 fix. A new public `Occlusion::SunCasterNeeded(obj, bound)` wraps the `SunUnneeded` logic. Then try the
  bit-40 filter the vanilla sun feed uses, and keep it only if the cascade registrations don't drop (4.5).
- The stack group is not DrawWorld's (its owner differs), so CBRO's `Block::Add` filter ignores its entries (the
  `mainPass` check). Keep it that way.
- The sun list is built at the main feed and used at +0x1BF in the same `Render_PreUI`. Both are on the main thread
  with no game update in between. Clear it at the cull begin. If the main feed didn't run this frame but the sun
  feed is called (it shouldn't be), call the previous target and count it.

### 4.3 Runtime changes (`src/Core/Runtime.cpp`)
- With `bPrevisFeed=1`:
  - `ApplyMode` never queues `SetPrevisEnabled(false)`. `previsRequest` stays unused.
  - `bDisablePrevis` is ignored (log once).
  - The "previs switched outside CBRO (tpc)" follower must not flip CBRO's mode. `tpc` only turns `kFeed` into
    `kClassic` (previs inactive).
- `bStartActive` keeps its meaning (which owner at load). There is no flush at load any more.
- Timing buckets: extend `g_timing.mode` from {0 previs, 1 CBRO} to {0 previs, 1 CBRO feed, 2 CBRO classic}. Print
  the three columns in the CPU/GPU-by-mode lines and in the A/B lines (feed vs previs is the headline).
- `CullGroups::SetRegistrationMode` gets the same three-way mode, or two with classic merged into CBRO: say which.

### 4.4 Frustum-plane self-check (once per session, and after each load)
On the first `kFeed`-eligible frame, before feeding, read G2's six planes and test:
- a point 100 units ahead of the camera (`NiCamera::world.translate + 100 * forward`; use `RE::Main::WorldRootCamera()`);
- a point 100 units behind it.

Find the sign convention for which "ahead" is inside all six planes and "behind" is outside at least one. If neither
convention fits, log `"previs feed: frustum planes self-check failed"` and stay in `kClassic` for the session. Log
the convention found.

### 4.5 The completeness audit (phase 2 gate; keep it in the build afterwards, off by default)
Setting `iFeedAuditInterval` (frames; 0 = off, default 600 during phases 2-3).
- On an audit frame, force `kClassic` (the engine's previs-off walk runs).
- Right before the cull (at the cull begin, same scene state), run `EnumerateCandidates` into a set A (object →
  route, site class).
- During the cull, the existing `GroupObserver` records every DrawWorld main-pass `Group::Add` (group, object,
  return address) into set E.
- Compare only the adds from the replica's sites (the return addresses in 2.6):
  - `missing` = E \ A (the replica forgot something: **a bug that would hide geometry in feed mode**);
  - `extra` = A \ E (the replica adds too much: harmless but costly);
  - `route mismatch`.

  Log the counts and up to 10 samples per class: type, name, parent's name, return address.
- Second audit, for the sun: on an audit frame and on the next `kFeed` frame at a still camera ("camera still"
  logic exists), compare the per-accumulator registration counts of the four `kShadowmapDir` accumulators. Expect
  equal within ~1%.
- Third audit, for the main view: on an audit frame, record the set of objects the main accumulator registers
  (`kClassic`, CBRO deciding). On the following `kFeed` frame at a still camera, record the same. Any object
  registered in classic but not in feed whose CBRO verdict was not "hidden/outside" in both → `main missing`. Log it
  with samples.

### 4.6 Fallback summary
| Situation | Path | Why |
|---|---|---|
| Interior, override root, portal path | `kClassic` | the walk's interior path is not replicated |
| Gate A or B non-zero | `kClassic` | the engine would run its own expansion or root loop as well (duplicates, unknown state) |
| Previs inactive (INI off, `tpc`, workshop suspended) | `kClassic` without touching the bytes | the engine already walks |
| A feed site not `E8` at install / self-check failed | `kClassic` for the session | can't feed safely |
| No usable depth this frame | `kFeed`, frustum-only | the engine's frustum test is off on this path |
| Audit frame | `kClassic` | the reference walk |

---

## 5. Phases (each ends with a build, a user run, and a log check before the next)

Build (PowerShell tool): `cmd /c "call ""D:\Softwares\Visual Studio\Product\VC\Auxiliary\Build\vcvars64.bat"" >nul
&& cd /d ""D:\Projects\Confidence-Based Runtime Occlusion System"" && cmake --preset release && cmake --build --preset
release"`. The post-build step copies `CBRO.dll`, `CBRO.pdb`, `res/CBRO.ini` and `meta.ini` into
`D:\MO2\Fallout 4\mods\Confidence-Based Runtime Occlusion`. **This overwrites the installed CBRO.ini**, which the user
had changed (`startActive=true` at 00:56 on 2026-09-30), so tell the user. Bump the version in `CMakeLists.txt`
(`project(CBRO VERSION ...)`) each build. Offline tests live in `tools/tests`. The log is at
`%USERPROFILE%\Documents\My Games\Fallout4\F4SE\CBRO.log`.

### Phase 0: fix the sun-path check (v1.29.0; tiny, independent)
- In `CallsTo`, resolve the site's current target through jump stubs before comparing: an `FF 25 00000000` + u64 or
  an `E9 rel32`, up to 3 hops. `Util::DescribeCodeAddress` already follows these, so factor out a
  `Util::FollowJumps(uintptr_t)`. Accept the site if the resolved target is CBRO's thunk and
  `OriginalOf(kSunCascades) == a_target`.
  - Alternative: have `WriteCall5` return the stub address it wrote.
- Log check: `sun shadows per interval: on N` > 0 in daylight; `path changed 0`.
- Expect the CBRO sun-cascade bucket to fall. Report the before/after at the same spot.

### Phase 1: PrevisFeed hooks, pass-through only (v1.30.0)
- `Hooks/PrevisFeed` with both wrappers installed and owner always `previs`. The accessors, `EnumerateCandidates`
  (not yet called), `CullGroups::AddDirect`, and the settings `bPrevisFeed` (default 0 in this build) and
  `iFeedAuditInterval`.
- Log: the two "previs feed: … site … calls …" lines, the entry-byte descriptions, and the gates' values once per
  interval: gate A, gate B, exterior, previs active, suspended.
- Gate: with `bPrevisFeed=0` every v1.29 number stays the same at the same spot: previs-mode registrations, hook
  calls, A/B. In previs mode the wrappers must call through (Addictol or vanilla) with no measurable cost.

### Phase 2: replica + audit, observe only (v1.31.0)
- Implement `kClassic` with the per-frame suspend/unsuspend (4.2), still with **no feeding**. Implement
  `EnumerateCandidates` and the audit (4.5, first part).
- User run: the load spot, location 54 (the bridge in the 2026-09-30 screenshots), a dense street, a walk across
  several cells, one fast travel, night (no sun), one interior (must show "classic: interior"), workshop mode.
- Gate: `missing 0` in every audit interval of the run. `route mismatch 0`. Explain any `extra`. Fix the replica
  until then. Also report classic-path frame time vs v1.28 CBRO mode at the same spot: the suspend must not be
  slower than the flush path. The v1.28 notes suspect the "applied then suspended" state files extra entries, so
  measure it.

### Phase 3: feed mode (v1.32.0)
- `kFeed` as designed: main feed with frustum + CBRO decisions, conservative sun feed, the frustum self-check, F8 =
  owner switch, the three-way timing modes, and the sun and main audits (4.5, parts 2-3).
- `bPrevisFeed=1` in `res/CBRO.ini`. Explain the setting there: what it does, that 0 restores v1.28 behaviour, and
  that previs is never switched off with it.
- New log lines per interval:
  - `feed paths: feed N | classic N (interior, gateA, gateB, previs inactive, audit, unavailable) | previs N`
  - `feed per frame: candidates N (R1 a, R2 b, R3 c, R4 d) | frustum out N | CBRO hidden N | fed G2 N / G1 N | sun list N | build X ms`
  - `feed audit: main missing N | sun registrations classic/feed …`
- User run: as in phase 2, plus same-view A/B screenshots at location 54 and the load spot:
  1. load with `bStartActive=0`;
  2. take the previs screenshot before any F8;
  3. press F8 once and take the CBRO screenshot.
- Gates:
  - no missing geometry or shadows in the user's view (their call);
  - `main missing 0`;
  - sun registrations equal within ~1%;
  - cull stage in feed mode ≤ ~1.2 ms at both spots (v1.28 CBRO: 2.7-2.9);
  - frame time vs real previs reported honestly, whichever way it goes.

### Phase 4: optimizations (v1.33+; one per build, each measured)
1. **Sun feed through CBRO's sun test** (needs phase 0). Then check whether the bit-40 filter keeps the cascade
   registrations; keep it only if it does.
2. **Per-cell candidate cache.** Rebuild a cell's list only when the cell node's child count/array pointer, or a
   container's, changes. Otherwise iterate the cached arrays. Verify with the audit.
3. **Coarse cell rejection.** Keep a per-cell union sphere of its candidates (the engine's node bounds are
   unmaintained, notes §5.3), refreshed with the cache. A cell entirely outside the frustum planes skips its
   candidates for the main feed. It **never** skips them for the sun list.
4. **Optional:** skip the vanilla query at `Render_PreUI+0x80` (`E8` → 0x14284F1D0) in CBRO-owner frames. Only
   after the third view's consumer (2.4) is identified and fed, or shown not to matter. Otherwise that view loses
   its geometry. Measure the query's cost first: it runs before the cull stage, in the "rest" bucket.

---

## 6. Numbers to compare against (v1.28 logs)
- Load spot, real previs (bStartActive=0, before any F8): 6.46 ms, main 1,045 registrations, cull stage 0.37 ms.
- CBRO classic at the same spot: 9.0 ms, main ~1,850 kept, cull 2.7 ms (bookkeeping 1.37 ms), pre-pass GPU 2.0 ms,
  sun cascades 1.85 ms (sun culling off because of the bug).
- Location 54 (2026-09-30 01:19-01:20, startActive=true, so "previs" there was the weakened, flushed state):
  previs mode 15.1 ms / main 4,711; CBRO 12.4 ms / main 3,266, cull 2.88 ms, sun cascades GPU 3.6 ms. Re-measure
  this spot with real previs in phase 3.
- Estimated feed-mode cull stage: the candidate pass ~0.3 ms (~7k objects at the measured ~40-60 ns per cached
  verdict) plus the engine filing the kept ~2k objects (previs-scale, ~0.4-0.6 ms). This is an estimate: measure it.

## 7. Open questions to keep an eye on (log them, don't guess)
- The writers of gate A (root+0x14C) and gate B (root+0x180). Count the frames where they are non-zero.
- The identity of `[0x14609A030]` (added to G2 every previs frame) and of the third view.
- Whether anything else files into G0 outside the walk in previs-off mode that the main view needs. The main audit
  would show it as `main missing` from a non-replica site. One known G0 add outside the walk is at 0x1428544D2
  (objects from the array at 0x1467231C0, +0x208: probably lights).
- The cost of the cascades' stack-group path (Process per cascade) vs the previs-off view-manager path. Compare the
  sun buckets.
