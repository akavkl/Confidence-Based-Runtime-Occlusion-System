# Fallout 4 engine notes (OG 1.10.163): a checkpoint

Everything learned about Fallout 4's engine, runtime and modding ecosystem while building CBRO, written so it can be reused for any other project on this game. It is about the game, not about the mod: how the engine culls, renders, lays out its objects, where its switches are, what other mods do to it, and how the facts were verified. Where a fact came from disassembly rather than a public header, it says so. Where something is still unverified, it says so.

Date: 2026-09-29. Game: `Fallout4.exe` 1.10.163.0 (GOG build, "OG"). Unless stated otherwise every numeric ID is an **Address Library ID for 1.10.163** (`version-1-10-163-0.bin`), every virtual address assumes the image base `0x140000000`, and every offset is a byte offset from the start of the named object or function.

---

## 1. How to read and verify these notes

### 1.1 Address Library IDs
- The 1.10.163 database is `version-1-10-163-0.bin` (V0 format): a `u64` count followed by `(u64 id, u64 rva)` pairs. Resolving an ID is a linear scan; `rva + 0x140000000` is the VA. A 12-line PowerShell script does it (read the count, loop, match).
- The IDs in `CommonLibF4-DM/include/RE/IDs.h` are namespace-qualified names for a subset of them; the IDs below that have no name there came from disassembly.
- The database matches the GOG exe: every call site used was checked to be an `E8 rel32` call and every vtable ID to resolve to its MSVC RTTI name before anything was written.

### 1.2 Tools that worked
- `dumpbin /disasm:nobytes /range:<va>,<va> Fallout4.exe` (MSVC 14.51: `D:\Softwares\Visual Studio\Product\VC\Tools\MSVC\14.51.36231\bin\Hostx64\x64\dumpbin.exe`) is a good enough disassembler for targeted reading.
- `tools/re/*.ps1` (PowerShell + C# `Add-Type`, no Python on this machine): `rtti_scan.ps1` (MSVC RTTI class names matching a pattern, with their vtable IDs), `xref_scan.ps1` / `calls_in.ps1` (callers of a function, function-bounded), `rip_refs.ps1` (RIP-relative data references to a global), `verify_offsets.ps1` (byte checks at ID + offset). `resolve_ids.ps1` and `read_rva.ps1` (bytes at an RVA, with string dereference) live in the session scratchpad.
- `NiObject::GetRTTI` (vtable slot 2) is always `lea rax, [rip+ms_RTTI]; ret`, so an object's class name can be decoded from those bytes without calling anything. Slots 3/4 are `IsNode`-style queries.
- Reading engine memory of unknown type is done under SEH (`__try`), never through virtual calls.

### 1.3 Sources to read before disassembling
`external/CommonLibF4RD` is the only tree under `external/` (build dependency; since v1.26). Its headers are the original CommonLibF4 layout with `REL::ID(OG, NG, AE)` ids and vtable / RTTI / NiRTTI id tables; DM-only layouts CBRO uses are in `src/Engine/Compat.h`. The former reference trees (CommonLibF4-DM, libxse CommonLibF4, CommonLibF4-NG (alandtse), CommonLibF4-original (Ryan McKenzie), f4se (ianpatt), fo4test (jarari, the Upscaling mod)) were deleted on 2026-09-29 at the user's request; facts below that cite them stand, and the upstream repos can be cloned to a scratch location when a second opinion is needed.

---

## 2. Runtime, loader and ecosystem facts

### 2.1 F4SE 0.6.23 (the OG loader)
- OG F4SE calls only `F4SEPlugin_Query` and `F4SEPlugin_Load`; `F4SEPlugin_Version` is read by the NG/AE loaders (CBRO exports all three itself since v1.26).
- There is **no trampoline interface** on OG F4SE. `F4SE::Init` allocates nothing; a plugin must create its own pool (`F4SE::AllocTrampoline(size)` in CommonLibF4RD falls back to `Trampoline::create`).
- **`F4SE::TaskInterface::AddTask` does not run on the main thread on OG.** Its queue is drained inside F4SE's hook on the engine's message-queue `ProcessTask` (`f4se/Hooks_Threads.cpp`, `MessageQueueProcessTask_Hook`), which the engine runs on its task-pool threads. Five different thread IDs were seen executing such tasks in one session. Only `AddUITask` (the `ProcessEventQueue` hook) runs on the main thread. Console commands (e.g. `tpc`) run on the main thread. Anything that must not race the render/cull code belongs in a render-stage hook or in `AddUITask`.
- F4SE messaging listeners (e.g. post-load-game) run on a message-handler thread, not the main thread.
- F4SE logs land in `%USERPROFILE%\Documents\My Games\Fallout4\F4SE\` even with MO2 profile-local settings.
- `RE::Main::GetSingleton()->threadID` is the main thread; `RE::Main::WorldRootCamera()` is the world `NiCamera`.
- **CommonLibF4RD (since CBRO v1.26):** engine ids resolve through `Data/F4SE/Plugins/f4rd-runtime.bin` (MO2 mod "Runtime Database"; the file is not on the fork's GitHub releases). `REL::ID(OG, NG, AE)`: one argument is the AE id, so OG-only ids must be written `REL::ID{ og, REL::ID::INVALID_ID }` (CBRO: `Engine::OG(id)`); on OG the resolver first tries the AE id in the runtime table, then the OG id. The RD `RE::VTABLE::*` arrays repeat the OG number as the AE id. Its RE headers are the original CommonLibF4 layout: no `BSGraphics::State`, `NiLight`, `TES`, `GridCellArray`, `EXTERIOR_DATA`, `NiCullingProcess` bodies (src/Engine/Compat.h carries the DM layouts). Resolution failures log "F4RD FAIL <id> <reason>"; a `CBRO.trace` file next to the DLL records every id requested.

### 2.2 Hooking rules learned the hard way
- **Never write data into code pages.** Counters placed inside the executable trampoline page, next to hot gateways that every culling thread runs, caused self-/cross-modifying-code machine clears: frame time went 2-3x in every mode (v1.14: 31 FPS instead of 151).
- Function-entry detours: verify the exact prologue bytes first (another plugin may have patched the entry), relocate whole position-independent instructions (`Block::Add` starts `48 89 5C 24 08`, which relocates cleanly), keep a gateway.
- A hook that must be removable at run time can be swapped with one 8-byte compare-exchange (entries are 16-byte aligned) so a thread mid-entry sees one complete version or the other; a hook another plugin stacks on top later makes the swap fail and is never undone.
- A mid-function `jmp rel32` patch must sit inside one 8-byte word (`src % 8` is 0 or 1) to be swappable atomically; replaced instructions are repeated in a stub in the trampoline.
- Call-site wrappers (`write_call` on `E8 rel32`) chain cleanly with other mods' wrappers at the same site (Upscaling wraps five DrawWorld sites; both chains fire).
- **Do not hook `IDXGISwapChain::Present`** in this setup: Upscaling's frame generation presents through a D3D12 proxy swapchain and inserts extra frames.

### 2.3 Other mods and what they do to the engine (Fallout London profile)
| Mod | Effect on the engine / on hooks |
|---|---|
| **Upscaling Custom v1.7 (= jarari/fo4test)** | DLSS SR at render 1280x800, output 2560x1600; DLSS-G frame generation via a D3D12 proxy swapchain; Reflex hooks on `Main` OnIdle/Swap. Installs entry-gateway detours on `DrawWorld::Imagespace`, `DrawWorld::FrameGenerationForward`, `Renderer::Begin`, `AcquireRenderTarget` and wraps `Render_PreUI` +0x17F/+0x1BA/+0x1C9 and `ForwardAlphaImpl` +0x1DC/+0x253. Its pre-pass thunk only overrides sampler states around the chained call. Its log reports ENB "depth logical=1". Later in the frame it **temporarily swaps `depthStencilTargets[2]`'s views for a 2560x1600 texture**, so the main depth must be captured inside the pre-pass hook, not read from `RendererData` later. |
| **ENBSeries 0.501** (`d3d11.dll`, + ENB Helper) | Wraps the D3D11 device and context; `RendererData->context` is ENB's proxy. Adds AO/SSS passes reading depth. Tracks output-merger bindings (so do not unbind the engine's depth target; copy it instead). DXVK logs are stale (`EnableProxyLibrary=false`). |
| **Addictol 1.7** | Its "BSPreCulledObjects" module **replaces the vanilla previs feed**: it calls `Group::Add` itself (Addictol.dll+0xEC9F4/+0xECA75/+0xECE0F/+0xECE86). Do not hook the vanilla feed (997287); anything downstream of `Group::Add` is unaffected. Its crash logger symbolizes with a plugin's PDB. |
| **Shadow Boost FO4** | A frame-time-driven budget manager: target 58.4 FPS, moves shadow distance (2000-4000 observed), LOD fade and grass distances every frame. Any other adaptive controller keyed on frame time will oscillate with it. |
| **FO4CloudShadows** | A compute pre-pass each frame; hooks `ID3D11DeviceContext::Draw*` on ENB's context vtable. |
| **Fake Through Scope** | Hooks `DrawIndexed` on the context vtable; renders a scope view to texture (another camera overwrites `cameraState`). |
| **Dynamic Cubemap F4** | Extra render-to-texture passes (another camera). |
| **Precombine And Previs Guardian** | Stops the engine from dropping a cell's precombines when a mod edits a precombined reference. |
| **Full Body First Person / True Third Person** | Change what is in the depth buffer near the camera. |
| **High FPS Physics Fix, Smooth Cell Loading, Faster Loadscreens** | Change loading-loop and cell-attach timing. |
| xSE PluginPreloader, CKPE | Present; no interaction found. |

MO2 2.5.3 rewrites `profiles\<profile>\modlist.txt` on exit; a DLL-only mod's priority does not matter; launch through MO2's F4SE executable so USVFS mounts `Data\`.

### 2.4 INI settings that matter for visibility and rendering
| Setting | What it does | Evidence |
|---|---|---|
| `[Display] bUsePreCulledObjects` | The previs (pre-culled objects) master switch; setting value ID 1472203 (byte at 0x1438C7890). Console `tpc` (TogglePreCulling) flips the enabled byte. | disassembly, run 3 |
| `[General] bUseCombinedObjects` | Precombines on/off (1 here). Off multiplies draws for everything visible. | exe strings, Phase 0 |
| `[General] bEnableBoundingVolumeOcclusion` | Authored occlusion planes/boxes (1 here). | Phase 0 dump |
| `[General] bCullingBatch` | "GroupsEnabled" (ID 938585 = the `Setting`'s value byte at +8; static initializer at 0x142BFE6B0; default 0; not set in the London INIs). Selects which of two code paths sets up the sun's cascades (see §6). | disassembly |
| `fDirShadowDistance` | INI says 14000 here, but the applied cascade range (global 777729, written by 545406; 1000 if unset) read 4000 at run time (Shadow Boost moves it). | disassembly + log |
| `uGridsToLoad` 5 | 25 exterior cells loaded (5x5). | survey |

Game units: 1 unit is about 1.43 cm; an exterior cell is 4096 units wide.

---

## 3. Threading model

- **`Main::Update` and every `DrawWorld` render stage run on the main thread** (the pre-pass, forward, depth-resolve and first-person-alpha stages all had `tid == Main::threadID`; 0 draws on other threads or deferred contexts). There is no separate D3D11 render thread like Unreal's RHI thread.
- **Culling runs on the main thread in practice**: `Block::Add` was on the main thread for 99.99% of calls (at most 62 per 600 frames elsewhere). Culling groups are processed as BSJobs (`Group::Process`, 626862), so job workers *can* run them; anything on that path must be lock-free.
- Per-object work that touches shared atomics from all those threads is measurably slow (cache-line bouncing); per-thread counters summed at log time are not.
- Cross-session frame times at the same spot vary by 6-20 ms in both modes (time of day, weather, Shadow Boost state). Only a same-session, same-view comparison means anything.

---

## 4. The frame: DrawWorld render stages and draw budgets

### 4.1 `DrawWorld::Render_PreUI` (ID 984743), in call order
| Offset | Call | Notes |
|---|---|---|
| +0x0 | fixes the directional-shadows byte (ID 241042) for the frame: `flag = byte 0x146721F58 ? 0 : flag` | that byte's only setter (0x142857980) runs from the one-time render setup 0x140D3DA57, which also fills DrawWorld's callback slots |
| +0x16A | **DrawWorld cull** (718911) | the main camera's visibility pass (§5) |
| +0x16F | `call [0x1467232C0]` | unknown |
| +0x175 | `call 339369` | batched sun-cascade setup (§6.1) |
| +0x17F | **deferred opaque pre-pass** | the world's depth is complete after this call; fo4test's `DeferredPrePass` site |
| +0x1BA | HBAO | never called in this setup (HBAO off) |
| +0x1BF | `call 1108521` | unbatched sun-cascade setup (§6.1) |
| +0x1C9 | forward | |
Related: `ForwardAlphaImpl` (338205) +0x1DC post-resolve-depth, +0x253 first-person alpha; `DrawWorld::Imagespace` (587723); `DrawWorld::FrameGenerationForward` (656535, hooked by Upscaling); `DrawWorld::DeferredComposite` (728427). Main's render function (408683) calls 502840 right before `Render_PreUI`, which calls `SetFrustum` on DrawWorld's two cullers. DrawWorld init (1570173) builds those two `BSGeometryListCullingProcess` objects and an `NiCamera`. `Interface3D::Renderer::Create` is 88488. (OG/AE pairs from fo4test: `Render_PreUI` {984743, 2318321}, `Imagespace` {587723, 2318322}.)

### 4.2 First-person geometry and the depth buffer
- The first-person weapon **is drawn into the main depth buffer**, with the viewport depth range switched to **[0, 0.01]**; the world uses **[0.01, 1]**. 6-8 first-person-range draws per frame happen outside the wrapped stages, 0-1 inside the pre-pass. Any depth-based test must treat `d < 0.01` as "no world surface here".
- **Standard Z (0 = near, 1 = far), not reversed.** fo4test's linear-depth shader is `lin = f*n / (f - d*(f-n))`; ReShade's FO4 preset is `IS_REVERSED=0`.
- Linear view depth from the render camera's own numbers: `ndc = (d - 0.01) / 0.99`, `z = depthB / (ndc - depthA)` with `depthA`, `depthB` derived from the view-projection (see §4.4); `d >= 0.999999` or `ndc >= depthA` is sky/infinite.

### 4.3 Draw-call budget (immediate draws per frame, this setup)
- Daytime exterior, previs on: 900-2,900 draws; the pre-pass 330-1,030; the **sun's shadow cascades 500-1,900 (55-65% of all draws)**, drawn outside the wrapped stages into `depthStencilTargets[8]`, a 2560x2560 3-slice array.
- Same, previs off: pre-pass 6,100-6,900 (steady), shadow draws unchanged (**previs trims only the main camera**).
- Dense pier at night: ~3.5k main draws and ~13.6k point-light shadow-map draws per frame (about 80% of draws are lamp shadow maps; one shadow accumulator per lamp, 7-11 lamps per frame).
- Each spot has a floor of fixed passes (post-processing, ENB, upscaling, UI, lamps' and cascades' fixed costs) that visibility cannot lower; with almost the whole scene culled the heavy spot still ran at 13.7 ms.

### 4.4 Cameras and projection
- `BSGraphics::State` (singleton `VariantID{600795, 2704621}`) holds `cameraState` and `cameraDataCache` (a `BSTArray<CameraStateData>`, 0x250 bytes per entry). **`cameraState` after the pre-pass is often another pass's camera** (a 37 deg FOV / near-15 scope camera, cubemap cameras). The world camera's entry is the one whose `referenceCamera == Main::WorldRootCamera()`; prefer the entry with `useJitter == false`. `camViewData` has `viewProjUnjittered`, `viewDir`, `viewRight`, `viewUp`; `posAdjust` is the camera-relative offset.
- Convention verified by a self-check (a point ahead of the camera lands at the screen centre with the right/up signs correct): **`clip = (p - posAdjust) * viewProjUnjittered`, row-vector, camera-relative.**
- The view-projection decomposes exactly into eye + orthonormal basis + per-axis scales (`ndc.x = scaleX * x/z`, `ndc.y = scaleY * y/z`) + a depth curve (`ndc.z = depthA + depthB / z`, `depthB < 0` for standard Z); verified on an off-axis point each capture.
- TAA/DLSS jitter is sub-pixel; use the unjittered matrix. The jitter still moves silhouette texels between an occluder's depth and the background from frame to frame.
- Camera near/far globals: `REL::ID{57985, 2712882}` and `{958877, 2712883}` (fo4test).
- `NiCamera::viewFrustum` is at **+0x160**: `left, right, top, bottom, near, far` (floats) then `ortho` (bool). Vertical FOV is about 50 degrees at the default settings.
- `NiCamera::world.rotate` (rows vs columns) maps onto the render basis with an ambiguity: facing within ~5.7 degrees of a world axis both readings fit. The render camera can trail the NiCamera by a frame while turning.
- The rotation angle between two orientations is best computed as `2*asin(||R1 - R2||_F / (2*sqrt(2)))` (exact 0 for an unchanged matrix; `acos(trace)` reads ~0.03 degrees of noise).
- **The camera sways every frame even when the player stands still** (sub-unit translation, hundredths of a degree); a "still camera" test needs a tolerance (2 units / 0.06 degrees per frame worked). Frames with a large camera jump (fast travel, load) show as >1 s frames.
- Dynamic resolution was off (ratio 1.0); under Upscaling the depth is rendered at 1280x800 in a 1280x800 texture.

### 4.5 Render targets (`BSGraphics::RendererData`, `RE::BSGraphics::GetRendererData()`)
| Target | Facts |
|---|---|
| `depthStencilTargets[2]` (`kMain` in fo4test's enum) | The main depth in this setup (= Upscaling's "logical 1"): 1280x800 `R24G8_TYPELESS`, DSV `D24_UNORM_S8_UINT`, SRV `R24_UNORM_X8_TYPELESS`. ~99% of pre-pass draws use `depth[2].dsView[0]`. Under ENB + Upscaling the index is not guaranteed: identify the bound DSV with `OMGetRenderTargets(0, nullptr, &dsv)` inside the pre-pass hook and match `dsv->GetResource()` against `depthStencilTargets[i].texture`. A bound depth target cannot be read by a shader, but `CopyResource` into a private texture of the same typeless format works without unbinding it. |
| `depthStencilTargets[8]` | The sun's cascade array: 2560x2560, 3 slices. |
| `renderTargets[39]` (`kMainDepthMips`) | 1280x800 `R32_FLOAT`, **11 mips**, SRV\|RTV\|UAV. Contents (raw or linear, min or max) **not verified**. |
| `RendererData->device / context` | The engine's device and immediate context; with ENB active the context is ENB's proxy (so `Draw*` counted there are the game's submissions). |
Compute work between engine passes must save/restore the CS bindings it touches; a staging ring mapped with `D3D11_MAP_FLAG_DO_NOT_WAIT` never stalls (skip the frame on `DXGI_ERROR_WAS_STILL_DRAWING`); after a hitch several readbacks complete at once. A mapped staging buffer can stay mapped and be read from any thread until `Unmap` (CBRO v1.25 publishes the Hi-Z straight out of the mapping). Feature level 11.0 guarantees typed UAV loads only for single-component R32 formats: `RWTexture2D<float2>` reads are not portable, use two R32_FLOAT textures or a raw buffer.

---

## 5. How the main camera is culled (disassembly, verified by runs)

### 5.1 It is not the `NiCullingProcess` virtuals
`NiCullingProcess` vtable (slots verified against the bytes): 0x18 deleting destructor, **0x19 `Process(NiAVObject*)`** (reads `obj+0xBC` worldBound radius and `obj+0x108` flags), **0x1A `Process(const NiCamera*, NiAVObject*, NiVisibleArray*)`**, 0x1B `AppendVirtual(BSGeometry*)` (BS classes), 0x1C `AppendNonAccum(NiAVObject*)`, 0x1D `TestBaseVisibility(BSMultiBound*)` (calls `bound->shape->WithinFrustum(planes)`), 0x1E `TestBaseVisibility(BSOcclusionPlane*)`, 0x1F `TestBaseVisibility(const NiBound*)` (reads `bCustomCullPlanes` at +0x11F, planes at +0x3C / +0xAC). Vtable IDs: `NiCullingProcess` 547317, `BSCullingProcess` 556243, `BSGeometryListCullingProcess` 1398386, `BSParabolicCullingProcess` 845854, `BSFadeNodeCuller` 1334420. Constructors: `BSCullingProcess` 423200, `BSGeometryListCullingProcess` 125924, `BSParabolicCullingProcess` 1288195. `BSCullingProcess::cullMode` is at +0x158 (kNormal 0, kAllPass 1, kAllFail 2, kIgnoreMultiBounds 3, kForceMultiBoundsNoUpdate 4); `updateAccumulateFlag` at +0x11D; `OnVisible` is vtable +0x1C8; the accumulator restart is vfunc +0x140; the culler's finish loop appends visible entries through vfunc +0x168.

Measured traffic on those virtuals: the main camera generates **zero** calls. `BSParabolicCullingProcess` handles ~8k objects per frame for lamp shadows (§6.2); `BSFadeNodeCuller` makes 11 calls per frame on the WorldRoot camera and only tests one fade node's sphere against six planes and flips flag bit 39; `BSCullingProcess` ~80 calls per interval (special cases). `__MainCullingCamera` (`(anonymous)::MainCullingCamera`, 708657) is a small interface vtable.

### 5.2 The real pipeline: DrawWorld cull, scene walk, culling groups, blocks, jobs, accumulators
1. **DrawWorld cull** (718911, from `Render_PreUI`+0x16A) resets its two `BSGeometryListCullingProcess` objects (globals 865470 / 1084947), initializes its culling groups with DrawWorld's camera (global 81406), and calls the **scene walk** (1138818) on the scene root (global 1327069, which is also the `ShadowSceneNode` the batched cascade path reads its light from) with `(culler, group 0, group 1, group 2, &group array, previsActive)`.
2. The walk descends the top of the graph and calls **`Group::Add(group, obj, &obj->worldBound, flags)`** (1175493).
3. A **culling group** is a non-polymorphic **0x170-byte struct with no RTTI**: six frustum planes at +0x00, a node block list at +0x98 (current block +0xD0), a geometry block list at +0x118 (current +0x150), an owner at +0x158, a flag byte at +0x16A. DrawWorld's groups: **group 0 = 1117782 (shadow casters: NiAVObject flags bit 40 clear)**, **group 1 = 133326 (non-casters)**, **group 2 = 1328670 (the previs visible list)**, plus a **group array** at 459440 (`{group* data; ...; u32 count @+0x10}`). All share one owner value. **Group 3 = 731482** holds special always-draw objects with radius-1 bounds and its own camera, registers with a second accumulator (1430301) and is reprocessed by a later pass (0x142850A27). A stack-allocated group with a null owner gets ~1,000 adds per frame from the previs code earlier in the frame (probably a shadow pass).
4. `Group::Add` sorts the object into a **block** through **`Block::Add(block, obj, bound, startIndex)`** (1143206; -1 = fresh add, >= 0 continues a multi-instance mesh). A block is **0x3A70 bytes**: the group's planes copied in; **SoA bound packets of four at +0x60** (`x[4] y[4] z[4] r[4]`); objects at +0x2060; per-entry **result byte at +0x3060 + 5*i** and **force-visible byte at +0x3061 + 5*i**; group markers at +0x3A6C..+0x3A6E; the owner copied to +0x3A60; the group's +0x16A flag copied to +0x3A6F; count at +0x3A68 (max 0x200 entries). Block::Add checks `BSPreCulledObjects::IsActive` (0x142809E30): **with previs active and the block's +0x3A6F flag 0, main-pass entries are marked force-visible and merge-instanced entries are not expanded**, and the finish loop (0x1CCEBF0) appends them regardless of the frustum result.
5. **`Group::Process`** (626862) submits the blocks as **BSJobs**; each job runs the SIMD frustum test over the SoA bounds. Job workers push a visible node's children back in through **`ChildPush`** (357475: `(group, obj, bound, r9b flag, byte, byte, context)`, one object per call; the loop over the children is in its callers) → `Block::Add`.
6. Each view then **registers** its visible geometry with its own accumulator: **`BSShaderAccumulator::RegisterObject(BSGeometry*)`, vtable 357329 slot 45** (0x14282CED0; called only through the vtable; the RTTI shows no subclass; all four register loops ignore its result). DrawWorld's main accumulator is the global **1211381**. With a camera on the accumulator (+0x10; the main one carries DrawWorld's camera) the entry is re-tested against that camera first (0x141CC4870). `BSShaderAccumulator::renderMode` is at **+0x560** (`BSShaderManager::etRenderMode`: 24 = the main "deferred gbuffer" pass, 0xF kShadowmap, **0x10 kShadowmapDir** (the four sun cascades), **0x11 kShadowmapPB** (one per lamp)), `shadowLight` at +0x568. Groups 1, 2 and the array register with the main accumulator only; **group 0 is also registered by the sun's cascades** (§6.1) through a view manager (0x141CD0880; 0x68-byte records `{group, camera, planes, accumulator}`; registration 0x141CCB8C0).

### 5.3 The scene walk (1138818) in detail, exterior path with previs off
- The exterior path runs when byte `[[culler+0x150]+0x138]` is set and the override-root global 0x146723208 is null. The **world node is the scene root's child 3**; each of its children from index 1 on is a **cell node** (the walk sets its flag bit 42). Of a cell node's children only **2, 3 and 9** are visited: child 2 → its children go whole into group 1 (path B, +0x4C0..+0x4F9, ~104 adds per frame); **child 3 = the cell's precombined chunks and static references**; child 9's role is unknown (never touched). The world node's child 0 goes whole into group 1; the tail adds the scene root's children 6, 7 and 10 to group 1.
- For node 3 (and 9): loop start at +0x5B8 (`cmp dx, word [rax+132h]`: the child count is a word at node+0x132; node in r14, index in r15d, cell in r12, other state in rbp/rbx/rdi/r13). For each child, +0x5F1 fetches its RTTI (`mov rax,[rsi]; mov rcx,rsi; call [rax+10h]`) and compares it with **`NiNode`'s NiRTTI (ID 191219 = 0x145C08EC0)**: an **exact `NiNode`** child ("container") has its own children added to group 0 one by one with flags 0 (loop +0x602..+0x633, Group::Add return address Fallout4.exe+0x284E581: **~5,700 adds per frame**, the main feeder); anything else is added whole (+0x635, return address +0x284E59A: ~1,340 per frame). The exact-NiNode containers under node 3 are the **precombined NIF roots**, each holding about ten `BSFadeNode` chunks. `BSMultiBoundNode`'s NiRTTI is 0x14609A070.
- The override-root path (global 0x146723208 non-null) has the same shape at +0xA00 (node in r15, index in r12d) and +0xA31 (container check with `cmp rax, rbx`). The portal/room path (`[root+0x238]`, byte `[[culler+0x150]+0x139]`) exists for interiors and was left alone.
- DrawWorld's cull also has its **own loop over the roots registered with the culling camera** (`[[culler+0x150]+0x10]+0x58`, count at +0x68; 0x14284F61C-0x14284F705, loop test at 718911+0x309): for each root that is a node and not flag-14 special it adds every child to group 0 or group 1 by the shadow-caster flag (~13 adds per frame; return address +0x284F6E5).
- **With previs on none of this expansion runs**: the cell multibound roots go whole into group 1 and group 2 receives the previs visible list. Per frame at a light exterior: previs on ~300-1,000 main-pass `Block::Add` entries and ~830 objects processed in the cull stage (0.4 ms); previs off 9,000-11,700 entries, ~7,200 top-level `Group::Add`s, ~9,700 objects (1.8-3.2 ms).
- Grouping nodes (cell nodes and containers) have flags `0x80000000280E` (bit 11 always-draw set) and **an unmaintained `worldBound` (radius 1)**. The always-draw bit on such a node means "never cull me by my bound"; only the bounds of the entries the engine files (their children) are maintained.

### 5.4 Merge-instanced geometry in the blocks
- A `BSGeometry` of **type byte 0x0F** ("merge-instanced", not the `BSMergeInstancedTriShape` class) has an instance array at obj+0x1C0 (stride 0xF0, world `NiBound` at +0xC0 of each record) and a count at obj+0x1D0. `Block::Add` writes the mesh's own entry and then one entry per instance until the block is full; the caller continues in a new block with `startIndex` = the next instance. The finish loop ignores the mesh's own entry and appends the whole mesh if any instance entry passed. Such entries were never seen in the main pass with previs off.
- **`BSMergeInstancedTriShape`** (vtable 245837, size 0x1B0) is different: the precombine builder (657056 = 0x1428444A0) concatenates several source meshes' vertices *in their own local spaces* into one vertex buffer and stores each instance's `NiTransform` (translation relative to the group centre, which becomes the node's local translate) in a GPU buffer at +0x170 (80-byte records: `NiTransform` at +0, grayscale-palette scale at +0x40; count at +0x34, kind at +0x38; LOD triangle counts at +0x1A0). **The CPU copy of those records is freed**, so the raw vertex buffer says nothing about where the instances are drawn. 16-28% of precombined leaf shapes are of this class; 85 of 218 kept meshes at one wall.

### 5.5 Previs (`BSPreCulledObjects`) internals
- `IsActive()` (917969, inlined copy at 0x142809E30) = `enabled (byte 0x146722288, ID 493183) && INI (0x1438C7890, ID 1472203) && !suspended (0x146722289, ID 718924)`; queried in 35 places.
- `SetEnabled(bool)` (1090712): **enable is only a byte write; disable runs the flush** (61939), which calls the callbacks in two registries (0x1438C78F8 keyed by reference form ID, 0x1438C7928) with `true`. `SetSuspended(bool, flush)` (1263609). The game suspends previs itself in some modes (callers 0xE9E742 and inside the cascade cull 1390075). `IsEnabled` is 652211. Console: `tpc`.
- The previs feed into DrawWorld group 2 is 997287 (replaced by Addictol when present). Other previs classes seen: `BSPrecomputedVisibility::BSVisDB` (the `.uvd` loader), `MultiCellVisibilityData`, `BGSObjectVisibilityManager`, `BSPreCulledObjects::ObjectRecord{obj, flags}`, `ExtraCellPrevisRefs`, `TES::UpdateMultiBoundVisibility` {1281872, 2192134}. `TESObjectCELL` has `visibilityData`, `rootVisibilityCellID`, `visCalcDate`, `preCombineDate`.
- Previs marks hidden objects with flag bit 26 (pre-processed) and bit 39 (not visible); the lamp culler (§6.2) honours those bits, so **previs also trims lamp shadow casters** while it leaves the sun's cascades to their own path.
- Toggling previs off does not leave it degraded (enable is a byte write), and switching it on the main thread with nothing reading previs data (right before DrawWorld's cull) is safe; switching it from a worker thread races the cull.
- Previs is position-only and conservative over a cell cluster; pre-generated previs can wrongly hide objects a mod added or moved.

---

## 6. Shadows

### 6.1 The sun (directional cascades)
- **`BSShadowDirectionalLight`** (vtable 97945). Its **slot 14 update** (1242204 = 0x1428CACC0) normalizes **row 0 of its `NiLight`'s world rotation** (the light is at +0xB8 of the shadow light; `world.rotate` is NiAVObject +0x70), eases direction changes over time (state at +0x200/+0x210/+0x220), and builds its **shadow camera (+0x2C0)** with local rotation row 0 = that direction, **placed at `view camera - 15000 * direction`** (constant 0x14309DD00). So the shadow camera's rotation row 0 is the light's travel direction, and `(view - cameraPos) . row0 == +15000` verifies the sign at run time. The direction points down (z < 0) whenever the sun is up; observed `(-0.57, 0.62, -0.54)`.
- The cascade range is the global 777729 (0x1467333DC): `fDirShadowDistance` as applied (1000 when unset), written by 545406; slot 14 clamps the cascade far to `min(camera far, range)`.
- **Cascade cull: 1390075 (light, group 0).** With previs off it points group 0's camera and frustum at the sun's culler and **registers every cascade as another view of group 0** (four `kShadowmapDir` accumulators); with previs on it builds its own stack group from previs data instead. **So anything removed from group 0 also disappears from the sun's shadow maps** (hidden or off-screen casters lose their shadows).
- Two paths select on `bCullingBatch` (ID 938585, read as `movzx eax, byte [0x14384E778]` at 0x141CCB850) and both require the directional-shadows byte (241042):
  - batched (1): `Render_PreUI`+0x175 → 339369, which at +0x185 calls 432406 (runs the light's slot-14 update, then tail-jumps to 1390075 with group 0, `lea rdx, [0x1467233C0]`); light from `ShadowSceneNode` 1327069 +0x208.
  - unbatched (0, the London default): `Render_PreUI`+0x1BF → 1108521, which at +0xE49 calls 1390075 after `light->update(camera)` when `dirShadows && !bCullingBatch`; light from `ShadowSceneNode` 879298 +0x208.
  - 259940 is 432406's code with no caller. DrawWorld's callback slot (global 473367) holds a no-op (`ret` at 0x1403A8020, set through 0x1428579B0 from 0x140D3DA9A), so hooking it sees nothing.
- With the directional-shadows byte clear (night, interiors) neither path runs and nothing but the main camera reads group 0. The byte is fixed at `Render_PreUI`+0x0, before the cull, so whatever it says at cull time holds for the cascades that frame.
- Cascade cost: 500-1,900 draws per frame in a daytime exterior, into `depthStencilTargets[8]`. With previs on the cascades draw only previs's (small) shadow list; with previs off they draw every group-0 caster in range, including everything behind the walls in front of the camera.
- Shadow maps filter over a few texels; the direction eases over time; cascade splits run along view depth.

### 6.2 Point and spot lights ("lamps", paraboloid shadow maps)
- `BSShadowParabolicLight` vtable slot 9 (656208) culls a lamp's casters through `BSParabolicCullingProcess::Process(camera, scene, set)` (vtable 845854 slot 0x1A): sets the camera and frustum, restarts the accumulator (vfunc +0x140), then `scene->Cull`. Before that, its setup `0x1429775C0(culler, camera, ?, radius, flag)` (called at +0x190 with `radius = [light+0xB8]->+0x138`, i.e. `NiLight::spec.r`) puts the **light's culling sphere on the culler: centre at +0x1C0 (= the shadow camera's world translate), radius at +0x1CC (also +0x1A0)**. The shadow camera's own frustum far is 1 (the paraboloid projection is normalized), so it says nothing about the reach.
- Per-object `Process` (slot 0x19, 0x1429772B0): `cullMode` 1/2 pass/fail everything; **flags bit 11 (always-draw) passes without a test; bit 26 (pre-processed) is decided by bit 39 (not-visible)**; otherwise its frustum test (0x142977950), then `OnVisible` (vtable +0x1C8), which files geometry or walks a node's children; a failing object drops its subtree and, when culler byte +0x11D is set, has its bit 42 (accumulated) cleared.
- Lamp culls run **every frame per lamp** (the count of culls matches the lamp count), so no lamp shadow map outlives the view it was culled for. Shadow cameras are **pooled across lights** (the camera pointer is not a light's identity; position + reach is). Observed reach 742 units for a street lamp with `worldBound` radius 163-296. About 8k objects per frame pass through these cullers at a light exterior; at a dense night scene 11 lamps produce ~13.6k shadow draws and 15-20k registrations per frame; with previs off every object within a lamp's reach casts, previs on trims them via bits 26/39.
- Geometric fact used: for a lamp inside the view pyramid, a caster entirely outside the (slightly widened) pyramid cannot shadow a visible pixel, because the lamp-to-pixel segment stays inside the convex pyramid. Shadow filtering reaches about 5 degrees seen from the light.

### 6.3 Lights in general
- Lights live in the scene's light list, not in the culling groups: with previs off, lights never pass through the main-camera cull blocks. `NiPointLight::worldBound` radius ranged 42-4,793; `NiLight::modelBound` (+0x150) is either -1 or equal to the world bound; the actual radius is **`NiLight::spec.r` at +0x138**, often far larger than the bounds.
- Hiding a node drops the lights beneath it from lighting visible surfaces. The engine does not appear to have a "light budget" that previs protects: an earlier theory that switching previs off overran a renderer light budget was disproven (the lighting breakage came from wrong-camera rejections).

---

## 7. Scene-graph object layouts (Gamebryo / Bethesda classes)

### 7.1 `NiAVObject`
| Offset | Field | Source |
|---|---|---|
| +0x30 | `local.rotate` (3x3, row 0 first), +0x60 `local.translate`, then scale | CommonLib layout, used for the shadow camera |
| +0x70 | `world.rotate`, +0xA0 `world.translate` | same |
| +0xB0 | `worldBound` (`NiBound`: centre, radius at +0xBC) | CommonLib, confirmed by `Process(obj)` reading +0xBC |
| +0xC0 | `previousWorld` (`UpdateWorldData` copies `world` into it before recomputing; an object that moved in its last update has the two differ; a never-set one has scale 0 or the identity) | header + observation |
| +0x108 | `flags` (u64) | confirmed by `Process(obj)` reading +0x108 |
| `userData` | for a reference's 3D root, the `TESObjectREFR` (form type `kACHR` = an actor) | header |
| node +0x132 | child count (word) as the scene walk reads it | disassembly |

Flag bits (F4SE `NiObjects.h` names): **11 AlwaysDraw** (Block::Add forces such entries visible; the lamp culler passes them untested; on grouping nodes it marks an unmaintained bound), 14 (a "special" root DrawWorld's root loop skips), **26 PreProcessedNode**, **39 NotVisible** (previs / fade culler), **40 ShadowCaster** (CommonLib's `ShadowCaster()` tests it *clear*), **42 Accumulated** (set by the walk on cell nodes; cleared on lamp-cull failure).

Object types met in exteriors: `NiNode`, `BSFadeNode` (each precombined chunk is one, with a single shape child), `BSLeafAnimNode` (swaying leaves), `BSMultiBoundNode` (cell roots), `BSTriShape`, `BSMeshLODTriShape`, `BSMergeInstancedTriShape`, `BSCombinedTriShape`, `BSLODTriShape`, `BSSubIndexTriShape` (segmented: object LOD and most actor parts), `BSDynamicTriShape` (actor parts), `BSMultiStreamInstanceTriShape` (instance entries use their own bounds), `NiPointLight`, `NiSpotLight`, particles. Big "near" meshes: `Land` (radius ~93k), `obj`/`obj-at` object LOD (12-24k), clouds.

### 7.2 `BSGeometry` / `BSTriShape` / GPU data
| Offset | Field |
|---|---|
| +0x120 | `modelBound` (local `NiBound`) |
| +0x138 | `properties[1]`: the shader property (`BSLightingShaderProperty` vtable 241915); its `flags` at +0x30 (bits: 1 skinned, 25 tessellate, 45 billboard, 61 tree anim: shaders that move vertices) |
| +0x140 | `skinInstance` |
| +0x148 | `rendererData` → `BSGraphics::TriShape` |
| +0x160 | `numTriangles`, +0x164 `numVertices` (u16) |
`BSGraphics::TriShape`: `vertexDesc` +0, `vertexBuffer` +8, `indexBuffer` +0x10. `BSGraphics::Buffer`: `ID3D11Buffer*` +0, CPU `data` +8, `dataSize` +0x34, `dataOffset` +0x48. `vertexDesc`: stride = `(desc & 0xF) * 4`, position offset = `(desc >> 2) & 0x3C`, flags = `desc >> 44` (`VF_FULLPREC` = bit 10 → float xyzw, else half xyzw). **Nearly every mesh keeps a CPU copy of its vertices** (1,965 of 1,986 in one dump; `dataSize == numVertices * stride`). The engine makes an object's world bound by moving the model bound with the world transform (`worldBound.r == modelBound.r * scale` for plain meshes).

### 7.3 Cells, references, precombines
- `TES::GetSingleton()->gridCells / interiorCell`, `TESObjectCELL::references`, `LOADED_CELL_DATA::combinedObjects` at +0x1E0 holds the precombined chunks (survey: 5,738 chunks in 25 London cells; chunk bound radius mean 452 units, max 2,145, 97% under 2,048; plus 1,020 non-precombined references with their own 3D, mean radius 229). **Precombined references have no 3D of their own** (read the 3D field, do not call `Get3D()` expecting one).
- Fallout London ships 135,620 precombined NIFs and 888 previs `.uvd` files (`LondonWorldSpace - MeshesExtra.ba2`); vanilla 124,871 and 966. Precombines are draw-call batching per material, not visibility.
- Authored occlusion: `BSMultiBoundRoom`, `BSPortalGraph`, occlusion planes/boxes (`bEnableBoundingVolumeOcclusion`).
- Actors: the engine updates actor bounds every frame; an NPC walking under a node can leave the node's bound stale for a frame; hiding an actor's parts from the main camera only (not from the shadow views) does not break its shadow.

---

## 8. Engine behaviour observed at run time (numbers worth remembering)
- Light exterior, standing still: previs on 6.5 ms (cull stage 0.4-0.5 ms, ~900 pre-pass draws); previs off with no culling ~6,100-6,900 pre-pass draws.
- The main-pass cull stage with previs off is dominated by the engine filing and frustum-testing ~9,700 objects (~70-150 ns per object of any extra per-object work on the main thread adds up to milliseconds).
- Per-frame registrations at a light exterior: main view 1,560-1,660; the four sun cascades 1,436 + 1,346 + 451 + 121 (previs on: main 1,253, cascades 2,801 in total).
- Sky texels are cleared to the far plane (1.0) in the depth buffer.
- Loading, fast travel and cell transitions show as frames longer than 1 s; the engine's scene structure after a load is rebuilt (object addresses are reused).
- Hi-Z built from the pre-pass depth at 1/4 resolution (320x200) is enough to hide ~85% of the previs-off main pass at a wall; sphere bounds of large chunks are what stays ("near" objects whose spheres reach the eye plane).

---

## 9. Leads and open questions
- `renderTargets[39]` (`kMainDepthMips`, 11 mips): what it contains and when it is built.
- The role of a cell node's child 9, of DrawWorld's `call [0x1467232C0]` at +0x16F, and of group 3 / its second accumulator and later pass.
- `BSPackedCombinedSharedGeomDataExtra`: per-object bounds inside a precombined chunk (a way to split cell-sized chunks).
- The portal/room path of the scene walk (interiors) and the override-root path were not exercised.
- Whether any water-reflection or cubemap pass reads DrawWorld's groups (none showed up in the per-accumulator registration counts; all extra views were the cascades and the lamps).
- NG (1.10.984) and AE (1.11.x) equivalents of every ID above are unverified; only the fo4test pairs listed in §4.1 carry AE IDs.

### 5.5a Previs enable/disable, read from the code (2026-09-29)
- `SetEnabled(bool)` (1090712, 0x142809E70): writes the enabled byte (0x146722288); with `false`, or with the INI byte (0x1438C7890) clear, or while suspended (0x146722289), it tail-jumps to the flush (61939, 0x14280AAB0). With `true` and not suspended it returns: **re-enabling is only the byte write.**
- `SetSuspended(bool suspend, bool flush)` (1263609, 0x142809EB0): writes the suspended byte; flushes only when `flush` and (suspending, or previs not enabled/INI off). Unsuspending with previs enabled never flushes. The cascade cull (1390075 +0x1AA) suspends without flush around its own group feed (0x14280A160); the workshop code (0x140E9E742) calls `SetSuspended(!x, true)`.
- The flush (0x14280AAB0): under the lock at 0x1467222F0 it walks two hash registries of 0x20-byte entries `{u32 key, fn, ctx, next}` (registry 1 at 0x1438C78F8, count 0x1438C78DC, keyed by reference form id; registry 2 at 0x1438C7928) and calls each `fn(ctx, true)`. Registry 1 entries are added by 350676 (0x142809310) from cell code (0x14039CF00, 0x1403A1974), removed by 136788 (0x142809630) from 0x1403A2042, re-keyed by 1041766 (0x142809410) from 0x1400FC5F0 (a form-id change). The console `tpc` handler is 189985 (0x14052CAC0): `SetEnabled(!IsEnabled())`.
- The vanilla visible list: a `BSTArray<u32>` at 0x1467222A8 (count 0x1467222B8, lock 0x1467222A0), appended by 1225481 (0x142809AA0, no direct caller in the exe: filled through callbacks or by Addictol), cleared by 1131946 (0x142809B10). The vanilla feed 997287 (0x142809EE0) walks it each frame: for each id it tries load-order indices, looks the object up (0x14280A5E0), and `Group::Add`s it to the caster or non-caster group by flag bit 40, then calls 1167260 (0x14280A690, form id). Addictol's "BSPreCulledObjects" module replaces this feed.
- Observed (CBRO v1.28 log): after CBRO's `SetEnabled(false)` at load and `SetEnabled(true)` on F8, "previs mode" registers 3,270 main-view objects a frame with a 3.4 ms GPU pre-pass, versus 2,940 / 2.4 ms in CBRO mode: previs culls nothing after re-enable until something re-applies it (presumably a cell change). Not yet found: the function that (re)builds the visible list and marks hidden objects on a cell change. `TES::UpdateMultiBoundVisibility` (1281872, 0x1400FE130) has one caller, 0x140D39A60 (1013897) from 0x140D39450.
