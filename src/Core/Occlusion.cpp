#include "Core/Occlusion.h"

#include "Core/DropSet.h"
#include "Core/MeshProxy.h"
#include "Hooks/CullGroups.h"
#include "Settings.h"
#include "Util/Gamebryo.h"

#include <intrin.h>

namespace CBRO::Core::Occlusion
{
	namespace
	{
		// ---- tunables (copied from Settings once; read on the hot path) --------------------

		struct Tunables
		{
			std::uint32_t confirmFrames{ 2 };
			float         nearDistance{ 48.0f };
			float         depthTolerance{ 0.002f };  // relative, on linear view depth
			float         depthSlack{ 4.0f };        // absolute, game units
			float         depthMin{ 0.01f };
			float         depthRange{ 0.99f };
			bool          observeOnly{ false };
			bool          cullActors{ false };
			bool          meshShapes{ true };
		};
		Tunables g_tunables;

		// A Hi-Z texel that isn't clearly nearer is re-checked at up to this many finer levels.
		constexpr std::uint32_t kRefineLevels = 3;
		// ... and when a cached hidden verdict's evidence is re-checked against a newer depth (ViewValid): fewer, since a
		// failure only means the full test runs.
		constexpr std::uint32_t kRecheckRefine = 2;
		// A cached kept (drawn) verdict is re-evaluated every this many frames (staggered by object), never sooner: no
		// depth change can make drawing an object unsafe, so this only bounds how long a newly hidden one stays drawn.
		constexpr std::uint32_t kKeptRecheckFrames = 16;
		// Meshes at least this big (bound radius) get a shape test when their sphere can't be settled.
		constexpr float kShapeMinRadius = 200.0f;
		// Merged meshes with more instances than this are only judged by their whole bound.
		constexpr std::uint32_t kMaxInstanceTests = 1024;

		std::atomic<bool>           g_active{ false };
		std::atomic<bool>           g_observeOverride{ false };
		std::array<FrameContext, 2> g_contexts{};
		std::atomic<int>            g_context{ -1 };
		std::atomic<bool>           g_resetRequested{ false };

		// Classification is redone every epoch so lights attached after the first look are noticed.
		constexpr std::uint32_t kEpochFrames = 600;

		// ---- stats: one block per thread, single writer, summed at log time -------------------
		// The tests run on the main thread and every BSJobs worker; shared atomic counters would
		// bounce one cache line between all of them for every object.

		enum Counter : std::size_t
		{
			kTested,
			kRejected,
			kWouldReject,
			kConfirming,
			kVisible,
			kEdge,
			kOutside,
			kBehind,
			kNear,
			kInvalid,
			kExemptType,
			kExemptLight,
			kExemptActor,
			kLightsTested,
			kLightsRejected,
			kNoContext,
			kTableFull,
			kMerged,
			kMergedRejected,
			kMergedConfirming,
			kInstanceTests,
			kInstanceEntries,
			kInstanceEntriesRejected,
			kFramesCulling,
			kFramesBlocked,
			kSkippedHidden,
			kSkippedOutside,
			kShapeTests,
			kShapeHidden,
			kShapeOutside,
			kShapeCells,
			kShapeMoving,
			kShapeAnimated,
			kShapeNoTransform,
			kShared,             // entries of an unknown group or group 0 offered to the filter
			kDropInherited,      // shared entries dropped with their parent (no test)
			kDropFull,           // drops the drop table had no room for (the object stays drawn)
			kRegistered,         // main-view registrations while DrawWorld culled
			kRegistrationsDropped,
			kSunTests,           // group-0 objects the main view doesn't need, whose sun shadow was tested
			kSunOutside,         // ... shadow can't reach the view (or the cascade range)
			kSunHidden,          // ... shadow only falls behind visible surfaces (confirmed)
			kSunConfirming,
			kSunNeeded,
			kSunOff,             // ... sun shadows off this frame: nothing to test
			kCasterRejected,     // group-0 entries rejected in every view (neither the main camera nor the sun needs them)
			kCasterSkipped,      // group-0 top-level objects never filed, for the same reason
			kCellNodesSeen,       // cells' child node 3 (precombined chunks + static refs) offered by the scene walk
			kCellNodesSkipped,    // ... left out whole: outside the view, sun shadow can't reach it
			kCellNodesInView,     // ... walked: the node's bound meets the view
			kCellNodesSunNeeded,  // ... walked: outside the view, but its sun shadow may reach the view
			kCellNodesOther,      // cells' child node 9 offered (never pruned: its role is unknown)
			kRootsSeen,           // roots registered with DrawWorld's cull, offered by its root loop (pruned like node 3)
			kContainersSeen,      // exact-NiNode containers under a cell's node 3, offered by the walk's container loops
			kCellNodesHeld,       // ... holding an always-draw entry or an actor (or too big to scan): walked
			kCellNodeEntriesSkipped,  // entries never filed because their node was skipped whole
			kCellNodesAlwaysDraw, // ... the node's own flags carry always-draw (counted only)
			kLampVolumeOutside,   // lamp casters whose shadow volume lies outside the view (or beyond the lamp's reach)
			kLampVolumeMisses,    // ... lies entirely behind, or entirely in front of, every visible surface (confirmed)
			kLampVolumeConfirming,
			kLampVolumeNeeded,
			kLampVolumeUnknown,
			kCacheHits,           // verdict cache: outcomes reused from last frame's record
			kCacheNew,            // ... objects with no record in last frame's stream (new, or the order changed)
			kCacheEpoch,          // ... re-evaluated: the camera left the epoch
			kCacheBound,          // ... re-evaluated: the object's bound changed
			kCacheDepth,          // ... re-evaluated: a Hi-Z block it read changed
			kCacheNoCache,        // ... re-evaluated: the record was marked never-reuse (a light's reach, table full)
			kCacheRecheck,        // hidden verdicts reused after their evidence was re-checked against the current depth (their blocks had changed)
			kCacheKeptRecheck,    // kept (drawn) verdicts re-evaluated on their periodic turn (never because the depth changed)
			kCacheSunDepth,       // sun outcomes re-evaluated under a reused view outcome (the blocks their shadow test read changed)
			kCycles,              // TSC cycles in the per-object tests (sampled frames, scaled)
			kCyclesEvaluate,      // ... of which: full view evaluations (sphere tests, mesh shapes, lights)
			kCyclesShape,         // ... of which: the mesh-shape tests inside those evaluations
			kCyclesSun,           // ... of which: sun-shadow evaluations
			kCounterCount
		};

		struct alignas(64) ThreadStats
		{
			std::array<std::atomic<std::uint64_t>, kCounterCount> values{};
		};
		constexpr std::size_t                  kStatSlots = 128;
		std::array<ThreadStats, kStatSlots>    g_threadStats{};
		std::atomic<std::uint32_t>             g_nextStatSlot{ 0 };
		std::array<std::uint64_t, kCounterCount> g_reported{};  // totals at the last LogStats (main thread)
		std::uint64_t                          g_cyclesReported{ 0 };
		ThreadStats*                           g_mainStats{ nullptr };  // the main thread's block (set at BeginFrame)
		std::uint64_t                          g_mainCyclesReported{ 0 };

		ThreadStats& LocalStats() noexcept
		{
			// Constant-initialized, so no per-access thread-local init guard (a dynamic initializer would cost one
			// on every Bump; there are several per object tested).
			thread_local ThreadStats* stats = nullptr;
			if (!stats) [[unlikely]] {
				stats = &g_threadStats[g_nextStatSlot.fetch_add(1, std::memory_order_relaxed) % kStatSlots];
			}
			return *stats;
		}

		void Bump(Counter a_counter, std::uint64_t a_amount = 1) noexcept
		{
			auto& value = LocalStats().values[a_counter];
			value.store(value.load(std::memory_order_relaxed) + a_amount, std::memory_order_relaxed);
		}

		std::uint64_t Total(Counter a_counter) noexcept
		{
			std::uint64_t total = 0;
			for (const auto& slot : g_threadStats) {
				total += slot.values[a_counter].load(std::memory_order_relaxed);
			}
			return total;
		}

		// Cycles -> milliseconds, calibrated against QPC between log calls.
		struct TscCalibration
		{
			std::uint64_t tsc{ 0 };
			std::int64_t  qpc{ 0 };
			double        cyclesPerMs{ 0.0 };
		};
		TscCalibration g_calibration;

		void Calibrate() noexcept
		{
			LARGE_INTEGER now{}, frequency{};
			QueryPerformanceCounter(&now);
			QueryPerformanceFrequency(&frequency);
			const auto tsc = __rdtsc();
			if (g_calibration.qpc != 0 && now.QuadPart > g_calibration.qpc) {
				const double ms = static_cast<double>(now.QuadPart - g_calibration.qpc) * 1000.0 / static_cast<double>(frequency.QuadPart);
				if (ms > 100.0) {
					g_calibration.cyclesPerMs = static_cast<double>(tsc - g_calibration.tsc) / ms;
				}
			}
			g_calibration.tsc = tsc;
			g_calibration.qpc = now.QuadPart;
		}

		// ---- per-object history: lock-free open addressing keyed by NiAVObject* ------------
		// value bits: [0,32) clock of the last verdict, [32,40) hidden streak, 40 classified,
		//             41 exempt, [42,44) exemption reason, 44 light, 45 merge-instanced,
		//             46 merged mesh rejected at `clock`, [48,56) classification epoch,
		//             [56,64) merged mesh: index of the instance last seen visible (search hint)

		constexpr std::size_t   kTableBits = 17;
		constexpr std::size_t   kTableSize = std::size_t{ 1 } << kTableBits;
		constexpr std::size_t   kMaxProbe = 64;
		constexpr std::uint64_t kLowBits = 0xFFFFFFFFFFull;  // clock + streak
		constexpr std::uint64_t kClassified = std::uint64_t{ 1 } << 40;
		constexpr std::uint64_t kExempt = std::uint64_t{ 1 } << 41;
		constexpr std::uint64_t kIsLight = std::uint64_t{ 1 } << 44;
		constexpr std::uint64_t kIsMerged = std::uint64_t{ 1 } << 45;
		constexpr std::uint64_t kMergedReject = std::uint64_t{ 1 } << 46;
		constexpr int           kEpochShift = 48;
		constexpr int           kHintShift = 56;

		struct alignas(32) Entry
		{
			std::atomic<std::uintptr_t> key{ 0 };
			std::atomic<std::uint64_t>  value{ 0 };
			std::atomic<std::uint64_t>  sun{ 0 };   // sun-shadow streak: [0,32) clock of the last hidden verdict, [32,40) streak
			std::atomic<std::uint64_t>  lamp{ 0 };  // lamp-shadow streak (any lamp): same layout
		};
		static_assert(sizeof(Entry) == 32);
		std::unique_ptr<Entry[]>   g_table;
		std::atomic<std::uint32_t> g_used{ 0 };

		Entry* FindEntry(std::uintptr_t a_key) noexcept
		{
			const auto hash = static_cast<std::size_t>(((a_key >> 4) * 0x9E3779B97F4A7C15ull) >> (64 - kTableBits));
			for (std::size_t probe = 0; probe < kMaxProbe; ++probe) {
				auto& entry = g_table[(hash + probe) & (kTableSize - 1)];
				auto  current = entry.key.load(std::memory_order_acquire);
				if (current == a_key) {
					return &entry;
				}
				if (current == 0) {
					if (entry.key.compare_exchange_strong(current, a_key, std::memory_order_acq_rel)) {
						g_used.fetch_add(1, std::memory_order_relaxed);
						return &entry;
					}
					if (current == a_key) {
						return &entry;
					}
				}
			}
			return nullptr;
		}

		void ClearTable()
		{
			for (std::size_t i = 0; i < kTableSize; ++i) {
				g_table[i].key.store(0, std::memory_order_relaxed);
				g_table[i].value.store(0, std::memory_order_relaxed);
				g_table[i].sun.store(0, std::memory_order_relaxed);
				g_table[i].lamp.store(0, std::memory_order_relaxed);
			}
			g_used.store(0);
		}

		// ---- main-view drops (shared groups) -------------------------------------------------------
		// Entries of a shared group (group 0: the sun's shadow cascades read it too) stay in the group; the
		// ones the main view needn't draw this frame are recorded here, and the main accumulator's
		// registration of them is skipped (MainView).
		DropSet       g_drops;
		std::uint32_t g_dropCycle{ 0 };  // clock / kTagCycle at the last frame start (main thread)

		// ---- object types -----------------------------------------------------------------------
		// Static scenery (precombined chunks, reference roots, their nodes and meshes) and point/spot
		// lights may be rejected. A light is only rejected when its whole influence sphere is behind
		// visible surfaces, i.e. it can't reach a single visible pixel. Directional/ambient lights,
		// particles, instanced meshes and everything under an Actor stay with the engine.
		// BSSubIndexTriShape (segmented meshes) and BSDynamicTriShape are mostly actor parts, which the
		// actor check keeps; the rest are static scenery (v1.5 run: ~200-400 hidden per frame were kept).

		constexpr std::array kCullableTypes{
			"NiNode"sv,
			"BSFadeNode"sv,
			"BSLeafAnimNode"sv,
			"BSMultiBoundNode"sv,
			"BSTriShape"sv,
			"BSMeshLODTriShape"sv,
			"BSMergeInstancedTriShape"sv,
			"BSCombinedTriShape"sv,
			"BSLODTriShape"sv,
			"BSSubIndexTriShape"sv,
			"BSDynamicTriShape"sv,
			"NiPointLight"sv,
			"NiSpotLight"sv,
		};

		enum TypeFlags : std::uint8_t
		{
			kTypeKnown = 1 << 0,
			kTypeCullable = 1 << 1,
			kTypeLight = 1 << 2,
			kTypeLeafAnim = 1 << 3,  // its meshes sway: never judged by their shape
		};

		constexpr std::size_t                               kTypeSlots = 512;
		std::array<std::atomic<std::uintptr_t>, kTypeSlots> g_typeKeys{};
		std::array<std::atomic<std::uint8_t>, kTypeSlots>   g_typeFlags{};

		std::uint8_t ClassifyTypeName(std::string_view a_name) noexcept
		{
			std::uint8_t flags = kTypeKnown;
			if (std::ranges::find(kCullableTypes, a_name) != kCullableTypes.end()) {
				flags |= kTypeCullable;
			}
			if (a_name.find("Light"sv) != std::string_view::npos) {
				flags |= kTypeLight;
			}
			if (a_name == "BSLeafAnimNode"sv) {
				flags |= kTypeLeafAnim;
			}
			return flags;
		}

		std::uint8_t TypeOf(const RE::NiAVObject* a_object) noexcept
		{
			const auto vtable = Util::TryReadVtable(a_object);
			if (!vtable) {
				return kTypeKnown;
			}
			const auto hash = static_cast<std::size_t>(((vtable >> 3) * 0x9E3779B97F4A7C15ull) >> 55) % kTypeSlots;
			for (std::size_t probe = 0; probe < kTypeSlots; ++probe) {
				const auto i = (hash + probe) % kTypeSlots;
				auto       current = g_typeKeys[i].load(std::memory_order_acquire);
				if (current == vtable) {
					if (const auto flags = g_typeFlags[i].load(std::memory_order_acquire)) {
						return flags;
					}
					break;  // another thread is classifying it: decide locally
				}
				if (current == 0) {
					char name[64]{};
					Util::TryGetRTTIName(a_object, name, sizeof(name));
					const auto flags = ClassifyTypeName(name);
					if (g_typeKeys[i].compare_exchange_strong(current, vtable, std::memory_order_acq_rel)) {
						g_typeFlags[i].store(flags, std::memory_order_release);
						logger::info(
							"occlusion: type {} -> {}{}", name[0] ? name : "(unknown)",
							(flags & kTypeCullable) ? "may be culled" : "never culled", (flags & kTypeLight) ? " (light)" : "");
						return flags;
					}
					if (current == vtable) {
						return flags;
					}
				}
			}
			char name[64]{};
			Util::TryGetRTTIName(a_object, name, sizeof(name));
			return ClassifyTypeName(name);
		}

		// userData of a reference's 3D root points at the TESObjectREFR.
		int FormTypeOf(std::uintptr_t a_form) noexcept
		{
			__try {
				return static_cast<int>(reinterpret_cast<const RE::TESForm*>(a_form)->GetFormType());
			} __except (EXCEPTION_EXECUTE_HANDLER) {
				return -1;
			}
		}

		bool IsActorRoot(const RE::NiAVObject* a_object) noexcept
		{
			return a_object->userData && FormTypeOf(a_object->userData) == static_cast<int>(RE::ENUM_FORM_ID::kACHR);
		}

		bool InsideActor(const RE::NiAVObject* a_object) noexcept
		{
			auto node = a_object;
			for (int depth = 0; node && depth < 32; ++depth) {
				if (IsActorRoot(node)) {
					return true;
				}
				node = node->parent;
			}
			return false;
		}

		// What a node's subtree holds that must not go with the node: rejecting a node also drops the
		// lights beneath it (they then stop lighting visible surfaces), and an NPC walking under it can
		// leave the node's bound behind for a frame. Too large to scan counts as a light (conservative).
		enum SubtreeContents : std::uint8_t
		{
			kSubtreePlain = 0,
			kSubtreeLight = 1 << 0,
			kSubtreeActor = 1 << 1,
			kSubtreeAlwaysDraw = 1 << 2,  // a descendant the engine registers whatever the frustum says (flags bit 11)
		};

		std::uint8_t ScanSubtree(RE::NiAVObject* a_object, std::size_t a_budget = 512) noexcept
		{
			std::array<RE::NiNode*, 64> stack{};
			std::size_t                 top = 0;
			std::size_t                 visited = 0;
			std::uint8_t                found = kSubtreePlain;

			auto root = a_object->IsNode();
			if (!root) {
				return kSubtreePlain;
			}
			stack[top++] = root;
			while (top > 0) {
				const auto node = stack[--top];
				for (auto& child : node->children) {
					const auto object = child.get();
					if (!object) {
						continue;
					}
					if (++visited > a_budget) {
						return found | kSubtreeLight;
					}
					if ((object->GetFlags() >> 11) & 1) {
						found |= kSubtreeAlwaysDraw;
					}
					if (TypeOf(object) & kTypeLight) {
						return found | kSubtreeLight;
					}
					if (IsActorRoot(object)) {
						found |= kSubtreeActor;  // (its own parts are the actor's business)
						continue;
					}
					if (const auto childNode = object->IsNode()) {
						if (top == stack.size()) {
							return found | kSubtreeLight;
						}
						stack[top++] = childNode;
					}
				}
			}
			return found;
		}

		// ---- the test -------------------------------------------------------------------------

		enum class Verdict
		{
			kVisible,
			kHidden,
			kEdge,     // part of it is in the current view but was never rendered in the depth frame
			kOutside,  // not in the current view at all: the engine's frustum test drops it
			kBehind,   // behind the depth frame's camera
			kNear,     // reaches the camera: always visible
			kInvalid,  // bad bound
		};

		// Linear view depth of a Hi-Z value; +inf for cleared/sky texels.
		float LinearDepth(const HiZ::Camera& a_camera, float a_buffer) noexcept
		{
			const float ndc = (a_buffer - g_tunables.depthMin) / g_tunables.depthRange;
			if (a_buffer >= 0.999999f || ndc >= a_camera.depthA) {
				return std::numeric_limits<float>::infinity();
			}
			return a_camera.depthB / (ndc - a_camera.depthA);
		}

		using ShadowGeometry::SphereExtent;

		// Clips an NDC rectangle of the depth frame to the part the current camera can see.
		// kHidden here means "judge it": the clipped rectangle lies within the depth frame.
		Verdict ClipToView(const FrameContext& a_context, float& a_x0, float& a_x1, float& a_y0, float& a_y1) noexcept
		{
			a_x0 = std::max(a_x0, a_context.view[0]);
			a_x1 = std::min(a_x1, a_context.view[1]);
			a_y0 = std::max(a_y0, a_context.view[2]);
			a_y1 = std::min(a_y1, a_context.view[3]);
			if (a_x0 >= a_x1 || a_y0 >= a_y1) {
				return Verdict::kOutside;
			}
			// A visible part the depth frame never rendered: turning the camera brought it into view.
			if (a_x0 < -1.0f || a_x1 > 1.0f || a_y0 < -1.0f || a_y1 > 1.0f) {
				return Verdict::kEdge;
			}
			return Verdict::kHidden;
		}

		// A sphere (view-space center, radius) entirely outside the current view: behind the eye, or beyond one of
		// the view's four edge planes (planes through the eye; the view lies in front of the depth frame's eye).
		// Only decidable with the view known.
		bool SphereOutsideView(const FrameContext& a_context, float a_x, float a_y, float a_z, float a_radius) noexcept
		{
			if (!std::isfinite(a_context.view[0])) {
				return false;
			}
			const auto& camera = a_context.snapshot->camera;
			const auto  outsidePlane = [&](float a_scale, float a_coord, float a_edge, bool a_upper) {
				const float side = a_scale * a_coord - a_edge * a_z;  // >= 0 on the inner side of a lower edge
				const float reach = a_radius * std::sqrt(a_scale * a_scale + a_edge * a_edge);
				return a_upper ? side > reach : side < -reach;
			};
			return a_z + a_radius < 0.0f ||
			       outsidePlane(camera.scaleX, a_x, a_context.view[0], false) || outsidePlane(camera.scaleX, a_x, a_context.view[1], true) ||
			       outsidePlane(camera.scaleY, a_y, a_context.view[2], false) || outsidePlane(camera.scaleY, a_y, a_context.view[3], true);
		}

		// What a test read of the depth, for the verdict cache. Add records a level-0 texel rectangle read, widened by
		// the coarsest texel the query could have touched (the level where the rectangle spans <= 4 texels, as the
		// Snapshot queries choose it): which Hi-Z blocks the outcome rests on. AddHidden also records the evidence of a
		// hidden verdict: the rectangle whose texels were all nearer than the threshold, and the smallest such
		// threshold, so the verdict can be re-checked against a later depth with one coarse query (ViewValid).
		struct DepthRect
		{
			float x0{ 0.0f }, y0{ 0.0f }, x1{ 0.0f }, y1{ 0.0f };  // widened union of everything read
			bool  used{ false };
			float hx0{ 0.0f }, hy0{ 0.0f }, hx1{ 0.0f }, hy1{ 0.0f };  // union of the hidden evidence (not widened)
			float threshold{ 1.0f };                                    // buffer depth every texel of it was nearer than
			bool  hidden{ false };

			void Add(float a_x0, float a_y0, float a_x1, float a_y1) noexcept
			{
				float span = std::max(a_x1 - a_x0, a_y1 - a_y0);
				float texel = 1.0f;
				while (span > 4.0f && texel < 4096.0f) {
					span *= 0.5f;
					texel *= 2.0f;
				}
				a_x0 -= texel;
				a_y0 -= texel;
				a_x1 += texel;
				a_y1 += texel;
				if (!used) {
					x0 = a_x0;
					y0 = a_y0;
					x1 = a_x1;
					y1 = a_y1;
					used = true;
				} else {
					x0 = std::min(x0, a_x0);
					y0 = std::min(y0, a_y0);
					x1 = std::max(x1, a_x1);
					y1 = std::max(y1, a_y1);
				}
			}

			void AddHidden(float a_x0, float a_y0, float a_x1, float a_y1, float a_threshold) noexcept
			{
				Add(a_x0, a_y0, a_x1, a_y1);
				if (!hidden) {
					hx0 = a_x0;
					hy0 = a_y0;
					hx1 = a_x1;
					hy1 = a_y1;
					hidden = true;
				} else {
					hx0 = std::min(hx0, a_x0);
					hy0 = std::min(hy0, a_y0);
					hx1 = std::max(hx1, a_x1);
					hy1 = std::max(hy1, a_y1);
				}
				threshold = std::min(threshold, a_threshold);
			}
		};

		// With the verdict cache on, an object counts as hidden only with the cache's extra depth room (HiZ marks a
		// block changed only past half of it, so a reused verdict keeps a margin either way).
		float CacheLimit(const FrameContext& a_context, float a_limit) noexcept
		{
			return a_context.cacheEnabled ? (a_limit - HiZ::kCacheDepthSlack) / (1.0f + HiZ::kCacheDepthTolerance) : a_limit;
		}

		float CacheLimitFar(const FrameContext& a_context, float a_limit) noexcept
		{
			return a_context.cacheEnabled ? a_limit * (1.0f + HiZ::kCacheDepthTolerance) + HiZ::kCacheDepthSlack : a_limit;
		}

		Verdict TestViewSpace(const FrameContext& a_context, const RE::NiBound& a_bound, float a_radius, DepthRect* a_rect = nullptr) noexcept
		{
			const auto& snapshot = *a_context.snapshot;
			const auto& camera = snapshot.camera;

			const float dx = a_bound.center.x - camera.origin[0];
			const float dy = a_bound.center.y - camera.origin[1];
			const float dz = a_bound.center.z - camera.origin[2];
			const float z = dx * camera.viewDir[0] + dy * camera.viewDir[1] + dz * camera.viewDir[2];
			const float x = dx * camera.viewRight[0] + dy * camera.viewRight[1] + dz * camera.viewRight[2];
			const float y = dx * camera.viewUp[0] + dy * camera.viewUp[1] + dz * camera.viewUp[2];
			// (a reused verdict must hold for a camera turned by the cache tolerance: at this depth that shifts the
			// object by up to this much)
			a_radius += std::max(z, 0.0f) * a_context.angularSlack;

			// With the current view known, anything entirely outside it is decided here, wherever it is
			// (beside or behind the camera too).
			if (SphereOutsideView(a_context, x, y, z, a_radius)) {
				return Verdict::kOutside;
			}
			if (z + a_radius < 0.0f) {
				return Verdict::kBehind;
			}
			const float nearest = z - a_radius;
			if (nearest < g_tunables.nearDistance) {
				return Verdict::kNear;
			}

			float x0, x1, y0, y1;
			SphereExtent(x, z, a_radius, x0, x1);
			SphereExtent(y, z, a_radius, y0, y1);
			x0 *= camera.scaleX;
			x1 *= camera.scaleX;
			y0 *= camera.scaleY;
			y1 *= camera.scaleY;
			if (const auto clip = ClipToView(a_context, x0, x1, y0, y1); clip != Verdict::kHidden) {
				return clip;
			}

			const float width = static_cast<float>(snapshot.width[0]);
			const float height = static_cast<float>(snapshot.height[0]);
			const float px0 = (x0 * 0.5f + 0.5f) * width;
			const float px1 = (x1 * 0.5f + 0.5f) * width;
			const float py0 = (0.5f - y1 * 0.5f) * height;
			const float py1 = (0.5f - y0 * 0.5f) * height;

			// Hidden when every occluder in the rectangle is nearer than the object's nearest point
			// (with tolerance). Compared in buffer depth so Hi-Z texels can be checked directly.
			const float limit = CacheLimit(a_context, (nearest - g_tunables.depthSlack) / (1.0f + g_tunables.depthTolerance));
			if (!(limit > 0.0f)) {
				return Verdict::kVisible;
			}
			if (camera.depthB >= 0.0f) {  // not standard Z: compare linear depths over the whole rectangle (no evidence recorded: never reused hidden)
				const float occluder = LinearDepth(camera, snapshot.MaxDepth(px0, py0, px1, py1));
				return occluder < limit ? Verdict::kHidden : Verdict::kVisible;
			}
			const float threshold = std::min(g_tunables.depthMin + g_tunables.depthRange * (camera.depthA + camera.depthB / limit), 0.999999f);

			// Only texels the sphere's own rays pass through can show it (not the rectangle's corners).
			HiZ::SphereRays rays{};
			rays.centerX = (camera.scaleX * x / z * 0.5f + 0.5f) * width;
			rays.centerY = (0.5f - camera.scaleY * y / z * 0.5f) * height;
			rays.depth = z;
			rays.radius = a_radius;
			rays.texelsPerTanX = camera.scaleX * width * 0.5f;
			rays.texelsPerTanY = camera.scaleY * height * 0.5f;
			rays.axisX = width * 0.5f;
			rays.axisY = height * 0.5f;
			if (!snapshot.AllNearer(px0, py0, px1, py1, threshold, kRefineLevels, &rays)) {
				return Verdict::kVisible;
			}
			if (a_rect) {
				a_rect->AddHidden(px0, py0, px1, py1, threshold);
			}
			return Verdict::kHidden;
		}

		// Fallback when the view-space form couldn't be verified: project the bound's box corners.
		Verdict TestCorners(const FrameContext& a_context, const RE::NiBound& a_bound, float a_radius, DepthRect* a_rect = nullptr) noexcept
		{
			const auto& snapshot = *a_context.snapshot;
			const auto& camera = snapshot.camera;
			const auto& m = camera.viewProj;

			const float cx = a_bound.center.x - camera.posAdjust[0];
			const float cy = a_bound.center.y - camera.posAdjust[1];
			const float cz = a_bound.center.z - camera.posAdjust[2];

			float xmin = 1.0e30f, xmax = -1.0e30f, ymin = 1.0e30f, ymax = -1.0e30f, zmin = 1.0e30f;
			for (int i = 0; i < 8; ++i) {
				const float px = cx + ((i & 1) ? a_radius : -a_radius);
				const float py = cy + ((i & 2) ? a_radius : -a_radius);
				const float pz = cz + ((i & 4) ? a_radius : -a_radius);
				const float w = px * m[0][3] + py * m[1][3] + pz * m[2][3] + m[3][3];
				if (w < g_tunables.nearDistance) {
					return Verdict::kNear;
				}
				const float inv = 1.0f / w;
				xmin = std::min(xmin, (px * m[0][0] + py * m[1][0] + pz * m[2][0] + m[3][0]) * inv);
				xmax = std::max(xmax, (px * m[0][0] + py * m[1][0] + pz * m[2][0] + m[3][0]) * inv);
				ymin = std::min(ymin, (px * m[0][1] + py * m[1][1] + pz * m[2][1] + m[3][1]) * inv);
				ymax = std::max(ymax, (px * m[0][1] + py * m[1][1] + pz * m[2][1] + m[3][1]) * inv);
				zmin = std::min(zmin, (px * m[0][2] + py * m[1][2] + pz * m[2][2] + m[3][2]) * inv);
			}
			if (const auto clip = ClipToView(a_context, xmin, xmax, ymin, ymax); clip != Verdict::kHidden) {
				return clip;
			}

			const float depth = g_tunables.depthMin + g_tunables.depthRange * std::clamp(zmin, 0.0f, 1.0f);
			const float width = static_cast<float>(snapshot.width[0]);
			const float height = static_cast<float>(snapshot.height[0]);
			const float px0 = (xmin * 0.5f + 0.5f) * width;
			const float py0 = (0.5f - ymax * 0.5f) * height;
			const float px1 = (xmax * 0.5f + 0.5f) * width;
			const float py1 = (0.5f - ymin * 0.5f) * height;
			const float threshold = depth - 1.0e-6f;
			if (!snapshot.AllNearer(px0, py0, px1, py1, threshold, kRefineLevels)) {
				return Verdict::kVisible;
			}
			if (a_rect) {
				a_rect->AddHidden(px0, py0, px1, py1, threshold);
			}
			return Verdict::kHidden;
		}

		// ---- sun shadows: the capsule a caster sweeps along the light --------------------------------------

		enum class SunVerdict
		{
			kNeeded,   // its shadow may fall on something visible
			kOutside,  // the sweep never reaches the current view within the cascade range
			kHidden,   // every visible surface where it could land is in front of it
		};

		// Whether a caster's sun shadow can fall on anything visible. The shadow lies in the capsule swept from
		// the caster's bound along the light (its radius growing with the direction's uncertainty). Receivers are
		// in front of the eye, within the cascade range and inside the current view, so the sweep is clipped to
		// where its cross-section reaches that region; that part is then tested against the depth frame like an
		// object: hidden only if every visible surface over its screen box is in front of its nearest point.
		SunVerdict TestSun(const FrameContext& a_context, const RE::NiBound& a_bound, DepthRect* a_rect = nullptr) noexcept
		{
			const auto& snapshot = *a_context.snapshot;
			const auto& camera = snapshot.camera;
			const auto& sun = a_context.sun;
			const float radius = a_bound.fRadius;
			if (!camera.viewSpace || !(camera.depthB < 0.0f) || !(radius > 0.0f) || !(radius < 1.0e6f) ||
				!std::isfinite(a_bound.center.x) || !std::isfinite(a_bound.center.y) || !std::isfinite(a_bound.center.z)) {
				return SunVerdict::kNeeded;
			}
			const float dx = a_bound.center.x - camera.origin[0];
			const float dy = a_bound.center.y - camera.origin[1];
			const float dz = a_bound.center.z - camera.origin[2];
			const float center[3]{
				dx * camera.viewRight[0] + dy * camera.viewRight[1] + dz * camera.viewRight[2],
				dx * camera.viewUp[0] + dy * camera.viewUp[1] + dz * camera.viewUp[2],
				dx * camera.viewDir[0] + dy * camera.viewDir[1] + dz * camera.viewDir[2]
			};
			// (+ the cache's turn tolerance over the whole receiver range: a reused verdict must hold for a camera
			// turned by that much, which shifts a point at depth z by z x the slack)
			const float r0 = radius + a_context.dilateMove + sun.margin + (std::isfinite(sun.reach) ? std::max(sun.reach, 0.0f) * a_context.angularSlack : 0.0f);

			for (int i = 0; i < sun.reachCount; ++i) {  // the cheap part of the sweep first (most casters end here)
				const auto& plane = sun.reachPlanes[i];
				if (ShadowGeometry::Dot(plane.n, center) + plane.d + r0 < 0.0f) {
					return SunVerdict::kOutside;
				}
			}
			float t0 = 0.0f, t1 = 0.0f;
			if (!ShadowGeometry::SweepClip(center, sun.dir, r0, sun.spread, 1.0e6f, sun.planes, sun.planeCount, t0, t1)) {
				return SunVerdict::kOutside;
			}

			const float q0[3]{ center[0] + t0 * sun.dir[0], center[1] + t0 * sun.dir[1], center[2] + t0 * sun.dir[2] };
			const float q1[3]{ center[0] + t1 * sun.dir[0], center[1] + t1 * sun.dir[1], center[2] + t1 * sun.dir[2] };
			const float ra = r0 + t0 * sun.spread;
			const float rb = r0 + t1 * sun.spread;
			// Depth minus radius is linear along the capsule: its nearest point is at one of the ends.
			const float nearest = std::min(q0[2] - ra, q1[2] - rb);
			if (nearest < g_tunables.nearDistance) {
				return SunVerdict::kNeeded;
			}
			// The capsule projects within the box around both end spheres' projections (all in front of the eye).
			float x0, x1, y0, y1, u0, u1, v0, v1;
			SphereExtent(q0[0], q0[2], ra, x0, x1);
			SphereExtent(q0[1], q0[2], ra, y0, y1);
			SphereExtent(q1[0], q1[2], rb, u0, u1);
			SphereExtent(q1[1], q1[2], rb, v0, v1);
			x0 = std::min(x0, u0) * camera.scaleX;
			x1 = std::max(x1, u1) * camera.scaleX;
			y0 = std::min(y0, v0) * camera.scaleY;
			y1 = std::max(y1, v1) * camera.scaleY;
			switch (ClipToView(a_context, x0, x1, y0, y1)) {
			case Verdict::kOutside:
				return SunVerdict::kOutside;
			case Verdict::kEdge:
				return SunVerdict::kNeeded;
			default:
				break;
			}
			const float limit = CacheLimit(a_context, (nearest - g_tunables.depthSlack) / (1.0f + g_tunables.depthTolerance));
			if (!(limit > 0.0f)) {
				return SunVerdict::kNeeded;
			}
			const float farthest = std::max(q0[2] + ra, q1[2] + rb);
			const float limitFar = CacheLimitFar(a_context, (farthest + g_tunables.depthSlack) * (1.0f + g_tunables.depthTolerance));
			const float width = static_cast<float>(snapshot.width[0]);
			const float height = static_cast<float>(snapshot.height[0]);
			const auto  bufferDepth = [&](float a_z) { return std::min(g_tunables.depthMin + g_tunables.depthRange * (camera.depthA + camera.depthB / a_z), 0.999999f); };
			if (a_rect) {
				a_rect->Add((x0 * 0.5f + 0.5f) * width - 0.25f, (0.5f - y1 * 0.5f) * height - 0.25f, (x1 * 0.5f + 0.5f) * width + 0.25f, (0.5f - y0 * 0.5f) * height + 0.25f);
			}
			// No visible surface inside the capsule's depth range over its screen box: at every texel the surfaces are
			// all in front of it or all behind it (the sky, cleared to the far plane, counts as behind: nothing there
			// receives a shadow).
			return snapshot.NoSurfaceBetween(
					   (x0 * 0.5f + 0.5f) * width - 0.25f, (0.5f - y1 * 0.5f) * height - 0.25f,
					   (x1 * 0.5f + 0.5f) * width + 0.25f, (0.5f - y0 * 0.5f) * height + 0.25f,
					   bufferDepth(limit), bufferDepth(limitFar), kRefineLevels) ?
			           SunVerdict::kHidden :
			           SunVerdict::kNeeded;
		}

		// Where sun-shadow receivers can be this frame (TestSun clips each caster's sweep to it): in front of the
		// eye, within reach, and inside the current view's edges (planes through the eye, as TestViewSpace uses).
		void SetSunPlanes(FrameContext& a_context) noexcept
		{
			auto& sun = a_context.sun;
			sun.planeCount = 0;
			if (sun.state != FrameContext::Sun::State::kOn || !a_context.snapshot) {
				return;
			}
			const auto& camera = a_context.snapshot->camera;
			int         count = 0;
			sun.planes[count++] = { { 0.0f, 0.0f, 1.0f }, 0.0f };
			sun.planes[count++] = { { 0.0f, 0.0f, -1.0f }, sun.reach };
			if (std::isfinite(a_context.view[0])) {
				const auto edge = [&](int a_axis, float a_scale, float a_edge, bool a_upper) {
					const float length = std::sqrt(a_scale * a_scale + a_edge * a_edge);
					auto&       plane = sun.planes[count++];
					plane = {};
					plane.n[a_axis] = (a_upper ? -a_scale : a_scale) / length;
					plane.n[2] = (a_upper ? a_edge : -a_edge) / length;
				};
				edge(0, camera.scaleX, a_context.view[0], false);
				edge(0, camera.scaleX, a_context.view[1], true);
				edge(1, camera.scaleY, a_context.view[2], false);
				edge(1, camera.scaleY, a_context.view[3], true);
			}
			sun.planeCount = count;
			// Along the light a sweep only moves away from these planes (its radius growth included), so a caster
			// sphere already beyond one of them can't reach any receiver: SweepClip's answer, without the divisions.
			sun.reachCount = 0;
			for (int i = 0; i < count; ++i) {
				if (ShadowGeometry::Dot(sun.planes[i].n, sun.dir) + sun.spread <= 0.0f) {
					sun.reachPlanes[sun.reachCount++] = sun.planes[i];
				}
			}
		}

		// Where lamp-shadow receivers can be this frame: in front of the eye and inside the current view's edges
		// (as SetSunPlanes, without the cascade range: a lamp's reach bounds its sweep instead).
		void SetLampPlanes(FrameContext& a_context) noexcept
		{
			a_context.lampPlaneCount = 0;
			if (!a_context.snapshot || !a_context.snapshot->camera.viewSpace) {
				return;
			}
			const auto& camera = a_context.snapshot->camera;
			int         count = 0;
			a_context.lampPlanes[count++] = { { 0.0f, 0.0f, 1.0f }, 0.0f };
			if (std::isfinite(a_context.view[0])) {
				const auto edge = [&](int a_axis, float a_scale, float a_edge, bool a_upper) {
					const float length = std::sqrt(a_scale * a_scale + a_edge * a_edge);
					auto&       plane = a_context.lampPlanes[count++];
					plane = {};
					plane.n[a_axis] = (a_upper ? -a_scale : a_scale) / length;
					plane.n[2] = (a_upper ? a_edge : -a_edge) / length;
				};
				edge(0, camera.scaleX, a_context.view[0], false);
				edge(0, camera.scaleX, a_context.view[1], true);
				edge(1, camera.scaleY, a_context.view[2], false);
				edge(1, camera.scaleY, a_context.view[3], true);
			}
			a_context.lampPlaneCount = count;
		}

		// ---- mesh shapes: boxes in the depth frame's view space --------------------------------------

		// A view-space box (center, half extents) against the depth frame. Only the part beyond the near
		// distance can be drawn, so the box is clipped there first.
		Verdict TestViewBox(const FrameContext& a_context, const float a_center[3], const float a_halfIn[3], DepthRect* a_rect = nullptr) noexcept
		{
			const auto& snapshot = *a_context.snapshot;
			const auto& camera = snapshot.camera;
			const float zNear = g_tunables.nearDistance;
			const float zFar = a_center[2] + a_halfIn[2];
			if (zFar < zNear) {
				return Verdict::kOutside;  // entirely before the near plane or behind the eye
			}
			// (+ the cache's turn tolerance at the box's farthest depth)
			const float slack = zFar * a_context.angularSlack;
			const float a_half[3]{ a_halfIn[0] + slack, a_halfIn[1] + slack, a_halfIn[2] + slack };
			if (std::isfinite(a_context.view[0])) {
				const auto outsidePlane = [&](float a_scale, float a_coord, float a_extent, float a_edge, bool a_upper) {
					const float side = a_scale * a_coord - a_edge * a_center[2];
					const float reach = a_scale * a_extent + std::abs(a_edge) * a_half[2];
					return a_upper ? side > reach : side < -reach;
				};
				if (outsidePlane(camera.scaleX, a_center[0], a_half[0], a_context.view[0], false) ||
					outsidePlane(camera.scaleX, a_center[0], a_half[0], a_context.view[1], true) ||
					outsidePlane(camera.scaleY, a_center[1], a_half[1], a_context.view[2], false) ||
					outsidePlane(camera.scaleY, a_center[1], a_half[1], a_context.view[3], true)) {
					return Verdict::kOutside;
				}
			}
			const float zNearest = std::max(a_center[2] - a_half[2], zNear);
			// x/z and y/z over the clipped box peak at its corners.
			float x0 = std::numeric_limits<float>::infinity(), x1 = -x0, y0 = x0, y1 = -x0;
			for (const float z : { zNearest, zFar }) {
				for (const float sign : { -1.0f, 1.0f }) {
					const float x = camera.scaleX * (a_center[0] + sign * a_half[0]) / z;
					const float y = camera.scaleY * (a_center[1] + sign * a_half[1]) / z;
					x0 = std::min(x0, x);
					x1 = std::max(x1, x);
					y0 = std::min(y0, y);
					y1 = std::max(y1, y);
				}
			}
			if (const auto clip = ClipToView(a_context, x0, x1, y0, y1); clip != Verdict::kHidden) {
				return clip;
			}
			const float limit = CacheLimit(a_context, (zNearest - g_tunables.depthSlack) / (1.0f + g_tunables.depthTolerance));
			if (!(limit > 0.0f) || camera.depthB >= 0.0f) {
				return Verdict::kVisible;
			}
			const float width = static_cast<float>(snapshot.width[0]);
			const float height = static_cast<float>(snapshot.height[0]);
			const float threshold = std::min(g_tunables.depthMin + g_tunables.depthRange * (camera.depthA + camera.depthB / limit), 0.999999f);
			// (a quarter texel of slack covers the render's sub-pixel jitter, as for spheres)
			const float px0 = (x0 * 0.5f + 0.5f) * width - 0.25f;
			const float py0 = (0.5f - y1 * 0.5f) * height - 0.25f;
			const float px1 = (x1 * 0.5f + 0.5f) * width + 0.25f;
			const float py1 = (0.5f - y0 * 0.5f) * height + 0.25f;
			if (!snapshot.AllNearer(px0, py0, px1, py1, threshold, kRefineLevels)) {
				return Verdict::kVisible;
			}
			if (a_rect) {
				a_rect->AddHidden(px0, py0, px1, py1, threshold);
			}
			return Verdict::kHidden;
		}

		// How NiTransform applies its rotation: 0 = rows dot vector (R * v), 1 = columns (R^T * v). Learned
		// from meshes whose model bound, moved by their world transform, lands on their world bound; settled
		// only after kRotationVotes agreeing meshes without dissent (dissent restarts the count, and
		// repeated dissent turns shapes off: the reading can't be trusted).
		constexpr std::uint32_t    kRotationVotes = 32;
		constexpr std::uint32_t    kRotationConflicts = 3;
		std::atomic<int>           g_rotationReading{ -1 };  // -1 learning, 0 rows, 1 columns, 2 off
		std::atomic<std::uint32_t> g_rotationVotes[2]{};
		std::atomic<std::uint32_t> g_rotationConflicts{ 0 };

		bool WorldRotation(const RE::NiAVObject* a_object, const RE::NiBound& a_model, float a_rotate[3][3]) noexcept
		{
			const auto& transform = a_object->world;
			const auto& world = a_object->worldBound;
			const float scale = transform.scale;
			// The engine makes the world bound by moving the model bound with the world transform. Where the
			// radius doesn't match, the bound comes from somewhere else and the transform isn't the whole story.
			if (!(scale > 0.0f) || !(std::abs(world.fRadius - a_model.fRadius * scale) <= world.fRadius * 0.01f + 1.0f)) {
				return false;
			}
			const float c[3]{ a_model.center.x * scale, a_model.center.y * scale, a_model.center.z * scale };
			const float target[3]{ world.center.x - transform.translate.x, world.center.y - transform.translate.y, world.center.z - transform.translate.z };
			const float tolerance = world.fRadius * 0.01f + 1.0f;
			const auto  fits = [&](int a_reading) {
				float error = 0.0f;
				for (int i = 0; i < 3; ++i) {
					float v = 0.0f;
					for (int j = 0; j < 3; ++j) {
						v += (a_reading == 0 ? transform.rotate.entry[i].pt[j] : transform.rotate.entry[j].pt[i]) * c[j];
					}
					error = std::max(error, std::abs(v - target[i]));
				}
				return error <= tolerance;
			};
			const auto reading = g_rotationReading.load(std::memory_order_acquire);
			if (reading < 0) {
				const bool rows = fits(0);
				const bool columns = fits(1);
				if (rows != columns) {  // both fit for a centered bound or a symmetric rotation: no evidence
					const int vote = rows ? 0 : 1;
					if (g_rotationVotes[1 - vote].load(std::memory_order_relaxed) > 0) {
						g_rotationVotes[0].store(0, std::memory_order_relaxed);
						g_rotationVotes[1].store(0, std::memory_order_relaxed);
						if (g_rotationConflicts.fetch_add(1, std::memory_order_relaxed) + 1 >= kRotationConflicts) {
							int expected = -1;
							g_rotationReading.compare_exchange_strong(expected, 2, std::memory_order_acq_rel);
						}
					} else if (g_rotationVotes[vote].fetch_add(1, std::memory_order_relaxed) + 1 >= kRotationVotes) {
						int expected = -1;
						g_rotationReading.compare_exchange_strong(expected, vote, std::memory_order_acq_rel);
					}
				}
				return false;  // no shapes until the reading is settled
			}
			if (reading > 1 || !fits(reading)) {
				return false;  // shapes off, or this object's bound doesn't follow its transform
			}
			for (int i = 0; i < 3; ++i) {
				for (int j = 0; j < 3; ++j) {
					a_rotate[i][j] = reading == 0 ? transform.rotate.entry[i].pt[j] : transform.rotate.entry[j].pt[i];
				}
			}
			return true;
		}

		// NiAVObject::UpdateWorldData copies world into previousWorld before recomputing it, so the two differ
		// for an object that moved in its last update. Such an object can sit behind its own image in the
		// older depth frame, which a tight shape would take for an occluder. A previousWorld that was never
		// set (zero scale, or the default identity at the origin) says nothing.
		bool Moved(const RE::NiAVObject* a_object) noexcept
		{
			const auto& now = a_object->world;
			const auto& before = a_object->previousWorld;
			if (!(before.scale > 0.0f)) {
				return false;
			}
			bool same = now.scale == before.scale && now.translate.x == before.translate.x && now.translate.y == before.translate.y &&
			            now.translate.z == before.translate.z;
			bool identity = before.scale == 1.0f && before.translate.x == 0.0f && before.translate.y == 0.0f && before.translate.z == 0.0f;
			for (int i = 0; i < 3; ++i) {
				for (int j = 0; j < 3; ++j) {
					same = same && now.rotate.entry[i].pt[j] == before.rotate.entry[i].pt[j];
					identity = identity && before.rotate.entry[i].pt[j] == (i == j ? 1.0f : 0.0f);
				}
			}
			return !same && !identity;
		}

		// A mesh judged by its shape: the whole box first, then its occupied cells. Hidden once every
		// occupied cell is hidden or out of view; kept as soon as one isn't. a_hint (in/out) is the cell last seen
		// visible: the scan starts there, so a mesh that stays visible usually stops at its first cell.
		Verdict TestShape(const FrameContext& a_context, const RE::NiAVObject* a_object, const RE::NiBound& a_model, const MeshProxy::Shape& a_shape, std::uint32_t& a_cellsTested, std::uint8_t& a_hint, DepthRect* a_rect = nullptr) noexcept
		{
			float rotate[3][3];
			if (!WorldRotation(a_object, a_model, rotate)) {
				return Verdict::kInvalid;
			}
			const auto&  camera = a_context.snapshot->camera;
			const auto&  transform = a_object->world;
			const float* view[3]{ camera.viewRight, camera.viewUp, camera.viewDir };
			float        m[3][3], t[3];
			for (int i = 0; i < 3; ++i) {
				for (int j = 0; j < 3; ++j) {
					m[i][j] = transform.scale * (view[i][0] * rotate[0][j] + view[i][1] * rotate[1][j] + view[i][2] * rotate[2][j]);
				}
				t[i] = view[i][0] * (transform.translate.x - camera.origin[0]) + view[i][1] * (transform.translate.y - camera.origin[1]) +
				       view[i][2] * (transform.translate.z - camera.origin[2]);
			}
			const float dilate = a_context.dilateMove;
			const auto  box = [&](const float a_lo[3], const float a_size[3]) {
				float center[3], half[3];
				for (int i = 0; i < 3; ++i) {
					float c = t[i], h = dilate;
					for (int j = 0; j < 3; ++j) {
						const float local = a_lo[j] + a_size[j] * 0.5f;
						c += m[i][j] * local;
						h += std::abs(m[i][j]) * a_size[j] * 0.5f;
					}
					center[i] = c;
					half[i] = h;
				}
				return TestViewBox(a_context, center, half, a_rect);
			};

			const float size[3]{ a_shape.cell[0] * a_shape.dims[0], a_shape.cell[1] * a_shape.dims[1], a_shape.cell[2] * a_shape.dims[2] };
			const auto  whole = box(a_shape.min, size);
			if (whole == Verdict::kHidden || whole == Verdict::kOutside || a_shape.count == 0) {
				return whole;
			}
			bool                anyHidden = false;
			const std::uint32_t dims0 = a_shape.dims[0];
			const std::uint32_t dims1 = a_shape.dims[1];
			const std::uint32_t cells = dims0 * dims1 * a_shape.dims[2];
			const std::uint32_t first = a_hint < cells ? a_hint : 0u;
			for (std::uint32_t k = 0; k < cells; ++k) {
				auto bit = first + k;
				if (bit >= cells) {
					bit -= cells;
				}
				if (!(a_shape.occupied[bit >> 6] & (std::uint64_t{ 1 } << (bit & 63)))) {
					continue;
				}
				const auto  x = bit % dims0;
				const auto  y = (bit / dims0) % dims1;
				const auto  z = bit / (dims0 * dims1);
				const float lo[3]{ a_shape.min[0] + a_shape.cell[0] * static_cast<float>(x), a_shape.min[1] + a_shape.cell[1] * static_cast<float>(y), a_shape.min[2] + a_shape.cell[2] * static_cast<float>(z) };
				++a_cellsTested;
				const auto cell = box(lo, a_shape.cell);
				if (cell == Verdict::kHidden) {
					anyHidden = true;
				} else if (cell != Verdict::kOutside) {
					a_hint = static_cast<std::uint8_t>(bit);
					return cell;  // part of the mesh can be seen
				}
			}
			return anyHidden ? Verdict::kHidden : Verdict::kOutside;
		}

		Verdict Test(const FrameContext& a_context, const RE::NiBound& a_bound, DepthRect* a_rect = nullptr) noexcept
		{
			auto radius = a_bound.fRadius;
			if (!(radius > 0.0f) || !std::isfinite(radius) || radius > 1.0e6f ||
				!std::isfinite(a_bound.center.x) || !std::isfinite(a_bound.center.y) || !std::isfinite(a_bound.center.z)) {
				return Verdict::kInvalid;
			}
			// Grow the bound by how far the camera moved since the depth was rendered (parallax can
			// uncover it). The same growth covers the shift of the current view's footprint.
			radius += a_context.dilateMove;
			if (a_context.snapshot->camera.viewSpace) {
				return TestViewSpace(a_context, a_bound, radius, a_rect);
			}
			// (no view-space form: the turn tolerance is covered by the distance from the eye, which bounds the depth)
			const auto& eye = a_context.snapshot->camera.eye;
			const float dx = a_bound.center.x - eye[0], dy = a_bound.center.y - eye[1], dz = a_bound.center.z - eye[2];
			radius += std::sqrt(dx * dx + dy * dy + dz * dz) * a_context.angularSlack;
			return TestCorners(a_context, a_bound, radius, a_rect);
		}

		void CountVerdict(Verdict a_verdict) noexcept
		{
			switch (a_verdict) {
			case Verdict::kVisible:
				Bump(kVisible);
				break;
			case Verdict::kEdge:
				Bump(kEdge);
				break;
			case Verdict::kOutside:
				Bump(kOutside);
				break;
			case Verdict::kBehind:
				Bump(kBehind);
				break;
			case Verdict::kNear:
				Bump(kNear);
				break;
			case Verdict::kInvalid:
				Bump(kInvalid);
				break;
			default:
				break;
			}
		}

		std::atomic<float> g_lastRejected{ 0.0f };
		std::atomic<float> g_lastTested{ 0.0f };
		std::atomic<float> g_lastLights{ 0.0f };
		std::atomic<float> g_lastLightsRejected{ 0.0f };

		enum class Exemption : std::uint8_t
		{
			kNone,
			kType,
			kLight,
			kActor,
		};

		Exemption Classify(RE::NiAVObject* a_object, bool& a_isLight) noexcept
		{
			const auto type = TypeOf(a_object);
			a_isLight = (type & kTypeLight) != 0;
			if (!(type & kTypeCullable)) {
				return Exemption::kType;
			}
			if (!g_tunables.cullActors && InsideActor(a_object)) {
				return Exemption::kActor;
			}
			// A node that carries lights is never cut as a whole: its bound needn't cover their reach. Nor is
			// one carrying an NPC (whatever bCullActors says): the NPC is judged by its own, current bound.
			if (!a_isLight) {
				const auto contents = ScanSubtree(a_object);
				if (contents & kSubtreeLight) {
					return Exemption::kLight;
				}
				if (contents & kSubtreeActor) {
					return Exemption::kActor;
				}
			}
			return Exemption::kNone;
		}

		// A light's reach: its world position with the largest of the scene-graph bound radius, its own
		// model bound (NiLight +0x150) and its light radius, all scaled to world. The engine keeps the
		// radius in NiLight::spec.r (+0x138; the paraboloid shadow setup reads it from there too), and
		// it can be far larger than the bounds (742 vs 163 in the v1.6 run). Tiny reach means unknown:
		// never cull it.
		std::atomic<int> g_lightSamplesLogged{ 0 };

		bool LightReach(RE::NiAVObject* a_light, const RE::NiBound& a_bound, RE::NiBound& a_out) noexcept
		{
			const auto  light = reinterpret_cast<const CBRO::Engine::NiLightView*>(a_light);
			const float scale = std::isfinite(a_light->world.scale) && a_light->world.scale > 0.0f ? a_light->world.scale : 1.0f;
			const float model = light->modelBound.fRadius * scale;
			const float radius = light->spec.r * scale;
			const float world = a_bound.fRadius;
			const auto  valid = [](float a_value) { return std::isfinite(a_value) ? a_value : 0.0f; };
			a_out.center = a_light->world.translate;
			a_out.fRadius = std::max({ valid(model), valid(radius), valid(world) });

			if (g_lightSamplesLogged.load(std::memory_order_relaxed) < 12 && g_lightSamplesLogged.fetch_add(1) < 12) {
				char type[48]{};
				Util::TryGetRTTIName(a_light, type, sizeof(type));
				logger::info(
					"occlusion: light sample {} worldBound r={:.0f} modelBound r={:.0f} radius={:.0f} scale={:.2f} -> reach {:.0f}",
					type, world, light->modelBound.fRadius, light->spec.r, scale, a_out.fRadius);
			}
			return a_out.fRadius >= 16.0f && a_out.fRadius < 1.0e6f;
		}

		const FrameContext* CurrentContext() noexcept
		{
			if (!g_active.load(std::memory_order_relaxed)) {
				return nullptr;
			}
			const auto index = g_context.load(std::memory_order_acquire);
			if (index < 0) {
				return nullptr;
			}
			const auto& context = g_contexts[index];
			return context.cull && context.snapshot ? &context : nullptr;
		}

		// Looks up (and once per epoch re-classifies) an object's history entry.
		Entry* Lookup(const FrameContext& a_context, RE::NiAVObject* a_object, std::uint64_t& a_value) noexcept
		{
			const auto key = reinterpret_cast<std::uintptr_t>(a_object);
			const auto entry = FindEntry(key);
			if (!entry) {
				Bump(kTableFull);
				return nullptr;
			}
			// Re-classify each object once per epoch, staggered by address so it never happens all at once.
			const auto epoch = static_cast<std::uint64_t>(((a_context.clock + static_cast<std::uint32_t>((key >> 4) % kEpochFrames)) / kEpochFrames) & 0xFF);
			auto       value = entry->value.load(std::memory_order_relaxed);
			if (!(value & kClassified) || ((value >> kEpochShift) & 0xFF) != epoch) {
				bool       isLight = false;
				const auto exemption = Classify(a_object, isLight);
				const bool merged = exemption == Exemption::kNone && Hooks::CullGroups::IsMergeInstanced(a_object);
				value = (value & (kLowBits | (std::uint64_t{ 0xFF } << kHintShift))) | kClassified | (epoch << kEpochShift) |
				        (isLight ? kIsLight : 0) | (merged ? kIsMerged : 0);
				if (exemption != Exemption::kNone) {
					value |= kExempt | (static_cast<std::uint64_t>(exemption) << 42);  // bits 42-43: why, for the stats
				}
				entry->value.store(value, std::memory_order_relaxed);
			}
			a_value = value;
			return entry;
		}

		std::uint32_t NextStreak(std::uint64_t a_value, bool a_hidden, std::uint32_t a_clock) noexcept
		{
			if (!a_hidden) {
				return 0;
			}
			const auto streak = static_cast<std::uint32_t>((a_value >> 32) & 0xFF);
			return static_cast<std::uint32_t>(a_value) != a_clock ? std::min(streak + 1, 255u) : streak;
		}

		bool Observing() noexcept
		{
			return g_tunables.observeOnly || g_observeOverride.load(std::memory_order_relaxed);
		}

		// ---- the Block::Add filter (an object's own entry) --------------------------------------

		// ---- kept-object dump (diagnostics, one sampled frame per location) ---------------------------
		// Records every object the main pass keeps on one frame (visible, near, edge, bad bound, exempt,
		// confirming) with its type, name, size and, for meshes, where its vertex data lives, so the
		// leftover draw calls can be attributed. Details are read while the object is certainly alive.

		struct KeptItem
		{
			const char*    reason{ "" };
			float          radius{ 0.0f };
			float          distance{ 0.0f };
			char           type[40]{};
			char           name[48]{};
			bool           geometry{ false };
			bool           skinned{ false };
			bool           dropped{ false };  // not kept: a mesh shape settled it (listed apart, to audit shapes)
			std::uint64_t  vertexDesc{ 0 };
			std::uintptr_t buffer{ 0 };
			std::uintptr_t data{ 0 };
			std::uint32_t  dataSize{ 0 };
			std::uint32_t  dataOffset{ 0 };
			std::uint32_t  vertices{ 0 };
			std::uint32_t  byteWidth{ 0 };
			std::uint32_t  usage{ 0 };
			// BSMergeInstancedTriShape only: its per-instance transform buffer (+0x170, 80-byte records that
			// start with an NiTransform relative to the node, per the precombine builder), to judge its
			// instances one by one later. Raw words, since the buffer object's layout isn't confirmed.
			bool           merged{ false };
			std::uintptr_t instanceObject{ 0 };
			std::uint64_t  instanceWords[5]{};
			std::uint32_t  instanceCount{ 0 };  // +0x34
			std::uint32_t  instanceKind{ 0 };   // +0x38
			std::uint32_t  lodTriangles[4]{};   // +0x1A0
			float          localTranslate[3]{};
			float          modelBound[4]{};
			std::uint32_t  instanceBytes{ 0 };
			std::uint32_t  instanceStride{ 0 };
			std::uint32_t  instanceBind{ 0 };
			std::uint32_t  instanceMisc{ 0 };
			std::uint32_t  instanceUsage{ 0 };
			std::int32_t   instanceWord{ -1 };  // which word held the ID3D11Buffer
			char           instanceModule[24]{};
		};

		std::atomic<std::uint32_t> g_dumpClock{ 0 };
		std::mutex                 g_keptLock;
		std::vector<KeptItem>      g_kept;
		std::uintptr_t             g_mergedVtable{ 0 };

		// BSGeometry: skinInstance +0x140, rendererData (BSGraphics::TriShape*) +0x148; TriShape:
		// vertexDesc +0, vertexBuffer +8; Buffer: ID3D11Buffer* +0, data +8, dataSize +0x34,
		// dataOffset +0x48; BSTriShape::numVertices +0x164.
		void DescribeGeometry(const RE::NiAVObject* a_object, KeptItem& a_item) noexcept
		{
			__try {
				const auto geometry = const_cast<RE::NiAVObject*>(a_object)->IsGeometry();
				if (!geometry) {
					return;
				}
				const auto base = reinterpret_cast<const std::byte*>(geometry);
				a_item.geometry = true;
				a_item.skinned = *reinterpret_cast<const std::uintptr_t*>(base + 0x140) != 0;
				a_item.vertices = *reinterpret_cast<const std::uint16_t*>(base + 0x164);
				const auto triShape = *reinterpret_cast<const std::byte* const*>(base + 0x148);
				if (!triShape) {
					return;
				}
				a_item.vertexDesc = *reinterpret_cast<const std::uint64_t*>(triShape);
				const auto vertexBuffer = *reinterpret_cast<const std::byte* const*>(triShape + 0x8);
				if (!vertexBuffer) {
					return;
				}
				a_item.buffer = *reinterpret_cast<const std::uintptr_t*>(vertexBuffer);
				a_item.data = *reinterpret_cast<const std::uintptr_t*>(vertexBuffer + 0x8);
				a_item.dataSize = *reinterpret_cast<const std::uint32_t*>(vertexBuffer + 0x34);
				a_item.dataOffset = *reinterpret_cast<const std::uint32_t*>(vertexBuffer + 0x48);
			} __except (EXCEPTION_EXECUTE_HANDLER) {
				a_item.geometry = false;
			}
		}

		void DescribeMerged(const RE::NiAVObject* a_object, KeptItem& a_item) noexcept
		{
			__try {
				const auto base = reinterpret_cast<const std::byte*>(a_object);
				if (!g_mergedVtable || *reinterpret_cast<const std::uintptr_t*>(base) != g_mergedVtable) {
					return;
				}
				a_item.merged = true;
				a_item.localTranslate[0] = a_object->local.translate.x;
				a_item.localTranslate[1] = a_object->local.translate.y;
				a_item.localTranslate[2] = a_object->local.translate.z;
				const auto& model = *reinterpret_cast<const RE::NiBound*>(base + 0x120);
				a_item.modelBound[0] = model.center.x;
				a_item.modelBound[1] = model.center.y;
				a_item.modelBound[2] = model.center.z;
				a_item.modelBound[3] = model.fRadius;
				for (int i = 0; i < 4; ++i) {
					a_item.lodTriangles[i] = *reinterpret_cast<const std::uint32_t*>(base + 0x1A0 + 4 * i);
				}
				const auto object = *reinterpret_cast<const std::byte* const*>(base + 0x170);
				a_item.instanceObject = reinterpret_cast<std::uintptr_t>(object);
				if (!object) {
					return;
				}
				for (int i = 0; i < 5; ++i) {
					a_item.instanceWords[i] = *reinterpret_cast<const std::uint64_t*>(object + 8 * i);
				}
				a_item.instanceCount = *reinterpret_cast<const std::uint32_t*>(object + 0x34);
				a_item.instanceKind = *reinterpret_cast<const std::uint32_t*>(object + 0x38);
			} __except (EXCEPTION_EXECUTE_HANDLER) {
			}
		}

		// Finds which of the instance buffer object's words is an ID3D11Buffer: only an object whose vtable
		// lies in a module named d3d11.dll (the runtime or a wrapper such as ENB's) is ever called.
		void DescribeInstanceBuffer(KeptItem& a_item) noexcept
		{
			for (int i = 0; i < 5; ++i) {
				const auto candidate = static_cast<std::uintptr_t>(a_item.instanceWords[i]);
				const auto vtable = Util::TryReadVtable(reinterpret_cast<const void*>(candidate));
				HMODULE    module = nullptr;
				if (!vtable || !GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT, reinterpret_cast<LPCWSTR>(vtable), &module)) {
					continue;
				}
				char path[MAX_PATH]{};
				GetModuleFileNameA(module, path, MAX_PATH);
				const std::string_view full(path);
				const auto             slash = full.find_last_of("\\/");
				const auto             file = slash == std::string_view::npos ? full : full.substr(slash + 1);
				strncpy_s(a_item.instanceModule, file.data(), _TRUNCATE);
				if (_stricmp(a_item.instanceModule, "d3d11.dll") != 0) {
					continue;
				}
				ID3D11Buffer* buffer = nullptr;
				if (FAILED(reinterpret_cast<IUnknown*>(candidate)->QueryInterface(__uuidof(ID3D11Buffer), reinterpret_cast<void**>(&buffer))) || !buffer) {
					continue;
				}
				D3D11_BUFFER_DESC desc{};
				buffer->GetDesc(&desc);
				buffer->Release();
				a_item.instanceWord = i;
				a_item.instanceBytes = desc.ByteWidth;
				a_item.instanceStride = desc.StructureByteStride;
				a_item.instanceBind = desc.BindFlags;
				a_item.instanceMisc = desc.MiscFlags;
				a_item.instanceUsage = static_cast<std::uint32_t>(desc.Usage);
				return;
			}
		}

		void RecordKept(const FrameContext& a_context, const RE::NiAVObject* a_object, const RE::NiBound& a_bound, const char* a_reason, bool a_dropped = false) noexcept
		{
			if (a_context.clock != g_dumpClock.load(std::memory_order_relaxed)) {
				return;
			}
			KeptItem item{};
			item.reason = a_reason;
			item.dropped = a_dropped;
			item.radius = a_bound.fRadius;
			const auto& camera = a_context.snapshot->camera;
			const float dx = a_bound.center.x - camera.eye[0];
			const float dy = a_bound.center.y - camera.eye[1];
			const float dz = a_bound.center.z - camera.eye[2];
			item.distance = std::sqrt(dx * dx + dy * dy + dz * dz);
			Util::TryGetRTTIName(a_object, item.type, sizeof(item.type));
			Util::TryGetObjectName(a_object, item.name, sizeof(item.name));
			DescribeGeometry(a_object, item);
			if (item.buffer) {
				D3D11_BUFFER_DESC desc{};
				reinterpret_cast<ID3D11Buffer*>(item.buffer)->GetDesc(&desc);
				item.byteWidth = desc.ByteWidth;
				item.usage = static_cast<std::uint32_t>(desc.Usage);
			}
			DescribeMerged(a_object, item);
			if (item.merged && item.instanceObject) {
				DescribeInstanceBuffer(item);
			}
			std::scoped_lock lock(g_keptLock);
			if (g_kept.size() < 8192) {
				g_kept.push_back(item);
			}
		}

		const char* VerdictName(Verdict a_verdict) noexcept
		{
			switch (a_verdict) {
			case Verdict::kVisible:
				return "visible";
			case Verdict::kEdge:
				return "edge";
			case Verdict::kNear:
				return "near";
			case Verdict::kInvalid:
				return "bad-bound";
			default:
				return "?";
			}
		}

		// Hidden streak for a hidden verdict now: consecutive only if the object tested hidden in the
		// previous frame too (a visible verdict writes nothing, so it breaks the chain by omission).
		std::uint32_t HiddenStreak(std::uint64_t a_value, std::uint32_t a_clock) noexcept
		{
			const auto last = static_cast<std::uint32_t>(a_value);
			const auto streak = static_cast<std::uint32_t>((a_value >> 32) & 0xFF);
			if (last == a_clock) {
				return std::max(streak, 1u);  // added twice this frame
			}
			return last + 1 == a_clock ? std::min(streak + 1, 255u) : 1u;
		}

		// Exempt types that tested hidden, sampled every 32nd frame, for the log (which kinds are kept); and the types
		// CBRO leaves out of group 0 entirely (never filed or rejected in every view), to name what another reader
		// of group 0 may be missing.
		std::mutex                                     g_exemptLock;
		std::unordered_map<std::uintptr_t, std::uint32_t> g_exemptTypes;
		std::unordered_map<std::uintptr_t, std::uint32_t> g_leftOutTypes;

		void SampleExemptType(const FrameContext& a_context, const RE::NiAVObject* a_object) noexcept
		{
			if (a_context.clock % 32 != 0) {
				return;
			}
			if (const auto vtable = Util::TryReadVtable(a_object)) {
				std::scoped_lock lock(g_exemptLock);
				++g_exemptTypes[vtable];
			}
		}

		void SampleLeftOutType(const FrameContext& a_context, const RE::NiAVObject* a_object) noexcept
		{
			if (a_context.clock % 32 != 0) {
				return;
			}
			if (const auto vtable = Util::TryReadVtable(a_object)) {
				std::scoped_lock lock(g_exemptLock);
				++g_leftOutTypes[vtable];
			}
		}

		// CPU time is sampled on every 8th frame (and scaled up in the report): reading the TSC twice per
		// object costs more than the rest of a quick verdict.
		constexpr std::uint32_t kTimingStride = 8;

		bool Timed(const FrameContext& a_context) noexcept
		{
			return a_context.clock % kTimingStride == 0;
		}

		// The decision made at Group::Add for the object now being added (same thread): the Block::Add
		// that follows inside the engine's Group::Add reuses it instead of testing again.
		struct TopDecision
		{
			const RE::NiAVObject* object{ nullptr };
			std::uint32_t         clock{ 0 };
			const RE::NiBound*    result{ nullptr };
		};
		thread_local TopDecision t_top;

		// Whether the parent of an object being added was dropped from the main view this frame. The engine
		// adds a node's children one after another, so the answer is kept for the siblings (per thread).
		struct ParentDecision
		{
			const RE::NiAVObject* parent{ nullptr };
			std::uint32_t         clock{ 0 };
			bool                  dropped{ false };
		};
		thread_local ParentDecision t_parent;

		bool ParentDropped(const RE::NiAVObject* a_object, std::uint32_t a_clock, std::uint32_t a_tag) noexcept
		{
			const auto parent = a_object->parent;
			if (!parent) {
				return false;
			}
			if (t_parent.parent != parent || t_parent.clock != a_clock) {
				t_parent = { parent, a_clock, g_drops.Contains(parent, a_tag) };
			}
			return t_parent.dropped;
		}

		// ---- the verdict cache -----------------------------------------------------------------------------
		// The v1.23 run put all of the per-object test time (1.7-2.0 ms per frame at the light spot) on the main
		// thread: every object the engine offers is tested from scratch every frame although, standing still,
		// nothing about it changed. So each object's last outcome is kept and reused while everything it depended
		// on still holds: the camera (and the depth frame's camera) within the cache tolerance of the epoch's
		// reference (Runtime keeps the epoch id; tested bounds are dilated by the tolerance and by depth x the turn
		// tolerance, so a verdict stays conservative anywhere inside it), the object's own bound unchanged (16-byte
		// compare), the sun's direction/state unchanged (its epoch id) and, for outcomes that rest on the depth, no
		// change in the 8x8 Hi-Z blocks the test read since its readback (HiZ marks the blocks whose farthest or
		// nearest depth changed against the previous readback). A hidden object reused for a frame counts that
		// frame toward its confirmation streak (the evidence still holds); one that was merely out of view keeps
		// its streak (that says nothing about the depth); one seen visible, near or at the edge starts over.
		//
		// The engine offers the objects in the same order every frame (the scene walk, then the block jobs), so the
		// cache is a per-thread sequential stream of last frame's records read with a cursor (a short look-ahead
		// resyncs after an insertion): no hashing, prefetch-friendly. One stream for the top-level adds
		// (Group::Add), one for the children (Block::Add from the jobs).

		enum class Outcome : std::uint8_t
		{
			kNone,
			kOutside,  // entirely outside the current view (geometry only)
			kHidden,   // behind the depth (rejected once its streak reaches confirmFrames)
			kKept,     // drawn: visible, near, at the edge, exempt, a light that reaches in, or a bad bound
		};

		enum class SunOutcome : std::uint8_t
		{
			kNone,      // not evaluated (the main view needed the object anyway)
			kUnneeded,  // sun shadows off, or the shadow can't reach the view (geometry)
			kBehind,    // the shadow falls only behind visible surfaces (needs its streak)
			kNeeded,
			kUnknown,   // the sun's state couldn't be verified: needed
		};

		enum RecordFlags : std::uint8_t
		{
			kRecordDepth = 1 << 0,     // a hidden view outcome with its evidence recorded (rect, threshold, blocks)
			kRecordLight = 1 << 2,     // the object is a light (never skipped as out of view)
			kRecordNoCache = 1 << 3,   // never reuse (a light's reach test, a table-full case)
		};

		struct Record
		{
			const void*   object{ nullptr };
			float         bound[4]{};         // the bound tested (before dilation)
			std::uint32_t viewEpoch{ 0 };
			std::uint32_t readback{ 0 };      // the depth a hidden view outcome holds against (its evaluation's, or its last re-check's)
			std::uint32_t sunReadback{ 0 };   // the depth the sun outcome read
			std::uint16_t rect[4]{};          // hidden evidence: level-0 texels x0, y0, x1, y1 that were all nearer than `threshold`
			float         threshold{ 0.0f };  // ... in buffer depth
			std::uint8_t  blocks[4]{ 255, 0, 0, 0 };     // Hi-Z blocks the evidence covers, widened (255 = none)
			std::uint8_t  sunBlocks[4]{ 255, 0, 0, 0 };  // Hi-Z blocks the sun's shadow test read, widened (255 = none)
			Outcome       outcome{ Outcome::kNone };
			std::uint8_t  verdict{ 0 };       // the Verdict behind the outcome (for the stats)
			SunOutcome    sun{ SunOutcome::kNone };
			std::uint8_t  sunEpoch{ 0 };
			std::uint8_t  streak{ 0 };        // consecutive frames hidden
			std::uint8_t  sunStreak{ 0 };     // consecutive frames with the shadow behind surfaces
			std::uint8_t  flags{ 0 };
			std::uint8_t  cellHint{ 0 };      // mesh shapes: the cell last seen visible (TestShape starts there)
		};
		static_assert(sizeof(Record) == 64);

		struct Stream
		{
			std::vector<Record> prev;
			std::vector<Record> cur;
			std::size_t         cursor{ 0 };
			std::uint32_t       clock{ 0 };

			void Begin(std::uint32_t a_clock)
			{
				if (clock != a_clock) {
					// A frame in which this thread judged nothing (a blocked frame, no depth yet) keeps the older
					// records: the validity checks still guard their reuse, and the streaks survive the gap.
					if (!cur.empty()) {
						prev.swap(cur);
					}
					cur.clear();
					cursor = 0;
					clock = a_clock;
				}
			}

			// Last frame's record for a_object if it is where the sequence says (or within a few slots after an
			// insertion); the cursor moves past it.
			const Record* Find(const void* a_object) noexcept
			{
				// The common case: the next record, or one of the next few (an object left the sequence). Then a
				// wider scan (a whole node's worth left it, e.g. a pruned container); a miss leaves the cursor.
				const auto count = prev.size();
				for (std::size_t k = 0; k < 512 && cursor + k < count; ++k) {
					if (prev[cursor + k].object == a_object) {
						const Record* found = &prev[cursor + k];
						cursor += k + 1;
						return found;
					}
				}
				return nullptr;
			}

			void Push(const Record& a_record)
			{
				if (cur.size() < 65536) {
					cur.push_back(a_record);
				}
			}

			void Clear()
			{
				prev.clear();
				cur.clear();
				cursor = 0;
			}
		};

		struct ThreadStreams
		{
			Stream        top;
			Stream        child;
			std::uint32_t generation{ 0 };
		};
		std::atomic<std::uint32_t> g_streamGeneration{ 1 };  // bumped by ResetHistory: every thread's streams start over

		ThreadStreams& Streams(const FrameContext& a_context)
		{
			thread_local ThreadStreams* streams = nullptr;  // (one per thread, for the process lifetime)
			if (!streams) [[unlikely]] {
				streams = new ThreadStreams();
			}
			if (const auto generation = g_streamGeneration.load(std::memory_order_relaxed); streams->generation != generation) {
				streams->generation = generation;
				streams->top.Clear();
				streams->child.Clear();
			}
			streams->top.Begin(a_context.clock);
			streams->child.Begin(a_context.clock);
			return *streams;
		}

		// The Hi-Z blocks a rectangle (widened) covers, into a record's block field.
		void SetBlocks(const FrameContext& a_context, std::uint8_t a_blocks[4], const DepthRect& a_rect) noexcept
		{
			if (!a_rect.used || a_context.blocksW == 0 || a_context.blocksH == 0) {
				return;
			}
			const auto block = [](float a_texel, std::uint32_t a_count) {
				return static_cast<std::uint8_t>(std::clamp(static_cast<int>(a_texel) / 8, 0, static_cast<int>(a_count) - 1));
			};
			a_blocks[0] = block(a_rect.x0, a_context.blocksW);
			a_blocks[1] = block(a_rect.y0, a_context.blocksH);
			a_blocks[2] = block(a_rect.x1, a_context.blocksW);
			a_blocks[3] = block(a_rect.y1, a_context.blocksH);
		}

		// A hidden view outcome's evidence into the record: the texel rectangle and threshold to re-check it by, and
		// the blocks it rests on.
		void SetEvidence(const FrameContext& a_context, Record& a_record, const DepthRect& a_rect) noexcept
		{
			if (!a_rect.hidden) {
				return;
			}
			const auto texel = [](float a_value, bool a_up) {
				return static_cast<std::uint16_t>(std::clamp(a_up ? std::ceil(a_value) : std::floor(a_value), 0.0f, 65535.0f));
			};
			a_record.rect[0] = texel(a_rect.hx0, false);
			a_record.rect[1] = texel(a_rect.hy0, false);
			a_record.rect[2] = texel(a_rect.hx1, true);
			a_record.rect[3] = texel(a_rect.hy1, true);
			a_record.threshold = a_rect.threshold;
			a_record.flags |= kRecordDepth;
			SetBlocks(a_context, a_record.blocks, a_rect);
		}

		// Whether any of the blocks moved in the invalidating direction (farthest farther, nearest nearer) since the
		// capture a_readback (HiZ's change map: a block's mark is the index of the capture that last changed it).
		bool BlocksChanged(const FrameContext& a_context, const std::uint8_t a_blocks[4], std::uint32_t a_readback) noexcept
		{
			if (a_blocks[0] == 255 || !a_context.blockChangedAt) {
				return true;  // nothing recorded: can't tell
			}
			for (std::uint32_t y = a_blocks[1]; y <= a_blocks[3]; ++y) {
				const auto* row = a_context.blockChangedAt + static_cast<std::size_t>(y) * a_context.blocksW;
				for (std::uint32_t x = a_blocks[0]; x <= a_blocks[2]; ++x) {
					if (row[x] > a_readback) {
						return true;
					}
				}
			}
			return false;
		}

		// The full view test, with the classification and the streak, into a fresh record. a_old is last frame's
		// record of the same object (streak continuity, the shape's cell hint) or null.
		void EvaluateView(const FrameContext& a_context, const Hooks::CullGroups::BlockAdd& a_add, const Record* a_old, Record& a_out, bool a_timed)
		{
			a_out = Record{};
			a_out.object = a_add.object;
			a_out.bound[0] = a_add.bound->center.x;
			a_out.bound[1] = a_add.bound->center.y;
			a_out.bound[2] = a_add.bound->center.z;
			a_out.bound[3] = a_add.bound->fRadius;
			a_out.viewEpoch = a_context.viewEpoch;
			a_out.readback = a_context.readback;

			Bump(kTested);
			DepthRect rect{};
			auto      verdict = Test(a_context, *a_add.bound, &rect);

			// A big static mesh the sphere can't settle: judge it by its actual shape (MeshProxy decides
			// which meshes have one; a mesh that moved or sways is left to its sphere).
			if (g_tunables.meshShapes && (verdict == Verdict::kVisible || verdict == Verdict::kNear || verdict == Verdict::kEdge) &&
				a_add.bound->fRadius >= kShapeMinRadius && a_context.snapshot->camera.viewSpace && (TypeOf(a_add.object) & kTypeCullable)) {
				RE::NiBound model{};
				if (const auto shape = MeshProxy::Find(a_add.object, model)) {
					if (Moved(a_add.object)) {
						Bump(kShapeMoving);
					} else if (a_add.object->parent && (TypeOf(a_add.object->parent) & kTypeLeafAnim)) {
						Bump(kShapeAnimated);
					} else {
						Bump(kShapeTests);
						std::uint32_t cells = 0;
						std::uint8_t  hint = a_old ? a_old->cellHint : std::uint8_t{ 0 };
						const auto    shapeStart = a_timed ? __rdtsc() : 0;
						const auto    shaped = TestShape(a_context, a_add.object, model, *shape, cells, hint, &rect);
						if (a_timed) {
							Bump(kCyclesShape, (__rdtsc() - shapeStart) * kTimingStride);
						}
						a_out.cellHint = hint;
						Bump(kShapeCells, cells);
						if (shaped == Verdict::kHidden || shaped == Verdict::kOutside) {
							Bump(shaped == Verdict::kHidden ? kShapeHidden : kShapeOutside);
							RecordKept(a_context, a_add.object, *a_add.bound, shaped == Verdict::kHidden ? "shape: hidden" : "shape: out of view", true);
							// Either way it goes through confirmation and the reject bound: the engine's own
							// frustum test would still pass its sphere, so "out of view" alone drops nothing.
							verdict = Verdict::kHidden;
						} else if (shaped == Verdict::kInvalid) {
							Bump(kShapeNoTransform);
						}
					}
				}
			}
			a_out.verdict = static_cast<std::uint8_t>(verdict);

			if (verdict == Verdict::kOutside) {
				a_out.outcome = Outcome::kOutside;
				Bump(kOutside);
				if (TypeOf(a_add.object) & kTypeLight) {
					a_out.flags |= kRecordLight;  // (its light can reach into the view: never skipped as out of view)
				}
			} else if (verdict != Verdict::kHidden) {
				a_out.outcome = Outcome::kKept;
				CountVerdict(verdict);
				if (verdict == Verdict::kVisible || verdict == Verdict::kNear || verdict == Verdict::kEdge || verdict == Verdict::kInvalid) {
					RecordKept(a_context, a_add.object, *a_add.bound, VerdictName(verdict));
				}
			} else {
				std::uint64_t value = 0;
				const auto    entry = Lookup(a_context, a_add.object, value);
				if (!entry) {
					a_out.outcome = Outcome::kKept;
					a_out.flags |= kRecordNoCache;
				} else if (value & kExempt) {
					a_out.outcome = Outcome::kKept;
					switch (static_cast<Exemption>((value >> 42) & 3)) {
					case Exemption::kLight:
						Bump(kExemptLight);
						RecordKept(a_context, a_add.object, *a_add.bound, "hidden, kept: carries a light");
						break;
					case Exemption::kActor:
						Bump(kExemptActor);
						RecordKept(a_context, a_add.object, *a_add.bound, "hidden, kept: actor part");
						break;
					default:
						Bump(kExemptType);
						SampleExemptType(a_context, a_add.object);
						RecordKept(a_context, a_add.object, *a_add.bound, "hidden, kept: type");
						break;
					}
				} else {
					// A light reaches past its scene bound: it is hidden only if its whole reach is (never cached:
					// the reach's depth rectangle isn't tracked).
					bool hidden = true;
					if (value & kIsLight) {
						a_out.flags |= kRecordLight | kRecordNoCache;
						Bump(kLightsTested);
						RE::NiBound reach{};
						if (!LightReach(a_add.object, *a_add.bound, reach)) {
							Bump(kInvalid);
							hidden = false;
						} else if (const auto lightVerdict = Test(a_context, reach); lightVerdict != Verdict::kHidden) {
							CountVerdict(lightVerdict);
							RecordKept(a_context, a_add.object, reach, "light reaches the view");
							hidden = false;
						}
					}
					if (!hidden) {
						a_out.outcome = Outcome::kKept;
					} else {
						a_out.outcome = Outcome::kHidden;
						// Consecutive frames hidden: last frame's record continues the count when it was hidden (or only
						// out of view, which says nothing about the depth); anything seen starts over.
						a_out.streak = a_old && (a_old->outcome == Outcome::kHidden || a_old->outcome == Outcome::kOutside) ?
						                   static_cast<std::uint8_t>(std::min<std::uint32_t>(a_old->streak + 1u, 255u)) :
						                   static_cast<std::uint8_t>(1u);
						if (a_out.streak < g_tunables.confirmFrames) {
							Bump(kConfirming);
							RecordKept(a_context, a_add.object, *a_add.bound, "hidden, confirming");
						} else if (Observing()) {
							Bump(kWouldReject);
						} else {
							Bump(kRejected);
							if (value & kIsLight) {
								Bump(kLightsRejected);
							}
						}
					}
				}
			}
			if (a_out.outcome == Outcome::kHidden) {
				SetEvidence(a_context, a_out, rect);  // (what a later frame re-checks the verdict by)
			}
		}

		// The sun part (group 0: whether the cascades need the object's shadow), into the record.
		void EvaluateSun(const FrameContext& a_context, const RE::NiBound& a_bound, const Record* a_old, Record& a_out)
		{
			a_out.sunEpoch = a_context.sunEpoch;
			a_out.sunReadback = a_context.readback;
			a_out.sunBlocks[0] = 255;
			switch (a_context.sun.state) {
			case FrameContext::Sun::State::kOff:
				// The sun's cascades don't read group 0 this frame, but something else does: in the v1.31 interior run,
				// leaving hidden and out-of-view entries out of group 0 with the sun off made the lamp shadow maps
				// vanish (seven "shadowmap PB" accumulators a frame in previs mode, none in CBRO mode). Until that
				// reader is identified, a sun-off frame removes nothing from group 0 (main-view drops only).
				Bump(kSunOff);
				a_out.sun = SunOutcome::kUnknown;
				return;
			case FrameContext::Sun::State::kUnknown:
				a_out.sun = SunOutcome::kUnknown;
				return;
			default:
				break;
			}
			Bump(kSunTests);
			DepthRect  rect{};
			const auto verdict = TestSun(a_context, a_bound, &rect);
			if (verdict == SunVerdict::kOutside) {
				Bump(kSunOutside);
				a_out.sun = SunOutcome::kUnneeded;
				return;
			}
			if (rect.used) {
				SetBlocks(a_context, a_out.sunBlocks, rect);
			}
			if (verdict != SunVerdict::kHidden) {
				Bump(kSunNeeded);
				a_out.sun = SunOutcome::kNeeded;
				return;
			}
			a_out.sun = SunOutcome::kBehind;
			a_out.sunStreak = a_old && a_old->sun == SunOutcome::kBehind ? static_cast<std::uint8_t>(std::min<std::uint32_t>(a_old->sunStreak + 1u, 255u)) : static_cast<std::uint8_t>(1u);
			Bump(a_out.sunStreak < g_tunables.confirmFrames ? kSunConfirming : kSunHidden);
		}

		bool SunUnneededIn(const Record& a_record) noexcept
		{
			return a_record.sun == SunOutcome::kUnneeded || (a_record.sun == SunOutcome::kBehind && a_record.sunStreak >= g_tunables.confirmFrames);
		}

		enum class Reuse : std::uint8_t
		{
			kNo,
			kYes,
			kRechecked,  // a hidden outcome whose evidence was re-checked against the current depth (and held)
		};

		// Whether last frame's record can stand for this frame's view outcome. The camera epoch and the object's bound
		// must hold for any outcome. Then by outcome:
		//   - out of view: geometry alone (the epoch's dilation covers the camera);
		//   - kept (drawn): no depth change can make drawing unsafe, so it stands until its periodic turn comes
		//     (kKeptRecheckFrames), whatever the depth did. Up to v1.26 any change in its blocks re-evaluated it, and
		//     under a camera's idle sway (silhouettes moving a pixel a frame) that was every depth-reading object every
		//     frame: the v1.26 log had 2,700 of 3,100 evaluations a frame down to "depth changed" while standing still;
		//   - hidden: the evidence must still hold. It does when none of its blocks moved since the depth it was last
		//     verified against; otherwise the evidence rectangle is re-checked against the current depth at its own
		//     threshold (one coarse query; a pass is as good as a re-evaluation, and cheaper than one by the full test's
		//     refinement, rays and, for mesh shapes, every cell).
		Reuse ViewValid(const FrameContext& a_context, const Record& a_old, const RE::NiBound& a_bound) noexcept
		{
			if (a_old.flags & kRecordNoCache) {
				Bump(kCacheNoCache);
				return Reuse::kNo;
			}
			if (a_old.viewEpoch != a_context.viewEpoch) {
				Bump(kCacheEpoch);
				return Reuse::kNo;
			}
			if (a_old.bound[0] != a_bound.center.x || a_old.bound[1] != a_bound.center.y || a_old.bound[2] != a_bound.center.z || a_old.bound[3] != a_bound.fRadius) {
				Bump(kCacheBound);
				return Reuse::kNo;
			}
			switch (a_old.outcome) {
			case Outcome::kOutside:
				return Reuse::kYes;
			case Outcome::kKept:
				if ((a_context.clock + static_cast<std::uint32_t>(reinterpret_cast<std::uintptr_t>(a_old.object) >> 4)) % kKeptRecheckFrames == 0) {
					Bump(kCacheKeptRecheck);
					return Reuse::kNo;
				}
				return Reuse::kYes;
			case Outcome::kHidden:
				if (!(a_old.flags & kRecordDepth)) {
					Bump(kCacheDepth);  // no evidence recorded: can't tell
					return Reuse::kNo;
				}
				if (a_old.readback == a_context.readback || !BlocksChanged(a_context, a_old.blocks, a_old.readback)) {
					return Reuse::kYes;
				}
				if (a_context.snapshot->AllNearer(a_old.rect[0], a_old.rect[1], a_old.rect[2], a_old.rect[3], a_old.threshold, kRecheckRefine)) {
					Bump(kCacheRecheck);
					return Reuse::kRechecked;
				}
				Bump(kCacheDepth);
				return Reuse::kNo;
			default:
				return Reuse::kNo;
			}
		}

		// Whether a reused record's sun outcome still stands (its epoch already matched): geometry outcomes always; one
		// that rests on the depth (the shadow falls only behind visible surfaces) while none of the blocks its test
		// read moved since.
		bool SunValid(const FrameContext& a_context, const Record& a_old) noexcept
		{
			if (a_old.sun != SunOutcome::kBehind) {
				return true;
			}
			return a_old.sunBlocks[0] != 255 && (a_old.sunReadback == a_context.readback || !BlocksChanged(a_context, a_old.sunBlocks, a_old.sunReadback));
		}

		void ReuseView(const Record& a_old, Record& a_out) noexcept
		{
			a_out = a_old;
			Bump(kTested);
			Bump(kCacheHits);
			switch (a_out.outcome) {
			case Outcome::kOutside:
				Bump(kOutside);
				break;
			case Outcome::kHidden:
				a_out.streak = static_cast<std::uint8_t>(std::min<std::uint32_t>(a_out.streak + 1u, 255u));
				if (a_out.streak < g_tunables.confirmFrames) {
					Bump(kConfirming);
				} else if (Observing()) {
					Bump(kWouldReject);
				} else {
					Bump(kRejected);
				}
				break;
			default:
				CountVerdict(static_cast<Verdict>(a_out.verdict));
				break;
			}
		}

		void ReuseSun(const FrameContext& a_context, Record& a_out) noexcept
		{
			switch (a_out.sun) {
			case SunOutcome::kBehind:
				a_out.sunStreak = static_cast<std::uint8_t>(std::min<std::uint32_t>(a_out.sunStreak + 1u, 255u));
				Bump(a_out.sunStreak < g_tunables.confirmFrames ? kSunConfirming : kSunHidden);
				break;
			case SunOutcome::kUnneeded:
				Bump(kSunOutside);
				break;
			case SunOutcome::kNeeded:
				Bump(kSunNeeded);
				break;
			case SunOutcome::kUnknown:
				if (a_context.sun.state == FrameContext::Sun::State::kOff) {
					Bump(kSunOff);
				}
				break;
			default:
				break;
			}
		}

		// The object's outcome for this frame: reused from last frame's record when everything it depended on
		// still holds, evaluated otherwise; the sun part too when a_wantSun and the main view doesn't need the
		// object. The record is pushed to this frame's stream either way.
		void Judge(const FrameContext& a_context, Stream& a_stream, const Hooks::CullGroups::BlockAdd& a_add, bool a_wantSun, Record& a_out)
		{
			const Record* old = a_stream.Find(a_add.object);
			const bool    reusable = a_context.cacheEnabled && a_context.clock != g_dumpClock.load(std::memory_order_relaxed);
			const bool    timed = Timed(a_context);
			bool          reused = false;
			if (old && reusable) {
				if (const auto reuse = ViewValid(a_context, *old, *a_add.bound); reuse != Reuse::kNo) {
					ReuseView(*old, a_out);
					if (reuse == Reuse::kRechecked) {
						a_out.readback = a_context.readback;  // the evidence was verified against this depth
					}
					reused = true;
				}
			}
			if (!reused) {
				if (!old) {
					Bump(kCacheNew);
				}
				const auto start = timed ? __rdtsc() : 0;
				EvaluateView(a_context, a_add, old, a_out, timed);
				if (timed) {
					Bump(kCyclesEvaluate, (__rdtsc() - start) * kTimingStride);
				}
			}
			if (a_wantSun && !(a_out.flags & kRecordLight)) {
				const bool unneededByView = a_out.outcome == Outcome::kOutside || (a_out.outcome == Outcome::kHidden && a_out.streak >= g_tunables.confirmFrames);
				const bool sunReusable = reused && old->sun != SunOutcome::kNone && old->sunEpoch == a_context.sunEpoch;
				if (!unneededByView) {
					a_out.sun = SunOutcome::kNone;
				} else if (sunReusable && SunValid(a_context, *old)) {
					ReuseSun(a_context, a_out);
				} else {
					if (sunReusable) {
						Bump(kCacheSunDepth);
					}
					const auto start = timed ? __rdtsc() : 0;
					EvaluateSun(a_context, *a_add.bound, old, a_out);
					if (timed) {
						Bump(kCyclesSun, (__rdtsc() - start) * kTimingStride);
					}
				}
			}
			a_stream.Push(a_out);
		}

		// Whether the sun's cascades can do without a caster (node pruning, on a node's entries' sphere). "Outside"
		// is geometry alone and holds at once; "hidden" rests on the depth frame, so like an object's it must hold
		// for confirmFrames consecutive frames (streak in the history table).
		bool SunUnneeded(const FrameContext& a_context, RE::NiAVObject* a_object, const RE::NiBound& a_bound) noexcept
		{
			switch (a_context.sun.state) {
			case FrameContext::Sun::State::kOff:
				Bump(kSunOff);
				return false;  // (group 0 has other readers with the sun off: see EvaluateSun)
			case FrameContext::Sun::State::kUnknown:
				return false;
			default:
				break;
			}
			Bump(kSunTests);
			const auto verdict = TestSun(a_context, a_bound);
			if (verdict == SunVerdict::kOutside) {
				Bump(kSunOutside);
				return true;
			}
			if (verdict != SunVerdict::kHidden) {
				Bump(kSunNeeded);
				return false;
			}
			const auto entry = FindEntry(reinterpret_cast<std::uintptr_t>(a_object));
			if (!entry) {
				Bump(kTableFull);
				return false;
			}
			const auto streak = HiddenStreak(entry->sun.load(std::memory_order_relaxed), a_context.clock);
			entry->sun.store((static_cast<std::uint64_t>(streak) << 32) | a_context.clock, std::memory_order_relaxed);
			if (streak < g_tunables.confirmFrames) {
				Bump(kSunConfirming);
				return false;
			}
			Bump(kSunHidden);
			return true;
		}

		// ---- the Block::Add filter (children, and top-level entries not left out at Group::Add) -------------

		const RE::NiBound* FilterDecide(const FrameContext& a_context, const Hooks::CullGroups::BlockAdd& a_add)
		{
			auto&  streams = Streams(a_context);
			Record out{};
			switch (a_add.kind) {
			case Hooks::CullGroups::GroupKind::kMainOnly: {
				Judge(a_context, streams.child, a_add, false, out);
				return out.outcome == Outcome::kHidden && out.streak >= g_tunables.confirmFrames && !Observing() ? &a_context.reject : nullptr;
			}
			case Hooks::CullGroups::GroupKind::kSunShared: {
				// Group 0 (the shadow casters; the sun's cascades read it too): rejected in every view only when neither
				// the main camera nor the sun's shadow needs the object; dropped from the main view alone when only the
				// shadow is needed. A child of a main-view drop inherits it (its parent's shadow was needed, so it stays
				// for the cascades).
				Bump(kShared);
				if ((a_add.object->GetFlags() >> 11) & 1) {
					return nullptr;  // the engine's always-draw mark: Block::Add forces such entries visible
				}
				const auto tag = DropSet::Tag(a_context.clock);
				if (ParentDropped(a_add.object, a_context.clock, tag) && !IsActorRoot(a_add.object)) {
					Bump(kDropInherited);
					if (!g_drops.Insert(a_add.object, tag)) {
						Bump(kDropFull);
					}
					return nullptr;
				}
				Judge(a_context, streams.child, a_add, true, out);
				if (Observing()) {
					return nullptr;
				}
				const bool hiddenConfirmed = out.outcome == Outcome::kHidden && out.streak >= g_tunables.confirmFrames;
				if ((hiddenConfirmed || out.outcome == Outcome::kOutside) && !(out.flags & kRecordLight) && SunUnneededIn(out)) {
					Bump(kCasterRejected);
					SampleLeftOutType(a_context, a_add.object);
					return &a_context.reject;
				}
				if (hiddenConfirmed && !g_drops.Insert(a_add.object, tag)) {
					Bump(kDropFull);
				}
				return nullptr;
			}
			default: {
				// A group with other possible readers: only ever dropped from the main view.
				Bump(kShared);
				if ((a_add.object->GetFlags() >> 11) & 1) {
					return nullptr;
				}
				const auto tag = DropSet::Tag(a_context.clock);
				if (ParentDropped(a_add.object, a_context.clock, tag) && !IsActorRoot(a_add.object)) {
					Bump(kDropInherited);
					if (!g_drops.Insert(a_add.object, tag)) {
						Bump(kDropFull);
					}
					return nullptr;
				}
				Judge(a_context, streams.child, a_add, false, out);
				if (out.outcome == Outcome::kHidden && out.streak >= g_tunables.confirmFrames && !Observing() && !g_drops.Insert(a_add.object, tag)) {
					Bump(kDropFull);
				}
				return nullptr;
			}
			}
		}

		const RE::NiBound* Filter(const Hooks::CullGroups::BlockAdd& a_add)
		{
			const auto context = CurrentContext();
			if (!context) {
				if (g_active.load(std::memory_order_relaxed)) {
					Bump(kNoContext);
				}
				return nullptr;
			}
			// Decided at Group::Add already (same thread): not again.
			if (t_top.object == a_add.object && t_top.clock == context->clock) {
				t_top.object = nullptr;
				return t_top.result;
			}
			if (!Timed(*context)) {
				return FilterDecide(*context, a_add);
			}
			const auto start = __rdtsc();
			const auto result = FilterDecide(*context, a_add);
			Bump(kCycles, (__rdtsc() - start) * kTimingStride);
			return result;
		}

		// The main accumulator registering an object while DrawWorld culls: leave out what was dropped.
		// (a_object is never dereferenced: only its address is looked up.)
		bool MainView(const RE::NiAVObject* a_object)
		{
			const auto context = CurrentContext();
			if (!context) {
				return false;
			}
			Bump(kRegistered);
			if (!g_drops.Contains(a_object, DropSet::Tag(context->clock))) {
				return false;
			}
			Bump(kRegistrationsDropped);
			return true;
		}

		// Group::Add of a main-pass object into a main-only group or group 0 with no group markers pending: an
		// object nothing needs is left out entirely, so the engine never files, tests or walks it (with previs
		// off it would otherwise process every object in the loaded cells). In a main-only group that is an
		// object out of view or confirmed hidden; in group 0 its sun shadow must also be unneeded. Never for an
		// object the engine marks always-draw, and never in observe-only mode.
		bool SkipTopLevel(RE::NiAVObject* a_object, const RE::NiBound* a_bound, Hooks::CullGroups::GroupKind a_kind)
		{
			const auto context = CurrentContext();
			if (!context) {
				return false;
			}
			const bool timed = Timed(*context);
			const auto start = timed ? __rdtsc() : 0;

			bool skip = false;
			if (!((a_object->GetFlags() >> 11) & 1)) {
				auto&      streams = Streams(*context);
				const bool group0 = a_kind == Hooks::CullGroups::GroupKind::kSunShared;
				const Hooks::CullGroups::BlockAdd add{ nullptr, a_object, a_bound, -1, 0, true, a_kind, 0 };
				Record                            out{};
				if (group0) {
					Bump(kShared);
				}
				Judge(*context, streams.top, add, group0, out);
				const bool hiddenConfirmed = out.outcome == Outcome::kHidden && out.streak >= g_tunables.confirmFrames;
				const bool light = (out.flags & kRecordLight) != 0;
				if (!Observing()) {
					if (group0) {
						if ((hiddenConfirmed || out.outcome == Outcome::kOutside) && !light && SunUnneededIn(out)) {
							Bump(kCasterSkipped);
							SampleLeftOutType(*context, a_object);
							skip = true;
						} else if (hiddenConfirmed && !g_drops.Insert(a_object, DropSet::Tag(context->clock))) {
							Bump(kDropFull);
						}
					} else if (hiddenConfirmed) {
						Bump(kSkippedHidden);
						skip = true;
					} else if (out.outcome == Outcome::kOutside && !light) {
						// (a light's bound can be off-view while its light still reaches into view)
						Bump(kSkippedOutside);
						skip = true;
					}
				}
			}
			if (!skip) {
				t_top = { a_object, context->clock, nullptr };
			}
			if (timed) {
				Bump(kCycles, (__rdtsc() - start) * kTimingStride);
			}
			return skip;
		}

		// ---- whole cell nodes (the scene walk, before it adds a cell node's children one by one) --------------
		// A cell's child node 3 holds its precombined chunks and static refs (the walk adds them to group 0 one by
		// one). When that node lies entirely outside the view and its sun shadow can't reach the view either, it is
		// never walked: none of its objects is filed with any view. Its worldBound is the engine's own node bound
		// over its children, the same bound vanilla previs mode frustum-tests cell roots by (whole, as single entries
		// of group 1); the shadow condition is CBRO's addition. Only "outside" counts here, never "hidden behind
		// surfaces". Child 9's role is unknown: it is only counted.
		// What holds a node offered by a prune site. Only its ENTRIES matter: the objects the engine files one by one,
		// i.e. the node's children and, for a node the scene walk expands two levels (a cell's node 3, a_depth 2),
		// the children of its exact-NiNode children as well. An entry with the engine's always-draw bit (flags bit
		// 11) is registered whatever the frustum says (Block::Add forces it), and an actor's root moves: either holds
		// the node. The grouping nodes' own always-draw bits (the v1.20 run: cells' node 3 and the containers under
		// it all carry it) mean nothing, since the walk never files those nodes; they only stopped v1.17's and
		// v1.20's pruning. Lights don't hold either: the top-level early skip has left out light-carrying nodes
		// outside the view since v1.7 (lights live in the scene's light list, not in the culling groups). Scanned
		// once per epoch and cached in the node's history entry at bits [56,59) (such a node is never an entry, so
		// the object classification never shares its key). Too big to scan holds too.
		constexpr int  kNodeContentsShift = 56;
		std::uintptr_t g_niNodeVtable{ 0 };

		std::uint8_t EntryScan(RE::NiAVObject* a_node, int a_depth) noexcept
		{
			const auto root = a_node->IsNode();
			if (!root) {
				return kSubtreePlain;
			}
			constexpr std::size_t kBudget = 8192;
			std::uint8_t          found = kSubtreePlain;
			std::size_t           visited = 0;
			const auto            entry = [&](RE::NiAVObject* a_object) {
				if ((a_object->GetFlags() >> 11) & 1) {
					found |= kSubtreeAlwaysDraw;
				}
				if (IsActorRoot(a_object)) {
					found |= kSubtreeActor;
				}
			};
			for (auto& child : root->children) {
				const auto object = child.get();
				if (!object) {
					continue;
				}
				if (++visited > kBudget) {
					return found | kSubtreeLight;
				}
				const auto inner = a_depth >= 2 && Util::TryReadVtable(object) == g_niNodeVtable ? object->IsNode() : nullptr;
				if (!inner) {
					entry(object);
					continue;
				}
				for (auto& grandchild : inner->children) {
					const auto object2 = grandchild.get();
					if (!object2) {
						continue;
					}
					if (++visited > kBudget) {
						return found | kSubtreeLight;
					}
					entry(object2);
				}
			}
			return found;
		}

		std::uint8_t NodeContents(const FrameContext& a_context, RE::NiAVObject* a_node, int a_depth) noexcept
		{
			const auto key = reinterpret_cast<std::uintptr_t>(a_node);
			const auto entry = FindEntry(key);
			if (!entry) {
				Bump(kTableFull);
				return kSubtreeLight;
			}
			const auto epoch = static_cast<std::uint64_t>(((a_context.clock + static_cast<std::uint32_t>((key >> 4) % kEpochFrames)) / kEpochFrames) & 0xFF);
			auto       value = entry->value.load(std::memory_order_relaxed);
			if (!(value & kClassified) || ((value >> kEpochShift) & 0xFF) != epoch) {
				const auto found = EntryScan(a_node, a_depth);
				value = (value & kLowBits) | kClassified | (epoch << kEpochShift) | (static_cast<std::uint64_t>(found & 7) << kNodeContentsShift);
				entry->value.store(value, std::memory_order_relaxed);
			}
			return static_cast<std::uint8_t>((value >> kNodeContentsShift) & 7);
		}

		// A grouping node's own worldBound is NOT maintained by the engine: the v1.20/v1.21 samples showed r=1 on every
		// cell node and container, and that is what their always-draw bit stands for ("never cull me by my bound").
		// v1.21 judged them by that bound and most of the scene vanished. So a node is judged by its ENTRIES' bounds
		// (the objects the engine files, whose bounds it culls by), all of them, every frame: skippable only if every
		// entry lies entirely outside the current view (with the usual dilation), with the sphere around all of them
		// (for the sun sweep) returned. An entry whose bound looks unmaintained too (radius <= 1, e.g. a nested
		// container) or invalid holds the node; the pass stops at the first entry that isn't outside.
		bool EntriesOutside(const FrameContext& a_context, RE::NiAVObject* a_node, int a_depth, RE::NiBound& a_union, std::uint32_t& a_count) noexcept
		{
			const auto root = a_node->IsNode();
			if (!root || !std::isfinite(a_context.view[0])) {
				return false;
			}
			const auto& camera = a_context.snapshot->camera;
			if (!camera.viewSpace) {
				return false;
			}
			bool       any = false;
			const auto consider = [&](RE::NiAVObject* a_object) -> bool {
				const auto& bound = a_object->worldBound;
				if (!(bound.fRadius > 1.0f) || !(bound.fRadius < 1.0e6f) || !std::isfinite(bound.center.x) || !std::isfinite(bound.center.y) || !std::isfinite(bound.center.z)) {
					return false;  // unmaintained or bad: the node is walked normally
				}
				const float dx = bound.center.x - camera.origin[0];
				const float dy = bound.center.y - camera.origin[1];
				const float dz = bound.center.z - camera.origin[2];
				const float z = dx * camera.viewDir[0] + dy * camera.viewDir[1] + dz * camera.viewDir[2];
				const float x = dx * camera.viewRight[0] + dy * camera.viewRight[1] + dz * camera.viewRight[2];
				const float y = dx * camera.viewUp[0] + dy * camera.viewUp[1] + dz * camera.viewUp[2];
				if (!SphereOutsideView(a_context, x, y, z, bound.fRadius + a_context.dilateMove + std::max(z, 0.0f) * a_context.angularSlack)) {
					return false;
				}
				if (!any) {
					a_union = bound;
					any = true;
				} else {
					// Grow the union sphere to hold this one.
					const float cx = bound.center.x - a_union.center.x;
					const float cy = bound.center.y - a_union.center.y;
					const float cz = bound.center.z - a_union.center.z;
					const float d = std::sqrt(cx * cx + cy * cy + cz * cz);
					if (d + bound.fRadius > a_union.fRadius) {
						if (d + a_union.fRadius <= bound.fRadius) {
							a_union = bound;
						} else {
							const float radius = (d + a_union.fRadius + bound.fRadius) * 0.5f;
							const float t = (radius - a_union.fRadius) / d;
							a_union.center.x += cx * t;
							a_union.center.y += cy * t;
							a_union.center.z += cz * t;
							a_union.fRadius = radius;
						}
					}
				}
				++a_count;
				return true;
			};
			for (auto& child : root->children) {
				const auto object = child.get();
				if (!object) {
					continue;
				}
				const auto inner = a_depth >= 2 && Util::TryReadVtable(object) == g_niNodeVtable ? object->IsNode() : nullptr;
				if (!inner) {
					if (!consider(object)) {
						return false;
					}
					continue;
				}
				for (auto& grandchild : inner->children) {
					const auto object2 = grandchild.get();
					if (object2 && !consider(object2)) {
						return false;
					}
				}
			}
			return any;
		}

		// The first few nodes offered, for the log: what they are.
		std::atomic<int> g_nodeSamples{ 0 };

		void SampleNode(RE::NiAVObject* a_node, std::uint32_t a_index, std::uint8_t a_contents, bool a_outside, std::uint32_t a_entries, float a_unionRadius)
		{
			if (g_nodeSamples.load(std::memory_order_relaxed) >= 8 || g_nodeSamples.fetch_add(1) >= 8) {
				return;
			}
			char type[48]{};
			char name[48]{};
			Util::TryGetRTTIName(a_node, type, sizeof(type));
			Util::TryGetObjectName(a_node, name, sizeof(name));
			const auto node = a_node->IsNode();
			logger::info(
				"node pruning: sample index {} type {} name '{}' flags {:#x} children {} own r={:.0f} (unmaintained: not used) contents {:#x} (1 too big, 2 actor entry, 4 always-draw entry) {}: {} entries outside, their sphere r={:.0f}",
				a_index, type[0] ? type : "?", name, a_node->GetFlags(), node ? node->children.size() : 0u, a_node->worldBound.fRadius, a_contents,
				a_outside ? "all entries outside the view" : "meets the view (or held)", a_entries, a_unionRadius);
		}

		bool SkipCellNode(RE::NiAVObject* a_node, std::uint32_t a_index)
		{
			const auto context = CurrentContext();
			if (!context || Observing()) {
				return false;
			}
			if (a_index == 9) {
				Bump(kCellNodesOther);
				return false;
			}
			const bool timed = Timed(*context);
			const auto start = timed ? __rdtsc() : 0;
			// A cell's node 3 is expanded two levels by the walk (its exact-NiNode children are containers); a
			// container or a DrawWorld root has its own children filed.
			const int depth = a_index == 3 ? 2 : 1;
			Bump(a_index == 3 ? kCellNodesSeen : a_index == Hooks::CullGroups::kContainerIndex ? kContainersSeen : kRootsSeen);
			// (the node's own always-draw bit means nothing here: the node itself is never an entry; its entries'
			// bits are what the scan looks for)
			if ((a_node->GetFlags() >> 11) & 1) {
				Bump(kCellNodesAlwaysDraw);
			}
			bool          skip = false;
			bool          outside = false;
			RE::NiBound   unionBound{};
			std::uint32_t entries = 0;
			// (never the node's own worldBound: unmaintained, see EntriesOutside)
			const auto contents = NodeContents(*context, a_node, depth);
			if (contents != 0) {
				Bump(kCellNodesHeld);
			} else if (!(outside = EntriesOutside(*context, a_node, depth, unionBound, entries))) {
				Bump(kCellNodesInView);
			} else if (SunUnneeded(*context, a_node, unionBound)) {
				Bump(kCellNodesSkipped);
				Bump(kCellNodeEntriesSkipped, entries);
				skip = true;
			} else {
				Bump(kCellNodesSunNeeded);
			}
			if (g_nodeSamples.load(std::memory_order_relaxed) < 8) {
				SampleNode(a_node, a_index, contents, outside, entries, unionBound.fRadius);
			}
			if (timed) {
				Bump(kCycles, (__rdtsc() - start) * kTimingStride);
			}
			return skip;
		}

		// ---- per-frame spread: how steady the culled set is from one frame to the next -----------------
		// Sampled once per culled frame at the cull stage's end (main thread, jobs done), from the running
		// counters. "Other views" = every accumulator's registrations (the hook count, which includes the sun's
		// cascades registered later in the previous frame and the lamps) less the main view's.

		struct Spread
		{
			double        min{ std::numeric_limits<double>::infinity() };
			double        max{ -std::numeric_limits<double>::infinity() };
			double        sum{ 0.0 };
			double        sumSq{ 0.0 };
			double        steps{ 0.0 };  // |value - previous frame's value|, summed
			std::uint32_t count{ 0 };
			double        last{ 0.0 };
			bool          hasLast{ false };

			void Add(double a_value) noexcept
			{
				min = std::min(min, a_value);
				max = std::max(max, a_value);
				sum += a_value;
				sumSq += a_value * a_value;
				if (hasLast) {
					steps += std::abs(a_value - last);
				}
				last = a_value;
				hasLast = true;
				++count;
			}

			std::string Describe() const
			{
				if (!count) {
					return "-";
				}
				const double avg = sum / count;
				const double sd = std::sqrt(std::max(0.0, sumSq / count - avg * avg));
				return std::format("{:.0f}/{:.0f}/{:.0f} sd {:.0f} step {:.0f}", min, avg, max, sd, count > 1 ? steps / (count - 1) : 0.0);
			}

			void Clear() noexcept { *this = Spread{}; }
		};

		struct FrameTotals
		{
			std::uint64_t registered{ 0 };
			std::uint64_t dropped{ 0 };
			std::uint64_t rejected{ 0 };
			std::uint64_t confirming{ 0 };
			std::uint64_t allRegistrations{ 0 };
		};
		FrameTotals g_frameTotals;
		bool        g_frameTotalsKnown{ false };
		Spread      g_spreadKept;
		Spread      g_spreadOther;
		Spread      g_spreadRejected;
		Spread      g_spreadConfirming;

		// ---- merge-instanced meshes (their instance entries) ------------------------------------

		// Hidden only if every instance is hidden (or outside the current view): the engine draws the
		// whole mesh as soon as one instance entry passes.
		bool DecideMerged(const FrameContext& a_context, const Hooks::CullGroups::BlockAdd& a_add, Entry& a_entry, std::uint64_t a_value)
		{
			Bump(kMerged);
			auto verdict = Test(a_context, *a_add.bound);  // the whole mesh: hidden there means hidden everywhere
			if (verdict == Verdict::kOutside) {
				a_entry.value.store(a_value & ~kMergedReject, std::memory_order_relaxed);
				return false;  // the frustum test drops it anyway
			}

			auto hint = static_cast<std::uint32_t>(a_value >> kHintShift);
			if (verdict != Verdict::kHidden) {
				const auto count = Hooks::CullGroups::InstanceCount(a_add.object);
				if (count > 0 && count <= kMaxInstanceTests) {
					verdict = Verdict::kHidden;
					// Start at the instance last seen visible: a visible mesh usually stops at the first test.
					for (std::uint32_t k = 0; k < count; ++k) {
						const auto i = (hint + k) % count;
						Bump(kInstanceTests);
						const auto instance = Test(a_context, *Hooks::CullGroups::InstanceBound(a_add.object, i));
						if (instance != Verdict::kHidden && instance != Verdict::kOutside) {
							verdict = instance;
							hint = i & 0xFF;
							break;
						}
					}
				}
			}

			const bool hidden = verdict == Verdict::kHidden;
			const auto streak = NextStreak(a_value, hidden, a_context.clock);
			const bool reject = hidden && streak >= g_tunables.confirmFrames && !Observing();
			if (hidden) {
				Bump(streak < g_tunables.confirmFrames ? kMergedConfirming : reject ? kMergedRejected : kWouldReject);
			}
			a_entry.value.store(
				(a_value & ~(kLowBits | kMergedReject | (std::uint64_t{ 0xFF } << kHintShift))) |
					(static_cast<std::uint64_t>(streak) << 32) | a_context.clock |
					(reject ? kMergedReject : 0) | (static_cast<std::uint64_t>(hint & 0xFF) << kHintShift),
				std::memory_order_relaxed);
			return reject;
		}

		void Instances(const Hooks::CullGroups::BlockAdd& a_add, std::uint32_t a_first, std::uint32_t a_end)
		{
			const auto context = CurrentContext();
			if (!context) {
				return;
			}
			const auto start = __rdtsc();
			Bump(kInstanceEntries, a_end - a_first);

			bool          reject = false;
			std::uint64_t value = 0;
			if (const auto entry = Lookup(*context, a_add.object, value); entry && (value & kIsMerged) && !(value & kExempt)) {
				if (a_add.startIndex < 0) {
					reject = DecideMerged(*context, a_add, *entry, value);
				} else {
					// A continuation in a new block: follow the decision made at the fresh add this frame.
					reject = (value & kMergedReject) && static_cast<std::uint32_t>(value) == context->clock;
				}
			}
			if (reject) {
				for (auto i = a_first; i < a_end; ++i) {
					Hooks::CullGroups::WriteEntryBound(a_add.block, i, context->reject);
				}
				Bump(kInstanceEntriesRejected, a_end - a_first);
			}
			Bump(kCycles, __rdtsc() - start);
		}
	}

	void Install()
	{
		const auto& settings = Settings::Get();
		g_tunables.confirmFrames = settings.confirmFrames;
		g_tunables.nearDistance = settings.nearDistance;
		g_tunables.depthTolerance = settings.depthTolerance;
		g_tunables.depthSlack = settings.depthSlack;
		g_tunables.depthMin = settings.worldDepthMin;
		g_tunables.depthRange = settings.worldDepthMax - settings.worldDepthMin;
		g_tunables.observeOnly = settings.observeOnly;
		g_tunables.cullActors = settings.cullActors;
		g_tunables.meshShapes = settings.meshShapes;
		g_mergedVtable = RE::VTABLE::BSMergeInstancedTriShape[0].address();
		g_niNodeVtable = RE::VTABLE::NiNode[0].address();
		MeshProxy::Install();

		g_table = std::make_unique<Entry[]>(kTableSize);
		g_drops.Allocate();
		Calibrate();
		Hooks::CullGroups::SetFilter(&Filter);
		Hooks::CullGroups::SetInstanceFilter(&Instances);
		Hooks::CullGroups::SetGroupFilter(&SkipTopLevel);
		Hooks::CullGroups::SetMainViewFilter(&MainView);
		Hooks::CullGroups::SetCellNodeFilter(&SkipCellNode);
	}

	void EndFrameSample(std::uint64_t a_registrationsTotal)
	{
		const FrameTotals now{
			Total(kRegistered), Total(kRegistrationsDropped), Total(kRejected) + Total(kDropInherited),
			Total(kConfirming) + Total(kSunConfirming) + Total(kMergedConfirming), a_registrationsTotal
		};
		const bool culled = CurrentContext() != nullptr;
		if (g_frameTotalsKnown && culled) {
			const auto   d = [](std::uint64_t a_now, std::uint64_t a_then) { return static_cast<double>(a_now - a_then); };
			const double registered = d(now.registered, g_frameTotals.registered);
			g_spreadKept.Add(registered - d(now.dropped, g_frameTotals.dropped));
			g_spreadOther.Add(d(now.allRegistrations, g_frameTotals.allRegistrations) - registered);
			g_spreadRejected.Add(d(now.rejected, g_frameTotals.rejected));
			g_spreadConfirming.Add(d(now.confirming, g_frameTotals.confirming));
		}
		g_frameTotals = now;
		g_frameTotalsKnown = culled;  // (two culled frames in a row give a per-frame delta)
	}

	void BeginFrame(const FrameContext& a_context)
	{
		g_mainStats = &LocalStats();  // (BeginFrame runs on the main thread)
		if (g_resetRequested.exchange(false) || g_used.load() > kTableSize * 7 / 10) {
			ClearTable();
		}
		// Drop tags repeat every kTagCycle frames: forget the old ones before a tag is reused.
		if (const auto cycle = a_context.clock / DropSet::kTagCycle; cycle != g_dropCycle) {
			g_dropCycle = cycle;
			g_drops.Clear();
		}
		MeshProxy::BeginFrame();

		const auto next = (g_context.load() + 1 + 2) % 2;
		g_contexts[next] = a_context;
		SetSunPlanes(g_contexts[next]);
		SetLampPlanes(g_contexts[next]);
		g_context.store(next, std::memory_order_release);
		Bump(a_context.cull ? kFramesCulling : kFramesBlocked);
	}

	void SetActive(bool a_active)
	{
		g_active.store(a_active);
	}

	void SetObserveOnly(bool a_observeOnly)
	{
		g_observeOverride.store(a_observeOnly);
	}

	SphereVerdict TestSphere(const RE::NiPoint3& a_center, float a_radius) noexcept
	{
		const auto context = CurrentContext();
		if (!context) {
			return SphereVerdict::kUnknown;
		}
		RE::NiBound bound{};
		bound.center = a_center;
		bound.fRadius = a_radius;
		switch (Test(*context, bound)) {
		case Verdict::kHidden:
			return SphereVerdict::kHidden;
		case Verdict::kVisible:
			return SphereVerdict::kVisible;
		case Verdict::kOutside:
			return SphereVerdict::kOutOfView;
		default:
			return SphereVerdict::kUnknown;
		}
	}

	std::uint32_t Clock() noexcept
	{
		const auto index = g_context.load(std::memory_order_acquire);
		return index < 0 ? 0 : g_contexts[index].clock;
	}

	// A lamp caster's shadow is the cone from the lamp through the caster's sphere, from the caster's far side out
	// to the lamp's reach: a sweep along (caster - lamp) whose radius grows with the distance (SweepClip). Clipped to
	// where receivers can be (in front of the eye, inside the view), the kept part is a segment between two spheres;
	// it shadows nothing visible if every visible surface over its screen box is in front of its nearest point
	// (the max pyramid), or behind its farthest point (the nearest pyramid). Either way no surface is inside the
	// volume. Depth verdicts hold only after confirmFrames consecutive frames, like an object's.
	LampVerdict TestLampCaster(RE::NiAVObject* a_caster, const RE::NiPoint3& a_lamp, float a_reach, const RE::NiBound& a_bound) noexcept
	{
		const auto context = CurrentContext();
		if (!context || context->lampPlaneCount == 0) {
			return LampVerdict::kUnknown;
		}
		const auto& snapshot = *context->snapshot;
		const auto& camera = snapshot.camera;
		const float radius = a_bound.fRadius;
		if (!camera.viewSpace || !(camera.depthB < 0.0f) || !(radius > 0.0f) || !(radius < 1.0e6f) || !(a_reach > 0.0f) ||
			!std::isfinite(a_bound.center.x) || !std::isfinite(a_bound.center.y) || !std::isfinite(a_bound.center.z)) {
			return LampVerdict::kUnknown;
		}
		const auto toView = [&](const RE::NiPoint3& a_p, float a_out[3]) {
			const float dx = a_p.x - camera.origin[0];
			const float dy = a_p.y - camera.origin[1];
			const float dz = a_p.z - camera.origin[2];
			a_out[0] = dx * camera.viewRight[0] + dy * camera.viewRight[1] + dz * camera.viewRight[2];
			a_out[1] = dx * camera.viewUp[0] + dy * camera.viewUp[1] + dz * camera.viewUp[2];
			a_out[2] = dx * camera.viewDir[0] + dy * camera.viewDir[1] + dz * camera.viewDir[2];
		};
		float lamp[3], center[3];
		toView(a_lamp, lamp);
		toView(a_bound.center, center);
		const float r = radius + context->dilateMove;
		const float d[3]{ center[0] - lamp[0], center[1] - lamp[1], center[2] - lamp[2] };
		const float dist = std::sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2]);
		if (dist <= r + 1.0f) {
			return LampVerdict::kNeeded;  // the lamp is inside the caster: shadow everywhere
		}
		if (dist - r >= a_reach) {
			return LampVerdict::kOutside;  // entirely beyond the lamp's reach
		}
		const float dir[3]{ d[0] / dist, d[1] / dist, d[2] / dist };
		const float spread = r / dist;             // the cone widens with the distance from the lamp
		const float tMax = a_reach - dist + r;     // no light beyond the reach: no shadow either
		float       t0 = 0.0f, t1 = 0.0f;
		if (!ShadowGeometry::SweepClip(center, dir, r, spread, tMax, context->lampPlanes, context->lampPlaneCount, t0, t1)) {
			return LampVerdict::kOutside;
		}
		const float q0[3]{ center[0] + t0 * dir[0], center[1] + t0 * dir[1], center[2] + t0 * dir[2] };
		const float q1[3]{ center[0] + t1 * dir[0], center[1] + t1 * dir[1], center[2] + t1 * dir[2] };
		const float ra = r + t0 * spread;
		const float rb = r + t1 * spread;
		const float zMin = std::min(q0[2] - ra, q1[2] - rb);
		const float zMax = std::max(q0[2] + ra, q1[2] + rb);
		if (zMin < g_tunables.nearDistance) {
			return LampVerdict::kNeeded;  // reaches the camera: can't be judged
		}
		float x0, x1, y0, y1, u0, u1, v0, v1;
		SphereExtent(q0[0], q0[2], ra, x0, x1);
		SphereExtent(q0[1], q0[2], ra, y0, y1);
		SphereExtent(q1[0], q1[2], rb, u0, u1);
		SphereExtent(q1[1], q1[2], rb, v0, v1);
		x0 = std::min(x0, u0) * camera.scaleX;
		x1 = std::max(x1, u1) * camera.scaleX;
		y0 = std::min(y0, v0) * camera.scaleY;
		y1 = std::max(y1, v1) * camera.scaleY;
		switch (ClipToView(*context, x0, x1, y0, y1)) {
		case Verdict::kOutside:
			return LampVerdict::kOutside;
		case Verdict::kEdge:
			return LampVerdict::kNeeded;
		default:
			break;
		}
		const float width = static_cast<float>(snapshot.width[0]);
		const float height = static_cast<float>(snapshot.height[0]);
		const float px0 = (x0 * 0.5f + 0.5f) * width - 0.25f;
		const float py0 = (0.5f - y1 * 0.5f) * height - 0.25f;
		const float px1 = (x1 * 0.5f + 0.5f) * width + 0.25f;
		const float py1 = (0.5f - y0 * 0.5f) * height + 0.25f;
		const auto  bufferDepth = [&](float a_z) { return std::min(g_tunables.depthMin + g_tunables.depthRange * (camera.depthA + camera.depthB / a_z), 0.999999f); };

		const float limitBehind = (zMin - g_tunables.depthSlack) / (1.0f + g_tunables.depthTolerance);
		const float limitFront = (zMax + g_tunables.depthSlack) * (1.0f + g_tunables.depthTolerance);
		// No visible surface inside the volume's depth range at any texel of its box: each texel's surfaces are all in
		// front of the volume or all behind it (the sky counts as behind). Whole-box "all behind" or "all in front"
		// (v1.18) almost never held: a volume crossing a wall's silhouette into the sky failed both.
		if (!(limitBehind > 0.0f) || !snapshot.NoSurfaceBetween(px0, py0, px1, py1, bufferDepth(limitBehind), bufferDepth(limitFront), kRefineLevels)) {
			return LampVerdict::kNeeded;
		}
		const auto entry = FindEntry(reinterpret_cast<std::uintptr_t>(a_caster));
		if (!entry) {
			Bump(kTableFull);
			return LampVerdict::kNeeded;
		}
		const auto streak = HiddenStreak(entry->lamp.load(std::memory_order_relaxed), context->clock);
		entry->lamp.store((static_cast<std::uint64_t>(streak) << 32) | context->clock, std::memory_order_relaxed);
		return streak < g_tunables.confirmFrames ? LampVerdict::kConfirming : LampVerdict::kMisses;
	}

	void CountLampVerdict(LampVerdict a_verdict) noexcept
	{
		switch (a_verdict) {
		case LampVerdict::kOutside:
			Bump(kLampVolumeOutside);
			break;
		case LampVerdict::kMisses:
			Bump(kLampVolumeMisses);
			break;
		case LampVerdict::kConfirming:
			Bump(kLampVolumeConfirming);
			break;
		case LampVerdict::kNeeded:
			Bump(kLampVolumeNeeded);
			break;
		default:
			Bump(kLampVolumeUnknown);
			break;
		}
	}

	bool ViewCone(ShadowGeometry::Cone& a_cone, float& a_push) noexcept
	{
		const auto context = CurrentContext();
		if (!context || !context->coneValid) {
			return false;
		}
		a_cone = context->cone;
		a_push = context->conePush;
		return true;
	}

	bool Deciding() noexcept
	{
		return Observing();
	}

	bool Active() noexcept
	{
		return g_active.load(std::memory_order_relaxed);
	}

	void ResetHistory()
	{
		g_resetRequested.store(true);
		g_streamGeneration.fetch_add(1, std::memory_order_relaxed);  // (every thread's verdict streams start over)
	}

	void RequestKeptDump()
	{
		std::uint32_t expected = 0;
		g_dumpClock.compare_exchange_strong(expected, Clock() + 1);
	}

	void FlushKeptDump()
	{
		const auto dumpClock = g_dumpClock.load();
		if (dumpClock == 0 || Clock() <= dumpClock) {
			return;
		}
		std::vector<KeptItem> items;
		{
			std::scoped_lock lock(g_keptLock);
			items.swap(g_kept);
		}
		g_dumpClock.store(0);

		// Meshes a shape settled on this frame (not drawn): listed apart, largest first, so a wrongly hidden
		// mesh can be traced to its type and size.
		std::vector<KeptItem> dropped;
		std::erase_if(items, [&](const KeptItem& a_item) {
			if (a_item.dropped) {
				dropped.push_back(a_item);
			}
			return a_item.dropped;
		});
		if (!dropped.empty()) {
			std::ranges::sort(dropped, [](const KeptItem& a_left, const KeptItem& a_right) { return a_left.radius > a_right.radius; });
			std::map<std::string, std::uint32_t> byType;
			for (const auto& item : dropped) {
				++byType[std::format("{} ({})", item.type, item.reason)];
			}
			std::string text;
			for (const auto& [key, count] : byType) {
				text += std::format(" | {} x{}", key, count);
			}
			logger::info("==== settled by mesh shape on the same frame: {} meshes ===={}", dropped.size(), text);
			for (std::size_t i = 0; i < dropped.size() && i < 24; ++i) {
				const auto& item = dropped[i];
				logger::info(
					"  {:<20} {:<20} '{}' r={:.0f} dist={:.0f} verts={} | desc=0x{:X}",
					item.reason, item.type, item.name, item.radius, item.distance, item.vertices, item.vertexDesc);
			}
		}

		std::map<std::string, std::uint32_t> meshReasons;
		std::uint32_t                        meshes = 0, nodes = 0, withCpuData = 0, skinned = 0;
		for (const auto& item : items) {
			if (item.geometry) {
				++meshes;
				++meshReasons[item.reason];
				withCpuData += item.data != 0;
				skinned += item.skinned;
			} else {
				++nodes;
			}
		}
		std::string reasons;
		for (const auto& [reason, count] : meshReasons) {
			reasons += std::format(" {} {} |", reason, count);
		}
		logger::info("==== kept on one still frame: {} meshes (each is at least one draw), {} nodes (their children are judged on their own) ====", meshes, nodes);
		logger::info("meshes by reason:{}", reasons);
		logger::info("meshes with CPU-side vertex data: {} of {} | skinned: {}", withCpuData, meshes, skinned);

		// Grouped by type + name (precombined chunks have no name; LOD blocks are 'obj'/'obj-at'/'Land').
		std::map<std::string, std::pair<std::uint32_t, float>> groups;
		for (const auto& item : items) {
			if (item.geometry) {
				auto& group = groups[std::format("{} '{}'", item.type, item.name)];
				++group.first;
				group.second = std::max(group.second, item.radius);
			}
		}
		std::vector<std::pair<std::uint32_t, std::string>> ranked;
		for (const auto& [key, value] : groups) {
			ranked.emplace_back(value.first, std::format("{} x{} (max r {:.0f})", key, value.first, value.second));
		}
		std::ranges::sort(ranked, std::greater{});
		std::string text;
		for (std::size_t i = 0; i < ranked.size() && i < 16; ++i) {
			text += " | " + ranked[i].second;
		}
		logger::info("meshes by name (top 16):{}", text);

		std::ranges::sort(items, [](const KeptItem& a_left, const KeptItem& a_right) {
			if (a_left.geometry != a_right.geometry) {
				return a_left.geometry;
			}
			return a_left.radius > a_right.radius;
		});
		for (std::size_t i = 0; i < items.size() && i < 60; ++i) {
			const auto& item = items[i];
			if (!item.geometry) {
				break;
			}
			logger::info(
				"  {:<28} {:<24} '{}' r={:.0f} dist={:.0f} verts={} skinned={} | desc=0x{:X} buffer={:#x} data={:#x} size={} offset={} bytes={} usage={}",
				item.reason, item.type, item.name, item.radius, item.distance, item.vertices, item.skinned,
				item.vertexDesc, item.buffer, item.data, item.dataSize, item.dataOffset, item.byteWidth, item.usage);
		}

		// Merge-instanced meshes: where their instance transforms live (for judging instances one by one).
		std::uint32_t shown = 0;
		for (const auto& item : items) {
			if (!item.merged || shown >= 8) {
				continue;
			}
			++shown;
			logger::info(
				"  merge-instanced r={:.0f} dist={:.0f} verts={} lod tris {}/{}/{}/{} | local translate ({:.0f},{:.0f},{:.0f}) model bound ({:.1f},{:.1f},{:.1f}) r={:.0f} | instance object {:#x}: words {:#x} {:#x} {:#x} {:#x} {:#x}, +0x34 {} +0x38 {} | d3d buffer at word {} ({}): bytes {} stride {} bind {:#x} misc {:#x} usage {}",
				item.radius, item.distance, item.vertices, item.lodTriangles[0], item.lodTriangles[1], item.lodTriangles[2], item.lodTriangles[3],
				item.localTranslate[0], item.localTranslate[1], item.localTranslate[2], item.modelBound[0], item.modelBound[1], item.modelBound[2], item.modelBound[3],
				item.instanceObject, item.instanceWords[0], item.instanceWords[1], item.instanceWords[2], item.instanceWords[3], item.instanceWords[4],
				item.instanceCount, item.instanceKind, item.instanceWord, item.instanceModule, item.instanceBytes, item.instanceStride,
				item.instanceBind, item.instanceMisc, item.instanceUsage);
		}
	}

	TestTime TakeTestMilliseconds() noexcept
	{
		Calibrate();
		const auto total = Total(kCycles);
		const auto delta = total - g_cyclesReported;
		g_cyclesReported = total;
		const auto mainTotal = g_mainStats ? g_mainStats->values[kCycles].load(std::memory_order_relaxed) : 0;
		const auto mainDelta = mainTotal - g_mainCyclesReported;
		g_mainCyclesReported = mainTotal;
		const auto part = [](Counter a_counter) {
			const auto now = Total(a_counter);
			const auto since = now - g_reported[a_counter];
			g_reported[a_counter] = now;
			return since;
		};
		const auto   evaluate = part(kCyclesEvaluate);
		const auto   shape = part(kCyclesShape);
		const auto   sun = part(kCyclesSun);
		const double perMs = g_calibration.cyclesPerMs;
		const auto   ms = [perMs](std::uint64_t a_cycles) { return perMs > 0.0 ? static_cast<double>(a_cycles) / perMs : 0.0; };
		return { ms(delta), ms(mainDelta), ms(evaluate), ms(shape), ms(sun) };
	}

	float RejectedPerFrame() noexcept
	{
		return g_lastRejected.load();
	}

	float TestedPerFrame() noexcept
	{
		return g_lastTested.load();
	}

	float LightsPerFrame() noexcept
	{
		return g_lastLights.load();
	}

	float LightsRejectedPerFrame() noexcept
	{
		return g_lastLightsRejected.load();
	}

	void LogStats(std::uint32_t a_frames)
	{
		const double frames = std::max(1u, a_frames);
		std::array<double, kCounterCount> delta{};
		for (std::size_t i = 0; i < kCounterCount; ++i) {
			if (i == kCycles || i == kCyclesEvaluate || i == kCyclesShape || i == kCyclesSun) {
				continue;  // reported by TakeTestMilliseconds
			}
			const auto total = Total(static_cast<Counter>(i));
			delta[i] = static_cast<double>(total - g_reported[i]);
			g_reported[i] = total;
		}
		const auto per = [&](Counter a_counter) { return delta[a_counter] / frames; };

		const bool observing = Observing();
		g_lastRejected.store(static_cast<float>(observing ? per(kWouldReject) : per(kRejected) + per(kMergedRejected) + per(kDropInherited)));
		g_lastTested.store(static_cast<float>(per(kTested) + per(kMerged)));
		g_lastLights.store(static_cast<float>(per(kLightsTested)));
		g_lastLightsRejected.store(static_cast<float>(per(kLightsRejected)));

		logger::info(
			"occlusion per frame: tested {:.0f} | rejected {:.0f}{} | confirming {:.0f} | visible {:.0f} | kept: edge {:.0f} near {:.0f} | not in view: outside {:.0f} behind {:.0f} | hidden but exempt: type {:.0f} light {:.0f} actor {:.0f} | invalid {:.0f} | no-context {:.0f} | table-full {:.0f}",
			per(kTested), per(kRejected),
			observing ? std::format(" (decide-only: would reject {:.0f})", per(kWouldReject)) : std::string{},
			per(kConfirming), per(kVisible), per(kEdge), per(kNear), per(kOutside), per(kBehind),
			per(kExemptType), per(kExemptLight), per(kExemptActor),
			per(kInvalid), per(kNoContext), per(kTableFull));
		logger::info(
			"occlusion merged meshes per frame: {:.0f} decided | hidden {:.0f} | confirming {:.0f} | instance tests {:.0f} | instance entries {:.0f}, rejected {:.0f} || previs-forced entries hidden {:.0f}",
			per(kMerged), per(kMergedRejected), per(kMergedConfirming), per(kInstanceTests), per(kInstanceEntries), per(kInstanceEntriesRejected),
			static_cast<double>(Hooks::CullGroups::TakeForcedCleared()) / frames);
		logger::info(
			"occlusion lights per frame: tested {:.0f} | rejected {:.0f} (point/spot lights whose whole reach is hidden)",
			per(kLightsTested), per(kLightsRejected));
		logger::info(
			"occlusion main view only (groups the sun's shadow cascades read too) per frame: entries {:.0f} | dropped with their parent {:.0f} | registrations {:.0f}, left out {:.0f} | drop table full {:.0f}",
			per(kShared), per(kDropInherited), per(kRegistered), per(kRegistrationsDropped), per(kDropFull));
		logger::info(
			"occlusion sun shadows per frame (group-0 objects the main view doesn't need): shadow tested {:.0f} | not needed: out of reach {:.0f}, behind surfaces {:.0f} | sun off (kept in group 0 for its other readers) {:.0f} | confirming {:.0f} | needed {:.0f} || objects nothing needs: never filed {:.0f}, rejected {:.0f}",
			per(kSunTests), per(kSunOutside), per(kSunHidden), per(kSunOff), per(kSunConfirming), per(kSunNeeded), per(kCasterSkipped), per(kCasterRejected));
		logger::info(
			"occlusion early skips per frame (main-view-only groups, never filed with the engine): hidden {:.0f} | out of view {:.0f} | top-level adds considered {:.0f}",
			per(kSkippedHidden), per(kSkippedOutside), static_cast<double>(Hooks::CullGroups::TakeGroupAddsConsidered()) / frames);
		logger::info(
			"occlusion node pruning per frame: offered by the scene walk: cell node 3 {:.1f}, container {:.1f}; by DrawWorld's root loop {:.1f} | skipped whole {:.1f} ({:.0f} entries never filed) | walked: an entry meets the view {:.1f}, holds an always-draw entry or an actor {:.1f}, sun shadow may reach the view {:.1f} || cell node 9 seen {:.1f} (never pruned) | nodes with their own always-draw bit {:.1f}",
			per(kCellNodesSeen), per(kContainersSeen), per(kRootsSeen), per(kCellNodesSkipped), per(kCellNodeEntriesSkipped), per(kCellNodesInView), per(kCellNodesHeld), per(kCellNodesSunNeeded), per(kCellNodesOther), per(kCellNodesAlwaysDraw));
		logger::info("Group::Add callers of the main-pass top-level adds considered: {}", Hooks::CullGroups::TakeGroupAddSites(frames));
		logger::info(
			"occlusion lamp shadow volumes per frame (casters of point lights): outside the view {:.0f} | misses every visible surface {:.0f} | confirming {:.0f} | needed {:.0f} | unknown {:.0f}",
			per(kLampVolumeOutside), per(kLampVolumeMisses), per(kLampVolumeConfirming), per(kLampVolumeNeeded), per(kLampVolumeUnknown));
		logger::info(
			"verdict cache per frame: reused {:.0f} (hidden re-checked against the depth {:.0f}) | evaluated: new {:.0f}, camera epoch {:.0f}, bound changed {:.0f}, depth changed {:.0f}, kept on its turn {:.0f}, never-reuse {:.0f} | sun re-evaluated under a reused view {:.0f}",
			per(kCacheHits), per(kCacheRecheck), per(kCacheNew), per(kCacheEpoch), per(kCacheBound), per(kCacheDepth), per(kCacheKeptRecheck), per(kCacheNoCache), per(kCacheSunDepth));
		logger::info(
			"per-frame spread over culled frames (min/avg/max, sd, avg change between consecutive frames): main view kept {} | other views' registrations {} | rejected {} | confirming (verdict flips) {}",
			g_spreadKept.Describe(), g_spreadOther.Describe(), g_spreadRejected.Describe(), g_spreadConfirming.Describe());
		g_spreadKept.Clear();
		g_spreadOther.Clear();
		g_spreadRejected.Clear();
		g_spreadConfirming.Clear();
		const auto reading = g_rotationReading.load();
		logger::info(
			"occlusion by mesh shape per frame: tested {:.0f} (sphere said visible/near/edge) | settled hidden {:.0f} | out of view {:.0f} | cells tested {:.0f} | left to the sphere: moved {:.1f}, swaying {:.1f}, transform unconfirmed {:.1f} | rotation reading {} (votes {}/{}, conflicts {})",
			per(kShapeTests), per(kShapeHidden), per(kShapeOutside), per(kShapeCells), per(kShapeMoving), per(kShapeAnimated), per(kShapeNoTransform),
			reading == 0 ? "rows"sv : reading == 1 ? "columns"sv : reading == 2 ? "CONFLICTING (shapes off)"sv : "not settled yet"sv,
			g_rotationVotes[0].load(), g_rotationVotes[1].load(), g_rotationConflicts.load());
		MeshProxy::LogStats(a_frames);
		logger::info(
			"occlusion frames: culling {:.0f} / blocked {:.0f} (stale depth, camera jump, or inactive) | history {} objects",
			delta[kFramesCulling], delta[kFramesBlocked], g_used.load());

		const auto logTypes = [](std::unordered_map<std::uintptr_t, std::uint32_t>& a_types, std::string_view a_what) {
			std::vector<std::pair<std::uint32_t, std::uintptr_t>> ranked;
			{
				std::scoped_lock lock(g_exemptLock);
				for (const auto& [vtable, count] : a_types) {
					ranked.emplace_back(count, vtable);
				}
				a_types.clear();
			}
			if (ranked.empty()) {
				return;
			}
			std::ranges::sort(ranked, std::greater{});
			std::string text;
			for (std::size_t i = 0; i < ranked.size() && i < 10; ++i) {
				char                 name[64]{};
				const std::uintptr_t vtableOnly = ranked[i].second;  // an "object" whose first qword is the vtable
				Util::TryGetRTTIName(&vtableOnly, name, sizeof(name));
				text += std::format(" {} x{}", name[0] ? name : "?", ranked[i].first);
			}
			logger::info("{} (sampled every 32nd frame):{}", a_what, text);
		};
		logTypes(g_exemptTypes, "hidden but exempt by type"sv);
		logTypes(g_leftOutTypes, "left out of group 0 by type (never filed or rejected in every view)"sv);
	}
}
