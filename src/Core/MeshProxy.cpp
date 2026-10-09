#include "Core/MeshProxy.h"

#include "Core/ShapeFit.h"
#include "Settings.h"

namespace CBRO::Core::MeshProxy
{
	namespace
	{
		// Engine layout, OG 1.10.163 (confirmed by the v1.9 dump: Buffer::dataSize == vertices * stride).
		constexpr std::size_t kModelBound = 0x120;     // BSGeometry::modelBound (local NiBound)
		constexpr std::size_t kShaderProperty = 0x138;  // BSGeometry::properties[1] (the precombine builder reads it there)
		constexpr std::size_t kSkin = 0x140;            // BSGeometry::skinInstance
		constexpr std::size_t kRendererData = 0x148;    // BSGeometry::rendererData -> BSGraphics::TriShape
		constexpr std::size_t kTriangles = 0x160;       // BSTriShape::numTriangles
		constexpr std::size_t kVertices = 0x164;        // BSTriShape::numVertices (u16)
		constexpr std::size_t kShaderFlags = 0x30;      // BSShaderProperty::flags
		// TriShape: vertexDesc +0, vertexBuffer +8, indexBuffer +0x10. Buffer: data +8, dataSize +0x34.
		// vertexDesc: stride = (desc & 0xF) * 4, position offset = (desc >> 2) & 0x3C, flags = desc >> 44.
		constexpr std::uint64_t kFlagSkinned = 1ull << 6;
		constexpr std::uint64_t kFlagInstance = 1ull << 9;
		constexpr std::uint64_t kFlagFullPrecision = 1ull << 10;
		// Shader flags whose vertex shader moves vertices away from the CPU copy: skinning, tessellation
		// (displacement), billboards (turned to face the camera) and tree animation (wind sway).
		constexpr std::uint64_t kShaderMovesVertices = (1ull << 1) | (1ull << 25) | (1ull << 45) | (1ull << 61);
		constexpr std::uint32_t kMaxTriangles = 1u << 20;
		constexpr float         kHuge = std::numeric_limits<float>::max();

		// Exact classes whose vertex data is what gets drawn (never a subclass: instanced or merged
		// geometry derives from BSTriShape too).
		std::array<std::uintptr_t, 3> g_meshVtables{};
		std::uintptr_t                g_mergedVtable{ 0 };
		std::uintptr_t                g_lightingShaderVtable{ 0 };

		enum class Skip : std::uint8_t
		{
			kType,     // not one of the plain mesh classes (nodes, dynamic, combined, ...)
			kMerged,   // BSMergeInstancedTriShape: its vertex data isn't what is drawn
			kShader,   // no lighting shader, or one that moves vertices
			kLayout,   // skinned/instanced vertex format, or sizes that don't add up
			kUnread,   // memory not readable
			kCount
		};

		struct Source
		{
			std::uintptr_t   key{ 0 };  // the TriShape (shared by every reference of the mesh)
			const std::byte* vertices{ nullptr };
			std::uint32_t    vertexCount{ 0 };
			std::uint32_t    stride{ 0 };
			std::uint32_t    positionOffset{ 0 };
			bool             fullPrecision{ false };
			const std::byte* indices{ nullptr };
			std::uint32_t    triangleCount{ 0 };
			RE::NiBound      modelBound{};
			std::uint64_t    check{ 0 };  // pointers, sizes and the first vertex: detects a reused address
		};

		std::uint64_t Mix(std::uint64_t a_hash, std::uint64_t a_value) noexcept
		{
			a_hash ^= a_value + 0x9E3779B97F4A7C15ull + (a_hash << 6) + (a_hash >> 2);
			return a_hash;
		}

		// All reads of engine memory; nothing here needs C++ unwinding, so SEH can guard it. No engine
		// code is called: the class is told by the exact vtable.
		bool ReadSource(const RE::NiAVObject* a_object, Source& a_out, Skip& a_skip) noexcept
		{
			__try {
				const auto base = reinterpret_cast<const std::byte*>(a_object);
				const auto vtable = *reinterpret_cast<const std::uintptr_t*>(base);
				if (std::ranges::find(g_meshVtables, vtable) == g_meshVtables.end()) {
					a_skip = vtable == g_mergedVtable ? Skip::kMerged : Skip::kType;
					return false;
				}
				const auto shader = *reinterpret_cast<const std::byte* const*>(base + kShaderProperty);
				if (!shader || *reinterpret_cast<const std::uintptr_t*>(shader) != g_lightingShaderVtable ||
					(*reinterpret_cast<const std::uint64_t*>(shader + kShaderFlags) & kShaderMovesVertices)) {
					a_skip = Skip::kShader;
					return false;
				}
				a_skip = Skip::kLayout;
				if (*reinterpret_cast<const std::uintptr_t*>(base + kSkin) != 0) {
					return false;
				}
				const auto triShape = *reinterpret_cast<const std::byte* const*>(base + kRendererData);
				if (!triShape) {
					return false;
				}
				const auto desc = *reinterpret_cast<const std::uint64_t*>(triShape);
				const auto flags = desc >> 44;
				if (flags & (kFlagSkinned | kFlagInstance)) {
					return false;
				}
				const auto vertexBuffer = *reinterpret_cast<const std::byte* const*>(triShape + 0x8);
				const auto indexBuffer = *reinterpret_cast<const std::byte* const*>(triShape + 0x10);
				if (!vertexBuffer) {
					return false;
				}
				a_out.key = reinterpret_cast<std::uintptr_t>(triShape);
				a_out.vertices = *reinterpret_cast<const std::byte* const*>(vertexBuffer + 0x8);
				const auto vertexBytes = *reinterpret_cast<const std::uint32_t*>(vertexBuffer + 0x34);
				a_out.stride = static_cast<std::uint32_t>(desc & 0xF) * 4;
				a_out.positionOffset = static_cast<std::uint32_t>((desc >> 2) & 0x3C);
				a_out.fullPrecision = (flags & kFlagFullPrecision) != 0;
				a_out.vertexCount = *reinterpret_cast<const std::uint16_t*>(base + kVertices);
				const auto positionBytes = a_out.fullPrecision ? 12u : 6u;
				// The buffer holds exactly this mesh's vertices (the dump showed size == count * stride).
				if (!a_out.vertices || a_out.vertexCount == 0 || a_out.stride < a_out.positionOffset + positionBytes ||
					static_cast<std::uint64_t>(a_out.vertexCount) * a_out.stride != vertexBytes) {
					return false;
				}
				if (indexBuffer) {
					// Every triangle in the index buffer (LOD levels and segments included), never fewer
					// than the mesh says it draws.
					a_out.indices = *reinterpret_cast<const std::byte* const*>(indexBuffer + 0x8);
					const auto indexBytes = *reinterpret_cast<const std::uint32_t*>(indexBuffer + 0x34);
					a_out.triangleCount = indexBytes / 6;
					if (!a_out.indices || a_out.triangleCount < *reinterpret_cast<const std::uint32_t*>(base + kTriangles) ||
						a_out.triangleCount > kMaxTriangles) {
						return false;
					}
				}
				a_out.modelBound = *reinterpret_cast<const RE::NiBound*>(base + kModelBound);

				std::uint64_t hash = Mix(0, a_out.key);
				hash = Mix(hash, reinterpret_cast<std::uintptr_t>(a_out.vertices));
				hash = Mix(hash, (static_cast<std::uint64_t>(vertexBytes) << 32) | a_out.vertexCount);
				hash = Mix(hash, reinterpret_cast<std::uintptr_t>(a_out.indices));
				hash = Mix(hash, a_out.triangleCount);
				hash = Mix(hash, *reinterpret_cast<const std::uint64_t*>(a_out.vertices));
				hash = Mix(hash, *reinterpret_cast<const std::uint64_t*>(a_out.vertices + 8));
				a_out.check = hash | 1;
				return true;
			} __except (EXCEPTION_EXECUTE_HANDLER) {
				a_skip = Skip::kUnread;
				return false;
			}
		}

		bool CopyGuarded(void* a_destination, const void* a_source, std::size_t a_bytes) noexcept
		{
			__try {
				std::memcpy(a_destination, a_source, a_bytes);
				return true;
			} __except (EXCEPTION_EXECUTE_HANDLER) {
				return false;
			}
		}

		float HalfToFloat(std::uint16_t a_half) noexcept
		{
			const std::uint32_t sign = static_cast<std::uint32_t>(a_half & 0x8000u) << 16;
			std::uint32_t       exponent = (a_half >> 10) & 0x1Fu;
			std::uint32_t       mantissa = a_half & 0x3FFu;
			std::uint32_t       bits;
			if (exponent == 0) {
				if (mantissa == 0) {
					bits = sign;
				} else {
					exponent = 127 - 15 + 1;
					while (!(mantissa & 0x400u)) {
						mantissa <<= 1;
						--exponent;
					}
					bits = sign | (exponent << 23) | ((mantissa & 0x3FFu) << 13);
				}
			} else if (exponent == 31) {
				bits = sign | 0x7F800000u | (mantissa << 13);
			} else {
				bits = sign | ((exponent + 127 - 15) << 23) | (mantissa << 13);
			}
			return std::bit_cast<float>(bits);
		}

		// ---- the cache: lock-free table keyed by TriShape, shapes in a fixed array --------------

		enum : std::uint32_t
		{
			kEmpty,
			kQueued,
			kReady,
			kFailed,
		};

		struct alignas(32) Slot
		{
			std::atomic<std::uintptr_t> key{ 0 };
			std::atomic<std::uint32_t>  state{ kEmpty };
			std::atomic<std::uint32_t>  shape{ 0 };
			std::atomic<std::uint64_t>  check{ 0 };
		};
		constexpr std::size_t kSlotBits = 17;
		constexpr std::size_t kSlotCount = std::size_t{ 1 } << kSlotBits;
		constexpr std::size_t kMaxShapes = 1u << 16;
		// Neither table ever frees an entry (meshes unload without notice, and a reused address gets a new
		// shape), so both fill up over a long session. Near full, everything is dropped at a frame start
		// and rebuilt on demand.
		constexpr std::uint32_t kResetShapes = kMaxShapes - 4096;
		constexpr std::uint32_t kResetSlots = kSlotCount / 2;

		std::unique_ptr<Slot[]>    g_slots;
		std::unique_ptr<Shape[]>   g_shapes;
		std::atomic<std::uint32_t> g_shapeCount{ 0 };
		std::atomic<std::uint32_t> g_slotsUsed{ 0 };
		bool                       g_enabled{ false };

		Slot* FindSlot(std::uintptr_t a_key) noexcept
		{
			const auto hash = static_cast<std::size_t>(((a_key >> 5) * 0x9E3779B97F4A7C15ull) >> (64 - kSlotBits));
			for (std::size_t probe = 0; probe < 32; ++probe) {
				auto& slot = g_slots[(hash + probe) & (kSlotCount - 1)];
				auto  current = slot.key.load(std::memory_order_acquire);
				if (current == a_key) {
					return &slot;
				}
				if (current == 0 && slot.key.compare_exchange_strong(current, a_key, std::memory_order_acq_rel)) {
					g_slotsUsed.fetch_add(1, std::memory_order_relaxed);
					return &slot;
				}
				if (current == a_key) {
					return &slot;
				}
			}
			return nullptr;
		}

		// ---- the builder ----------------------------------------------------------------------------

		std::mutex              g_queueLock;
		std::condition_variable g_queueSignal;
		std::vector<Source>     g_queue;
		// The builder commits a finished shape under this lock, and only if the cache wasn't reset since it
		// took the batch (the generation it saw then). Reset takes it (then the queue lock) too.
		std::mutex                 g_commitLock;
		std::atomic<std::uint32_t> g_generation{ 0 };
		std::atomic<std::uint64_t> g_resets{ 0 };

		enum class Failure : std::uint8_t
		{
			kCopy,     // the data changed or vanished while copying
			kDecode,   // non-finite positions or indices past the vertices
			kFit,      // ShapeFit rejected it (counted per reason below)
			kNoRoom,   // the shape array was full (a reset follows at the next frame start)
			kCount
		};

		std::atomic<std::uint64_t> g_lookups{ 0 };
		std::atomic<std::uint64_t> g_found{ 0 };
		std::atomic<std::uint64_t> g_built{ 0 };
		std::atomic<std::uint64_t> g_stale{ 0 };
		std::atomic<std::uint64_t> g_boxOnly{ 0 };
		std::atomic<std::uint64_t> g_cellsTotal{ 0 };
		std::atomic<std::uint64_t> g_boundMismatch{ 0 };
		std::array<std::atomic<std::uint64_t>, static_cast<std::size_t>(Skip::kCount)>              g_skipped{};
		std::array<std::atomic<std::uint64_t>, static_cast<std::size_t>(Failure::kCount)>           g_failures{};
		std::array<std::atomic<std::uint64_t>, static_cast<std::size_t>(ShapeFit::Result::kCount)> g_fitFailures{};

		bool Enqueue(const Source& a_source)
		{
			{
				std::scoped_lock lock(g_queueLock);
				if (g_queue.size() >= 16384) {
					return false;  // the caller frees the slot, so a later lookup retries
				}
				g_queue.push_back(a_source);
			}
			g_queueSignal.notify_one();
			return true;
		}

		bool BuildShape(const Source& a_source, Shape& a_shape, std::vector<std::byte>& a_vertexCopy, std::vector<std::uint16_t>& a_indexCopy, std::vector<float>& a_positions, Failure& a_failure)
		{
			// Copy first (the engine may unload the mesh at any time), then verify nothing changed.
			a_failure = Failure::kCopy;
			a_vertexCopy.resize(static_cast<std::size_t>(a_source.vertexCount) * a_source.stride);
			if (!CopyGuarded(a_vertexCopy.data(), a_source.vertices, a_vertexCopy.size())) {
				return false;
			}
			const auto triangles = a_source.indices ? a_source.triangleCount : 0u;
			a_indexCopy.resize(static_cast<std::size_t>(triangles) * 3);
			if (triangles && !CopyGuarded(a_indexCopy.data(), a_source.indices, a_indexCopy.size() * sizeof(std::uint16_t))) {
				return false;
			}
			std::uint64_t first[2]{};
			if (!CopyGuarded(first, a_source.vertices, sizeof(first)) || std::memcmp(first, a_vertexCopy.data(), sizeof(first)) != 0) {
				return false;
			}

			// Positions, their box, and the farthest one from the model bound's center.
			a_failure = Failure::kDecode;
			const auto& bound = a_source.modelBound;
			const float center[3]{ bound.center.x, bound.center.y, bound.center.z };
			a_positions.resize(static_cast<std::size_t>(a_source.vertexCount) * 3);
			float lo[3]{ kHuge, kHuge, kHuge }, hi[3]{ -kHuge, -kHuge, -kHuge };
			float farthest2 = 0.0f;
			for (std::uint32_t v = 0; v < a_source.vertexCount; ++v) {
				const auto* vertex = a_vertexCopy.data() + static_cast<std::size_t>(v) * a_source.stride + a_source.positionOffset;
				float       p[3];
				if (a_source.fullPrecision) {
					std::memcpy(p, vertex, sizeof(p));
				} else {
					std::uint16_t h[3];
					std::memcpy(h, vertex, sizeof(h));
					p[0] = HalfToFloat(h[0]);
					p[1] = HalfToFloat(h[1]);
					p[2] = HalfToFloat(h[2]);
				}
				float distance2 = 0.0f;
				for (int a = 0; a < 3; ++a) {
					if (!std::isfinite(p[a])) {
						return false;
					}
					lo[a] = std::min(lo[a], p[a]);
					hi[a] = std::max(hi[a], p[a]);
					a_positions[static_cast<std::size_t>(v) * 3 + a] = p[a];
					distance2 += (p[a] - center[a]) * (p[a] - center[a]);
				}
				farthest2 = std::max(farthest2, distance2);
			}

			// These must be the vertices the model bound was made from; anything else (wrong layout, or
			// only part of what is drawn) is not used.
			if (!std::isfinite(bound.fRadius)) {
				return false;
			}
			if (const auto fit = ShapeFit::Check(lo, hi, std::sqrt(farthest2), center, bound.fRadius); fit != ShapeFit::Result::kOk) {
				a_failure = Failure::kFit;
				g_fitFailures[static_cast<std::size_t>(fit)].fetch_add(1, std::memory_order_relaxed);
				return false;
			}
			a_shape.bound[0] = center[0];
			a_shape.bound[1] = center[1];
			a_shape.bound[2] = center[2];
			a_shape.bound[3] = bound.fRadius;

			float maxExtent = 0.0f;
			for (int a = 0; a < 3; ++a) {
				maxExtent = std::max(maxExtent, hi[a] - lo[a]);
			}
			// Grow for half-precision rounding and float error.
			const float grow = maxExtent * (a_source.fullPrecision ? 0.001f : 0.004f) + 0.5f;
			for (int a = 0; a < 3; ++a) {
				lo[a] -= grow;
				hi[a] += grow;
				a_shape.min[a] = lo[a];
			}

			// Grid: roughly cubic cells, at most 8 per axis and 128 in all.
			const float target = maxExtent / 8.0f + 2.0f * grow;
			std::uint32_t dims[3];
			for (int a = 0; a < 3; ++a) {
				dims[a] = std::clamp(static_cast<std::uint32_t>(std::ceil((hi[a] - lo[a]) / target)), 1u, 8u);
			}
			while (dims[0] * dims[1] * dims[2] > 128) {
				auto& largest = *std::max_element(dims, dims + 3);
				--largest;
			}
			for (int a = 0; a < 3; ++a) {
				a_shape.dims[a] = static_cast<std::uint8_t>(dims[a]);
				a_shape.cell[a] = (hi[a] - lo[a]) / static_cast<float>(dims[a]);
			}

			// Mark every cell a triangle's box touches (conservative). No index data: the whole box.
			a_shape.occupied[0] = a_shape.occupied[1] = 0;
			const auto cellsTotal = dims[0] * dims[1] * dims[2];
			if (!triangles) {
				a_shape.count = 0;
				return true;
			}
			for (std::uint32_t t = 0; t < triangles; ++t) {
				float tlo[3]{ kHuge, kHuge, kHuge }, thi[3]{ -kHuge, -kHuge, -kHuge };
				for (int k = 0; k < 3; ++k) {
					const auto index = a_indexCopy[static_cast<std::size_t>(t) * 3 + k];
					if (index >= a_source.vertexCount) {
						return false;
					}
					for (int a = 0; a < 3; ++a) {
						const float value = a_positions[static_cast<std::size_t>(index) * 3 + a];
						tlo[a] = std::min(tlo[a], value);
						thi[a] = std::max(thi[a], value);
					}
				}
				std::uint32_t c0[3], c1[3];
				for (int a = 0; a < 3; ++a) {
					const float inv = 1.0f / a_shape.cell[a];
					c0[a] = std::min(dims[a] - 1, static_cast<std::uint32_t>(std::max(0.0f, (tlo[a] - lo[a]) * inv)));
					c1[a] = std::min(dims[a] - 1, static_cast<std::uint32_t>(std::max(0.0f, (thi[a] - lo[a]) * inv)));
				}
				for (auto z = c0[2]; z <= c1[2]; ++z) {
					for (auto y = c0[1]; y <= c1[1]; ++y) {
						for (auto x = c0[0]; x <= c1[0]; ++x) {
							const auto bit = x + dims[0] * (y + dims[1] * z);
							a_shape.occupied[bit >> 6] |= std::uint64_t{ 1 } << (bit & 63);
						}
					}
				}
			}
			const auto occupied = static_cast<std::uint32_t>(std::popcount(a_shape.occupied[0]) + std::popcount(a_shape.occupied[1]));
			// Nearly full grids gain nothing over the box and cost a test per cell.
			a_shape.count = occupied * 10 >= cellsTotal * 9 ? 0 : static_cast<std::uint8_t>(occupied);
			return occupied > 0;
		}

		void BuilderLoop()
		{
			std::vector<std::byte>     vertexCopy;
			std::vector<std::uint16_t> indexCopy;
			std::vector<float>         positions;
			std::vector<Source>        batch;
			for (;;) {
				std::uint32_t generation = 0;
				{
					std::unique_lock lock(g_queueLock);
					g_queueSignal.wait(lock, [] { return !g_queue.empty(); });
					batch.swap(g_queue);
					generation = g_generation.load(std::memory_order_relaxed);
				}
				for (const auto& source : batch) {
					Shape      shape{};
					Failure    failure{ Failure::kCopy };
					const bool ok = BuildShape(source, shape, vertexCopy, indexCopy, positions, failure);

					std::scoped_lock commit(g_commitLock);
					if (g_generation.load(std::memory_order_relaxed) != generation) {
						break;  // the cache was reset while this batch was being built: its sources are gone
					}
					const auto slot = FindSlot(source.key);
					if (!slot) {
						continue;
					}
					const auto index = ok ? g_shapeCount.load(std::memory_order_relaxed) : 0u;  // (only the builder allocates)
					if (ok && index < kMaxShapes) {
						g_shapes[index] = shape;
						g_shapeCount.store(index + 1, std::memory_order_relaxed);
						slot->shape.store(index, std::memory_order_relaxed);
						slot->check.store(source.check, std::memory_order_relaxed);
						slot->state.store(kReady, std::memory_order_release);
						g_built.fetch_add(1, std::memory_order_relaxed);
						(shape.count == 0 ? g_boxOnly : g_cellsTotal).fetch_add(shape.count == 0 ? 1 : shape.count, std::memory_order_relaxed);
					} else {
						slot->check.store(source.check, std::memory_order_relaxed);
						slot->state.store(kFailed, std::memory_order_release);
						g_failures[static_cast<std::size_t>(ok ? Failure::kNoRoom : failure)].fetch_add(1, std::memory_order_relaxed);
					}
				}
				batch.clear();
			}
		}

		// Main thread at a frame start, with no culling running (only culling threads read the cache).
		void Reset()
		{
			std::scoped_lock commit(g_commitLock);
			std::scoped_lock queue(g_queueLock);
			g_queue.clear();
			for (std::size_t i = 0; i < kSlotCount; ++i) {
				auto& slot = g_slots[i];
				slot.key.store(0, std::memory_order_relaxed);
				slot.state.store(kEmpty, std::memory_order_relaxed);
				slot.shape.store(0, std::memory_order_relaxed);
				slot.check.store(0, std::memory_order_relaxed);
			}
			g_shapeCount.store(0, std::memory_order_relaxed);
			g_slotsUsed.store(0, std::memory_order_relaxed);
			g_generation.fetch_add(1, std::memory_order_release);
			g_resets.fetch_add(1, std::memory_order_relaxed);
		}
	}

	void BeginFrame()
	{
		if (!g_enabled) {
			return;
		}
		const auto shapes = g_shapeCount.load(std::memory_order_relaxed);
		const auto slots = g_slotsUsed.load(std::memory_order_relaxed);
		if (shapes >= kResetShapes || slots >= kResetSlots) {
			Reset();
			logger::info("mesh shapes: cache full ({} shapes, {} meshes seen); cleared, shapes are rebuilt as meshes come into view", shapes, slots);
		}
	}

	void Install()
	{
		g_enabled = Settings::Get().meshShapes;
		if (!g_enabled) {
			return;
		}
		g_meshVtables = {
			RE::VTABLE::BSTriShape[0].address(),
			RE::VTABLE::BSSubIndexTriShape[0].address(),
			RE::VTABLE::BSMeshLODTriShape[0].address(),
		};
		g_mergedVtable = RE::VTABLE::BSMergeInstancedTriShape[0].address();
		g_lightingShaderVtable = RE::VTABLE::BSLightingShaderProperty[0].address();
		g_slots = std::make_unique<Slot[]>(kSlotCount);
		g_shapes = std::make_unique<Shape[]>(kMaxShapes);
		std::thread(BuilderLoop).detach();
		logger::info("mesh shapes: builder started (plain static meshes with a lighting shader are judged by their box and occupied cells)");
	}

	const Shape* Find(RE::NiAVObject* a_object, RE::NiBound& a_modelBound) noexcept
	{
		if (!g_enabled) {
			return nullptr;
		}
		Source source{};
		Skip   skip{ Skip::kType };
		if (!ReadSource(a_object, source, skip)) {
			g_skipped[static_cast<std::size_t>(skip)].fetch_add(1, std::memory_order_relaxed);
			return nullptr;
		}
		g_lookups.fetch_add(1, std::memory_order_relaxed);
		const auto slot = FindSlot(source.key);
		if (!slot) {
			return nullptr;
		}
		auto state = slot->state.load(std::memory_order_acquire);
		if (state == kReady && slot->check.load(std::memory_order_relaxed) == source.check) {
			const auto& shape = g_shapes[slot->shape.load(std::memory_order_relaxed)];
			// Built from another object sharing this mesh: only valid if this one has the same bound.
			const float center[3]{ source.modelBound.center.x, source.modelBound.center.y, source.modelBound.center.z };
			if (!ShapeFit::SameBound(shape.bound, center, source.modelBound.fRadius)) {
				g_boundMismatch.fetch_add(1, std::memory_order_relaxed);
				return nullptr;
			}
			g_found.fetch_add(1, std::memory_order_relaxed);
			a_modelBound = source.modelBound;
			return &shape;
		}
		// Never seen, or the address now holds another mesh: (re)build it.
		const bool stale = (state == kReady || state == kFailed) && slot->check.load(std::memory_order_relaxed) != source.check;
		if (state == kEmpty || stale) {
			if (stale) {
				g_stale.fetch_add(1, std::memory_order_relaxed);
			}
			if (slot->state.compare_exchange_strong(state, kQueued, std::memory_order_acq_rel) && !Enqueue(source)) {
				slot->state.store(kEmpty, std::memory_order_release);
			}
		}
		return nullptr;
	}

	void Inspect(RE::NiAVObject* a_object, Inspection& a_out) noexcept
	{
		a_out = {};
		if (!g_enabled) {
			return;
		}
		Source source{};
		Skip   skip{ Skip::kType };
		if (!ReadSource(a_object, source, skip)) {
			return;
		}
		std::vector<std::byte> vertexCopy;
		Failure                failure{ Failure::kCopy };
		a_out.fresh = BuildShape(source, a_out.freshShape, vertexCopy, a_out.indices, a_out.positions, failure);
		a_out.read = a_out.fresh || failure != Failure::kCopy;
		a_out.vertexCount = source.vertexCount;
		// (the cache, without taking a slot)
		const auto hash = static_cast<std::size_t>(((source.key >> 5) * 0x9E3779B97F4A7C15ull) >> (64 - kSlotBits));
		for (std::size_t probe = 0; probe < 32; ++probe) {
			const auto& slot = g_slots[(hash + probe) & (kSlotCount - 1)];
			const auto  key = slot.key.load(std::memory_order_acquire);
			if (key == 0) {
				break;
			}
			if (key == source.key) {
				if (slot.state.load(std::memory_order_acquire) == kReady && slot.check.load(std::memory_order_relaxed) == source.check) {
					a_out.cached = true;
					a_out.cachedShape = g_shapes[slot.shape.load(std::memory_order_relaxed)];
				}
				break;
			}
		}
	}

	void LogStats(std::uint32_t a_frames)
	{
		if (!g_enabled) {
			return;
		}
		const double frames = std::max(1u, a_frames);
		const auto   lookups = g_lookups.exchange(0);
		const auto   found = g_found.exchange(0);
		const auto   mismatch = g_boundMismatch.exchange(0);
		const auto   built = g_built.load();
		const auto   boxOnly = g_boxOnly.load();
		const auto   withCells = built - std::min(built, boxOnly);
		const auto   per = [&](std::atomic<std::uint64_t>& a_counter) { return static_cast<double>(a_counter.exchange(0)) / frames; };
		const auto   total = [](const std::atomic<std::uint64_t>& a_counter) { return a_counter.load(); };
		logger::info(
			"mesh shapes: {} built ({} with cells, avg {:.0f} cells; {} box only), {} rebuilt after reuse | not built: fit {} (outside bound {}, loose bound {}, off-center {}, too small {}), decode {}, copy {}, no room {} | cache: {} shapes, {} meshes, cleared {} times",
			built, withCells, withCells ? static_cast<double>(g_cellsTotal.load()) / withCells : 0.0, boxOnly, g_stale.load(),
			total(g_failures[static_cast<std::size_t>(Failure::kFit)]),
			total(g_fitFailures[static_cast<std::size_t>(ShapeFit::Result::kOutsideBound)]),
			total(g_fitFailures[static_cast<std::size_t>(ShapeFit::Result::kLooseBound)]),
			total(g_fitFailures[static_cast<std::size_t>(ShapeFit::Result::kOffCenter)]),
			total(g_fitFailures[static_cast<std::size_t>(ShapeFit::Result::kTooSmall)]),
			total(g_failures[static_cast<std::size_t>(Failure::kDecode)]),
			total(g_failures[static_cast<std::size_t>(Failure::kCopy)]),
			total(g_failures[static_cast<std::size_t>(Failure::kNoRoom)]),
			g_shapeCount.load(), g_slotsUsed.load(), g_resets.load());
		logger::info(
			"mesh shapes per frame: lookups {:.0f}, ready {:.0f}, other bound {:.1f} | not candidates: merge-instanced {:.0f}, other types {:.0f}, shader {:.0f}, layout {:.1f}, unreadable {:.1f}",
			lookups / frames, found / frames, mismatch / frames,
			per(g_skipped[static_cast<std::size_t>(Skip::kMerged)]), per(g_skipped[static_cast<std::size_t>(Skip::kType)]),
			per(g_skipped[static_cast<std::size_t>(Skip::kShader)]), per(g_skipped[static_cast<std::size_t>(Skip::kLayout)]),
			per(g_skipped[static_cast<std::size_t>(Skip::kUnread)]));
	}
}
