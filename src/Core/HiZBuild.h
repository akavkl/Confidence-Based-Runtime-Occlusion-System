#pragma once

// The GPU side of the Hi-Z build, engine-free so the offline test (tools/tests/hiz_gpu_test.cpp) compiles it as is
// and runs the shaders against a CPU model. Everything the CPU used to do at readback time runs on the GPU at capture,
// in two dispatches:
//   build: one 8x8 thread group per 8x8 block of Hi-Z texels (= one verdict-cache block = one level-3 texel). Each
//          thread reduces its factor x factor depth texels to (nearest, farthest), stores them in this capture's ring
//          slice, merges the previous captures of a still camera in (min/max over the ring), writes level 0, and
//          compares its linear depths against the block's reference; the group then reduces levels 1-3 through
//          groupshared memory and, if any texel moved past the tolerance, marks the block with this capture's index
//          and takes the new depths as the reference.
//   tail:  one group finishing the remaining (tiny) levels from level 3 on.
// The output is one flat buffer: far pyramid, near pyramid, then the per-block change map (the capture index of each
// block's last change; it persists across captures, only marked blocks are written). A texel's reference is the
// nearest farthest depth (and the farthest nearest depth) since its block was last marked, so "unmarked since
// capture R" means: no texel of the block is farther (or nearer) now than it was at any capture since R, beyond the
// tolerance.
// First-person geometry (depth below g_nearReject) is no world occluder (farthest = 1) and a surface as near as can be
// (nearest = 0), so nothing can be proven in front of it either.
// Only single-component R32 typed UAV loads are guaranteed on a feature level 11.0 device, so the ring and the
// references are R32_FLOAT textures (pairs would need typed-load support the runtime doesn't promise).

#include <array>
#include <cstdint>

namespace CBRO::Core::HiZ::Build
{
	// Compiled with CBRO_SCALE = 1 + kCacheDepthTolerance / 2 and CBRO_SLACK = kCacheDepthSlack / 2 defined.
	constexpr char kSource[] = R"(
Texture2D<float>        g_depth    : register(t0);
RWTexture2DArray<float> g_ringNear : register(u0);  // level 0 of the last few captures, one slice each (temporal merge)
RWTexture2DArray<float> g_ringFar  : register(u1);
RWTexture2D<float>      g_refFar   : register(u2);  // per texel: linear depth when its block was last marked changed
RWTexture2D<float>      g_refNear  : register(u3);
RWByteAddressBuffer     g_out      : register(u4);  // far pyramid | near pyramid | per block: capture index of the last change

cbuffer Params : register(b0)
{
	uint2 g_srcSize;
	uint2 g_dstSize;
	uint  g_factor;
	float g_nearReject;     // depth below this is first-person geometry: it must not occlude the world
	float g_depthMin;       // viewport depth range of the world, for the linear conversion
	float g_invRange;
	float g_depthA;         // ndc.z = depthA + depthB / z_view
	float g_depthB;
	uint  g_captureIndex;
	uint  g_mergeCount;     // older ring slices merged into this capture's level 0
	uint  g_ringCur;        // this capture's slice
	uint  g_ringSize;
	uint  g_flags;          // FLAG_*
	uint  g_levels;
	uint  g_blocksW;
	uint  g_nearOffset;     // byte offset of the near pyramid in g_out
	uint  g_changedOffset;  // byte offset of the change map in g_out
	uint  g_pad;
	uint4 g_dims[16];       // per level: width, height, texel offset, 0
};

#define FLAG_RESET  1u  // no usable reference: every block counts as changed and takes this depth as its reference
#define FLAG_LINEAR 2u  // the camera has a linear-depth form (without it nothing can be compared: every block changed)

groupshared float s_near0[8][8];
groupshared float s_far0[8][8];
groupshared float s_near1[4][4];
groupshared float s_far1[4][4];
groupshared float s_near2[2][2];
groupshared float s_far2[2][2];
groupshared uint  s_changed;

// Buffer depth -> linear view depth (infinity for the sky and anything at or past the far plane).
float Linear(float d)
{
	float ndc = (d - g_depthMin) * g_invRange;
	bool  sky = d >= 0.999999 || ndc >= g_depthA;
	return sky ? asfloat(0x7F800000u) : g_depthB / (ndc - g_depthA);
}

void Store(uint level, uint2 p, float nearest, float farthest)
{
	if (level < g_levels && p.x < g_dims[level].x && p.y < g_dims[level].y) {
		uint i = g_dims[level].z + p.y * g_dims[level].x + p.x;
		g_out.Store(i * 4, asuint(farthest));
		g_out.Store(g_nearOffset + i * 4, asuint(nearest));
	}
}

[numthreads(8, 8, 1)]
void build(uint3 id : SV_DispatchThreadID, uint3 gid : SV_GroupID, uint3 tid : SV_GroupThreadID)
{
	if (tid.x == 0 && tid.y == 0) {
		s_changed = 0;
	}
	const bool inside = id.x < g_dstSize.x && id.y < g_dstSize.y;
	float farthest = 0.0;  // texels outside the image contribute neutral values to the reductions
	float nearest = 1.0;
	bool  changed = false;
	if (inside) {
		uint2 base = id.xy * g_factor;
		for (uint y = 0; y < g_factor; ++y) {
			for (uint x = 0; x < g_factor; ++x) {
				uint2 p = min(base + uint2(x, y), g_srcSize - 1);
				float d = g_depth.Load(int3(p, 0));
				bool  firstPerson = d < g_nearReject;
				farthest = max(farthest, firstPerson ? 1.0 : d);
				nearest = min(nearest, firstPerson ? 0.0 : d);
			}
		}
		g_ringNear[uint3(id.xy, g_ringCur)] = nearest;
		g_ringFar[uint3(id.xy, g_ringCur)] = farthest;
		// (a nearer surface ever seen is conservative for "in front"; a farther one for "hidden")
		for (uint k = 1; k <= g_mergeCount; ++k) {
			uint3 slice = uint3(id.xy, (g_ringCur + g_ringSize - k) % g_ringSize);
			nearest = min(nearest, g_ringNear[slice]);
			farthest = max(farthest, g_ringFar[slice]);
		}
		Store(0, id.xy, nearest, farthest);

		// Changed only in the directions that can invalidate a hidden verdict: the farthest depth getting farther,
		// the nearest getting nearer (the other directions only make hidden verdicts truer).
		if ((g_flags & FLAG_RESET) || !(g_flags & FLAG_LINEAR)) {
			changed = true;
		} else {
			float farNew = Linear(farthest);
			float nearNew = Linear(nearest);
			changed = farNew > g_refFar[id.xy] * CBRO_SCALE + CBRO_SLACK || nearNew * CBRO_SCALE + CBRO_SLACK < g_refNear[id.xy];
		}
	}
	s_near0[tid.y][tid.x] = nearest;
	s_far0[tid.y][tid.x] = farthest;
	GroupMemoryBarrierWithGroupSync();

	if (changed) {
		s_changed = 1;
	}
	if (tid.x < 4 && tid.y < 4) {
		uint2 q = tid.xy * 2;
		float n = min(min(s_near0[q.y][q.x], s_near0[q.y][q.x + 1]), min(s_near0[q.y + 1][q.x], s_near0[q.y + 1][q.x + 1]));
		float f = max(max(s_far0[q.y][q.x], s_far0[q.y][q.x + 1]), max(s_far0[q.y + 1][q.x], s_far0[q.y + 1][q.x + 1]));
		s_near1[tid.y][tid.x] = n;
		s_far1[tid.y][tid.x] = f;
		Store(1, gid.xy * 4 + tid.xy, n, f);
	}
	GroupMemoryBarrierWithGroupSync();

	if (tid.x < 2 && tid.y < 2) {
		uint2 q = tid.xy * 2;
		float n = min(min(s_near1[q.y][q.x], s_near1[q.y][q.x + 1]), min(s_near1[q.y + 1][q.x], s_near1[q.y + 1][q.x + 1]));
		float f = max(max(s_far1[q.y][q.x], s_far1[q.y][q.x + 1]), max(s_far1[q.y + 1][q.x], s_far1[q.y + 1][q.x + 1]));
		s_near2[tid.y][tid.x] = n;
		s_far2[tid.y][tid.x] = f;
		Store(2, gid.xy * 2 + tid.xy, n, f);
	}
	GroupMemoryBarrierWithGroupSync();

	if (tid.x == 0 && tid.y == 0) {
		float n = min(min(s_near2[0][0], s_near2[0][1]), min(s_near2[1][0], s_near2[1][1]));
		float f = max(max(s_far2[0][0], s_far2[0][1]), max(s_far2[1][0], s_far2[1][1]));
		Store(3, gid.xy, n, f);
		if (s_changed) {
			g_out.Store(g_changedOffset + (gid.y * g_blocksW + gid.x) * 4, g_captureIndex);
		}
	}
	if (inside && (g_flags & FLAG_LINEAR)) {
		// A marked block's new depth becomes its reference. An unmarked block's reference follows the nearest farthest
		// depth (and the farthest nearest one) seen since its mark: a verdict taken at any capture since then saw
		// depths no nearer than that, so the next mark comes before the depth drifts past what any such verdict allows.
		// (v1.24-v1.26 kept the depth at the mark: a texel that got nearer in between, e.g. a passer-by, could then
		// return to its old depth unmarked, and an object hidden behind the passer-by stayed hidden.)
		float farNew = Linear(farthest);
		float nearNew = Linear(nearest);
		g_refFar[id.xy] = s_changed ? farNew : min(g_refFar[id.xy], farNew);
		g_refNear[id.xy] = s_changed ? nearNew : max(g_refNear[id.xy], nearNew);
	}
}

// Levels 4 and up (40x25 texels and smaller at 320x200): one group, every thread a strided share of each level.
[numthreads(256, 1, 1)]
void tail(uint3 tid : SV_GroupThreadID)
{
	for (uint level = 4; level < g_levels; ++level) {
		uint4 src = g_dims[level - 1];
		uint4 dst = g_dims[level];
		uint  count = dst.x * dst.y;
		for (uint i = tid.x; i < count; i += 256) {
			uint x = i % dst.x;
			uint y = i / dst.x;
			uint x0 = 2 * x;
			uint x1 = min(2 * x + 1, src.x - 1);  // an odd source size leaves one destination texel over a single column/row
			uint y0 = 2 * y;
			uint y1 = min(2 * y + 1, src.y - 1);
			uint i00 = src.z + y0 * src.x + x0;
			uint i01 = src.z + y0 * src.x + x1;
			uint i10 = src.z + y1 * src.x + x0;
			uint i11 = src.z + y1 * src.x + x1;
			float f = max(max(asfloat(g_out.Load(i00 * 4)), asfloat(g_out.Load(i01 * 4))), max(asfloat(g_out.Load(i10 * 4)), asfloat(g_out.Load(i11 * 4))));
			float n = min(min(asfloat(g_out.Load(g_nearOffset + i00 * 4)), asfloat(g_out.Load(g_nearOffset + i01 * 4))),
			              min(asfloat(g_out.Load(g_nearOffset + i10 * 4)), asfloat(g_out.Load(g_nearOffset + i11 * 4))));
			g_out.Store((dst.z + i) * 4, asuint(f));
			g_out.Store(g_nearOffset + (dst.z + i) * 4, asuint(n));
		}
		DeviceMemoryBarrierWithGroupSync();
	}
}
)";

	// The constant buffer, laid out as the shader's cbuffer (fxc: scalars at 0-76, g_dims at 80).
	struct alignas(16) Params
	{
		std::uint32_t srcSize[2];
		std::uint32_t dstSize[2];
		std::uint32_t factor;
		float         nearReject;
		float         depthMin;
		float         invRange;
		float         depthA;
		float         depthB;
		std::uint32_t captureIndex;
		std::uint32_t mergeCount;
		std::uint32_t ringCur;
		std::uint32_t ringSize;
		std::uint32_t flags;
		std::uint32_t levels;
		std::uint32_t blocksW;
		std::uint32_t nearOffset;
		std::uint32_t changedOffset;
		std::uint32_t pad;
		std::uint32_t dims[16][4];
	};
	static_assert(sizeof(Params) == 80 + 16 * 16);

	constexpr std::uint32_t kFlagReset = 1;
	constexpr std::uint32_t kFlagLinear = 2;
	constexpr std::uint32_t kUAVs = 5;         // ring near, ring far, reference far, reference near, output
	constexpr std::uint32_t kGroupLevels = 4;  // pyramid levels the 8x8 groups produce (0-3); the tail pass does the rest
	constexpr std::uint32_t kRingSize = 8;     // recent captures kept on the GPU for the temporal merge (iHiZTemporalFrames <= 8)

	// Both pyramids' level sizes and texel offsets (level 0 first), as Snapshot carries them.
	struct Layout
	{
		std::uint32_t                 levels{ 0 };
		std::array<std::uint32_t, 16> width{};
		std::array<std::uint32_t, 16> height{};
		std::array<std::uint32_t, 16> offset{};
		std::uint32_t                 total{ 0 };  // texels per pyramid
	};

	inline void ComputeLayout(Layout& a_layout, std::uint32_t a_width, std::uint32_t a_height) noexcept
	{
		std::uint32_t total = 0;
		std::uint32_t level = 0;
		auto          w = a_width;
		auto          h = a_height;
		while (level < a_layout.width.size()) {
			a_layout.width[level] = w;
			a_layout.height[level] = h;
			a_layout.offset[level] = total;
			total += w * h;
			++level;
			if (w == 1 && h == 1) {
				break;
			}
			w = w > 1 ? (w + 1) / 2 : 1;
			h = h > 1 ? (h + 1) / 2 : 1;
		}
		a_layout.levels = level;
		a_layout.total = total;
	}

	// Byte offsets inside the output buffer (and its size, a multiple of 16).
	struct OutputLayout
	{
		std::uint32_t nearOffset{ 0 };
		std::uint32_t changedOffset{ 0 };
		std::uint32_t bytes{ 0 };
	};

	inline OutputLayout ComputeOutput(const Layout& a_layout, std::uint32_t a_blocks) noexcept
	{
		OutputLayout out{};
		out.nearOffset = a_layout.total * 4;
		out.changedOffset = a_layout.total * 8;
		out.bytes = (out.changedOffset + a_blocks * 4 + 15) & ~15u;
		return out;
	}
}
