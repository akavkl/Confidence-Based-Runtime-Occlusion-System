// Brute-force check of CBRO's mesh-shape test (box + occupied cells, near clipping, view rect).
// Camera at the origin looking down +z (view space = world space here, but the mesh has its own
// rotation/scale/translation). Occluders: a per-texel linear depth grid. The test may only say
// "hidden" if no sampled point of any triangle is in front of the occluder at its texel.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

constexpr float kScaleX = 1.3242f, kScaleY = 2.1187f, kNear = 8.0f, kSlack = 4.0f, kTol = 0.002f;
constexpr int   W = 80, H = 50;
std::vector<float> g_depth(W * H);  // linear occluder depth per texel (inf = sky)

enum class V { Visible, Hidden, Outside };

// Max occluder depth over the NDC rect (like AllNearer on level 0).
bool AllNearer(float x0, float x1, float y0, float y1, float limit)
{
	const int px0 = std::clamp(int((x0 * 0.5f + 0.5f) * W), 0, W - 1), px1 = std::clamp(int((x1 * 0.5f + 0.5f) * W), 0, W - 1);
	const int py0 = std::clamp(int((0.5f - y1 * 0.5f) * H), 0, H - 1), py1 = std::clamp(int((0.5f - y0 * 0.5f) * H), 0, H - 1);
	for (int y = py0; y <= py1; ++y)
		for (int x = px0; x <= px1; ++x)
			if (!(g_depth[y * W + x] < limit)) return false;
	return true;
}

V TestViewBox(const float c[3], const float h[3])
{
	const float zFar = c[2] + h[2];
	if (zFar < kNear) return V::Outside;
	const float view[4]{ -1, 1, -1, 1 };
	auto outsidePlane = [&](float s, float coord, float ext, float edge, bool upper) {
		const float side = s * coord - edge * c[2];
		const float reach = s * ext + std::abs(edge) * h[2];
		return upper ? side > reach : side < -reach;
	};
	if (outsidePlane(kScaleX, c[0], h[0], view[0], false) || outsidePlane(kScaleX, c[0], h[0], view[1], true) ||
		outsidePlane(kScaleY, c[1], h[1], view[2], false) || outsidePlane(kScaleY, c[1], h[1], view[3], true))
		return V::Outside;
	const float zN = std::max(c[2] - h[2], kNear);
	float x0 = INFINITY, x1 = -INFINITY, y0 = INFINITY, y1 = -INFINITY;
	for (float z : { zN, zFar })
		for (float s : { -1.f, 1.f }) {
			const float x = kScaleX * (c[0] + s * h[0]) / z, y = kScaleY * (c[1] + s * h[1]) / z;
			x0 = std::min(x0, x); x1 = std::max(x1, x); y0 = std::min(y0, y); y1 = std::max(y1, y);
		}
	x0 = std::max(x0, view[0]); x1 = std::min(x1, view[1]); y0 = std::max(y0, view[2]); y1 = std::min(y1, view[3]);
	if (x0 >= x1 || y0 >= y1) return V::Outside;
	const float limit = (zN - kSlack) / (1 + kTol);
	if (!(limit > 0)) return V::Visible;
	return AllNearer(x0, x1, y0, y1, limit) ? V::Hidden : V::Visible;
}

struct Shape { float min[3], cell[3]; int dims[3]; std::vector<bool> occ; };

int main()
{
	std::mt19937 rng(7);
	std::uniform_real_distribution<float> u(-1, 1), u01(0, 1);
	int wrong = 0, hiddenCount = 0, trials = 0, cellsHelped = 0;
	for (int t = 0; t < 3000; ++t) {
		// occluders: a wall at random depth covering a random part of the screen, rest sky or far
		const float wall = 20 + u01(rng) * 400;
		const float cover = 0.5f + 0.5f * u01(rng);
		for (int y = 0; y < H; ++y)
			for (int x = 0; x < W; ++x)
				g_depth[y * W + x] = (y < H * cover) ? wall : (u01(rng) < 0.5f ? INFINITY : 3000 + u01(rng) * 2000);

		// mesh: triangles clustered in a few blobs (like a precombined chunk), in local space
		std::vector<std::array<float, 9>> tris;
		const int blobs = 1 + rng() % 4;
		for (int b = 0; b < blobs; ++b) {
			const float bx = u(rng) * 800, by = u(rng) * 800, bz = u(rng) * 200, size = 20 + u01(rng) * 150;
			for (int k = 0; k < 20; ++k) {
				std::array<float, 9> tri;
				for (int v = 0; v < 3; ++v) {
					tri[v * 3 + 0] = bx + u(rng) * size; tri[v * 3 + 1] = by + u(rng) * size; tri[v * 3 + 2] = bz + u(rng) * size;
				}
				tris.push_back(tri);
			}
		}
		// shape: box + 8x8x8-ish grid of cells touched by triangle boxes
		Shape shape{};
		float lo[3]{ INFINITY, INFINITY, INFINITY }, hi[3]{ -INFINITY, -INFINITY, -INFINITY };
		for (auto& tri : tris) for (int v = 0; v < 3; ++v) for (int a = 0; a < 3; ++a) { lo[a] = std::min(lo[a], tri[v * 3 + a]); hi[a] = std::max(hi[a], tri[v * 3 + a]); }
		float maxExtent = 0; for (int a = 0; a < 3; ++a) maxExtent = std::max(maxExtent, hi[a] - lo[a]);
		const float grow = maxExtent * 0.001f + 0.5f;
		for (int a = 0; a < 3; ++a) { lo[a] -= grow; hi[a] += grow; shape.min[a] = lo[a]; }
		const float target = maxExtent / 8 + 2 * grow;
		for (int a = 0; a < 3; ++a) shape.dims[a] = std::clamp(int(std::ceil((hi[a] - lo[a]) / target)), 1, 8);
		while (shape.dims[0] * shape.dims[1] * shape.dims[2] > 128) { int* m = std::max_element(shape.dims, shape.dims + 3); --*m; }
		for (int a = 0; a < 3; ++a) shape.cell[a] = (hi[a] - lo[a]) / shape.dims[a];
		shape.occ.assign(shape.dims[0] * shape.dims[1] * shape.dims[2], false);
		for (auto& tri : tris) {
			float tl[3]{ INFINITY, INFINITY, INFINITY }, th[3]{ -INFINITY, -INFINITY, -INFINITY };
			for (int v = 0; v < 3; ++v) for (int a = 0; a < 3; ++a) { tl[a] = std::min(tl[a], tri[v * 3 + a]); th[a] = std::max(th[a], tri[v * 3 + a]); }
			int c0[3], c1[3];
			for (int a = 0; a < 3; ++a) {
				c0[a] = std::min(shape.dims[a] - 1, int(std::max(0.f, (tl[a] - lo[a]) / shape.cell[a])));
				c1[a] = std::min(shape.dims[a] - 1, int(std::max(0.f, (th[a] - lo[a]) / shape.cell[a])));
			}
			for (int z = c0[2]; z <= c1[2]; ++z) for (int y = c0[1]; y <= c1[1]; ++y) for (int x = c0[0]; x <= c1[0]; ++x)
				shape.occ[x + shape.dims[0] * (y + shape.dims[1] * z)] = true;
		}

		// world transform: random rotation (from a random quaternion), scale, translation in front-ish
		float q[4]{ u(rng), u(rng), u(rng), u(rng) }; float qn = std::sqrt(q[0]*q[0]+q[1]*q[1]+q[2]*q[2]+q[3]*q[3]); for (auto& c : q) c /= qn;
		const float R[3][3]{
			{ 1 - 2*(q[2]*q[2]+q[3]*q[3]), 2*(q[1]*q[2]-q[0]*q[3]), 2*(q[1]*q[3]+q[0]*q[2]) },
			{ 2*(q[1]*q[2]+q[0]*q[3]), 1 - 2*(q[1]*q[1]+q[3]*q[3]), 2*(q[2]*q[3]-q[0]*q[1]) },
			{ 2*(q[1]*q[3]-q[0]*q[2]), 2*(q[2]*q[3]+q[0]*q[1]), 1 - 2*(q[1]*q[1]+q[2]*q[2]) } };
		const float s = 0.5f + u01(rng) * 1.5f;
		const float T[3]{ u(rng) * 1500, u(rng) * 800, -300 + u01(rng) * 3000 };

		auto box = [&](const float blo[3], const float bsize[3]) {
			float c[3], h[3];
			for (int i = 0; i < 3; ++i) {
				c[i] = T[i]; h[i] = 1.0f;  // dilate 1 like the plugin
				for (int j = 0; j < 3; ++j) { c[i] += s * R[i][j] * (blo[j] + bsize[j] * 0.5f); h[i] += std::abs(s * R[i][j]) * bsize[j] * 0.5f; }
			}
			return TestViewBox(c, h);
		};
		const float size[3]{ shape.cell[0] * shape.dims[0], shape.cell[1] * shape.dims[1], shape.cell[2] * shape.dims[2] };
		V verdict = box(shape.min, size);
		bool usedCells = false;
		if (verdict == V::Visible) {
			bool anyHidden = false, kept = false;
			for (int z = 0; z < shape.dims[2] && !kept; ++z) for (int y = 0; y < shape.dims[1] && !kept; ++y) for (int x = 0; x < shape.dims[0] && !kept; ++x) {
				if (!shape.occ[x + shape.dims[0] * (y + shape.dims[1] * z)]) continue;
				const float blo[3]{ shape.min[0] + shape.cell[0] * x, shape.min[1] + shape.cell[1] * y, shape.min[2] + shape.cell[2] * z };
				const V cv = box(blo, shape.cell);
				if (cv == V::Hidden) anyHidden = true; else if (cv != V::Outside) kept = true;
			}
			if (!kept) { verdict = anyHidden ? V::Hidden : V::Outside; usedCells = true; }
		}
		++trials;
		if (verdict == V::Visible) continue;
		++hiddenCount;
		cellsHelped += usedCells;
		// brute force: sample points on every triangle; any point in view and in front of the occluder = visible
		bool seen = false;
		for (auto& tri : tris) {
			for (int k = 0; k < 60 && !seen; ++k) {
				float a = u01(rng), b = u01(rng); if (a + b > 1) { a = 1 - a; b = 1 - b; }
				float p[3];
				for (int i = 0; i < 3; ++i) p[i] = tri[i] + a * (tri[3 + i] - tri[i]) + b * (tri[6 + i] - tri[i]);
				float w[3];
				for (int i = 0; i < 3; ++i) { w[i] = T[i]; for (int j = 0; j < 3; ++j) w[i] += s * R[i][j] * p[j]; }
				if (w[2] < 15.0f) continue;  // real near plane (>= kNear) clips it
				const float nx = kScaleX * w[0] / w[2], ny = kScaleY * w[1] / w[2];
				if (nx < -1 || nx > 1 || ny < -1 || ny > 1) continue;
				const int px = std::clamp(int((nx * 0.5f + 0.5f) * W), 0, W - 1), py = std::clamp(int((0.5f - ny * 0.5f) * H), 0, H - 1);
				if (w[2] < g_depth[py * W + px]) seen = true;  // in front of the occluder: visible
			}
			if (seen) break;
		}
		if (seen) {
			++wrong;
			if (wrong < 5) std::printf("WRONG: trial %d judged %s but a triangle point is visible\n", t, verdict == V::Hidden ? "hidden" : "outside");
		}
	}
	std::printf("shape test: %d trials, %d judged hidden/outside (%d thanks to cells), %d wrongly (must be 0)\n", trials, hiddenCount, cellsHelped, wrong);
	return wrong != 0;
}
