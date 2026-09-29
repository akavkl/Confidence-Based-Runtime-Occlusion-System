// Offline check of ShapeFit: real meshes (bound made from their own vertices) must pass, vertex data
// that is only part of what a bound covers (a merge-instanced group's source meshes) must fail.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>

#include "Core/ShapeFit.h"

using namespace CBRO::Core;
using V3 = std::array<float, 3>;

std::mt19937 rng(11);
float U(float a, float b) { return std::uniform_real_distribution<float>(a, b)(rng); }

std::vector<V3> Mesh()
{
	std::vector<V3> v;
	const int blobs = 1 + rng() % 4;
	const float span = U(20, 1500);
	for (int b = 0; b < blobs; ++b) {
		const V3 c{ U(-span, span), U(-span, span), U(-span * 0.3f, span * 0.3f) };
		const float s = U(5, span * 0.5f);
		for (int k = 0; k < 30; ++k) v.push_back({ c[0] + U(-s, s), c[1] + U(-s, s), c[2] + U(-s, s) });
	}
	// an origin offset like real meshes (pivot at the base, at a corner, ...)
	const V3 o{ U(-span, span), U(-span, span), U(-span, span) };
	for (auto& p : v) for (int a = 0; a < 3; ++a) p[a] += o[a];
	return v;
}

struct Bound { V3 c; float r; };

Bound BoxBound(const std::vector<V3>& v)  // AABB center, farthest vertex (Gamebryo/nifly style)
{
	V3 lo{ 1e30f, 1e30f, 1e30f }, hi{ -1e30f, -1e30f, -1e30f };
	for (auto& p : v) for (int a = 0; a < 3; ++a) { lo[a] = std::min(lo[a], p[a]); hi[a] = std::max(hi[a], p[a]); }
	Bound b{ { (lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, (lo[2] + hi[2]) / 2 }, 0 };
	for (auto& p : v) b.r = std::max(b.r, std::sqrt((p[0] - b.c[0]) * (p[0] - b.c[0]) + (p[1] - b.c[1]) * (p[1] - b.c[1]) + (p[2] - b.c[2]) * (p[2] - b.c[2])));
	return b;
}

Bound Ritter(const std::vector<V3>& v)
{
	auto d = [](const V3& a, const V3& b) { return std::sqrt((a[0] - b[0]) * (a[0] - b[0]) + (a[1] - b[1]) * (a[1] - b[1]) + (a[2] - b[2]) * (a[2] - b[2])); };
	V3 x = v[0], y = x, z = x;
	for (auto& p : v) if (d(p, x) > d(y, x)) y = p;
	for (auto& p : v) if (d(p, y) > d(z, y)) z = p;
	Bound b{ { (y[0] + z[0]) / 2, (y[1] + z[1]) / 2, (y[2] + z[2]) / 2 }, d(y, z) / 2 };
	for (auto& p : v) {
		const float dist = d(p, b.c);
		if (dist > b.r) {
			const float nr = (b.r + dist) / 2, k = (nr - b.r) / dist;
			for (int a = 0; a < 3; ++a) b.c[a] += (p[a] - b.c[a]) * k;
			b.r = nr;
		}
	}
	return b;
}

ShapeFit::Result Fit(const std::vector<V3>& v, const Bound& b)
{
	float lo[3]{ 1e30f, 1e30f, 1e30f }, hi[3]{ -1e30f, -1e30f, -1e30f }, far2 = 0;
	for (auto& p : v) {
		float d2 = 0;
		for (int a = 0; a < 3; ++a) { lo[a] = std::min(lo[a], p[a]); hi[a] = std::max(hi[a], p[a]); d2 += (p[a] - b.c[a]) * (p[a] - b.c[a]); }
		far2 = std::max(far2, d2);
	}
	return ShapeFit::Check(lo, hi, std::sqrt(far2), b.c.data(), b.r);
}

V3 Rotate(const float q[4], const V3& p)
{
	const float R[3][3]{
		{ 1 - 2 * (q[2] * q[2] + q[3] * q[3]), 2 * (q[1] * q[2] - q[0] * q[3]), 2 * (q[1] * q[3] + q[0] * q[2]) },
		{ 2 * (q[1] * q[2] + q[0] * q[3]), 1 - 2 * (q[1] * q[1] + q[3] * q[3]), 2 * (q[2] * q[3] - q[0] * q[1]) },
		{ 2 * (q[1] * q[3] - q[0] * q[2]), 2 * (q[2] * q[3] + q[0] * q[1]), 1 - 2 * (q[1] * q[1] + q[2] * q[2]) } };
	return { R[0][0] * p[0] + R[0][1] * p[1] + R[0][2] * p[2], R[1][0] * p[0] + R[1][1] * p[1] + R[1][2] * p[2], R[2][0] * p[0] + R[2][1] * p[1] + R[2][2] * p[2] };
}

int main()
{
	int realFail[2]{}, trials = 20000;
	for (int t = 0; t < trials; ++t) {
		const auto v = Mesh();
		realFail[0] += Fit(v, BoxBound(v)) != ShapeFit::Result::kOk;
		realFail[1] += Fit(v, Ritter(v)) != ShapeFit::Result::kOk;
	}
	std::printf("real meshes: %d trials | rejected with box-center bounds %d, with Ritter bounds %d (should be ~0)\n", trials, realFail[0], realFail[1]);

	// Merge-instanced groups: sources in their own space; instances rotated/moved; bound around all
	// instances, recentred on the group (local translate = group center, as the builder does).
	for (int n = 1; n <= 8; ++n) {
		int accepted = 0, acceptedRotated = 0, groups = 5000;
		for (int t = 0; t < groups; ++t) {
			const int sources = 1 + rng() % 3;
			std::vector<std::vector<V3>> src(sources);
			for (auto& s : src) s = Mesh();
			std::vector<V3> drawn;
			bool rotated = false;
			for (int i = 0; i < n; ++i) {
				const auto& s = src[rng() % sources];
				float q[4]{ 1, 0, 0, 0 };
				if (rng() % 4 != 0) {  // most instances rotated
					q[0] = U(-1, 1); q[1] = U(-1, 1); q[2] = U(-1, 1); q[3] = U(-1, 1);
					const float l = std::sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]);
					for (auto& c : q) c /= l;
					rotated = true;
				}
				const V3 tr{ U(-2000, 2000), U(-2000, 2000), U(-300, 300) };
				for (auto& p : s) { auto r = Rotate(q, p); drawn.push_back({ r[0] + tr[0], r[1] + tr[1], r[2] + tr[2] }); }
			}
			auto b = BoxBound(drawn);
			// recentre: the group's node sits at the bound center; the model bound is then around 0
			const V3 center = b.c;
			b.c = { 0, 0, 0 };
			(void)center;
			std::vector<V3> buffer;  // what the vertex buffer holds: the sources, untransformed
			for (auto& s : src) buffer.insert(buffer.end(), s.begin(), s.end());
			if (Fit(buffer, b) == ShapeFit::Result::kOk) {
				++accepted;
				acceptedRotated += rotated;
			}
		}
		std::printf("merge-instanced groups of %d: %d of %d would pass the fit (%d with rotated instances) -> the type check must exclude them\n", n, accepted, groups, acceptedRotated);
	}
	return 0;
}
