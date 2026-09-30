// Offline brute-force check of ShadowGeometry.h (compiled as is), the geometry behind shadow-caster culling:
//  1. SweepClip never cuts away a part of a sun sweep whose cross-section holds a point of the receiver region;
//  2. the kept part of the sweep projects inside the box around its two end spheres' projections, and no point
//     of it is nearer than the nearer end's depth minus its radius (what TestSun compares against the Hi-Z);
//  3. CasterOutside never drops a point-light caster that comes within the filter's reach of a segment from the
//     light to a visible point, with the camera turned and moved since the cone was built.
// Build: cl /std:c++latest /O2 /EHsc /I src tools\tests\shadow_test.cpp   (VS x64 prompt)
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <random>

#include "Core/ShadowGeometry.h"

using namespace CBRO::Core::ShadowGeometry;
using V3 = std::array<float, 3>;

static std::mt19937 rng(20260928);
static float U(float a_lo, float a_hi) { return std::uniform_real_distribution<float>(a_lo, a_hi)(rng); }

static V3   Add(const V3& a, const V3& b) { return { a[0] + b[0], a[1] + b[1], a[2] + b[2] }; }
static V3   Sub(const V3& a, const V3& b) { return { a[0] - b[0], a[1] - b[1], a[2] - b[2] }; }
static V3   Mul(const V3& a, float s) { return { a[0] * s, a[1] * s, a[2] * s }; }
static float D(const V3& a, const V3& b) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; }
static float Len(const V3& a) { return std::sqrt(D(a, a)); }
static V3   Unit(const V3& a) { return Mul(a, 1.0f / Len(a)); }
static V3   Cross(const V3& a, const V3& b) { return { a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0] }; }
static V3   RandomUnit()
{
	for (;;) {
		const V3 v{ U(-1, 1), U(-1, 1), U(-1, 1) };
		const float l = Len(v);
		if (l > 0.1f && l <= 1.0f) {
			return Mul(v, 1.0f / l);
		}
	}
}
static V3 InBall(float a_r)
{
	for (;;) {
		const V3 v{ U(-1, 1), U(-1, 1), U(-1, 1) };
		if (D(v, v) <= 1.0f) {
			return Mul(v, a_r);
		}
	}
}
// Rotate v about unit axis k by angle a (Rodrigues).
static V3 Rotate(const V3& v, const V3& k, float a)
{
	const float c = std::cos(a), s = std::sin(a);
	return Add(Add(Mul(v, c), Mul(Cross(k, v), s)), Mul(k, D(k, v) * (1.0f - c)));
}

int main()
{
	int failures = 0;

	// ---- 1 + 2: sun sweeps in view space (x right, y up, z forward) ----------------------------------------
	{
		std::uint64_t sweeps = 0, clippedAway = 0, samplesIn = 0, clipMisses = 0, boxMisses = 0, depthMisses = 0;
		for (int trial = 0; trial < 20000; ++trial) {
			// the receiver region: in front of the eye, within the range, inside a view footprint (NDC box)
			const float reach = U(500.0f, 20000.0f);
			const float scaleX = U(0.8f, 2.5f), scaleY = U(1.2f, 3.0f);
			float       view[4]{ U(-1.4f, -0.2f), 0, U(-1.4f, -0.2f), 0 };
			view[1] = U(view[0] + 0.2f, 1.4f);
			view[3] = U(view[2] + 0.2f, 1.4f);
			Plane planes[6]{};
			int   count = 0;
			planes[count++] = { { 0, 0, 1 }, 0 };
			planes[count++] = { { 0, 0, -1 }, reach };
			const auto edge = [&](int a_axis, float a_scale, float a_edge, bool a_upper) {
				const float length = std::sqrt(a_scale * a_scale + a_edge * a_edge);
				auto&       plane = planes[count++];
				plane.n[a_axis] = (a_upper ? -a_scale : a_scale) / length;
				plane.n[2] = (a_upper ? a_edge : -a_edge) / length;
			};
			edge(0, scaleX, view[0], false);
			edge(0, scaleX, view[1], true);
			edge(1, scaleY, view[2], false);
			edge(1, scaleY, view[3], true);
			const auto inside = [&](const V3& q) {
				for (int i = 0; i < count; ++i) {
					if (D({ planes[i].n[0], planes[i].n[1], planes[i].n[2] }, q) + planes[i].d < -1.0e-3f) {
						return false;
					}
				}
				return true;
			};

			// a caster anywhere around (often behind or beside the eye), the light mostly pointing down
			const V3    c{ U(-8000, 8000), U(-8000, 8000), U(-6000, 16000) };
			V3          s = RandomUnit();
			const float r0 = U(1.0f, 600.0f);
			const float spread = (trial % 3) ? 0.0175f : 0.0f;
			float       t0 = 0, t1 = 0;
			const float cv[3]{ c[0], c[1], c[2] };
			const float sv[3]{ s[0], s[1], s[2] };
			const bool  kept = SweepClip(cv, sv, r0, spread, 1.0e6f, planes, count, t0, t1);
			++sweeps;
			clippedAway += !kept;

			// 1: brute force over the sweep, long enough to pass through the whole region (a receiver point is at
			// most ~2.4 x reach from the eye with these footprints, so t <= |c| + 2.4 reach)
			const float tEnd = Len(c) + reach * 3.0f + 2000.0f;
			for (int k = 0; k < 400; ++k) {
				const float t = U(0.0f, tEnd);
				const V3    q = Add(Add(c, Mul(s, t)), InBall(r0 + t * spread));
				if (!inside(q)) {
					continue;
				}
				++samplesIn;
				if (!kept || t < t0 - 1.0e-2f * (1.0f + t) || t > t1 + 1.0e-2f * (1.0f + t)) {
					if (clipMisses++ < 5) {
						std::printf("  sweep clip wrong: trial %d t=%.2f kept=%d [%.2f, %.2f]\n", trial, t, kept, t0, t1);
					}
				}
			}

			// 2: the kept part's projection box and nearest depth (only when TestSun would project it)
			if (!kept) {
				continue;
			}
			const V3    q0 = Add(c, Mul(s, t0)), q1 = Add(c, Mul(s, t1));
			const float ra = r0 + t0 * spread, rb = r0 + t1 * spread;
			const float nearest = std::min(q0[2] - ra, q1[2] - rb);
			if (!(nearest >= 8.0f)) {
				continue;
			}
			float x0, x1, y0, y1, u0, u1, v0, v1;
			SphereExtent(q0[0], q0[2], ra, x0, x1);
			SphereExtent(q0[1], q0[2], ra, y0, y1);
			SphereExtent(q1[0], q1[2], rb, u0, u1);
			SphereExtent(q1[1], q1[2], rb, v0, v1);
			const float bx0 = std::min(x0, u0), bx1 = std::max(x1, u1), by0 = std::min(y0, v0), by1 = std::max(y1, v1);
			for (int k = 0; k < 300; ++k) {
				const float t = U(t0, t1);
				const V3    q = Add(Add(c, Mul(s, t)), InBall(r0 + t * spread));
				if (q[2] < nearest - 1.0e-3f * (1.0f + q[2])) {
					if (depthMisses++ < 5) {
						std::printf("  nearest depth wrong: trial %d z=%.3f nearest=%.3f\n", trial, q[2], nearest);
					}
				}
				const float px = q[0] / q[2], py = q[1] / q[2];
				const float tol = 1.0e-4f * (1.0f + std::abs(px) + std::abs(py));
				if (px < bx0 - tol || px > bx1 + tol || py < by0 - tol || py > by1 + tol) {
					if (boxMisses++ < 5) {
						std::printf("  projection outside the box: trial %d (%.5f, %.5f) box [%.5f,%.5f]x[%.5f,%.5f]\n", trial, px, py, bx0, bx1, by0, by1);
					}
				}
			}
		}
		std::printf("1. sun sweeps: %llu, %llu entirely outside the receiver region; %llu receiver-region samples, %llu cut away wrongly (must be 0)\n",
			(unsigned long long)sweeps, (unsigned long long)clippedAway, (unsigned long long)samplesIn, (unsigned long long)clipMisses);
		std::printf("2. kept sweeps' projections: %llu points outside the box, %llu nearer than the nearest depth (both must be 0)\n",
			(unsigned long long)boxMisses, (unsigned long long)depthMisses);
		failures += static_cast<int>(clipMisses + boxMisses + depthMisses);
	}

	// ---- 3: point-light casters -----------------------------------------------------------------------------
	{
		std::uint64_t lights = 0, casters = 0, dropped = 0, wrong = 0, relevantPairs = 0;
		for (int trial = 0; trial < 20000; ++trial) {
			// the camera the cone was built from
			const V3    eye{ U(-5000, 5000), U(-5000, 5000), U(-500, 2000) };
			const V3    dir = RandomUnit();
			const V3    right = Unit(Cross(dir, std::abs(dir[2]) < 0.9f ? V3{ 0, 0, 1 } : V3{ 1, 0, 0 }));
			const V3    up = Cross(right, dir);
			const float tanX = U(0.3f, 1.3f), tanY = U(0.25f, 0.9f);
			// the current camera: turned by up to 3 degrees, moved up to 60 units since
			const float turn = U(0.0f, 0.0524f);
			const V3    axis = RandomUnit();
			const V3    dirNow = Rotate(dir, axis, turn), rightNow = Rotate(right, axis, turn), upNow = Rotate(up, axis, turn);
			const V3    move = InBall(60.0f);
			const V3    eyeNow = Add(eye, move);
			const float fEye[3]{ eye[0], eye[1], eye[2] }, fDir[3]{ dir[0], dir[1], dir[2] }, fRight[3]{ right[0], right[1], right[2] }, fUp[3]{ up[0], up[1], up[2] };
			const Cone  cone = MakeCone(fEye, fDir, fRight, fUp, tanX, tanY, turn + 0.0087f);
			const float conePush = Len(move) + 4.0f;

			// a light near the view (inside it about half the time), and its reach
			const float depth = U(-300.0f, 3000.0f);
			const V3    light = Add(eyeNow, Add(Mul(dirNow, depth), Add(Mul(rightNow, U(-1.6f, 1.6f) * std::abs(depth) + U(-400, 400)), Mul(upNow, U(-1.2f, 1.2f) * std::abs(depth) + U(-300, 300)))));
			const float reach = U(100.0f, 2000.0f);
			const float lightV[3]{ light[0], light[1], light[2] };
			const float filter = reach * 0.0875f + 16.0f;  // as ShadowLights (kFilterSlope, kFilterBase)
			const float push = OutsideBy(cone, lightV) + conePush + filter;
			++lights;

			// visible points of the current view within the light's reach
			std::array<V3, 160> visible{};
			int                 visibleCount = 0;
			for (int k = 0; k < 4000 && visibleCount < 160; ++k) {
				const float z = U(1.0f, depth + reach + 100.0f);
				const V3    p = Add(eyeNow, Add(Mul(dirNow, z), Add(Mul(rightNow, U(-tanX, tanX) * z), Mul(upNow, U(-tanY, tanY) * z))));
				if (Len(Sub(p, light)) <= reach) {
					visible[visibleCount++] = p;
				}
			}
			if (!visibleCount) {
				continue;
			}
			for (int k = 0; k < 40; ++k) {
				// casters around the light, many near the cone's boundary
				const V3    center = Add(light, InBall(reach * 1.2f));
				const float radius = U(1.0f, 300.0f);
				const float cv[3]{ center[0], center[1], center[2] };
				++casters;
				const bool drop = CasterOutside(cone, push, cv, radius);
				dropped += drop;
				// does it come within the filter's reach of a light-to-visible-point segment?
				bool matters = false;
				for (int v = 0; v < visibleCount && !matters; ++v) {
					const V3    seg = Sub(visible[v], light);
					const float t = std::clamp(D(Sub(center, light), seg) / std::max(D(seg, seg), 1.0e-6f), 0.0f, 1.0f);
					matters = Len(Sub(center, Add(light, Mul(seg, t)))) <= radius + filter;
				}
				relevantPairs += matters;
				if (drop && matters && wrong++ < 5) {
					std::printf("  caster dropped wrongly: trial %d\n", trial);
				}
			}
		}
		std::printf("3. point-light casters: %llu lights, %llu casters, %llu dropped, %llu near a visible point's light segment, %llu of those dropped (must be 0)\n",
			(unsigned long long)lights, (unsigned long long)casters, (unsigned long long)dropped, (unsigned long long)relevantPairs, (unsigned long long)wrong);
		failures += static_cast<int>(wrong);
	}

	std::printf(failures ? "FAILED: %d\n" : "all checks passed\n", failures);
	return failures ? 1 : 0;
}
