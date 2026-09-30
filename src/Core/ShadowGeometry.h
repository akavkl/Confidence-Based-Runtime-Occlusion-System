#pragma once

// Geometry for shadow-caster culling (Occlusion: the sun; ShadowLights: point lights). Engine-free, so the
// offline brute-force test compiles it as is.
//
// Sun: a caster's shadow is the capsule swept from its bound along the light direction. It can only matter
// where that capsule meets the region where shadow receivers can be (the current view, in front of the eye,
// within the cascade range). SweepClip finds that part of the sweep; the caller then tests it against the
// depth buffer.
//
// Point light: the light L and every visible point P lie in a convex region C (the view cone, its planes
// pushed out until they contain L). The segment L-P then lies in C too, and a caster only shadows P if it
// meets that segment (plus the shadow filter's reach). A caster entirely outside C, by more than the filter's
// reach, shadows nothing visible.

namespace CBRO::Core::ShadowGeometry
{
	// Half-space n.p + d >= 0 (n unit length).
	struct Plane
	{
		float n[3]{};
		float d{ 0.0f };
	};

	inline float Dot(const float a_a[3], const float a_b[3]) noexcept
	{
		return a_a[0] * a_b[0] + a_a[1] * a_b[1] + a_a[2] * a_b[2];
	}

	// Exact screen extent of a sphere on one view axis: the slopes x/z of the two lines through the eye that
	// touch the circle (a_x, a_z, a_r), with a_z > a_r.
	inline void SphereExtent(float a_x, float a_z, float a_r, float& a_min, float& a_max) noexcept
	{
		const float denom = a_z * a_z - a_r * a_r;
		const float root = a_r * std::sqrt(std::max(0.0f, a_x * a_x + denom));
		a_min = (a_x * a_z - root) / denom;
		a_max = (a_x * a_z + root) / denom;
	}

	// The sweep p(t) = c + t*s, t in [0, tMax], with radius r0 + t*spread (spread covers an uncertain
	// direction). Returns false if no cross-section B(p(t), r(t)) reaches every half-space; otherwise [t0, t1]
	// holds every t whose cross-section does. (B reaches n.q + d >= 0 iff n.p + d + r >= 0: linear in t.)
	inline bool SweepClip(const float a_c[3], const float a_s[3], float a_r0, float a_spread, float a_tMax, const Plane* a_planes, int a_count, float& a_t0, float& a_t1) noexcept
	{
		a_t0 = 0.0f;
		a_t1 = a_tMax;
		for (int i = 0; i < a_count; ++i) {
			const auto& plane = a_planes[i];
			const float a = Dot(plane.n, a_c) + plane.d + a_r0;
			const float b = Dot(plane.n, a_s) + a_spread;
			if (b > 1.0e-9f) {
				a_t0 = std::max(a_t0, -a / b);
			} else if (b < -1.0e-9f) {
				a_t1 = std::min(a_t1, -a / b);
			} else if (a < 0.0f) {
				return false;
			}
			if (!(a_t0 <= a_t1)) {
				return false;
			}
		}
		return true;
	}

	// ---- point lights -----------------------------------------------------------------------------------

	// A view cone: four planes through the eye with inward unit normals (a point with view coordinates
	// x (right), y (up), z (forward) is inside iff |x| <= tanX z and |y| <= tanY z).
	struct Cone
	{
		float eye[3]{};
		float n[4][3]{};
	};

	// a_extraAngle (radians) widens every side (camera rotation since the basis was captured, plus margin).
	inline Cone MakeCone(const float a_eye[3], const float a_dir[3], const float a_right[3], const float a_up[3], float a_tanX, float a_tanY, float a_extraAngle) noexcept
	{
		Cone cone{};
		for (int i = 0; i < 3; ++i) {
			cone.eye[i] = a_eye[i];
		}
		const float tx = std::tan(std::min(std::atan(a_tanX) + a_extraAngle, 1.5f));
		const float ty = std::tan(std::min(std::atan(a_tanY) + a_extraAngle, 1.5f));
		const float lx = std::sqrt(tx * tx + 1.0f);
		const float ly = std::sqrt(ty * ty + 1.0f);
		for (int i = 0; i < 3; ++i) {
			cone.n[0][i] = (tx * a_dir[i] - a_right[i]) / lx;  // x <= tanX z
			cone.n[1][i] = (tx * a_dir[i] + a_right[i]) / lx;  // x >= -tanX z
			cone.n[2][i] = (ty * a_dir[i] - a_up[i]) / ly;     // y <= tanY z
			cone.n[3][i] = (ty * a_dir[i] + a_up[i]) / ly;     // y >= -tanY z
		}
		return cone;
	}

	// How far a point lies outside the cone (the largest distance beyond one of its planes), 0 inside.
	inline float OutsideBy(const Cone& a_cone, const float a_p[3]) noexcept
	{
		const float v[3]{ a_p[0] - a_cone.eye[0], a_p[1] - a_cone.eye[1], a_p[2] - a_cone.eye[2] };
		float       outside = 0.0f;
		for (const auto& n : a_cone.n) {
			outside = std::max(outside, -Dot(n, v));
		}
		return outside;
	}

	// A caster sphere entirely beyond one of the cone's planes pushed out by a_push (the light's own distance
	// outside the cone plus the shadow filter's reach and any camera movement): it can't shadow a visible point.
	inline bool CasterOutside(const Cone& a_cone, float a_push, const float a_center[3], float a_radius) noexcept
	{
		const float v[3]{ a_center[0] - a_cone.eye[0], a_center[1] - a_cone.eye[1], a_center[2] - a_cone.eye[2] };
		for (const auto& n : a_cone.n) {
			if (Dot(n, v) < -(a_push + a_radius)) {
				return true;
			}
		}
		return false;
	}

	// Spot light: a sphere holding its lit volume, the points within a_length of the light that lie inside the shadow
	// frustum (the engine's own bound of the volume), i.e. within the frustum's corner angle of the axis, whose tangent
	// is a_tanCorner. The center lies a_along down the axis. Under 60 degrees the sphere through the light and the
	// cap's rim, R = L / (2 cos), holds the cone's cap and so every segment from the light to it; wider, the ball itself.
	inline void SpotSphere(float a_length, float a_tanCorner, float& a_along, float& a_radius) noexcept
	{
		const float cosCorner = 1.0f / std::sqrt(1.0f + a_tanCorner * a_tanCorner);
		if (cosCorner > 0.5f) {
			a_radius = 0.5f * a_length / cosCorner;
			a_along = a_radius;
		} else {
			a_radius = a_length;
			a_along = 0.0f;
		}
	}
}
