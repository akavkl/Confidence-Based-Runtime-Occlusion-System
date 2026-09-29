#pragma once

// Checks that a mesh's vertices are the ones its model bound was made from, before they become an
// occlusion shape (MeshProxy). Engine-free, so the offline brute-force test compiles it as is.
//
// A bound computed from a mesh's own vertices contains them all, reaches one of them, and sits at
// (or very near) the middle of their box. Vertex data that is only part of what the bound covers
// (e.g. one source mesh of an instanced group) fails at least one of these.

namespace CBRO::Core::ShapeFit
{
	enum class Result : std::uint8_t
	{
		kOk,
		kOutsideBound,  // a vertex lies outside the bound: wrong layout or not this mesh's data
		kLooseBound,    // no vertex comes near the bound's surface: the bound covers more than these
		kOffCenter,     // the vertices sit to one side of the bound: part of something bigger
		kTooSmall,      // the vertices fill too little of the bound
		kCount
	};

	// a_lo/a_hi: box of the vertices; a_farthest: largest vertex distance from the bound's center.
	inline Result Check(const float a_lo[3], const float a_hi[3], float a_farthest, const float a_center[3], float a_radius) noexcept
	{
		if (!(a_radius > 0.0f) || !(a_farthest <= a_radius * 1.02f + 2.0f)) {
			return Result::kOutsideBound;
		}
		if (a_farthest < a_radius * 0.8f - 2.0f) {
			return Result::kLooseBound;
		}
		float offset2 = 0.0f;
		float maxExtent = 0.0f;
		for (int a = 0; a < 3; ++a) {
			const float offset = (a_lo[a] + a_hi[a]) * 0.5f - a_center[a];
			offset2 += offset * offset;
			maxExtent = std::max(maxExtent, a_hi[a] - a_lo[a]);
		}
		const float offsetLimit = a_radius * 0.25f + 2.0f;
		if (offset2 > offsetLimit * offsetLimit) {
			return Result::kOffCenter;
		}
		if (maxExtent < a_radius * 0.5f) {
			return Result::kTooSmall;
		}
		return Result::kOk;
	}

	// The shape may serve another object only if that object's model bound is the one it was checked
	// against (a mesh shared with a different object, e.g. an instanced group, has another bound).
	inline bool SameBound(const float a_bound[4], const float a_center[3], float a_radius) noexcept
	{
		const float tolerance = a_bound[3] * 1.0e-4f + 1.0e-3f;
		return std::abs(a_bound[0] - a_center[0]) <= tolerance && std::abs(a_bound[1] - a_center[1]) <= tolerance &&
		       std::abs(a_bound[2] - a_center[2]) <= tolerance && std::abs(a_bound[3] - a_radius) <= tolerance;
	}
}
