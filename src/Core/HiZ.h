#pragma once

// Hierarchical depth buffer, built entirely on the GPU from the main depth right after the opaque pre-pass
// (reduction, temporal merge, the whole min/max pyramid and the block-change map in two compute dispatches)
// and read back asynchronously as one flat buffer (no stalls, no CPU-side processing: the published snapshot
// points straight into the mapped readback). FO4 uses standard Z (0 near, 1 far), so each texel stores the
// FARTHEST depth it covers: an object is hidden when its nearest depth is farther. First-person geometry
// (viewport depth [0, 0.01]) is treated as far so the weapon never occludes.

namespace CBRO::Core::HiZ
{
	// The verdict cache's depth tolerance: with the cache on, a test counts an object as hidden only with this much
	// more room (linear units, and relative to the occluder's depth) than the plain tolerance, and a depth block
	// counts as changed once it moved by more than half of it against its reference (see Blocks()).
	constexpr float kCacheDepthSlack = 6.0f;
	constexpr float kCacheDepthTolerance = 0.004f;

	// Camera the depth was rendered with. The projection used by the tests is captured with it,
	// never mixed with a newer camera.
	struct Camera
	{
		float viewProj[4][4]{};  // row-vector convention after Runtime's self-check: clip = p * viewProj
		float posAdjust[3]{};    // subtracted from world positions before the transform (0 if not camera-relative)
		float eye[3]{};          // WorldRoot camera world translate
		float rotate[3][3]{};    // WorldRoot camera world rotate
		float viewDir[3]{};
		float viewRight[3]{};
		float viewUp[3]{};

		// View-space form of viewProj, derived and verified against it at capture. When valid the
		// test projects spheres exactly and compares linear depths; otherwise it falls back to
		// projecting the bound's box corners through viewProj.
		bool  viewSpace{ false };
		float origin[3]{};       // view-space origin in world units
		float scaleX{ 0.0f };    // ndc.x = scaleX * x_view / z_view
		float scaleY{ 0.0f };    // ndc.y = scaleY * y_view / z_view
		float depthA{ 0.0f };    // ndc.z = depthA + depthB / z_view
		float depthB{ 0.0f };

		// How the WorldRoot NiCamera's rotation maps onto this render basis (Runtime): bit 0 = its
		// columns fit, bit 1 = its rows fit (both when facing along a world axis). For each fitting
		// reading, the column/row index and sign giving viewDir, viewRight, viewUp. footprint = a
		// later frame's view can be placed on this depth frame.
		bool         footprint{ false };
		bool         looseFit{ false };  // the rotation matches the render basis only within ~5.7 deg (e.g. caught mid-turn)
		std::uint8_t axisReadings{ 0 };
		std::int8_t  axisIndex[2][3]{};
		std::int8_t  axisSign[2][3]{};
		float        frustumX{ 0.0f };  // WorldRoot NiCamera frustum half-extents at capture (raw units), to detect zoom
		float        frustumY{ 0.0f };
	};

	// A sphere seen from the depth frame's camera, in level-0 Hi-Z texel coordinates. Lets a test skip
	// texels (parts of the sphere's screen rectangle) that no ray through the sphere passes: the
	// rectangle's corners, and at coarse levels much of the rest.
	struct SphereRays
	{
		float centerX{ 0.0f };        // projection of the center
		float centerY{ 0.0f };
		float depth{ 0.0f };          // view depth of the center (> radius)
		float radius{ 0.0f };
		float texelsPerTanX{ 0.0f };  // texels per unit of x/z
		float texelsPerTanY{ 0.0f };
		float axisX{ 0.0f };          // where the view axis lands
		float axisY{ 0.0f };

		// True if no ray through the rectangle can meet the sphere. Conservative: the distance from
		// the center to a ray is at least (lateral offset at the center's depth) * cos(ray angle).
		[[nodiscard]] bool Misses(float a_x0, float a_y0, float a_x1, float a_y1) const noexcept;
	};

	struct Snapshot
	{
		std::uint64_t                 frame{ 0 };  // render frame (Runtime counter) the depth came from
		Camera                        camera{};
		// Temporal merge: level 0 is the farthest depth over this many consecutive captures of a (nearly)
		// still camera, the newest one's camera being `camera`. The older captures' cameras lie within
		// mergeMove units of it: add that to the dilation of every bound tested (the merge only ever makes
		// more visible, so it is safe with the newest camera's projection; the extra dilation keeps the
		// older frames' parallax covered too).
		std::uint32_t                 mergedFrames{ 1 };
		float                         mergeMove{ 0.0f };
		std::uint32_t                 readbackIndex{ 0 };  // the capture's index (Blocks() is dated by it)
		std::uint32_t                 levels{ 0 };
		std::array<std::uint32_t, 16> width{};
		std::array<std::uint32_t, 16> height{};
		std::array<std::uint32_t, 16> offset{};
		// Both pyramids, all levels, level 0 first, as the GPU wrote them: they point into the mapped readback,
		// which stays mapped until this snapshot has been superseded several times over (Poll).
		const float*                  texels{ nullptr };   // max-reduced buffer depth (the farthest surface per texel)
		const float*                  nearest{ nullptr };  // min-reduced (the nearest surface per texel; first-person counts as 0)
		std::int32_t                  slot{ -1 };          // the readback slot holding the data (HiZ.cpp)

		// Farthest depth over a level-0 texel-space rectangle, using the level where it spans <= 4 texels.
		[[nodiscard]] float MaxDepth(float a_x0, float a_y0, float a_x1, float a_y1) const noexcept;

		// True if every texel covering the rectangle has its nearest surface at or beyond a_threshold (buffer
		// depth): nothing visible there is nearer. Same level choice and refinement as AllNearer.
		[[nodiscard]] bool AllFarther(float a_x0, float a_y0, float a_x1, float a_y1, float a_threshold, std::uint32_t a_refine) const noexcept;

		// True if no visible surface over the rectangle lies within the buffer-depth range [a_near, a_far]: at every
		// texel the farthest surface is nearer than a_near, or the nearest surface is farther than a_far (a sky
		// texel, cleared to the far plane, always is). Coarse texels that are mixed are refined like AllNearer.
		// With a_rays, texels the sphere's rays can't reach are skipped (as AllNearer): a surface seen there lies outside it.
		[[nodiscard]] bool NoSurfaceBetween(float a_x0, float a_y0, float a_x1, float a_y1, float a_near, float a_far, std::uint32_t a_refine, const SphereRays* a_rays = nullptr) const noexcept;

		// True if every texel covering the rectangle is nearer than a_threshold (buffer depth). Starts
		// at the level where the rectangle spans <= 4 texels; a texel that fails there is re-checked at
		// up to a_refine finer levels, over the part of the rectangle it covers only. With a_rays, texels
		// the sphere's rays can't reach are skipped.
		[[nodiscard]] bool AllNearer(float a_x0, float a_y0, float a_x1, float a_y1, float a_threshold, std::uint32_t a_refine, const SphereRays* a_rays = nullptr) const noexcept;

		// Diagnostic: the level-0 texels of the rectangle (only those the sphere's rays reach, with a_rays) at or beyond
		// a_threshold. No such texel means a coarse texel alone made AllNearer fail (its refinement ran out).
		struct FartherScan
		{
			std::uint32_t texels{ 0 };
			std::uint32_t farPlane{ 0 };     // ... of them at the far plane: cleared (no surface) or first-person pixels
			std::uint32_t firstPerson{ 0 };  // ... of those holding first-person geometry (nearest 0)
			float         farPlaneNearest{ 1.0f };  // the nearest surface in those far-plane texels (below 1: a texel mixing nothing drawn with surfaces)
			float         farthest{ 0.0f };
			float         x{ -1.0f };      // the first one found (level-0 texel coordinates)
			float         y{ -1.0f };
		};
		[[nodiscard]] FartherScan ScanFarther(float a_x0, float a_y0, float a_x1, float a_y1, float a_threshold, const SphereRays* a_rays) const noexcept;

	private:
		[[nodiscard]] bool AllNearerAt(std::uint32_t a_level, float a_x0, float a_y0, float a_x1, float a_y1, float a_threshold, std::uint32_t a_refine, const SphereRays* a_rays) const noexcept;
		[[nodiscard]] bool AllFartherAt(std::uint32_t a_level, float a_x0, float a_y0, float a_x1, float a_y1, float a_threshold, std::uint32_t a_refine) const noexcept;
		[[nodiscard]] bool NoSurfaceBetweenAt(std::uint32_t a_level, float a_x0, float a_y0, float a_x1, float a_y1, float a_near, float a_far, std::uint32_t a_refine, const SphereRays* a_rays) const noexcept;
	};

	// Creates GPU resources on first use. Returns false if the device path is unusable.
	bool Capture(std::uint64_t a_frame, const Camera& a_camera);  // after the pre-pass
	void Poll();                                                  // map finished readbacks, publish the newest
	[[nodiscard]] const Snapshot* Latest() noexcept;              // newest published snapshot or null
	void Reset();                                                 // forget published data (load, device change)

	[[nodiscard]] bool Failed() noexcept;
	[[nodiscard]] std::string Describe();

	// Per 8x8 block of level 0: the capture index at which its farthest depth last got farther, or its nearest depth
	// nearer, by more than half the cache tolerance against the block's reference depths (per texel: the nearest
	// farthest depth, and the farthest nearest depth, seen since the block was last marked; kept on the GPU, compared
	// there at every capture, so captures that were never published count too). So "unmarked since capture R" means
	// no texel of the block is farther (or nearer) now than at any capture since R, beyond the tolerance: a verdict
	// that read the depth at R still holds. Occlusion also re-checks a hidden verdict's own evidence against the
	// current depth when its blocks did change (the map is coarse; under a swaying camera most blocks mark each frame).
	struct BlockChanges
	{
		const std::uint32_t* changedAt{ nullptr };
		std::uint32_t        width{ 0 };   // in blocks
		std::uint32_t        height{ 0 };
	};
	[[nodiscard]] BlockChanges Blocks() noexcept;
	[[nodiscard]] std::uint32_t TakeBlocksChanged() noexcept;  // blocks marked changed since the last call (for the log)
	// Of those, what the newest capture holds at the block (diagnostic: where a still view's changes come from).
	struct BlockChangeKinds
	{
		std::uint32_t firstPerson{ 0 };  // first-person pixels in the block (nearest 0)
		std::uint32_t farPlane{ 0 };     // ... else the far plane (sky) in the block
		std::uint32_t lowerThird{ 0 };   // blocks in the lower third of the view (any kind)
	};
	[[nodiscard]] BlockChangeKinds TakeBlockChangeKinds() noexcept;
}
