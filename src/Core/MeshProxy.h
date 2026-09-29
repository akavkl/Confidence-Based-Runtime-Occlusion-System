#pragma once

// Occlusion shapes for big static meshes (precombined chunks, LOD blocks, terrain quads), built from the
// engine's CPU copy of their vertex and index data (BSGeometry::rendererData -> BSGraphics::TriShape ->
// vertex/index Buffer::data; present for nearly every static mesh per the v1.9 kept-object dump). A
// shape is the local box around the vertices plus a coarse grid (at most 8 cells per axis, 128 in all)
// of the cells the triangles touch. A cell-sized precombined chunk whose bounding sphere always grazes
// some sky can then be judged by the parts it really has. Shapes are built once per mesh on a
// background thread and checked against the mesh's own model bound before use; culling threads only
// look them up.
//
// Only meshes whose vertex data is exactly what gets drawn qualify: plain BSTriShape, BSSubIndexTriShape
// and BSMeshLODTriShape with a lighting shader that doesn't move vertices (no skinning, tessellation,
// billboarding or tree animation). BSMergeInstancedTriShape does NOT: its vertex buffer holds several
// source meshes in their own local spaces and the per-instance transforms live only in a GPU buffer
// (+0x170, 80 bytes per instance; see the precombine builder at Fallout4.exe+0x28444A0). A shape built
// from it lands in the wrong place (v1.10 hid a visible wall that way).

namespace CBRO::Core::MeshProxy
{
	struct Shape
	{
		float         min[3]{};       // local box corner (vertices, slightly grown)
		float         cell[3]{};      // cell size per axis
		std::uint8_t  dims[3]{};      // cells per axis
		std::uint8_t  count{ 0 };     // occupied cells; 0 = judge the whole box only
		std::uint64_t occupied[2]{};  // bit (x + dims0 * (y + dims1 * z))
		float         bound[4]{};     // the model bound (center, radius) the vertices were checked against
	};

	void Install();

	// Main thread at each frame start, before any culling: clears the cache when it is nearly full.
	void BeginFrame();

	// The shape of a static, unskinned mesh, or nullptr (not a candidate, or not built yet: then a build
	// is queued). a_modelBound receives the mesh's local bound, to check its world transform against.
	// Culling threads only (never concurrent with BeginFrame).
	[[nodiscard]] const Shape* Find(RE::NiAVObject* a_object, RE::NiBound& a_modelBound) noexcept;

	void LogStats(std::uint32_t a_frames);
}
