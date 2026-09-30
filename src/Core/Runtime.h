#pragma once

// Drives CBRO each frame (main thread, inside DrawWorld::Render_PreUI):
//   cull stage begin  -> read back finished Hi-Z, check the camera, publish the frame context
//   pre-pass end      -> capture the camera and queue the next Hi-Z build
// and owns the CBRO <-> previs mode switch (one visibility authority at a time).

namespace CBRO::Core::Runtime
{
	void Install();       // F4SEPlugin_Load (before the render-stage hooks are installed)
	void OnGameLoaded();  // new game / save loaded
}
