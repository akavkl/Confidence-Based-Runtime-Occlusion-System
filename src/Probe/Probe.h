#pragma once

// Phase 0 probe: logging-only hooks that answer the open RE questions in PLAN.md §4/§6
// (thread model, main depth target, main-camera culling pass, first-person ordering,
// hook coexistence, draw-call baseline, precombine bounds). It never changes rendering.

namespace CBRO::Probe
{
	void Install();           // F4SEPlugin_Load
	void OnPostPostLoad();    // all plugins loaded: report who else hooks our sites
	void OnGameDataReady();
	void OnGameLoaded();      // new game / save loaded: schedule trace, dump and survey
}
