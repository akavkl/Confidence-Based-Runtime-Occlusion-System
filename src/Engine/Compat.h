#pragma once

// Engine definitions CBRO uses that CommonLibF4RD doesn't carry, with the layouts CommonLibF4-DM had (verified on
// OG 1.10.163 by the Phase 0 probe runs), and the id helper for CBRO's own OG-verified ids.

#include <immintrin.h>

namespace CBRO::Engine
{
	// An Address Library id CBRO verified on OG 1.10.163 only: it resolves there and fails safely (unknown id) on
	// NG/AE. Once an NG/AE id is verified, pass it as the library's second/third argument instead
	// (REL::ID{ og, ng, ae }) and the runtime database resolves it on that family.
	[[nodiscard]] constexpr REL::ID OG(std::uint64_t a_og) noexcept
	{
		return REL::ID{ a_og, REL::ID::INVALID_ID };
	}

	namespace BSGraphics
	{
		// The render camera's matrices (BSGraphics::State::cameraState / cameraDataCache entries).
		class ViewData
		{
		public:
			RE::NiRect<float> viewPort;                     // 000
			RE::NiPoint2      viewDepthRange;               // 010
			__m128            viewUp;                       // 020
			__m128            viewRight;                    // 030
			__m128            viewDir;                      // 040
			__m128            viewMat[4];                   // 050
			__m128            projMat[4];                   // 090
			__m128            viewProjMat[4];               // 0D0
			__m128            viewProjUnjittered[4];        // 110
			__m128            currentViewProjUnjittered[4]; // 150
			__m128            previousViewProjUnjittered[4];  // 190
			__m128            inv1stPersonProjMat[4];       // 1D0
		};
		static_assert(sizeof(ViewData) == 0x210);

		class CameraStateData
		{
		public:
			ViewData            camViewData;        // 000
			RE::NiPoint3        posAdjust;          // 210
			RE::NiPoint3        currentPosAdjust;   // 21C
			RE::NiPoint3        previousPosAdjust;  // 228
			const RE::NiCamera* referenceCamera;    // 238
			bool                useJitter;          // 240
		};
		static_assert(offsetof(CameraStateData, posAdjust) == 0x210);
		static_assert(offsetof(CameraStateData, referenceCamera) == 0x238);
		static_assert(offsetof(CameraStateData, useJitter) == 0x240);
		static_assert(sizeof(CameraStateData) == 0x250);

		// BSGraphics::State (only the fields CBRO reads; the rest is padding).
		class State
		{
		public:
			[[nodiscard]] static State* GetSingleton()
			{
				static REL::Relocation<State*> singleton{ REL::ID(600795, 2704621) };
				return singleton.get();
			}

			std::uint8_t                  pad000[0x78];        // 000
			std::uint32_t                 backBufferWidth;     // 078
			std::uint32_t                 backBufferHeight;    // 07C
			std::uint32_t                 screenWidth;         // 080
			std::uint32_t                 screenHeight;        // 084
			RE::NiRect<float>             frameBufferViewport; // 088
			std::uint32_t                 frameCount;          // 098
			std::uint8_t                  pad09C[0x140 - 0x9C];  // 09C
			RE::BSTArray<CameraStateData> cameraDataCache;     // 140
			CameraStateData               cameraState;         // 160
		};
		static_assert(offsetof(State, backBufferWidth) == 0x78);
		static_assert(offsetof(State, frameBufferViewport) == 0x88);
		static_assert(offsetof(State, frameCount) == 0x98);
		static_assert(offsetof(State, cameraDataCache) == 0x140);
		static_assert(offsetof(State, cameraState) == 0x160);
	}

	// The fields CBRO reads on a light node (NiLight; reinterpret an NiAVObject known to be one).
	struct NiLightView
	{
		std::uint8_t pad000[0x138];  // NiAVObject, then amb/diff at 120/12C
		RE::NiColor  spec;           // 138
		float        dimmer;         // 144
		std::uint8_t pad148[0x150 - 0x148];
		RE::NiBound  modelBound;     // 150
	};
	static_assert(offsetof(NiLightView, spec) == 0x138);
	static_assert(offsetof(NiLightView, modelBound) == 0x150);

	// TES and the exterior cell grid (for the scene survey).
	struct GridCell
	{
		RE::TESObjectCELL* cell;  // 00
	};

	class GridCellArray
	{
	public:
		[[nodiscard]] GridCell* Get(std::uint32_t a_x, std::uint32_t a_y) const
		{
			using func_t = GridCell* (*)(const GridCellArray*, std::uint32_t, std::uint32_t);
			static REL::Relocation<func_t> func{ REL::ID(1330136, 2194566) };
			return func(this, a_x, a_y);
		}

		void*         vtable;     // 00
		std::int32_t  centerX;    // 08
		std::int32_t  centerY;    // 0C
		std::uint32_t dimension;  // 10
	};
	static_assert(offsetof(GridCellArray, dimension) == 0x10);

	class TES
	{
	public:
		[[nodiscard]] static TES* GetSingleton()
		{
			static REL::Relocation<TES**> singleton{ REL::ID(1194835, 2698044) };
			return *singleton;
		}

		std::uint8_t       pad000[0x18];         // 000
		GridCellArray*     gridCells;            // 018
		std::uint8_t       pad020[0x58 - 0x20];  // 020
		RE::TESObjectCELL* interiorCell;         // 058
	};
	static_assert(offsetof(TES, gridCells) == 0x18);
	static_assert(offsetof(TES, interiorCell) == 0x58);

	// An INI setting by "name:Section", preferences first (as the DM helper did).
	[[nodiscard]] inline RE::Setting* GetINISetting(std::string_view a_name)
	{
		if (const auto prefs = RE::INIPrefSettingCollection::GetSingleton()) {
			if (const auto setting = prefs->GetSetting(a_name)) {
				return setting;
			}
		}
		if (const auto ini = RE::INISettingCollection::GetSingleton()) {
			return ini->GetSetting(a_name);
		}
		return nullptr;
	}
}
