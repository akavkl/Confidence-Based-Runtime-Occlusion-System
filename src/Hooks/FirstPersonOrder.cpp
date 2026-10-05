#include "Hooks/FirstPersonOrder.h"

#include "Engine/Compat.h"
#include "Hooks/RenderStages.h"
#include "Util/Hooking.h"

namespace CBRO::Hooks::FirstPersonOrder
{
	namespace
	{
		// 56596 (0x142850CB0), the deferred pre-pass, writes first person's depth twice before the world's:
		//   +0x1E7 call 1491502, the depth-only z-prepass (null pixel shader) of four lists: first person's two (counts
		//          [163482], [382658]) under the first-person camera, then the world's (counts [844802], [1064092],
		//          [1283533], [602241]) under the world camera; shadow state +0xD0 is set to 0 around it, 1 after
		//   +0x304 call 163409(fp accumulator)   +0x31B call 1048494(fp accumulator, 4, false, -1)   +0x327 call 761249(fp accumulator)
		// then the world camera (185153 with the world's arguments) and the world block, which ends at
		//   +0x3C1 call 761249(world accumulator)
		// followed by an empty marker call, state resets and the world camera once more. The marker calls around the
		// blocks (777723 / 338650) are bare `ret`s in this build. Either first-person part drawn before the world block
		// leaves no world depth behind the weapon (v1.69's first run held only the block back: the z-prepass still drew
		// the weapon's depth first, ~as many draws as the block).
		constexpr std::uint64_t  kPrePassID = 56596;
		constexpr std::ptrdiff_t kZPrepass = 0x1E7;
		constexpr std::ptrdiff_t kFirstStart = 0x304;
		constexpr std::ptrdiff_t kFirstGroup = 0x31B;
		constexpr std::ptrdiff_t kFirstFinish = 0x327;
		constexpr std::ptrdiff_t kWorldFinish = 0x3C1;
		constexpr std::uint64_t  kZPrepassID = 1491502;  // (0x1461E0900, fp camera, world camera, float x4)
		constexpr std::uint64_t  kStartID = 163409;      // accumulator: render groups 2 and 3 (checks +0xC4)
		constexpr std::uint64_t  kGroupID = 1048494;     // accumulator: render pass group (4)
		constexpr std::uint64_t  kFinishID = 761249;     // accumulator: finish (checks +0xC4)

		// The z-prepass's list counts: first person's two, then the world's four (reset only at the frame's end).
		constexpr std::array<std::uint64_t, 2> kFirstListIDs{ 163482, 382658 };
		constexpr std::array<std::uint64_t, 4> kWorldListIDs{ 844802, 1064092, 1283533, 602241 };

		constexpr std::uint64_t kFirstAccumulatorID = 1430301;  // [0x146723358]
		constexpr std::uint64_t kWorldAccumulatorID = 1211381;  // [0x146723350]
		constexpr std::uint64_t kSetCameraID = 185153;          // (camera state, NiCamera*, bool, float, float): camera data + viewport
		constexpr std::uint64_t kCameraStateID = 600795;        // 0x146541EF0, the state 185153 writes
		constexpr std::uint64_t kFirstCameraID = 380177;        // [0x1467232F8] the first-person NiCamera
		constexpr std::uint64_t kWorldCameraID = 1444212;       // [0x146723238] the world NiCamera
		constexpr std::uint64_t kFirstArgID = 1098847;          // [0x1467231B8] 185153's 4th argument for first person
		constexpr std::uint64_t kWorldArgID = 417116;           // [0x1467231BC] ... for the world
		constexpr std::uint64_t kFirstArg5ID = 483683;          // 0x1438C9378 (1.0) 185153's 5th argument for first person
		constexpr std::uint64_t kWorldArg5ID = 1384353;         // 0x1438C937C (1.0) ... for the world
		constexpr std::uint64_t kInverseID = 1581726;           // DirectX::XMMatrixInverse
		constexpr std::uint64_t kTlsIndexID = 842564;           // [0x1467347B4] the renderer's TLS slot
		constexpr std::uint64_t kDefaultContextID = 33539;      // [0x1461DDC68] the context when the thread has none

		struct alignas(16) Matrix
		{
			__m128 r[4];
		};

		using AccumulatorFn = void (*)(std::uintptr_t);
		using GroupFn = void (*)(std::uintptr_t, std::uint32_t, bool, std::int32_t);
		using SetCameraFn = void (*)(std::uintptr_t, RE::NiCamera*, bool, float, float);
		using ZPrepassFn = void (*)(std::uintptr_t, RE::NiCamera*, RE::NiCamera*, float, float, float, float);
		using InverseFn = Matrix* (*)(Matrix*, __m128*, const Matrix*);

		Listener*      g_listener{ nullptr };
		bool           g_installed{ false };
		std::uintptr_t g_zPrepass{ 0 };  // the call targets the sites had (verified to be the engine's functions)
		std::uintptr_t g_start{ 0 };
		std::uintptr_t g_group{ 0 };
		std::uintptr_t g_finish{ 0 };
		std::uintptr_t g_worldFinish{ 0 };

		// Main thread only (the pre-pass runs there, FO4-ENGINE-NOTES 1).
		enum class Decision
		{
			kNone,     // this pre-pass hasn't reached a first-person part yet
			kHold,     // its first-person parts wait for the world block's end
			kInPlace,  // ... run where the engine put them
		};
		Decision       g_decision{ Decision::kNone };
		bool           g_held{ false };          // the first-person block is held
		std::uintptr_t g_heldAccumulator{ 0 };
		bool           g_zHeld{ false };         // the z-prepass's first-person lists are held
		std::uint32_t  g_zFirstCounts[2]{};      // ... their counts
		struct ZPrepassArgs
		{
			std::uintptr_t state{ 0 };
			RE::NiCamera*  first{ nullptr };
			RE::NiCamera*  world{ nullptr };
			float          a4{ 0.0f }, a5{ 0.0f }, a6{ 0.0f }, a7{ 0.0f };
		};
		ZPrepassArgs  g_zArgs;
		std::uint32_t g_phase{ 0 };  // Phase()

		struct Counters
		{
			std::uint64_t prePasses{ 0 };   // main pre-passes that reached a first-person part
			std::uint64_t held{ 0 };        // ... whose first-person parts were held back past the world block
			std::uint64_t zHeld{ 0 };       // ... whose z-prepass had first-person lists to hold
			std::uint64_t atWorldEnd{ 0 };  // held parts run at the world block's end
			std::uint64_t atStageEnd{ 0 };  // ... at the stage's end (the world block's end never came)
			std::uint64_t otherCalls{ 0 };  // first-person parts outside the main pre-pass (run in place)
		};
		Counters g_counters;

		std::uintptr_t Global(std::uint64_t a_id)
		{
			return *reinterpret_cast<std::uintptr_t*>(CBRO::Engine::OG(a_id).address());
		}

		float GlobalFloat(std::uint64_t a_id)
		{
			return *reinterpret_cast<const float*>(CBRO::Engine::OG(a_id).address());
		}

		std::uint32_t& Count(std::uint64_t a_id)
		{
			return *reinterpret_cast<std::uint32_t*>(CBRO::Engine::OG(a_id).address());
		}

		// The renderer's shadow state the pre-pass writes its matrices into: this thread's context (TLS slot +0xB20),
		// else the default one, +0x1B70 (as 56596 finds it).
		std::uintptr_t ShadowState()
		{
			const auto index = *reinterpret_cast<const std::uint32_t*>(CBRO::Engine::OG(kTlsIndexID).address());
			const auto slots = reinterpret_cast<const std::uintptr_t*>(__readgsqword(0x58));
			auto       context = *reinterpret_cast<const std::uintptr_t*>(slots[index] + 0xB20);
			if (!context) {
				context = Global(kDefaultContextID);
			}
			return context + 0x1B70;
		}

		// A shadow-state field as 56596 sets it: the value, and its dirty bit in the flags at +0 when it changes.
		void SetStateField(std::uintptr_t a_state, std::ptrdiff_t a_offset, std::uint32_t a_value, std::uint32_t a_dirty)
		{
			auto& field = *reinterpret_cast<std::uint32_t*>(a_state + a_offset);
			if (field != a_value) {
				*reinterpret_cast<std::uint32_t*>(a_state) |= a_dirty;
				field = a_value;
			}
		}

		// Whether this main pre-pass holds its first-person parts back (decided once, at the first of them).
		bool Hold()
		{
			if (g_decision == Decision::kNone) {
				++g_counters.prePasses;
				g_decision = g_listener->WantWorldDepth() ? Decision::kHold : Decision::kInPlace;
				if (g_decision == Decision::kHold) {
					++g_counters.held;
				}
			}
			return g_decision == Decision::kHold;
		}

		// The z-prepass's first-person lists as 56596 draws them (the world's lists left out): +0xD0 = 0 around it, the
		// same arguments. The counts are put back as they were, so whatever reads them later sees no change.
		void RunFirstPersonZPrepass()
		{
			const auto state = ShadowState();
			const auto d0 = *reinterpret_cast<const std::uint32_t*>(state + 0xD0);
			std::array<std::uint32_t, 2> first{};
			std::array<std::uint32_t, 4> world{};
			for (std::size_t i = 0; i < first.size(); ++i) {
				first[i] = std::exchange(Count(kFirstListIDs[i]), g_zFirstCounts[i]);
			}
			for (std::size_t i = 0; i < world.size(); ++i) {
				world[i] = std::exchange(Count(kWorldListIDs[i]), 0u);
			}
			SetStateField(state, 0xD0, 0, 0x10);
			reinterpret_cast<ZPrepassFn>(g_zPrepass)(g_zArgs.state, g_zArgs.first, g_zArgs.world, g_zArgs.a4, g_zArgs.a5, g_zArgs.a6, g_zArgs.a7);
			SetStateField(state, 0xD0, d0, 0x10);
			for (std::size_t i = 0; i < first.size(); ++i) {
				Count(kFirstListIDs[i]) = first[i];
			}
			for (std::size_t i = 0; i < world.size(); ++i) {
				Count(kWorldListIDs[i]) = world[i];
			}
		}

		// The first-person block as 56596 runs it: its camera, the transposed inverse of the camera's matrix (+0x760)
		// at +0x8A0, the three accumulator calls; then the world camera again, which is what the pass set after it.
		void RunFirstPerson(std::uintptr_t a_accumulator)
		{
			const auto setCamera = reinterpret_cast<SetCameraFn>(CBRO::Engine::OG(kSetCameraID).address());
			const auto cameraState = CBRO::Engine::OG(kCameraStateID).address();
			setCamera(cameraState, reinterpret_cast<RE::NiCamera*>(Global(kFirstCameraID)), true, GlobalFloat(kFirstArgID), GlobalFloat(kFirstArg5ID));

			const auto state = ShadowState();
			Matrix     inverse{};
			__m128     determinant{};
			const auto result = reinterpret_cast<InverseFn>(CBRO::Engine::OG(kInverseID).address())(&inverse, &determinant, reinterpret_cast<const Matrix*>(state + 0x760));
			Matrix     transposed = *result;
			_MM_TRANSPOSE4_PS(transposed.r[0], transposed.r[1], transposed.r[2], transposed.r[3]);
			std::memcpy(reinterpret_cast<void*>(state + 0x8A0), &transposed, sizeof(transposed));

			reinterpret_cast<AccumulatorFn>(g_start)(a_accumulator);
			reinterpret_cast<GroupFn>(g_group)(a_accumulator, 4, false, -1);
			reinterpret_cast<AccumulatorFn>(g_finish)(a_accumulator);

			setCamera(cameraState, reinterpret_cast<RE::NiCamera*>(Global(kWorldCameraID)), true, GlobalFloat(kWorldArgID), GlobalFloat(kWorldArg5ID));
		}

		// The world's depth is complete: capture it, then first person in the engine's order (z-prepass, block).
		void RunHeld()
		{
			const bool block = std::exchange(g_held, false);
			const auto accumulator = std::exchange(g_heldAccumulator, 0);
			const bool zPrepass = std::exchange(g_zHeld, false);
			g_listener->OnWorldDepth();
			g_phase = 2;
			if (zPrepass) {
				RunFirstPersonZPrepass();
			}
			if (block) {
				RunFirstPerson(accumulator);
			}
			g_phase = 3;
		}

		void ZPrepassThunk(std::uintptr_t a_state, RE::NiCamera* a_first, RE::NiCamera* a_world, float a_4, float a_5, float a_6, float a_7)
		{
			const auto call = [&] { reinterpret_cast<ZPrepassFn>(g_zPrepass)(a_state, a_first, a_world, a_4, a_5, a_6, a_7); };
			if (RenderStages::Current() != RenderStages::Stage::kPrePass) {
				++g_counters.otherCalls;
				call();
				return;
			}
			if (g_zHeld || !Hold()) {
				call();
				return;
			}
			// The world's lists now, first person's later: their counts are 0 for this call (the engine reads nothing
			// else of them; a call with every count 0 returns at once).
			std::uint32_t first[2]{};
			for (std::size_t i = 0; i < kFirstListIDs.size(); ++i) {
				first[i] = std::exchange(Count(kFirstListIDs[i]), 0u);
			}
			call();
			for (std::size_t i = 0; i < kFirstListIDs.size(); ++i) {
				Count(kFirstListIDs[i]) = first[i];
				g_zFirstCounts[i] = first[i];
			}
			if (first[0] || first[1]) {
				++g_counters.zHeld;
				g_zHeld = true;
				g_zArgs = { a_state, a_first, a_world, a_4, a_5, a_6, a_7 };
				g_phase = 1;
			}
		}

		void StartThunk(std::uintptr_t a_accumulator)
		{
			if (a_accumulator == Global(kFirstAccumulatorID) && !g_held) {
				if (RenderStages::Current() != RenderStages::Stage::kPrePass) {
					++g_counters.otherCalls;
				} else if (Hold()) {
					g_held = true;
					g_phase = 1;
					g_heldAccumulator = a_accumulator;
					return;
				}
			}
			reinterpret_cast<AccumulatorFn>(g_start)(a_accumulator);
		}

		void GroupThunk(std::uintptr_t a_accumulator, std::uint32_t a_group, bool a_flag, std::int32_t a_index)
		{
			if (g_held && a_accumulator == g_heldAccumulator) {
				return;
			}
			reinterpret_cast<GroupFn>(g_group)(a_accumulator, a_group, a_flag, a_index);
		}

		void FinishThunk(std::uintptr_t a_accumulator)
		{
			if (g_held && a_accumulator == g_heldAccumulator) {
				return;
			}
			reinterpret_cast<AccumulatorFn>(g_finish)(a_accumulator);
		}

		void WorldFinishThunk(std::uintptr_t a_accumulator)
		{
			reinterpret_cast<AccumulatorFn>(g_worldFinish)(a_accumulator);
			if ((g_held || g_zHeld) && a_accumulator == Global(kWorldAccumulatorID)) {
				++g_counters.atWorldEnd;
				RunHeld();
			}
		}
	}

	bool Install(Listener* a_listener)
	{
		if (g_installed || !a_listener) {
			return g_installed;
		}
		const auto base = CBRO::Engine::OG(kPrePassID).address();
		struct Site
		{
			std::ptrdiff_t  offset;
			std::uint64_t   expected;
			std::uintptr_t  thunk;
			std::uintptr_t* original;
			const char*     name;
		};
		const std::array<Site, 5> sites{ {
			{ kZPrepass, kZPrepassID, Util::FnAddr(&ZPrepassThunk), &g_zPrepass, "firstperson:z-prepass" },
			{ kFirstStart, kStartID, Util::FnAddr(&StartThunk), &g_start, "firstperson:start" },
			{ kFirstGroup, kGroupID, Util::FnAddr(&GroupThunk), &g_group, "firstperson:group" },
			{ kFirstFinish, kFinishID, Util::FnAddr(&FinishThunk), &g_finish, "firstperson:finish" },
			{ kWorldFinish, kFinishID, Util::FnAddr(&WorldFinishThunk), &g_worldFinish, "firstperson:world finish" },
		} };
		// All five or none: a site another plugin already took (or a different build) leaves the pass untouched.
		for (const auto& site : sites) {
			const auto target = Util::ReadCall5Target(base + site.offset);
			if (target != CBRO::Engine::OG(site.expected).address()) {
				logger::warn(
					"first person after the world: not installed (pre-pass +0x{:X} calls {}, expected the engine's {})",
					site.offset, target ? Util::DescribeCodeAddress(target) : "no call"s, site.expected);
				return false;
			}
		}
		for (const auto& site : sites) {
			*site.original = Util::WriteCall5(base + site.offset, site.thunk, site.name);
		}
		g_listener = a_listener;
		g_installed = g_zPrepass && g_start && g_group && g_finish && g_worldFinish;
		logger::info("first person after the world: {}", g_installed ? "installed (the pre-pass draws the world first in frames CBRO captures depth)" : "NOT installed (a site write failed)");
		return g_installed;
	}

	bool Installed() noexcept
	{
		return g_installed;
	}

	void Flush()
	{
		if (g_held || g_zHeld) {
			++g_counters.atStageEnd;
			RunHeld();
		}
		g_decision = Decision::kNone;
		g_phase = 0;
	}

	std::uint32_t Phase() noexcept
	{
		return g_phase;
	}

	void LogStats(std::uint32_t a_frames)
	{
		if (!g_installed) {
			return;
		}
		const double frames = std::max(1u, a_frames);
		logger::info(
			"first person after the world per frame: main pre-passes {:.2f} | first person held back past the world {:.2f} (z-prepass lists among them {:.2f}; run at the world block's end {:.2f}, at the stage's end {:.2f}) | outside the main pre-pass, run in place {:.2f}",
			g_counters.prePasses / frames, g_counters.held / frames, g_counters.zHeld / frames, g_counters.atWorldEnd / frames, g_counters.atStageEnd / frames,
			g_counters.otherCalls / frames);
		g_counters = {};
	}
}
