#pragma once

// Shared state for the Phase 0 probes. Nothing here changes what the game renders.

#include "Hooks/RenderStages.h"
#include "Util/Gamebryo.h"

namespace CBRO::Probe
{
	using Stage = Hooks::RenderStages::Stage;

	using Hooks::RenderStages::StageName;

	[[nodiscard]] inline Stage CurrentStage() noexcept
	{
		return Hooks::RenderStages::Current();
	}

	enum class ThreadKind : std::uint32_t
	{
		kMain,
		kRender,
		kOther,
		kCount
	};

	[[nodiscard]] std::string_view ThreadKindName(ThreadKind a_kind) noexcept;

	struct SharedState
	{
		std::atomic<std::uint32_t>  mainThreadId{ 0 };
		std::atomic<std::uint32_t>  renderThreadId{ 0 };  // thread that runs the deferred pre-pass
		std::atomic<std::uintptr_t> worldCamera{ 0 };     // Main::WorldRootCamera(), refreshed each frame
		std::atomic<std::uint64_t>  frame{ 0 };           // incremented at each deferred pre-pass
		std::atomic<bool>           tracing{ false };
	};

	[[nodiscard]] SharedState& State() noexcept;

	[[nodiscard]] inline ThreadKind ClassifyThread(std::uint32_t a_tid) noexcept
	{
		const auto& state = State();
		if (a_tid == state.mainThreadId.load(std::memory_order_relaxed)) {
			return ThreadKind::kMain;
		}
		if (a_tid == state.renderThreadId.load(std::memory_order_relaxed)) {
			return ThreadKind::kRender;
		}
		return ThreadKind::kOther;
	}

	// Records an ordered event while a trace window is open; no-op otherwise.
	void TraceEvent(std::string a_text);

	[[nodiscard]] inline bool Tracing() noexcept
	{
		return State().tracing.load(std::memory_order_relaxed);
	}

	using Util::TryGetObjectName;
	using Util::TryGetRTTIName;
	using Util::TryReadVtable;

	namespace Cull
	{
		void Install();
		void VerifyChain();
		void EndFrame(bool a_log);
	}

	namespace Groups
	{
		void Install();
		void EndFrame(bool a_log);
	}

	namespace Draw
	{
		void Install();  // lazily, from the render thread, once the device exists
		void EndFrame(bool a_log);
		[[nodiscard]] std::uint64_t ImmediateDrawsThisFrame() noexcept;
	}

	namespace Scene
	{
		void LogGameSettings();
		void RequestSurvey();  // runs on the main thread through the F4SE task queue
	}
}
