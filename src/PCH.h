#pragma once

#undef DEBUG

#define WIN32_LEAN_AND_MEAN
#define NOMINMAX

// CommonLibF4RD: the familiar CommonLibF4 API with runtime-aware ids (one build runs on OG, NG and AE; the
// engine ids resolve through Data/F4SE/Plugins/f4rd-runtime.bin at run time).
#pragma warning(push)
#include "F4SE/F4SE.h"
#include "RE/Fallout.h"
#include <spdlog/sinks/basic_file_sink.h>
#pragma warning(pop)

// Windows/D3D headers come after CommonLib so their macros can't break its declarations.
#include <Windows.h>
#include <d3d11.h>
#include <dxgi.h>

// wingdi.h's ERROR macro collides with enumerators.
#undef ERROR

#include <array>
#include <atomic>
#include <cstdint>
#include <filesystem>
#include <format>
#include <mutex>
#include <span>
#include <string>
#include <string_view>
#include <vector>

using namespace std::literals;

// The log formatter (fmt) only prints void pointers.
template <class T>
[[nodiscard]] inline const void* fmt_ptr(T* a_ptr) noexcept
{
	return a_ptr;
}

[[nodiscard]] inline const void* fmt_ptr(std::uintptr_t a_address) noexcept
{
	return reinterpret_cast<const void*>(a_address);
}

// CBRO.log (Documents\My Games\Fallout4\F4SE), set up in Plugin.cpp: "[time] [thread] [level] message".
namespace logger
{
	template <class... Args>
	void trace(fmt::format_string<Args...> a_fmt, Args&&... a_args)
	{
		spdlog::trace(a_fmt, std::forward<Args>(a_args)...);
	}

	template <class... Args>
	void debug(fmt::format_string<Args...> a_fmt, Args&&... a_args)
	{
		spdlog::debug(a_fmt, std::forward<Args>(a_args)...);
	}

	template <class... Args>
	void info(fmt::format_string<Args...> a_fmt, Args&&... a_args)
	{
		spdlog::info(a_fmt, std::forward<Args>(a_args)...);
	}

	template <class... Args>
	void warn(fmt::format_string<Args...> a_fmt, Args&&... a_args)
	{
		spdlog::warn(a_fmt, std::forward<Args>(a_args)...);
	}

	template <class... Args>
	void error(fmt::format_string<Args...> a_fmt, Args&&... a_args)
	{
		spdlog::error(a_fmt, std::forward<Args>(a_args)...);
	}
}

#include "Engine/Compat.h"
#include "Version.h"
