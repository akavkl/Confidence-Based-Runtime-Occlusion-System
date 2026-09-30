cmake_minimum_required(VERSION 3.21)

# ---- vcpkg (manifest mode: vcpkg.json next to this file) --------------------------------------------------
# VCPKG_ROOT from the environment, else the local checkout. The packages are installed into
# build/vcpkg_installed (VCPKG_INSTALLED_DIR in CMakePresets.json) on the first configure.
if(NOT DEFINED CMAKE_TOOLCHAIN_FILE)
	if(DEFINED ENV{VCPKG_ROOT})
		set(CMAKE_TOOLCHAIN_FILE "$ENV{VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake" CACHE STRING "")
	elseif(EXISTS "D:/vcpkg/scripts/buildsystems/vcpkg.cmake")
		set(CMAKE_TOOLCHAIN_FILE "D:/vcpkg/scripts/buildsystems/vcpkg.cmake" CACHE STRING "")
	endif()
endif()
set(VCPKG_TARGET_TRIPLET "x64-windows-static-md" CACHE STRING "")
set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL" CACHE STRING "")

project(CBRO VERSION 1.51.0 LANGUAGES CXX)

if(PROJECT_SOURCE_DIR STREQUAL PROJECT_BINARY_DIR)
	message(FATAL_ERROR "in-source builds are not allowed")
endif()

# MO2 mod folder the plugin (DLL, PDB, CBRO.ini, meta.ini) is copied into after every build; empty disables it.
set(CBRO_INSTALL_DIR "D:/MO2/Fallout 4/mods/Confidence-Based Runtime Occlusion" CACHE PATH "MO2 mod folder to install CBRO into")

# ---- dependencies ----------------------------------------------------------------------------------------
set(Boost_USE_STATIC_LIBS ON)
add_subdirectory(external/CommonLibF4RD/CommonLibF4 CommonLibF4)
find_package(fmt CONFIG REQUIRED)
find_package(spdlog CONFIG REQUIRED)

# ---- version ---------------------------------------------------------------------------------------------
configure_file(cmake/Version.h.in "${CMAKE_CURRENT_BINARY_DIR}/include/Version.h" @ONLY)
configure_file(cmake/version.rc.in "${CMAKE_CURRENT_BINARY_DIR}/version.rc" @ONLY)

# ---- the plugin ------------------------------------------------------------------------------------------
file(GLOB_RECURSE CBRO_SOURCES CONFIGURE_DEPENDS src/*.cpp src/*.h)
source_group(TREE "${CMAKE_CURRENT_SOURCE_DIR}" FILES ${CBRO_SOURCES})

add_library(CBRO SHARED ${CBRO_SOURCES} "${CMAKE_CURRENT_BINARY_DIR}/version.rc")

target_compile_features(CBRO PRIVATE cxx_std_23)
target_compile_definitions(CBRO PRIVATE _UNICODE UNICODE)
target_include_directories(CBRO PRIVATE src "${CMAKE_CURRENT_BINARY_DIR}/include")
target_link_libraries(CBRO PRIVATE CommonLibF4::CommonLibF4 fmt::fmt spdlog::spdlog d3d11 dxgi d3dcompiler user32)
target_precompile_headers(CBRO PRIVATE src/PCH.h)

if(MSVC)
	target_compile_options(
		CBRO
		PRIVATE
			/utf-8
			/Zi
			/permissive-
			/Zc:preprocessor
			/EHsc
			/W4
			/arch:AVX2
			# /GL + /LTCG (link options below): cross-file inlining, e.g. HiZ::Snapshot::AllNearer into Occlusion's tests.
			# /Gw: globals in their own COMDATs so /OPT:REF can drop unused ones. Kept out on purpose:
			#   /fp:fast  - the tests rely on NaN/infinity semantics (std::isfinite guards on engine bounds, infinity for
			#               sky texels in LinearDepth, min/max against +inf); fast math may fold those checks away and
			#               hide objects on bad data.
			#   /Qpar     - auto-parallelizes loops with the OpenMP runtime (vcomp140.dll): extra threads inside the
			#               engine's frame, a new DLL dependency, and no hot loop here is parallelizable anyway.
			"$<$<CONFIG:Release>:/O2;/GL;/Zc:inline;/JMC-;/Gw>"
	)
	target_link_options(
		CBRO
		PRIVATE
			# "$<$<CONFIG:Release>:/INCREMENTAL:NO;/OPT:REF;/OPT:ICF;/DEBUG:FULL>"
			"$<$<CONFIG:Release>:/INCREMENTAL:NO;/LTCG;/OPT:REF;/OPT:ICF;/DEBUG:FULL>"
			"$<$<CONFIG:Debug>:/DEBUG:FULL>"
	)
endif()

# ---- install into the MO2 mod folder ----------------------------------------------------------------------
if(CBRO_INSTALL_DIR)
	add_custom_command(
		TARGET CBRO
		POST_BUILD
		COMMAND ${CMAKE_COMMAND} -E make_directory "${CBRO_INSTALL_DIR}/F4SE/Plugins"
		COMMAND ${CMAKE_COMMAND} -E copy_if_different "$<TARGET_FILE:CBRO>" "${CBRO_INSTALL_DIR}/F4SE/Plugins/"
		COMMAND ${CMAKE_COMMAND} -E copy_if_different "$<TARGET_PDB_FILE:CBRO>" "${CBRO_INSTALL_DIR}/F4SE/Plugins/"
		# (res/CBRO.ini is NOT copied: the user's INI in the mod folder is theirs; new keys fall back to Settings.h's defaults)
		COMMAND ${CMAKE_COMMAND} -E copy_if_different "${CMAKE_CURRENT_SOURCE_DIR}/res/meta.ini" "${CBRO_INSTALL_DIR}/"
		COMMENT "Installing CBRO into ${CBRO_INSTALL_DIR}"
	)
endif()
