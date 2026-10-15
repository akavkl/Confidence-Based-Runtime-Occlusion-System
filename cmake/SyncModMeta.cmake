# Post-build: sets the version line of the MO2 mod folder's meta.ini to this build's version. An existing meta.ini is
# MO2's (category, notes, Nexus fields...): only its version line changes. A missing one is made from res/meta.ini.
# Usage: cmake -DMETA=<mod folder>/meta.ini -DTEMPLATE=<res/meta.ini> -DVERSION=<x.y.z.w> -P SyncModMeta.cmake
if(EXISTS "${META}")
	file(READ "${META}" text)
else()
	file(READ "${TEMPLATE}" text)
endif()
string(REGEX REPLACE "(^|\n)version=[^\r\n]*" "\\1version=${VERSION}" updated "${text}")
if(NOT EXISTS "${META}" OR NOT updated STREQUAL text)
	file(WRITE "${META}" "${updated}")
	message(STATUS "MO2 meta.ini: version=${VERSION}")
endif()
