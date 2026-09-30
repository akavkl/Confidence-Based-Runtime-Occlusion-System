#include "Settings.h"

namespace CBRO
{
	namespace
	{
		std::filesystem::path GetIniPath()
		{
			HMODULE module = nullptr;
			GetModuleHandleExW(
				GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
				reinterpret_cast<LPCWSTR>(&GetIniPath),
				&module);

			std::wstring buf(MAX_PATH, L'\0');
			const auto   len = GetModuleFileNameW(module, buf.data(), static_cast<DWORD>(buf.size()));
			buf.resize(len);

			std::filesystem::path path{ buf };
			path.replace_extension(L".ini");
			return path;
		}

		std::uint32_t ReadUInt(const std::filesystem::path& a_path, const wchar_t* a_section, const wchar_t* a_key, std::uint32_t a_default)
		{
			return GetPrivateProfileIntW(a_section, a_key, static_cast<INT>(a_default), a_path.c_str());
		}

		bool ReadBool(const std::filesystem::path& a_path, const wchar_t* a_section, const wchar_t* a_key, bool a_default)
		{
			return ReadUInt(a_path, a_section, a_key, a_default ? 1 : 0) != 0;
		}

		float ReadFloat(const std::filesystem::path& a_path, const wchar_t* a_section, const wchar_t* a_key, float a_default)
		{
			wchar_t buf[64]{};
			GetPrivateProfileStringW(a_section, a_key, L"", buf, static_cast<DWORD>(std::size(buf)), a_path.c_str());
			wchar_t*    end = nullptr;
			const float value = std::wcstof(buf, &end);
			return end != buf && std::isfinite(value) ? value : a_default;
		}
	}

	void Settings::Load()
	{
		const auto path = GetIniPath();
		const bool exists = std::filesystem::exists(path);

		enabled = ReadBool(path, L"General", L"bEnabled", enabled);

		occlusion = ReadBool(path, L"Occlusion", L"bEnabled", occlusion);
		disablePrevis = ReadBool(path, L"Occlusion", L"bDisablePrevis", disablePrevis);
		startActive = ReadBool(path, L"Occlusion", L"bStartActive", startActive);
		observeOnly = ReadBool(path, L"Occlusion", L"bObserveOnly", observeOnly);
		toggleHotkey = ReadUInt(path, L"Occlusion", L"iToggleHotkey", toggleHotkey);
		statusHotkey = ReadUInt(path, L"Occlusion", L"iStatusHotkey", statusHotkey);
		diagnosticHotkey = ReadUInt(path, L"Occlusion", L"iDiagnosticHotkey", diagnosticHotkey);
		notify = ReadBool(path, L"Occlusion", L"bNotify", notify);
		hiZDownsample = std::clamp(ReadUInt(path, L"Occlusion", L"iHiZDownsample", hiZDownsample), 1u, 16u);
		confirmFrames = std::clamp(ReadUInt(path, L"Occlusion", L"iConfirmFrames", confirmFrames), 1u, 120u);
		maxSnapshotAge = std::clamp(ReadUInt(path, L"Occlusion", L"iMaxSnapshotAge", maxSnapshotAge), 1u, 30u);
		maxCameraMove = ReadFloat(path, L"Occlusion", L"fMaxCameraMove", maxCameraMove);
		maxCameraAngle = std::clamp(ReadFloat(path, L"Occlusion", L"fMaxCameraAngle", maxCameraAngle), 0.0f, 80.0f);
		nearDistance = std::max(1.0f, ReadFloat(path, L"Occlusion", L"fNearDistance", nearDistance));
		cullActors = ReadBool(path, L"Occlusion", L"bCullActors", cullActors);
		meshShapes = ReadBool(path, L"Occlusion", L"bMeshShapes", meshShapes);
		sunShadowCulling = ReadBool(path, L"Occlusion", L"bSunShadowCulling", sunShadowCulling);
		lampShadowCulling = ReadBool(path, L"Occlusion", L"bLampShadowCulling", lampShadowCulling);
		lampShadowVolumes = ReadBool(path, L"Occlusion", L"bLampShadowVolumes", lampShadowVolumes);
		cellNodePruning = ReadBool(path, L"Occlusion", L"bCellNodePruning", cellNodePruning);
		previsFeed = ReadBool(path, L"Occlusion", L"bPrevisFeed", previsFeed);
		interiors = ReadBool(path, L"Occlusion", L"bInteriors", interiors);
		setDiff = ReadBool(path, L"Occlusion", L"bSetDiff", setDiff);
		async = ReadBool(path, L"Occlusion", L"bAsync", async);
		lampGroupTrim = ReadBool(path, L"Occlusion", L"bLampGroupTrim", lampGroupTrim);
		feedAuditInterval = ReadUInt(path, L"Occlusion", L"iFeedAuditInterval", feedAuditInterval);
		verdictCache = ReadBool(path, L"Occlusion", L"bVerdictCache", verdictCache);
		cacheMove = std::clamp(ReadFloat(path, L"Occlusion", L"fCacheMove", cacheMove), 0.5f, 64.0f);
		cacheAngle = std::clamp(ReadFloat(path, L"Occlusion", L"fCacheAngle", cacheAngle), 0.01f, 2.0f);
		hiZTemporalFrames = std::clamp(ReadUInt(path, L"Occlusion", L"iHiZTemporalFrames", hiZTemporalFrames), 1u, 8u);
		depthTolerance = ReadFloat(path, L"Occlusion", L"fDepthTolerance", depthTolerance);
		depthSlack = ReadFloat(path, L"Occlusion", L"fDepthSlack", depthSlack);
		worldDepthMin = ReadFloat(path, L"Occlusion", L"fWorldDepthMin", worldDepthMin);
		worldDepthMax = ReadFloat(path, L"Occlusion", L"fWorldDepthMax", worldDepthMax);

		probe = ReadBool(path, L"Probe", L"bEnabled", probe);
		renderStageHooks = ReadBool(path, L"Probe", L"bRenderStageHooks", renderStageHooks);
		cullingHooks = ReadBool(path, L"Probe", L"bCullingHooks", cullingHooks);
		cullingGroupHooks = ReadBool(path, L"Probe", L"bCullingGroupHooks", cullingGroupHooks);
		drawCallCounter = ReadBool(path, L"Probe", L"bDrawCallCounter", drawCallCounter);
		sceneSurvey = ReadBool(path, L"Probe", L"bSceneSurvey", sceneSurvey);
		summaryIntervalFrames = std::max(1u, ReadUInt(path, L"Probe", L"iSummaryIntervalFrames", summaryIntervalFrames));
		traceFrames = ReadUInt(path, L"Probe", L"iTraceFrames", traceFrames);
		traceDelayFramesAfterLoad = ReadUInt(path, L"Probe", L"iTraceDelayFramesAfterLoad", traceDelayFramesAfterLoad);
		traceHotkey = ReadUInt(path, L"Probe", L"iTraceHotkey", traceHotkey);
		surveyHotkey = ReadUInt(path, L"Probe", L"iSurveyHotkey", surveyHotkey);

		logger::info("settings: {} ({})", path.string(), exists ? "found" : "missing, using defaults");
		logger::info(
			"settings: occlusion={} disablePrevis={} previsFeed={} interiors={} setDiff={} async={} lampGroupTrim={} feedAuditInterval={} startActive={} observeOnly={} toggleHotkey=0x{:X} statusHotkey=0x{:X} notify={} hiZDownsample={} hiZTemporalFrames={} confirmFrames={} maxSnapshotAge={} maxCameraMove={} maxCameraAngle={} nearDistance={} cullActors={} meshShapes={} sunShadowCulling={} lampShadowCulling={} cellNodePruning={} depthTolerance={} depthSlack={} worldDepth=[{},{}]",
			occlusion, disablePrevis, previsFeed, interiors, setDiff, async, lampGroupTrim, feedAuditInterval, startActive, observeOnly, toggleHotkey, statusHotkey, notify, hiZDownsample, hiZTemporalFrames, confirmFrames, maxSnapshotAge,
			maxCameraMove, maxCameraAngle, nearDistance, cullActors, meshShapes, sunShadowCulling, lampShadowCulling, cellNodePruning, depthTolerance, depthSlack,
			worldDepthMin, worldDepthMax);
		logger::info(
			"settings: enabled={} probe={} renderStageHooks={} cullingHooks={} cullingGroupHooks={} drawCallCounter={} sceneSurvey={} summaryInterval={} traceFrames={} traceDelay={} traceHotkey=0x{:X} surveyHotkey=0x{:X}",
			enabled, probe, renderStageHooks, cullingHooks, cullingGroupHooks, drawCallCounter, sceneSurvey,
			summaryIntervalFrames, traceFrames, traceDelayFramesAfterLoad, traceHotkey, surveyHotkey);
	}
}
