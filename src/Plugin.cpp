#include "Core/Runtime.h"
#include "Hooks/CullGroups.h"
#include "Hooks/RenderStages.h"
#include "Probe/Probe.h"
#include "Settings.h"

namespace
{
	void OnMessage(F4SE::MessagingInterface::Message* a_msg)
	{
		const bool probe = CBRO::Settings::Get().probe;
		switch (a_msg->type) {
		case F4SE::MessagingInterface::kPostPostLoad:
			if (probe) {
				CBRO::Probe::OnPostPostLoad();
			} else {
				logger::info("---- hook chain after all plugins loaded ----");
				CBRO::Hooks::RenderStages::LogChain();
			}
			break;
		case F4SE::MessagingInterface::kGameDataReady:
			if (probe) {
				CBRO::Probe::OnGameDataReady();
			}
			break;
		case F4SE::MessagingInterface::kPostLoadGame:
		case F4SE::MessagingInterface::kNewGame:
			if (probe) {
				CBRO::Probe::OnGameLoaded();
			}
			CBRO::Core::Runtime::OnGameLoaded();
			break;
		default:
			break;
		}
	}

	bool InitializeLogger()
	{
		auto path = F4SE::log::log_directory();
		if (!path) {
			return false;
		}
		*path /= "CBRO.log"sv;
		auto sink = std::make_shared<spdlog::sinks::basic_file_sink_mt>(path->string(), true);
		auto log = std::make_shared<spdlog::logger>("global log"s, std::move(sink));
		log->set_level(spdlog::level::info);
		log->flush_on(spdlog::level::info);
		spdlog::set_default_logger(std::move(log));
		spdlog::set_pattern("[%H:%M:%S.%e] [%t] [%L] %v"s);
		return true;
	}

	// F4SE's packed plugin version: major.minor.patch.
	[[nodiscard]] constexpr std::uint32_t PackVersion(std::uint32_t a_major, std::uint32_t a_minor, std::uint32_t a_patch) noexcept
	{
		return ((a_major & 0xFF) << 24) | ((a_minor & 0xFF) << 16) | ((a_patch & 0xFFF) << 4);
	}

	[[nodiscard]] constexpr F4SE::PluginVersionData MakeVersionData() noexcept
	{
		F4SE::PluginVersionData data{};
		data.pluginVersion = PackVersion(CBRO::Version::MAJOR, CBRO::Version::MINOR, CBRO::Version::PATCH);
		for (std::size_t i = 0; i < CBRO::Version::PROJECT.size() && i < std::size(data.name) - 1; ++i) {
			data.name[i] = CBRO::Version::PROJECT[i];
		}
		// Engine ids come from the runtime database, not from an exact executable version, so the NG/AE loaders
		// may load this DLL on any patch; F4SEPlugin_Load then decides what it can safely do on that runtime.
		data.addressIndependence = F4SE::PluginVersionData::kAddressIndependence_Signatures;
		data.structureIndependence =
			F4SE::PluginVersionData::kStructureIndependence_1_10_980Layout |
			F4SE::PluginVersionData::kStructureIndependence_1_11_137Layout;
		return data;
	}

	// The runtime database CommonLibF4RD resolves every engine id through. Without it the library would stop
	// the game with a message box at F4SE::Init; CBRO logs the problem and stays inert instead.
	[[nodiscard]] bool RuntimeDatabasePresent(const REL::Version& a_version)
	{
		const auto plugins = std::filesystem::path("Data/F4SE/Plugins");
		std::error_code error;
		return std::filesystem::exists(plugins / "f4rd-runtime.bin", error) ||
		       std::filesystem::exists(plugins / std::format("f4rd-runtime-{}.bin", a_version.string()), error);
	}
}

// Read by the NG/AE loaders (and by F4SEPlugin_Query below).
extern "C" __declspec(dllexport) constinit F4SE::PluginVersionData F4SEPlugin_Version = MakeVersionData();

// The OG loader (F4SE 0.6.x) calls Query, then Load.
extern "C" __declspec(dllexport) bool F4SEAPI F4SEPlugin_Query(const F4SE::QueryInterface* a_f4se, F4SE::PluginInfo* a_info)
{
	if (!a_f4se || !a_info) {
		return false;
	}
	a_info->infoVersion = F4SE::PluginInfo::kVersion;
	a_info->name = F4SEPlugin_Version.name;
	a_info->version = F4SEPlugin_Version.pluginVersion;
	return !a_f4se->IsEditor();
}

extern "C" __declspec(dllexport) bool F4SEAPI F4SEPlugin_Load(const F4SE::LoadInterface* a_f4se)
{
	if (!a_f4se || !InitializeLogger()) {
		return false;
	}
	const auto runtime = a_f4se->RuntimeVersion();
	logger::info("CBRO v{} loading: game {}, F4SE {}", CBRO::Version::NAME, runtime.string(), a_f4se->F4SEVersion().string());
	if (a_f4se->IsEditor()) {
		return false;
	}

	auto& settings = CBRO::Settings::Get();
	settings.Load();
	if (!settings.enabled) {
		logger::info("disabled by CBRO.ini [General] bEnabled=0");
		return true;
	}
	if (!RuntimeDatabasePresent(runtime)) {
		logger::error("Data/F4SE/Plugins/f4rd-runtime.bin (the CommonLibF4RD Runtime Database) is missing: engine ids can't be resolved; installing nothing");
		return true;
	}

	F4SE::Init(a_f4se);
	const auto& module = REL::Module::get();
	logger::info("runtime family {} ({})", module.is_og() ? "OG" : module.is_ng() ? "NG" : "AE", module.version().string());
	if (!module.is_og()) {
		logger::warn("CBRO's interior hook offsets and prologues are verified for 1.10.163 (OG) only; installing nothing on this runtime");
		return true;
	}

	// OG F4SE has no trampoline interface: this creates CBRO's own executable pool (the NG/AE interface is used
	// where it exists).
	F4SE::AllocTrampoline(1024);

	// Register listeners/observers first, then install the shared hooks once. Without the probe only
	// the two stages CBRO needs (cull, pre-pass) are hooked. Occlusion installs the culling-group hooks
	// itself (and stays off without them); the group probe only observes them.
	if (settings.probe) {
		CBRO::Probe::Install();
	}
	CBRO::Core::Runtime::Install();
	CBRO::Hooks::RenderStages::Install(settings.probe && settings.renderStageHooks);
	if (settings.probe && settings.cullingGroupHooks && !CBRO::Hooks::CullGroups::Install()) {
		logger::error("culling-group hooks unavailable: the group probe sees nothing");
	}

	if (const auto messaging = F4SE::GetMessagingInterface()) {
		messaging->RegisterListener(OnMessage);
	}
	return true;
}
