#include "Probe/ProbeInternal.h"

// Surveys the loaded cells on the main thread: how many references carry their own 3D, and
// what the precombined geometry under LOADED_CELL_DATA::combinedObjects looks like (node
// types, child counts, world-bound radii). Bound size decides how often a combined chunk can
// be fully occluded (PLAN.md §2A "Bound size").

namespace CBRO::Probe::Scene
{
	namespace
	{
		// Radius bucket edges in game units (1 unit ~ 1.43 cm, an exterior cell is 4096 units wide).
		constexpr std::array<float, 6> kRadiusEdges{ 128.0f, 256.0f, 512.0f, 1024.0f, 2048.0f, 4096.0f };

		struct RadiusHistogram
		{
			std::array<std::uint32_t, kRadiusEdges.size() + 1> buckets{};
			std::uint32_t                                      count{ 0 };
			double                                             sum{ 0.0 };
			float                                              max{ 0.0f };

			void Add(float a_radius)
			{
				std::size_t bucket = 0;
				while (bucket < kRadiusEdges.size() && a_radius >= kRadiusEdges[bucket]) {
					++bucket;
				}
				++buckets[bucket];
				++count;
				sum += a_radius;
				max = std::max(max, a_radius);
			}

			void Merge(const RadiusHistogram& a_other)
			{
				for (std::size_t i = 0; i < buckets.size(); ++i) {
					buckets[i] += a_other.buckets[i];
				}
				count += a_other.count;
				sum += a_other.sum;
				max = std::max(max, a_other.max);
			}

			[[nodiscard]] std::string ToString() const
			{
				if (count == 0) {
					return "n=0";
				}
				std::string result = std::format("n={} mean={:.0f} max={:.0f} |", count, sum / count, max);
				for (std::size_t i = 0; i < buckets.size(); ++i) {
					if (i < kRadiusEdges.size()) {
						result += std::format(" <{:.0f}:{}", kRadiusEdges[i], buckets[i]);
					} else {
						result += std::format(" >={:.0f}:{}", kRadiusEdges.back(), buckets[i]);
					}
				}
				return result;
			}
		};

		struct TypeCounts
		{
			std::vector<std::pair<std::string, std::uint32_t>> entries;

			void Add(const RE::NiAVObject* a_object)
			{
				char name[48]{};
				if (!TryGetRTTIName(a_object, name, sizeof(name))) {
					strncpy_s(name, "(no NiRTTI)", _TRUNCATE);
				}
				for (auto& [type, count] : entries) {
					if (type == name) {
						++count;
						return;
					}
				}
				entries.emplace_back(name, 1);
			}

			void Merge(const TypeCounts& a_other)
			{
				for (const auto& [type, count] : a_other.entries) {
					auto it = std::ranges::find(entries, type, &std::pair<std::string, std::uint32_t>::first);
					if (it != entries.end()) {
						it->second += count;
					} else {
						entries.emplace_back(type, count);
					}
				}
			}

			[[nodiscard]] std::string ToString() const
			{
				std::string result;
				for (const auto& [type, count] : entries) {
					result += std::format(" {}x{}", type, count);
				}
				return result.empty() ? " none" : result;
			}
		};

		struct CombinedStats
		{
			std::uint32_t   chunks{ 0 };       // direct children of combinedObjects
			std::uint32_t   shapes{ 0 };       // grandchildren
			TypeCounts      chunkTypes;
			TypeCounts      shapeTypes;
			RadiusHistogram chunkRadii;
			RadiusHistogram shapeRadii;

			void Merge(const CombinedStats& a_other)
			{
				chunks += a_other.chunks;
				shapes += a_other.shapes;
				chunkTypes.Merge(a_other.chunkTypes);
				shapeTypes.Merge(a_other.shapeTypes);
				chunkRadii.Merge(a_other.chunkRadii);
				shapeRadii.Merge(a_other.shapeRadii);
			}
		};

		template <class F>
		void ForEachChild(RE::NiNode* a_node, F&& a_func)
		{
			if (!a_node) {
				return;
			}
			for (auto& child : a_node->children) {
				if (const auto object = child.get()) {
					a_func(object);
				}
			}
		}

		CombinedStats SurveyCombined(RE::NiNode* a_combined)
		{
			CombinedStats stats;
			ForEachChild(a_combined, [&](RE::NiAVObject* a_chunk) {
				++stats.chunks;
				stats.chunkTypes.Add(a_chunk);
				stats.chunkRadii.Add(a_chunk->worldBound.fRadius);
				ForEachChild(a_chunk->IsNode(), [&](RE::NiAVObject* a_shape) {
					++stats.shapes;
					stats.shapeTypes.Add(a_shape);
					stats.shapeRadii.Add(a_shape->worldBound.fRadius);
				});
			});
			return stats;
		}

		std::string CellLabel(RE::TESObjectCELL* a_cell)
		{
			if (a_cell->IsInterior()) {
				return std::format("interior {:08X}", a_cell->GetFormID());
			}
			// EXTERIOR_DATA starts with cellX, cellY (the RD headers only forward-declare it).
			if (const auto coords = reinterpret_cast<const std::int32_t*>(a_cell->cellDataExterior)) {
				return std::format("exterior {:08X} ({},{})", a_cell->GetFormID(), coords[0], coords[1]);
			}
			return std::format("exterior {:08X}", a_cell->GetFormID());
		}

		std::vector<RE::TESObjectCELL*> LoadedCells()
		{
			std::vector<RE::TESObjectCELL*> cells;
			const auto                      tes = CBRO::Engine::TES::GetSingleton();
			if (!tes) {
				return cells;
			}
			if (tes->interiorCell) {
				cells.push_back(tes->interiorCell);
				return cells;
			}
			const auto grid = tes->gridCells;
			if (!grid) {
				return cells;
			}
			for (std::uint32_t x = 0; x < grid->dimension; ++x) {
				for (std::uint32_t y = 0; y < grid->dimension; ++y) {
					const auto gridCell = grid->Get(x, y);
					if (gridCell && gridCell->cell && gridCell->cell->cellState.get() == RE::TESObjectCELL::CELL_STATE::kAttached) {
						cells.push_back(gridCell->cell);
					}
				}
			}
			return cells;
		}

		void RunSurvey()
		{
			const auto cells = LoadedCells();
			logger::info("==== scene survey: {} attached cells (thread {}) ====", cells.size(), GetCurrentThreadId());

			CombinedStats   total;
			RadiusHistogram totalRefRadii;
			std::uint32_t   totalRefs = 0;
			std::uint32_t   totalRefs3D = 0;
			std::uint32_t   cellsWithPrevis = 0;

			for (const auto cell : cells) {
				std::uint32_t   refs = 0;
				std::uint32_t   refs3D = 0;
				RadiusHistogram refRadii;
				for (const auto& ref : cell->references) {
					const auto object = ref.get();
					if (!object) {
						continue;
					}
					++refs;
					// Field read instead of the Get3D() virtual: precombined refs have no 3D of their own.
					const auto loaded = object->loadedData;
					if (const auto root = loaded ? loaded->data3D.get() : nullptr) {
						++refs3D;
						refRadii.Add(root->worldBound.fRadius);
					}
				}
				totalRefRadii.Merge(refRadii);

				const auto data = cell->loadedData;
				const auto combined = data ? data->combinedObjects.get() : nullptr;
				const auto stats = SurveyCombined(combined);
				total.Merge(stats);
				totalRefs += refs;
				totalRefs3D += refs3D;
				cellsWithPrevis += cell->visibilityData != nullptr;

				logger::info(
					"cell {}: refs={} with3D={} preCombinedArr={} combinedNode={} chunks={} shapes={} "
					"combinedAttached={} registered={} | previs visData={} rootVisCell={:08X} visCalcDate={} preCombineDate={}",
					CellLabel(cell), refs, refs3D, data ? data->preCombined.size() : 0, fmt_ptr(combined),
					stats.chunks, stats.shapes,
					data ? data->combinedObjectsAttached.load_unchecked() : 0u, data ? data->combinedObjectsRegistered : false,
					fmt_ptr(cell->visibilityData), cell->rootVisibilityCellID, cell->visCalcDate, cell->preCombineDate);
				logger::info("  ref 3D radii: {}", refRadii.ToString());
				if (stats.chunks) {
					logger::info("  chunk types:{}", stats.chunkTypes.ToString());
					logger::info("  chunk radii: {}", stats.chunkRadii.ToString());
					logger::info("  shape types:{}", stats.shapeTypes.ToString());
					logger::info("  shape radii: {}", stats.shapeRadii.ToString());
				}
			}

			logger::info(
				"survey total: refs={} with3D={} cellsWithPrevisData={} combined chunks={} shapes={}",
				totalRefs, totalRefs3D, cellsWithPrevis, total.chunks, total.shapes);
			logger::info("  all ref 3D radii: {}", totalRefRadii.ToString());
			logger::info("  all chunk radii: {}", total.chunkRadii.ToString());
			logger::info("  all shape radii: {}", total.shapeRadii.ToString());
		}

		void LogSetting(std::string_view a_name)
		{
			const auto setting = CBRO::Engine::GetINISetting(a_name);
			if (!setting) {
				logger::info("game setting {}: not found", a_name);
				return;
			}
			switch (setting->GetType()) {
			case RE::Setting::SETTING_TYPE::kBinary:
				logger::info("game setting {} = {}", a_name, setting->GetBinary());
				break;
			case RE::Setting::SETTING_TYPE::kInt:
				logger::info("game setting {} = {}", a_name, setting->GetInt());
				break;
			case RE::Setting::SETTING_TYPE::kUInt:
				logger::info("game setting {} = {}", a_name, setting->GetUInt());
				break;
			case RE::Setting::SETTING_TYPE::kFloat:
				logger::info("game setting {} = {}", a_name, setting->GetFloat());
				break;
			default:
				logger::info("game setting {}: type {}", a_name, static_cast<int>(setting->GetType()));
				break;
			}
		}
	}

	void LogGameSettings()
	{
		for (const auto name : {
				 "bUseCombinedObjects:General"sv,
				 "bEnableBoundingVolumeOcclusion:General"sv,
				 "bEnableBoundingVolumeOcclusion:Display"sv,
				 "bUsePreCulledObjects:Display"sv,
				 "fDirShadowDistance:Display"sv,
				 "uGridsToLoad:General"sv,
			 }) {
			LogSetting(name);
		}
	}

	void RequestSurvey()
	{
		const auto main = RE::Main::GetSingleton();
		if (main && GetCurrentThreadId() == main->threadID) {
			RunSurvey();
			return;
		}

		const auto tasks = F4SE::GetTaskInterface();
		if (!tasks) {
			logger::warn("scene survey: no F4SE task interface");
			return;
		}
		tasks->AddTask([]() {
			const auto main = RE::Main::GetSingleton();
			if (main && GetCurrentThreadId() != main->threadID) {
				logger::warn("scene survey: task ran on thread {} but main is {}; skipped", GetCurrentThreadId(), main->threadID);
				return;
			}
			RunSurvey();
		});
	}
}
