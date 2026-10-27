#include "Core/Recorder.h"

#include "Settings.h"

#include <chrono>
#include <condition_variable>
#include <fstream>
#include <memory>
#include <mutex>
#include <thread>
#include <unordered_map>

namespace CBRO::Core::Recorder
{
	std::atomic<bool> g_on{ false };

	namespace
	{
		using namespace RecorderFormat;

		bool          g_enabled{ false };
		std::uint32_t g_frames{ 600 };
		std::int32_t  g_hiZLevel{ 2 };
		std::string   g_version;

		// One frame of the ring: its candidates (a fixed array, wait-free appends), its header, lamps and hi-z.
		struct Slot
		{
			std::unique_ptr<Candidate[]> data;
			std::uint32_t                capacity{ 0 };
			std::atomic<std::uint32_t>   count{ 0 };  // appends (may run past capacity: the excess is the overflow)
			FrameRecord                  frame{};
			std::vector<LampRecord>      lamps;
			std::unique_ptr<float[]>     hiZ;
			std::uint32_t                hiZFloats{ 0 };
			bool                         hasHiZ{ false };
			bool                         used{ false };
		};

		// Object address -> index into the objects list (open addressing, main thread).
		struct PointerIndex
		{
			std::vector<std::uint64_t> keys;
			std::vector<std::uint32_t> values;
			std::size_t                count{ 0 };

			static std::size_t Hash(std::uint64_t a_key) noexcept
			{
				a_key ^= a_key >> 33;
				a_key *= 0xff51afd7ed558ccdull;
				a_key ^= a_key >> 33;
				return static_cast<std::size_t>(a_key);
			}

			void Reserve(std::size_t a_capacity)
			{
				std::size_t capacity = 1024;
				while (capacity < a_capacity * 2) {
					capacity *= 2;
				}
				std::vector<std::uint64_t> oldKeys = std::move(keys);
				std::vector<std::uint32_t> oldValues = std::move(values);
				keys.assign(capacity, 0);
				values.assign(capacity, 0);
				count = 0;
				for (std::size_t i = 0; i < oldKeys.size(); ++i) {
					if (oldKeys[i]) {
						Insert(oldKeys[i], oldValues[i]);
					}
				}
			}

			[[nodiscard]] bool Find(std::uint64_t a_key, std::uint32_t& a_value) const noexcept
			{
				if (keys.empty()) {
					return false;
				}
				const auto mask = keys.size() - 1;
				for (auto i = Hash(a_key) & mask;; i = (i + 1) & mask) {
					if (keys[i] == a_key) {
						a_value = values[i];
						return true;
					}
					if (!keys[i]) {
						return false;
					}
				}
			}

			void Insert(std::uint64_t a_key, std::uint32_t a_value)
			{
				if ((count + 1) * 2 > keys.size()) {
					Reserve(std::max<std::size_t>(count + 1, 1024));
				}
				const auto mask = keys.size() - 1;
				for (auto i = Hash(a_key) & mask;; i = (i + 1) & mask) {
					if (keys[i] == a_key) {
						values[i] = a_value;
						return;
					}
					if (!keys[i]) {
						keys[i] = a_key;
						values[i] = a_value;
						++count;
						return;
					}
				}
			}
		};

		struct Ring
		{
			std::vector<std::unique_ptr<Slot>>          slots;
			std::uint32_t                               head{ 0 };      // the slot in progress (== filled: frames go in order, from the start to the stop)
			std::uint32_t                               filled{ 0 };    // closed frames behind it
			std::uint32_t                               maxCount{ 0 };  // the most candidates a frame had
			std::uint64_t                               candidates{ 0 };
			std::uint64_t                               overflow{ 0 };
			std::uint32_t                               hiZWidth{ 0 };
			std::uint32_t                               hiZHeight{ 0 };
			PointerIndex                                index;
			std::vector<ObjectRecord>                   objects;
			std::vector<char>                           strings;
			std::unordered_map<std::string, std::uint32_t> stringIndex;
			std::uint32_t                               firstClock{ 0 };
			std::uint32_t                               lastClock{ 0 };
			std::chrono::steady_clock::time_point       started;
		};

		std::unique_ptr<Ring> g_ring;                  // main thread
		std::atomic<Slot*>    g_current{ nullptr };    // the slot appends go to (null: none)
		std::atomic<int>      g_saving{ 0 };

		struct Decision
		{
			Occlusion::Record record{};
			std::uint8_t      source{ 3 };  // not judged
			std::uint8_t      path{ kPathNone };
			bool              judged{ false };
		};
		thread_local Decision      t_decision;
		thread_local std::uint16_t t_walkSite{ 0 };
		thread_local std::uint16_t t_walkFlags{ 0 };
		thread_local std::uint16_t t_flags{ 0 };

		void Notify(std::string a_text)
		{
			logger::info("recorder: {}", a_text);
			if (!Settings::Get().notify) {
				return;
			}
			if (const auto tasks = F4SE::GetTaskInterface()) {
				tasks->AddTask([text = std::move(a_text)]() {
					RE::SendHUDMessage::ShowHUDMessage(text.c_str(), nullptr, false, false);  // (no sound: Runtime's Notify has why)
				});
			}
		}

		inline void Append(const Candidate& a_candidate) noexcept
		{
			const auto slot = g_current.load(std::memory_order_acquire);
			if (!slot) {
				return;
			}
			const auto index = slot->count.fetch_add(1, std::memory_order_relaxed);
			if (index < slot->capacity) {
				slot->data[index] = a_candidate;
			}
		}

		// ---- object metadata (main thread, the frame that saw the object: it is alive) ------------------------------
		struct Described
		{
			const char*   rtti{ nullptr };
			const char*   name{ nullptr };
			std::uint64_t flags{ 0 };
			std::uint64_t parent{ 0 };
			std::uint32_t refID{ 0 };
			std::uint32_t baseID{ 0 };
			std::uint8_t  baseType{ 0 };
			std::uint8_t  isNode{ 0 };
			std::uint8_t  isFadeNode{ 0 };
			std::uint8_t  isGeometry{ 0 };
			float         bound[4]{};
		};

		void DescribeBody(RE::NiAVObject* a_object, Described& a_out) noexcept
		{
			if (const auto rtti = a_object->GetRTTI()) {
				a_out.rtti = rtti->GetName();
			}
			a_out.name = a_object->name.c_str();
			a_out.flags = a_object->GetFlags();
			a_out.parent = reinterpret_cast<std::uint64_t>(a_object->parent);
			a_out.isNode = a_object->IsNode() != nullptr;
			a_out.isFadeNode = a_object->IsFadeNode() != nullptr;
			a_out.isGeometry = a_object->IsGeometry() != nullptr;
			a_out.bound[0] = a_object->worldBound.center.x;
			a_out.bound[1] = a_object->worldBound.center.y;
			a_out.bound[2] = a_object->worldBound.center.z;
			a_out.bound[3] = a_object->worldBound.fRadius;
			// The owning reference: the nearest ancestor whose userData is a REFR or ACHR (as Occlusion's FindReference).
			const RE::NiAVObject* node = a_object;
			for (int depth = 0; node && depth < 32; ++depth, node = node->parent) {
				if (!node->userData) {
					continue;
				}
				const auto form = reinterpret_cast<const RE::TESForm*>(node->userData);
				const auto type = form->GetFormType();
				if (type != RE::ENUM_FORM_ID::kREFR && type != RE::ENUM_FORM_ID::kACHR) {
					continue;
				}
				a_out.refID = form->GetFormID();
				if (const auto base = reinterpret_cast<const RE::TESObjectREFR*>(form)->GetObjectReference()) {
					a_out.baseID = base->GetFormID();
					a_out.baseType = static_cast<std::uint8_t>(base->GetFormType());
				}
				return;
			}
		}

		bool Describe(RE::NiAVObject* a_object, Described& a_out) noexcept
		{
			__try {
				DescribeBody(a_object, a_out);
				return true;
			} __except (EXCEPTION_EXECUTE_HANDLER) {
				return false;
			}
		}

		std::uint32_t InternString(Ring& a_ring, const char* a_text)
		{
			if (!a_text || !*a_text) {
				if (a_ring.strings.empty()) {
					a_ring.strings.push_back('\0');
				}
				return 0;
			}
			std::string key(a_text);
			if (const auto it = a_ring.stringIndex.find(key); it != a_ring.stringIndex.end()) {
				return it->second;
			}
			const auto offset = AddString(a_ring.strings, key);
			a_ring.stringIndex.emplace(std::move(key), offset);
			return offset;
		}

		void AddObjects(Ring& a_ring, const Slot& a_slot, std::uint32_t a_count, std::uint32_t a_clock)
		{
			for (std::uint32_t i = 0; i < a_count; ++i) {
				const auto key = a_slot.data[i].object;
				std::uint32_t found = 0;
				if (!key || a_ring.index.Find(key, found)) {
					continue;
				}
				ObjectRecord record{};
				record.object = key;
				record.firstClock = a_clock;
				Described described{};
				if (Describe(reinterpret_cast<RE::NiAVObject*>(key), described)) {
					record.parent = described.parent;
					record.flags = described.flags;
					record.refID = described.refID;
					record.baseID = described.baseID;
					record.baseType = described.baseType;
					record.isNode = described.isNode;
					record.isFadeNode = described.isFadeNode;
					record.isGeometry = described.isGeometry;
					std::copy_n(described.bound, 4, record.bound);
					record.rtti = InternString(a_ring, described.rtti);
					record.name = InternString(a_ring, described.name);
				} else {
					record.rtti = InternString(a_ring, "<unreadable>");
				}
				a_ring.index.Insert(key, static_cast<std::uint32_t>(a_ring.objects.size()));
				a_ring.objects.push_back(record);
			}
		}

		// ---- the frame header ---------------------------------------------------------------------------------------
		void FillFrame(FrameRecord& a_frame, const FrameInfo& a_info)
		{
			a_frame = {};
			a_frame.renderFrame = a_info.renderFrame;
			a_frame.location = a_info.location;
			a_frame.mode = static_cast<std::uint8_t>(std::clamp(a_info.mode, 0, 2));
			a_frame.settled = a_info.settled;
			a_frame.observeOnly = a_info.observeOnly;
			a_frame.hiZBlob = -1;
			if (a_info.pose) {
				std::copy_n(a_info.pose->eye, 3, a_frame.eye);
				for (int r = 0; r < 3; ++r) {
					std::copy_n(a_info.pose->rotate[r], 3, a_frame.rotate + 3 * r);
				}
				std::copy_n(a_info.pose->zoom, 2, a_frame.zoom);
			}
			if (a_info.lamps) {
				a_frame.lampsComplete = a_info.lamps->complete;
			}
			const auto context = a_info.context;
			if (!context) {
				return;
			}
			a_frame.clock = context->clock;
			a_frame.cull = context->cull;
			a_frame.asyncValid = context->asyncValid;
			a_frame.sunState = static_cast<std::uint8_t>(context->sun.state);
			a_frame.sunEpoch = context->sunEpoch;
			a_frame.viewEpoch = context->viewEpoch;
			a_frame.readback = context->readback;
			std::copy_n(context->view, 4, a_frame.view);
			std::copy_n(context->overhang, 4, a_frame.overhang);
			a_frame.dilateMove = context->dilateMove;
			a_frame.parallaxMove = context->parallaxMove;
			a_frame.angularSlack = context->angularSlack;
			std::copy_n(context->sun.dir, 3, a_frame.sunDir);
			a_frame.sunReach = context->sun.reach;
			a_frame.sunSpread = context->sun.spread;
			a_frame.sunMargin = context->sun.margin;
			if (!a_info.pose) {
				std::copy_n(context->eye, 3, a_frame.eye);
			}
			const auto snapshot = context->snapshot;
			if (!snapshot) {
				return;
			}
			const auto& camera = snapshot->camera;
			a_frame.depthFrame = snapshot->frame;
			a_frame.mergedFrames = snapshot->mergedFrames;
			a_frame.mergeMove = snapshot->mergeMove;
			for (int r = 0; r < 4; ++r) {
				std::copy_n(camera.viewProj[r], 4, a_frame.viewProj + 4 * r);
			}
			std::copy_n(camera.posAdjust, 3, a_frame.posAdjust);
			std::copy_n(camera.eye, 3, a_frame.depthEye);
			for (int r = 0; r < 3; ++r) {
				std::copy_n(camera.rotate[r], 3, a_frame.depthRotate + 3 * r);
			}
			std::copy_n(camera.viewDir, 3, a_frame.viewDir);
			std::copy_n(camera.viewRight, 3, a_frame.viewRight);
			std::copy_n(camera.viewUp, 3, a_frame.viewUp);
			std::copy_n(camera.origin, 3, a_frame.origin);
			a_frame.scaleX = camera.scaleX;
			a_frame.scaleY = camera.scaleY;
			a_frame.depthA = camera.depthA;
			a_frame.depthB = camera.depthB;
			a_frame.frustumX = camera.frustumX;
			a_frame.frustumY = camera.frustumY;
			a_frame.viewSpace = camera.viewSpace;
			a_frame.footprint = camera.footprint;
			a_frame.firstPersonAfter = snapshot->firstPersonAfter;
			a_frame.cameraState = camera.cameraState;
			a_frame.hiZWidth0 = snapshot->width[0];
			a_frame.hiZHeight0 = snapshot->height[0];
		}

		void CopyHiZ(Ring& a_ring, Slot& a_slot, const HiZ::Snapshot* a_snapshot)
		{
			a_slot.hasHiZ = false;
			if (!a_snapshot || g_hiZLevel < 0 || static_cast<std::uint32_t>(g_hiZLevel) >= a_snapshot->levels || !a_snapshot->texels) {
				return;
			}
			const auto level = static_cast<std::size_t>(g_hiZLevel);
			const auto width = a_snapshot->width[level];
			const auto height = a_snapshot->height[level];
			if (!width || !height) {
				return;
			}
			// (one size per recording: the first depth frame's; a later change leaves the frame without a blob)
			if (!a_ring.hiZWidth) {
				a_ring.hiZWidth = width;
				a_ring.hiZHeight = height;
			} else if (a_ring.hiZWidth != width || a_ring.hiZHeight != height) {
				return;
			}
			const auto floats = static_cast<std::uint32_t>(width) * height;
			if (a_slot.hiZFloats != floats * 2) {
				a_slot.hiZ = std::make_unique<float[]>(static_cast<std::size_t>(floats) * 2);
				a_slot.hiZFloats = floats * 2;
			}
			std::copy_n(a_snapshot->texels + a_snapshot->offset[level], floats, a_slot.hiZ.get());
			if (a_snapshot->nearest) {
				std::copy_n(a_snapshot->nearest + a_snapshot->offset[level], floats, a_slot.hiZ.get() + floats);
			} else {
				std::fill_n(a_slot.hiZ.get() + floats, floats, 0.0f);
			}
			a_slot.hasHiZ = true;
		}

		// ---- the save (its own thread) ----------------------------------------------------------------------------
		struct Job
		{
			std::unique_ptr<Ring> ring;
			std::string           json;
		};

		std::string Timestamp()
		{
			const auto now = std::chrono::system_clock::now();
			const auto time = std::chrono::system_clock::to_time_t(now);
			std::tm local{};
			localtime_s(&local, &time);
			char buf[32];
			std::strftime(buf, sizeof(buf), "%Y%m%d-%H%M%S", &local);
			return buf;
		}

		void Save(Job& a_job)
		{
			const auto start = std::chrono::steady_clock::now();
			auto&      ring = *a_job.ring;
			Recording  recording;
			recording.json = std::move(a_job.json);
			recording.hiZWidth = ring.hiZWidth;
			recording.hiZHeight = ring.hiZHeight;
			recording.frames.reserve(ring.filled);
			std::uint64_t candidateOffset = 0;
			for (std::uint32_t k = 0; k < ring.filled; ++k) {
				const auto& slot = ring.slots[k];
				if (!slot || !slot->used) {
					continue;
				}
				auto frame = slot->frame;
				frame.candidateOffset = candidateOffset;
				recording.candidateRuns.emplace_back(slot->data.get(), frame.candidateCount);
				candidateOffset += frame.candidateCount;
				frame.lampOffset = static_cast<std::uint32_t>(recording.lamps.size());
				frame.lampCount = static_cast<std::uint32_t>(slot->lamps.size());
				recording.lamps.insert(recording.lamps.end(), slot->lamps.begin(), slot->lamps.end());
				if (slot->hasHiZ && ring.hiZWidth) {
					frame.hiZBlob = static_cast<std::int32_t>(recording.hiZRuns.size());
					recording.hiZRuns.emplace_back(slot->hiZ.get(), slot->hiZFloats);
				} else {
					frame.hiZBlob = -1;
				}
				recording.frames.push_back(frame);
			}
			recording.objects = std::move(ring.objects);
			recording.strings = std::move(ring.strings);
			if (recording.strings.empty()) {
				recording.strings.push_back('\0');
			}
			if (!recording.hiZRuns.empty() && recording.candidateRuns.empty()) {
				recording.candidateRuns.emplace_back();  // (runs mode for both: a run of nothing)
			}

			std::error_code       error;
			const std::filesystem::path dir{ "Data/CBRO_Recordings" };
			std::filesystem::create_directories(dir, error);
			if (error) {
				Notify(std::format("save failed: cannot create {} ({})", dir.string(), error.message()));
				return;
			}
			const auto name = std::format("CBRO-{}.cbrorec", Timestamp());
			const auto path = dir / name;
			const auto part = dir / (name + ".part");
			{
				std::ofstream out;
				std::vector<char> buffer(1u << 20);
				out.rdbuf()->pubsetbuf(buffer.data(), static_cast<std::streamsize>(buffer.size()));
				out.open(part, std::ios::binary | std::ios::trunc);
				if (!out) {
					Notify(std::format("save failed: cannot open {}", part.string()));
					return;
				}
				if (!Write(out, recording)) {
					out.close();
					std::filesystem::remove(part, error);
					Notify(std::format("save failed: write error on {}", part.string()));
					return;
				}
			}
			std::filesystem::remove(path, error);
			std::filesystem::rename(part, path, error);
			if (error) {
				Notify(std::format("save failed: cannot rename {} ({})", part.string(), error.message()));
				return;
			}
			const auto bytes = std::filesystem::file_size(path, error);
			const auto ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
			Notify(std::format("saved {} frames ({} candidates, {} objects, {:.0f} MB) to {} in {:.0f} ms",
				recording.frames.size(), candidateOffset, recording.objects.size(), static_cast<double>(bytes) / (1024.0 * 1024.0), path.string(), ms));
		}

		std::string SettingsJson()
		{
			const auto& s = Settings::Get();
			std::string out = std::format("\"cbro\":\"{}\",\"recordedAt\":\"{}\",\"frameLimit\":{},\"hiZLevel\":{},\"settings\":{{", g_version, Timestamp(), g_frames, g_hiZLevel);
			out += std::format("\"confirmFrames\":{},\"nearDistance\":{},\"viewOverhang\":{},\"viewOverhangTurn\":{},\"depthTolerance\":{},\"depthSlack\":{},\"maxCameraMove\":{},\"maxCameraAngle\":{},",
				s.confirmFrames, s.nearDistance, s.viewOverhang, s.viewOverhangTurn, s.depthTolerance, s.depthSlack, s.maxCameraMove, s.maxCameraAngle);
			out += std::format("\"cullActors\":{},\"cullPlayer\":{},\"traceFeed\":{},\"meshShapes\":{},\"sunShadowCulling\":{},\"cellNodePruning\":{},\"entryWalk\":{},\"async\":{},\"parallaxMargin\":{},\"keepFadeState\":{}}}",
				s.cullActors, s.cullPlayer, s.traceFeed, s.meshShapes, s.sunShadowCulling, s.cellNodePruning, s.entryWalk, s.async, s.parallaxMargin, s.keepFadeState);
			return out;
		}

		// The next frame's slot: reused, or allocated, with room for the most candidates seen so far and a quarter more.
		// The frames' candidate buffers, made ahead on a helper thread (v1.96.1). Each frame opens a fresh buffer (~19k
		// candidates, 760 KB outdoors); allocated on the main thread it was zeroed there and page-faulted on first touch in
		// the appends: the recorder's largest cost (OpenSlot, 0.3-0.5 ms a frame of its ~1.2). The helper keeps a few
		// buffers allocated and touched; OpenSlot takes one under a short lock, or makes its own when none fits.
		struct Prefetch
		{
			std::mutex                                                          lock;
			std::condition_variable                                             wake;
			std::vector<std::pair<std::unique_ptr<Candidate[]>, std::uint32_t>> ready;  // buffer, its capacity
			std::uint32_t                                                       wanted{ 16384 };
			bool                                                                stop{ false };
			std::thread                                                         thread;
		};
		std::unique_ptr<Prefetch> g_prefetch;
		constexpr std::size_t     kPrefetchAhead = 8;

		std::unique_ptr<Candidate[]> TouchedBuffer(std::uint32_t a_capacity)
		{
			auto buffer = std::make_unique_for_overwrite<Candidate[]>(a_capacity);  // every used record is written whole
			const auto bytes = reinterpret_cast<volatile std::byte*>(buffer.get());
			const auto size = sizeof(Candidate) * a_capacity;
			for (std::size_t i = 0; i < size; i += 4096) {  // first touch: the pages fault in here, not in the appends
				bytes[i] = std::byte{ 0 };
			}
			return buffer;
		}

		void PrefetchLoop(Prefetch& a_prefetch)
		{
			std::unique_lock lock(a_prefetch.lock);
			while (!a_prefetch.stop) {
				if (a_prefetch.ready.size() >= kPrefetchAhead) {
					a_prefetch.wake.wait(lock);
					continue;
				}
				const auto wanted = a_prefetch.wanted;
				lock.unlock();
				auto buffer = TouchedBuffer(wanted);
				lock.lock();
				a_prefetch.ready.emplace_back(std::move(buffer), wanted);
			}
		}

		void StartPrefetch()
		{
			g_prefetch = std::make_unique<Prefetch>();
			g_prefetch->thread = std::thread(PrefetchLoop, std::ref(*g_prefetch));
		}

		void StopPrefetch()
		{
			if (!g_prefetch) {
				return;
			}
			{
				std::lock_guard lock(g_prefetch->lock);
				g_prefetch->stop = true;
			}
			g_prefetch->wake.notify_all();
			g_prefetch->thread.join();
			g_prefetch.reset();
		}

		void OpenSlot(Ring& a_ring)
		{
			auto& slot = a_ring.slots[a_ring.head];
			if (!slot) {
				slot = std::make_unique<Slot>();
			}
			const auto wanted = std::max<std::uint32_t>(16384, a_ring.maxCount + a_ring.maxCount / 4);
			if (slot->capacity < wanted) {
				bool taken = false;
				if (g_prefetch) {
					std::lock_guard lock(g_prefetch->lock);
					g_prefetch->wanted = wanted;
					if (!g_prefetch->ready.empty() && g_prefetch->ready.back().second >= wanted) {
						slot->data = std::move(g_prefetch->ready.back().first);
						slot->capacity = g_prefetch->ready.back().second;
						g_prefetch->ready.pop_back();
						taken = true;
					} else {
						g_prefetch->ready.clear();  // too small (the frames grew): the helper makes larger ones
					}
				}
				if (g_prefetch) {
					g_prefetch->wake.notify_one();
				}
				if (!taken) {
					slot->data = std::make_unique_for_overwrite<Candidate[]>(wanted);
					slot->capacity = wanted;
				}
			}
			slot->count.store(0, std::memory_order_relaxed);
			slot->used = false;
			g_current.store(slot.get(), std::memory_order_release);
		}
	}

	void Install(bool a_enabled, std::uint32_t a_frames, std::int32_t a_hiZLevel, std::string a_version)
	{
		g_enabled = a_enabled;
		g_frames = std::clamp<std::uint32_t>(a_frames, 1, 36000);
		g_hiZLevel = std::clamp<std::int32_t>(a_hiZLevel, -1, 15);
		g_version = std::move(a_version);
		if (g_enabled) {
			logger::info("recorder: ready (bRecorder=1): the hotkey starts a recording (up to {} frames, hi-z level {}) and, pressed again, stops and saves it; at the frame limit it saves itself; files go to Data\\CBRO_Recordings", g_frames, g_hiZLevel);
		}
	}

	bool Enabled() noexcept
	{
		return g_enabled;
	}

	void Start()
	{
		if (!g_enabled || g_on.load(std::memory_order_relaxed)) {
			return;
		}
		auto ring = std::make_unique<Ring>();
		ring->slots.resize(g_frames);
		ring->index.Reserve(1u << 15);
		ring->strings.push_back('\0');
		ring->started = std::chrono::steady_clock::now();
		g_ring = std::move(ring);
		StartPrefetch();
		OpenSlot(*g_ring);
		g_on.store(true, std::memory_order_release);
		Notify(std::format("recording (up to {} frames; press again to stop and save)", g_frames));
	}

	void Stop()
	{
		if (!g_on.load(std::memory_order_relaxed)) {
			return;
		}
		g_on.store(false, std::memory_order_release);
		g_current.store(nullptr, std::memory_order_release);
		StopPrefetch();
		auto job = std::make_unique<Job>();
		job->ring = std::move(g_ring);
		job->json = Json(static_cast<std::uint32_t>(std::max(g_hiZLevel, 0)), job->ring->hiZWidth, job->ring->hiZHeight, SettingsJson());
		const auto frames = job->ring->filled;
		const auto candidates = job->ring->candidates;
		Notify(std::format("saving {} frames ({} candidates a frame, {} overflowed)...", frames, frames ? candidates / frames : 0, job->ring->overflow));
		g_saving.fetch_add(1, std::memory_order_relaxed);
		std::thread([job = std::move(job)]() mutable {
			Save(*job);
			job.reset();
			g_saving.fetch_sub(1, std::memory_order_relaxed);
		}).detach();
	}

	void Toggle()
	{
		if (!g_enabled) {
			return;
		}
		if (g_on.load(std::memory_order_relaxed)) {
			Stop();
		} else {
			Start();
		}
	}

	std::string Status()
	{
		if (!g_enabled) {
			return "recorder off (bRecorder=0)";
		}
		if (g_on.load(std::memory_order_relaxed) && g_ring) {
			const auto frames = g_ring->filled;
			return std::format("recording: {} of {} frames, {} candidates a frame, {} objects", frames, g_frames, frames ? g_ring->candidates / frames : 0, g_ring->objects.size());
		}
		return g_saving.load(std::memory_order_relaxed) ? "recorder: saving" : "recorder: idle (the hotkey starts it)";
	}

	// ---- the hooks' notes -------------------------------------------------------------------------------------------
	void BeginDecision() noexcept
	{
		t_decision = {};
		t_flags = 0;
	}

	void NoteJudged(const Occlusion::Record& a_record, std::uint8_t a_source) noexcept
	{
		t_decision.record = a_record;
		t_decision.source = a_source;
		t_decision.judged = true;
	}

	void NotePath(std::uint8_t a_path) noexcept
	{
		t_decision.path = a_path;
	}

	void SetWalkSite(std::uint16_t a_site, std::uint16_t a_flags) noexcept
	{
		t_walkSite = a_site;
		t_walkFlags = a_flags;
	}

	void NoteFlags(std::uint16_t a_flags) noexcept
	{
		t_flags |= a_flags;
	}

	void NoteEntry(Origin a_origin, const RE::NiAVObject* a_object, const RE::NiBound& a_bound, std::uint8_t a_kind, std::uint16_t a_flags, std::uint16_t a_site) noexcept
	{
		Candidate candidate{};
		candidate.object = reinterpret_cast<std::uint64_t>(a_object);
		candidate.bound[0] = a_bound.center.x;
		candidate.bound[1] = a_bound.center.y;
		candidate.bound[2] = a_bound.center.z;
		candidate.bound[3] = a_bound.fRadius;
		candidate.origin = static_cast<std::uint8_t>(a_origin);
		candidate.kind = a_kind;
		candidate.flags = static_cast<std::uint16_t>(a_flags | t_flags | t_walkFlags);
		candidate.site = a_site ? a_site : t_walkSite;
		if (a_origin == Origin::kEngineTop && t_walkSite) {
			candidate.origin = static_cast<std::uint8_t>(Origin::kWalkTop);
		}
		const auto& decision = t_decision;
		if (decision.judged) {
			const auto& record = decision.record;
			candidate.outcome = static_cast<std::uint8_t>(record.outcome);
			candidate.verdict = record.verdict;
			candidate.sun = static_cast<std::uint8_t>(record.sun);
			candidate.streak = record.streak;
			candidate.extra = static_cast<std::uint32_t>(record.flags) | (static_cast<std::uint32_t>(record.sunStreak) << 8);
			candidate.flags |= kJudged;
			if (record.flags & Occlusion::kRecordLight) {
				candidate.flags |= kLight;
			}
		}
		candidate.source = decision.judged ? decision.source : 3;
		candidate.path = decision.path;
		if (decision.path != kPathNone) {
			candidate.flags |= kLeftOut;
		}
		if (decision.path == 3) {
			candidate.flags |= kInherited;  // "parent dropped"
		}
		Append(candidate);
		t_decision = {};
		t_flags = 0;
		t_walkSite = 0;
		t_walkFlags = 0;
	}

	void NoteNode(const RE::NiAVObject* a_node, std::uint32_t a_index, bool a_skipped, std::uint32_t a_entries, std::uint16_t a_flags, const RE::NiBound& a_bound) noexcept
	{
		Candidate candidate{};
		candidate.object = reinterpret_cast<std::uint64_t>(a_node);
		candidate.bound[0] = a_bound.center.x;
		candidate.bound[1] = a_bound.center.y;
		candidate.bound[2] = a_bound.center.z;
		candidate.bound[3] = a_bound.fRadius;
		candidate.origin = static_cast<std::uint8_t>(a_skipped ? Origin::kNodePruned : Origin::kNodeWalked);
		candidate.kind = 1;  // group 0
		candidate.flags = static_cast<std::uint16_t>(a_flags | (a_skipped ? kLeftOut : 0));
		candidate.source = 3;
		candidate.path = a_skipped ? 8 : kPathNone;  // "never filed: its node pruned whole"
		candidate.site = static_cast<std::uint16_t>(a_index > 0xFFFF ? (a_index == 0xFFFFFFFFu ? 255 : 254) : a_index);
		candidate.extra = a_entries;
		Append(candidate);
	}

	void NoteRegistration(const RE::NiAVObject* a_object, bool a_refused) noexcept
	{
		Candidate candidate{};
		candidate.object = reinterpret_cast<std::uint64_t>(a_object);
		candidate.origin = static_cast<std::uint8_t>(a_refused ? Origin::kRefused : Origin::kRegistered);
		candidate.kind = 2;
		candidate.source = 3;
		candidate.path = a_refused ? 2 : kPathNone;  // "main view drop"
		candidate.flags = a_refused ? kLeftOut : 0;
		Append(candidate);
	}

	// ---- the frame's end --------------------------------------------------------------------------------------------
	void EndFrame(const FrameInfo& a_info)
	{
		if (!g_on.load(std::memory_order_relaxed) || !g_ring) {
			return;
		}
		auto& ring = *g_ring;
		auto& slot = *ring.slots[ring.head];
		const auto appended = slot.count.load(std::memory_order_acquire);
		const auto count = std::min(appended, slot.capacity);
		FillFrame(slot.frame, a_info);
		slot.frame.candidateCount = count;
		slot.frame.overflow = appended - count;
		slot.lamps.clear();
		if (a_info.lamps) {
			for (std::uint32_t i = 0; i < a_info.lamps->count && i < ShadowLights::LampList::kMax; ++i) {
				const auto& lamp = a_info.lamps->items[i];
				LampRecord record{};
				record.position[0] = lamp.position.x;
				record.position[1] = lamp.position.y;
				record.position[2] = lamp.position.z;
				record.reach = lamp.reach;
				record.volume[0] = lamp.volume.center.x;
				record.volume[1] = lamp.volume.center.y;
				record.volume[2] = lamp.volume.center.z;
				record.volume[3] = lamp.volume.fRadius;
				record.flags = (lamp.spot ? 1u : 0u) | (lamp.emptied ? 2u : 0u) | (lamp.steady ? 4u : 0u);
				slot.lamps.push_back(record);
			}
		}
		CopyHiZ(ring, slot, a_info.context ? a_info.context->snapshot : nullptr);
		AddObjects(ring, slot, count, slot.frame.clock);
		slot.used = true;
		ring.maxCount = std::max(ring.maxCount, appended);
		ring.candidates += count;
		ring.overflow += appended - count;
		if (!ring.filled) {
			ring.firstClock = slot.frame.clock;
		}
		ring.lastClock = slot.frame.clock;
		ring.head = ++ring.filled;
		if (ring.filled >= static_cast<std::uint32_t>(ring.slots.size())) {
			// The frame limit: the recording saves itself (the walk's threads are done with this frame; nothing appends).
			Notify(std::format("frame limit reached ({} frames)", ring.filled));
			Stop();
			return;
		}
		OpenSlot(ring);
	}
}
