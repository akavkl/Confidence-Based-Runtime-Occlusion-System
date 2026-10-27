#pragma once

// The recording file (Core/Recorder, bRecorder; tools/recording.py reads it): engine-free, so tools/tests/recorder_test.cpp
// can write one and the Python reader can be checked against it without the game.
//
// Layout, all little-endian, sections back to back after the fixed header, each starting 8-byte aligned:
//   FileHeader | JSON (the enum names, the settings, and every record struct's field list, so a reader builds its
//   dtypes from the file) | frames (FrameRecord x frameCount) | candidates (Candidate x candidateCount, a frame's run
//   at its candidateOffset) | lamps (LampRecord x lampCount) | hi-z blobs (hiZBytes: per blob, max-reduced then
//   min-reduced depth of one pyramid level, width x height floats each, a frame's at hiZBlob x blobBytes) | objects
//   (ObjectRecord x objectCount: the metadata of every object the candidates name, by address) | strings (NUL-separated,
//   offsets from the records).
// A frame's candidates are what CBRO decided in that frame, in the order it decided them: enough to re-run the frame's
// decisions against the recorded depth later (replay is not part of this version).
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <ostream>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace CBRO::Core::RecorderFormat
{
	inline constexpr char          kMagic[8]{ 'C', 'B', 'R', 'O', 'R', 'E', 'C', '\0' };
	inline constexpr std::uint32_t kVersion = 1;

	// Where a candidate came from (Candidate::origin).
	enum class Origin : std::uint8_t
	{
		kWalkSkipped,   // the entry walk's cheap tests left it out unoffered (no verdict)
		kWalkMarker,    // the entry walk filed it unjudged: a group marker was pending on its block (judged at Block::Add as a child)
		kWalkTop,       // the entry walk offered it to the top-level decision
		kEngineTop,     // the engine's own Group::Add offered it to the top-level decision
		kChild,         // a Block::Add not decided at the top level (the engine's child pushes, unjudged filings)
		kRegistered,    // the main accumulator registered it (both modes: what the main view draws)
		kRefused,       // ... CBRO refused the registration (a main-view drop)
		kNodePruned,    // a cell node, container or root left out whole by node pruning (extra = its entry count)
		kNodeWalked,    // ... kept, its entries walked or filed (extra = its entry count where scanned)
		kCount
	};
	inline constexpr const char* kOriginNames[]{ "walk skipped", "walk marker", "walk top", "engine top", "child", "registered", "refused", "node pruned", "node walked" };

	// Candidate::flags.
	enum Flags : std::uint16_t
	{
		kForced = 1 << 0,       // filed force-visible (for the trace, or as the player's, or a sky mesh)
		kPlayer = 1 << 1,       // the player's 3D
		kAlwaysDraw = 1 << 2,   // the engine's always-draw bit (flags bit 11)
		kSky = 1 << 3,          // a sky mesh
		kLight = 1 << 4,        // a light (never skipped as out of view)
		kInherited = 1 << 5,    // its parent's decision inherited (no verdict of its own)
		kMoved = 1 << 6,        // the entry walk: the bound changed since the last walk
		kTraceTurn = 1 << 7,    // the entry walk: the node is the trace's this frame (nothing skipped cheaply)
		kJudged = 1 << 8,       // a verdict record exists (outcome, verdict, sun, streak, source are meaningful)
		kLeftOut = 1 << 9,      // left out of the main view (skipped, rejected, dropped, refused, pruned)
		kGroup1 = 1 << 10,      // a main-only group (else group 0, or unknown)
		kObserveOnly = 1 << 11,  // decide-only mode: nothing was hidden
		kOutside = 1 << 12,     // node pruning: every entry outside the view
		kLeased = 1 << 13,      // the entry walk: left out whole with its cluster, untouched (v1.98)
		kCertified = 1 << 14,   // the entry walk: left out on its own certificate, untouched (v1.99 only: certificates went in v1.99.1)
	};
	inline constexpr const char* kFlagNames[]{ "forced", "player", "always draw", "sky", "light", "inherited", "moved", "trace turn", "judged", "left out", "group 1", "observe only", "outside", "leased", "certified" };

	inline constexpr const char* kOutcomeNames[]{ "none", "outside", "hidden", "kept" };
	inline constexpr const char* kVerdictNames[]{ "visible", "hidden", "edge", "outside", "behind", "near", "invalid" };
	inline constexpr const char* kSunNames[]{ "none", "unneeded", "behind", "needed", "unknown", "lamp unneeded" };
	inline constexpr const char* kSourceNames[]{ "fresh", "reused", "async map", "not judged" };
	inline constexpr const char* kPathNames[]{ "rejected", "rejected in every view", "main view drop", "parent dropped", "never filed: hidden", "never filed: out of view", "never filed: no view needs it", "out of view (engine's frustum test)", "never filed: its node pruned whole", "none" };
	inline constexpr std::uint8_t kPathNone = 9;
	inline constexpr const char* kKindNames[]{ "main only", "sun shared", "unknown" };
	inline constexpr const char* kModeNames[]{ "previs", "CBRO feed", "CBRO classic" };
	inline constexpr const char* kSunStateNames[]{ "unknown", "off", "on" };

#pragma pack(push, 1)
	struct FileHeader
	{
		char          magic[8];
		std::uint32_t version;
		std::uint32_t jsonBytes;
		std::uint32_t frameCount;
		std::uint32_t objectCount;
		std::uint64_t candidateCount;
		std::uint32_t lampCount;
		std::uint32_t hiZBlobs;
		std::uint64_t hiZBytes;
		std::uint64_t stringBytes;
		std::uint32_t frameBytes;      // sizeof(FrameRecord) (a reader checks its dtype against it)
		std::uint32_t candidateBytes;  // sizeof(Candidate)
		std::uint32_t lampBytes;       // sizeof(LampRecord)
		std::uint32_t objectBytes;     // sizeof(ObjectRecord)
		std::uint32_t blobBytes;       // one hi-z blob (0: none recorded)
		std::uint32_t reserved;
	};
	static_assert(sizeof(FileHeader) == 80);

	struct Candidate
	{
		std::uint64_t object;    // NiAVObject address (the key into the objects section)
		float         bound[4];  // the bound decided with (center, radius); 0 for registrations
		std::uint8_t  origin;    // Origin
		std::uint8_t  kind;      // group kind (kKindNames)
		std::uint8_t  outcome;   // Occlusion::Outcome
		std::uint8_t  verdict;   // Occlusion's Verdict
		std::uint8_t  sun;       // Occlusion::SunOutcome
		std::uint8_t  streak;    // consecutive frames hidden
		std::uint8_t  source;    // fresh / reused / async map / not judged
		std::uint8_t  path;      // the left-out path (kPathNames; kPathNone when kept)
		std::uint16_t flags;     // Flags
		std::uint16_t site;      // walk entries: the engine loop they stand in (3 = cell node 3, 9 = node 9, 254 = a container's member, 255 = a DrawWorld root's child); nodes: the index offered
		std::uint32_t extra;     // nodes: entries; entries: the record's flags | sun streak << 8
	};
	static_assert(sizeof(Candidate) == 40);

	struct FrameRecord
	{
		std::uint32_t clock;          // CBRO's frame clock
		std::uint32_t location;       // the A/B location counter
		std::uint64_t renderFrame;
		std::uint64_t candidateOffset;  // index into the candidates section
		std::uint32_t candidateCount;
		std::uint32_t overflow;       // candidates dropped: the frame's buffer was full
		std::uint32_t lampOffset;     // index into the lamps section
		std::uint32_t lampCount;
		std::int32_t  hiZBlob;        // index into the hi-z section (-1: none)
		std::uint8_t  mode;           // kModeNames
		std::uint8_t  settled;        // past the load and switch settle, no menu
		std::uint8_t  cull;           // CBRO culled this frame
		std::uint8_t  asyncValid;     // the async worker's map was usable
		std::uint8_t  sunState;       // kSunStateNames
		std::uint8_t  sunEpoch;
		std::uint8_t  observeOnly;
		std::uint8_t  lampsComplete;
		std::uint32_t viewEpoch;
		std::uint32_t readback;       // the depth frame's readback index
		float         eye[3];         // the current camera (WorldRoot camera) world position
		float         rotate[9];      // ... its world rotation, rows
		float         zoom[2];        // ... its frustum half-extents
		float         view[4];        // the current view in the depth frame's NDC {x0, x1, y0, y1}
		float         overhang[4];
		float         dilateMove;
		float         parallaxMove;
		float         angularSlack;
		float         sunDir[3];      // the sun's travel direction in the depth frame's view space
		float         sunReach;
		float         sunSpread;
		float         sunMargin;
		// The depth frame's camera (HiZ::Camera): what the recorded hi-z was rendered with.
		std::uint64_t depthFrame;     // the render frame the depth came from (0: no depth frame)
		std::uint32_t mergedFrames;
		float         mergeMove;
		float         viewProj[16];   // row-vector convention: clip = p * viewProj
		float         posAdjust[3];
		float         depthEye[3];
		float         depthRotate[9];
		float         viewDir[3];
		float         viewRight[3];
		float         viewUp[3];
		float         origin[3];      // view-space form (viewSpace != 0): origin, scales, depth mapping
		float         scaleX;
		float         scaleY;
		float         depthA;
		float         depthB;
		float         frustumX;
		float         frustumY;
		std::uint8_t  viewSpace;
		std::uint8_t  footprint;
		std::uint8_t  firstPersonAfter;
		std::uint8_t  cameraState;
		std::uint32_t hiZWidth0;      // the pyramid's level-0 size (the recorded level's is in the JSON)
		std::uint32_t hiZHeight0;
		std::uint32_t reserved[3];
	};
	static_assert(sizeof(FrameRecord) == 420);

	struct LampRecord
	{
		float         position[3];
		float         reach;
		float         volume[4];  // spot lamps: the lit volume tested (center, radius)
		std::uint32_t flags;      // bit 0 spot, bit 1 emptied, bit 2 steady
		std::uint32_t reserved;
	};
	static_assert(sizeof(LampRecord) == 40);

	struct ObjectRecord
	{
		std::uint64_t object;
		std::uint64_t parent;
		std::uint64_t flags;       // NiAVObject flags (+0x108) when first seen
		std::uint32_t refID;       // the owning reference (0: none found)
		std::uint32_t baseID;      // ... its base form
		std::uint32_t rtti;        // string offset: the NiRTTI name
		std::uint32_t name;        // string offset: the object's name
		float         bound[4];    // world bound when first seen
		std::uint8_t  baseType;    // the base form's type (ENUM_FORM_ID)
		std::uint8_t  isNode;
		std::uint8_t  isFadeNode;
		std::uint8_t  isGeometry;
		std::uint32_t firstClock;  // the frame clock it was first seen at
	};
	static_assert(sizeof(ObjectRecord) == 64);
#pragma pack(pop)

	// Everything a file holds, in memory (the recorder fills it from its ring at the stop; the test fills it by hand).
	struct Recording
	{
		std::string                    json;  // the JSON section (Json() below builds the standard one)
		std::vector<FrameRecord>       frames;
		std::vector<Candidate>         candidates;
		std::vector<LampRecord>        lamps;
		std::vector<std::vector<float>> hiZ;  // blobs, each 2 x width x height floats (max then min pyramid)
		std::uint32_t                  hiZWidth{ 0 };
		std::uint32_t                  hiZHeight{ 0 };
		std::vector<ObjectRecord>      objects;
		std::vector<char>              strings;  // NUL-separated; offset 0 is ""
		// The recorder's ring hands its frames' candidates and blobs over as runs (no copy): when these are set they
		// are written, in order, instead of `candidates` and `hiZ`.
		std::vector<std::span<const Candidate>> candidateRuns;
		std::vector<std::span<const float>>     hiZRuns;
	};

	namespace detail
	{
		inline void Escape(std::string& a_out, std::string_view a_text)
		{
			for (const char c : a_text) {
				switch (c) {
				case '"': a_out += "\\\""; break;
				case '\\': a_out += "\\\\"; break;
				case '\n': a_out += "\\n"; break;
				case '\r': a_out += "\\r"; break;
				case '\t': a_out += "\\t"; break;
				default:
					if (static_cast<unsigned char>(c) < 0x20) {
						char buf[8];
						std::snprintf(buf, sizeof(buf), "\\u%04x", static_cast<unsigned>(static_cast<unsigned char>(c)));
						a_out += buf;
					} else {
						a_out += c;
					}
				}
			}
		}

		inline void Names(std::string& a_out, const char* a_key, const char* const* a_names, std::size_t a_count)
		{
			a_out += '"';
			a_out += a_key;
			a_out += "\":[";
			for (std::size_t i = 0; i < a_count; ++i) {
				a_out += i ? ",\"" : "\"";
				Escape(a_out, a_names[i]);
				a_out += '"';
			}
			a_out += ']';
		}

		// A struct's fields as [name, numpy dtype, count] triples: the reader builds its dtype from them.
		struct Field
		{
			const char* name;
			const char* type;
			int         count;
		};

		inline void Fields(std::string& a_out, const char* a_key, const Field* a_fields, std::size_t a_count)
		{
			a_out += '"';
			a_out += a_key;
			a_out += "\":[";
			for (std::size_t i = 0; i < a_count; ++i) {
				char buf[96];
				std::snprintf(buf, sizeof(buf), "%s[\"%s\",\"%s\",%d]", i ? "," : "", a_fields[i].name, a_fields[i].type, a_fields[i].count);
				a_out += buf;
			}
			a_out += ']';
		}

		inline constexpr Field kCandidateFields[]{
			{ "object", "u8", 1 }, { "bound", "f4", 4 }, { "origin", "u1", 1 }, { "kind", "u1", 1 }, { "outcome", "u1", 1 }, { "verdict", "u1", 1 },
			{ "sun", "u1", 1 }, { "streak", "u1", 1 }, { "source", "u1", 1 }, { "path", "u1", 1 }, { "flags", "u2", 1 }, { "site", "u2", 1 }, { "extra", "u4", 1 }
		};
		inline constexpr Field kFrameFields[]{
			{ "clock", "u4", 1 }, { "location", "u4", 1 }, { "renderFrame", "u8", 1 }, { "candidateOffset", "u8", 1 }, { "candidateCount", "u4", 1 }, { "overflow", "u4", 1 },
			{ "lampOffset", "u4", 1 }, { "lampCount", "u4", 1 }, { "hiZBlob", "i4", 1 }, { "mode", "u1", 1 }, { "settled", "u1", 1 }, { "cull", "u1", 1 }, { "asyncValid", "u1", 1 },
			{ "sunState", "u1", 1 }, { "sunEpoch", "u1", 1 }, { "observeOnly", "u1", 1 }, { "lampsComplete", "u1", 1 }, { "viewEpoch", "u4", 1 }, { "readback", "u4", 1 },
			{ "eye", "f4", 3 }, { "rotate", "f4", 9 }, { "zoom", "f4", 2 }, { "view", "f4", 4 }, { "overhang", "f4", 4 }, { "dilateMove", "f4", 1 }, { "parallaxMove", "f4", 1 },
			{ "angularSlack", "f4", 1 }, { "sunDir", "f4", 3 }, { "sunReach", "f4", 1 }, { "sunSpread", "f4", 1 }, { "sunMargin", "f4", 1 }, { "depthFrame", "u8", 1 },
			{ "mergedFrames", "u4", 1 }, { "mergeMove", "f4", 1 }, { "viewProj", "f4", 16 }, { "posAdjust", "f4", 3 }, { "depthEye", "f4", 3 }, { "depthRotate", "f4", 9 },
			{ "viewDir", "f4", 3 }, { "viewRight", "f4", 3 }, { "viewUp", "f4", 3 }, { "origin", "f4", 3 }, { "scaleX", "f4", 1 }, { "scaleY", "f4", 1 }, { "depthA", "f4", 1 },
			{ "depthB", "f4", 1 }, { "frustumX", "f4", 1 }, { "frustumY", "f4", 1 }, { "viewSpace", "u1", 1 }, { "footprint", "u1", 1 }, { "firstPersonAfter", "u1", 1 },
			{ "cameraState", "u1", 1 }, { "hiZWidth0", "u4", 1 }, { "hiZHeight0", "u4", 1 }, { "reserved", "u4", 3 }
		};
		inline constexpr Field kLampFields[]{
			{ "position", "f4", 3 }, { "reach", "f4", 1 }, { "volume", "f4", 4 }, { "flags", "u4", 1 }, { "reserved", "u4", 1 }
		};
		inline constexpr Field kObjectFields[]{
			{ "object", "u8", 1 }, { "parent", "u8", 1 }, { "flags", "u8", 1 }, { "refID", "u4", 1 }, { "baseID", "u4", 1 }, { "rtti", "u4", 1 }, { "name", "u4", 1 },
			{ "bound", "f4", 4 }, { "baseType", "u1", 1 }, { "isNode", "u1", 1 }, { "isFadeNode", "u1", 1 }, { "isGeometry", "u1", 1 }, { "firstClock", "u4", 1 }
		};
	}

	// The JSON section: the names, the field lists, the hi-z level and whatever the caller adds (a_extra: JSON members
	// without the braces, e.g. "\"cbro\":\"1.95.0\",\"frames\":600").
	inline std::string Json(std::uint32_t a_hiZLevel, std::uint32_t a_hiZWidth, std::uint32_t a_hiZHeight, std::string_view a_extra)
	{
		using namespace detail;
		std::string out = "{\"format\":\"CBRO recording\",\"version\":";
		out += std::to_string(kVersion);
		out += ",\"enums\":{";
		Names(out, "origin", kOriginNames, std::size(kOriginNames));
		out += ',';
		Names(out, "flags", kFlagNames, std::size(kFlagNames));
		out += ',';
		Names(out, "outcome", kOutcomeNames, std::size(kOutcomeNames));
		out += ',';
		Names(out, "verdict", kVerdictNames, std::size(kVerdictNames));
		out += ',';
		Names(out, "sun", kSunNames, std::size(kSunNames));
		out += ',';
		Names(out, "source", kSourceNames, std::size(kSourceNames));
		out += ',';
		Names(out, "path", kPathNames, std::size(kPathNames));
		out += ',';
		Names(out, "kind", kKindNames, std::size(kKindNames));
		out += ',';
		Names(out, "mode", kModeNames, std::size(kModeNames));
		out += ',';
		Names(out, "sunState", kSunStateNames, std::size(kSunStateNames));
		out += "},\"structs\":{";
		Fields(out, "frame", kFrameFields, std::size(kFrameFields));
		out += ',';
		Fields(out, "candidate", kCandidateFields, std::size(kCandidateFields));
		out += ',';
		Fields(out, "lamp", kLampFields, std::size(kLampFields));
		out += ',';
		Fields(out, "object", kObjectFields, std::size(kObjectFields));
		out += "},\"hiz\":{\"level\":";
		out += std::to_string(a_hiZLevel);
		out += ",\"width\":";
		out += std::to_string(a_hiZWidth);
		out += ",\"height\":";
		out += std::to_string(a_hiZHeight);
		out += ",\"planes\":[\"max\",\"min\"]}";
		if (!a_extra.empty()) {
			out += ',';
			out += a_extra;
		}
		out += '}';
		return out;
	}

	// Appends a string to the table and returns its offset (0 for the empty string; no dedupe here: the recorder dedupes).
	inline std::uint32_t AddString(std::vector<char>& a_table, std::string_view a_text)
	{
		if (a_table.empty()) {
			a_table.push_back('\0');
		}
		if (a_text.empty()) {
			return 0;
		}
		const auto offset = static_cast<std::uint32_t>(a_table.size());
		a_table.insert(a_table.end(), a_text.begin(), a_text.end());
		a_table.push_back('\0');
		return offset;
	}

	// Writes the file. False when the stream failed (the caller reports); the stream's own buffering carries the I/O.
	inline bool Write(std::ostream& a_out, const Recording& a_recording)
	{
		const auto pad = [&](std::uint64_t a_written) {
			static constexpr char zeros[8]{};
			const auto rest = (8 - (a_written & 7)) & 7;
			if (rest) {
				a_out.write(zeros, static_cast<std::streamsize>(rest));
			}
			return a_written + rest;
		};
		const auto blobFloats = static_cast<std::uint64_t>(a_recording.hiZWidth) * a_recording.hiZHeight * 2;
		const bool runs = !a_recording.candidateRuns.empty();
		const bool blobRuns = !a_recording.hiZRuns.empty();
		std::uint64_t candidateCount = 0;
		if (runs) {
			for (const auto& run : a_recording.candidateRuns) {
				candidateCount += run.size();
			}
		} else {
			candidateCount = a_recording.candidates.size();
		}
		FileHeader header{};
		std::memcpy(header.magic, kMagic, sizeof(kMagic));
		header.version = kVersion;
		header.jsonBytes = static_cast<std::uint32_t>(a_recording.json.size());
		header.frameCount = static_cast<std::uint32_t>(a_recording.frames.size());
		header.objectCount = static_cast<std::uint32_t>(a_recording.objects.size());
		header.candidateCount = candidateCount;
		header.lampCount = static_cast<std::uint32_t>(a_recording.lamps.size());
		header.hiZBlobs = static_cast<std::uint32_t>(blobRuns ? a_recording.hiZRuns.size() : a_recording.hiZ.size());
		header.blobBytes = static_cast<std::uint32_t>(blobFloats * sizeof(float));
		header.hiZBytes = header.blobBytes * static_cast<std::uint64_t>(header.hiZBlobs);
		header.stringBytes = a_recording.strings.size();
		header.frameBytes = sizeof(FrameRecord);
		header.candidateBytes = sizeof(Candidate);
		header.lampBytes = sizeof(LampRecord);
		header.objectBytes = sizeof(ObjectRecord);
		a_out.write(reinterpret_cast<const char*>(&header), sizeof(header));
		std::uint64_t written = sizeof(header);
		a_out.write(a_recording.json.data(), static_cast<std::streamsize>(a_recording.json.size()));
		written = pad(written + a_recording.json.size());
		const auto section = [&](const void* a_data, std::uint64_t a_bytes) {
			if (a_bytes) {
				a_out.write(static_cast<const char*>(a_data), static_cast<std::streamsize>(a_bytes));
			}
			written = pad(written + a_bytes);
		};
		section(a_recording.frames.data(), a_recording.frames.size() * sizeof(FrameRecord));
		if (runs) {
			for (const auto& run : a_recording.candidateRuns) {
				if (!run.empty()) {
					a_out.write(reinterpret_cast<const char*>(run.data()), static_cast<std::streamsize>(run.size() * sizeof(Candidate)));
				}
				written += run.size() * sizeof(Candidate);
			}
			written = pad(written);
		} else {
			section(a_recording.candidates.data(), a_recording.candidates.size() * sizeof(Candidate));
		}
		section(a_recording.lamps.data(), a_recording.lamps.size() * sizeof(LampRecord));
		const auto blob = [&](const float* a_data, std::uint64_t a_floats) {
			if (a_floats != blobFloats) {
				return false;
			}
			if (blobFloats) {
				a_out.write(reinterpret_cast<const char*>(a_data), static_cast<std::streamsize>(blobFloats * sizeof(float)));
			}
			written += blobFloats * sizeof(float);
			return true;
		};
		if (blobRuns) {
			for (const auto& run : a_recording.hiZRuns) {
				if (!blob(run.data(), run.size())) {
					return false;
				}
			}
		} else {
			for (const auto& item : a_recording.hiZ) {
				if (!blob(item.data(), item.size())) {
					return false;
				}
			}
		}
		written = pad(written);
		section(a_recording.objects.data(), a_recording.objects.size() * sizeof(ObjectRecord));
		section(a_recording.strings.data(), a_recording.strings.size());
		a_out.flush();
		return static_cast<bool>(a_out);
	}
}
