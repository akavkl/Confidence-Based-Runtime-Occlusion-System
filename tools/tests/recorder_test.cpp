// Offline check of the recording format (src/Core/RecorderFormat.h, compiled as is): writes a small recording the way
// Core/Recorder does (frames in two modes, candidates of every origin, lamps, hi-z blobs, an object table with a string
// table) to the path given, then tools/recording.py reads it back ("check" verifies the sizes and offsets; the other
// commands run on it). The file is the reader's fixture: the field lists in the JSON must decode every record exactly.
// Build: cl /std:c++latest /O2 /EHsc /I src tools\tests\recorder_test.cpp   (VS x64 prompt)
// Run:   recorder_test.exe <out.cbrorec> && python -I tools\recording.py <out.cbrorec> check
#include <cstdio>
#include <fstream>
#include <random>

#include "Core/RecorderFormat.h"

using namespace CBRO::Core::RecorderFormat;

int main(int argc, char** argv)
{
	if (argc < 2) {
		std::printf("usage: recorder_test <out.cbrorec>\n");
		return 2;
	}
	Recording rec;
	rec.hiZWidth = 8;
	rec.hiZHeight = 4;
	rec.json = Json(2, rec.hiZWidth, rec.hiZHeight, "\"cbro\":\"test\",\"recordedAt\":\"now\",\"frameLimit\":6,\"hiZLevel\":2");
	AddString(rec.strings, "");
	const auto rttiFade = AddString(rec.strings, "BSFadeNode");
	const auto rttiShape = AddString(rec.strings, "BSTriShape");
	const auto name1 = AddString(rec.strings, "Wall01");
	const auto name2 = AddString(rec.strings, "Rail \"quoted\"");

	std::mt19937 rng(7);
	const int frames = 6;
	for (int i = 0; i < frames; ++i) {
		FrameRecord f{};
		f.clock = 100 + i;
		f.renderFrame = 1000 + i;
		f.location = 1;
		f.mode = i < 3 ? 0 : 2;  // three previs frames, three CBRO frames
		f.settled = 1;
		f.cull = f.mode != 0;
		f.eye[0] = 10.0f * i;
		f.eye[1] = 5.0f;
		f.eye[2] = 100.0f;
		f.rotate[3] = 1.0f;  // yaw 0 for every frame (the diff matches by distance alone)
		f.view[0] = -1.0f;
		f.view[1] = 1.0f;
		f.view[2] = -1.0f;
		f.view[3] = 1.0f;
		f.sunState = 2;
		f.hiZWidth0 = 32;
		f.hiZHeight0 = 16;
		f.candidateOffset = rec.candidates.size();
		// the same five objects in every frame; previs frames register four, CBRO frames decide all and register three
		const std::uint64_t objects[]{ 0x1000, 0x2000, 0x3000, 0x4000, 0x5000 };
		if (f.mode == 0) {
			for (int k = 0; k < 4; ++k) {
				Candidate c{};
				c.object = objects[k];
				c.origin = static_cast<std::uint8_t>(Origin::kRegistered);
				c.kind = 2;
				c.source = 3;
				c.path = kPathNone;
				rec.candidates.push_back(c);
			}
		} else {
			for (int k = 0; k < 5; ++k) {
				Candidate c{};
				c.object = objects[k];
				c.bound[0] = 100.0f * k;
				c.bound[3] = 50.0f;
				c.origin = static_cast<std::uint8_t>(k == 4 ? Origin::kWalkSkipped : k == 3 ? Origin::kChild : Origin::kWalkTop);
				c.kind = 1;
				c.site = 3;
				if (k < 3) {
					c.outcome = 3;  // kept
					c.verdict = 0;
					c.sun = 3;
					c.source = 0;
					c.path = kPathNone;
					c.flags = kJudged;
				} else if (k == 3) {
					c.outcome = 2;  // hidden, confirmed: rejected in every view
					c.verdict = 1;
					c.streak = 2;
					c.sun = 1;
					c.source = 2;
					c.path = 1;
					c.flags = kJudged | kLeftOut;
				} else {
					c.source = 3;
					c.path = 5;
					c.flags = kLeftOut;
				}
				rec.candidates.push_back(c);
			}
			Candidate node{};
			node.object = 0x9000;
			node.origin = static_cast<std::uint8_t>(Origin::kNodePruned);
			node.kind = 1;
			node.site = 3;
			node.extra = 120;
			node.flags = kLeftOut | kOutside;
			node.path = 8;
			node.source = 3;
			rec.candidates.push_back(node);
			for (int k = 0; k < 3; ++k) {
				Candidate c{};
				c.object = objects[k];
				c.origin = static_cast<std::uint8_t>(Origin::kRegistered);
				c.kind = 2;
				c.source = 3;
				c.path = kPathNone;
				rec.candidates.push_back(c);
			}
			Candidate refused{};
			refused.object = objects[3];
			refused.origin = static_cast<std::uint8_t>(Origin::kRefused);
			refused.kind = 2;
			refused.source = 3;
			refused.path = 2;
			refused.flags = kLeftOut;
			rec.candidates.push_back(refused);
		}
		f.candidateCount = static_cast<std::uint32_t>(rec.candidates.size() - f.candidateOffset);
		f.lampOffset = static_cast<std::uint32_t>(rec.lamps.size());
		LampRecord lamp{};
		lamp.position[0] = 1.0f;
		lamp.reach = 300.0f;
		lamp.flags = 1;
		rec.lamps.push_back(lamp);
		f.lampCount = 1;
		if (f.mode != 0) {
			std::vector<float> blob(static_cast<std::size_t>(rec.hiZWidth) * rec.hiZHeight * 2);
			std::uniform_real_distribution<float> depth(0.0f, 1.0f);
			for (auto& v : blob) {
				v = depth(rng);
			}
			f.hiZBlob = static_cast<std::int32_t>(rec.hiZ.size());
			rec.hiZ.push_back(std::move(blob));
		} else {
			f.hiZBlob = -1;
		}
		rec.frames.push_back(f);
	}
	const std::uint64_t objects[]{ 0x1000, 0x2000, 0x3000, 0x4000, 0x5000, 0x9000 };
	for (int k = 0; k < 6; ++k) {
		ObjectRecord o{};
		o.object = objects[k];
		o.parent = k == 5 ? 0 : 0x9000;
		o.flags = 0x0E;
		o.refID = 0x00010000u + k;
		o.baseID = 0x00020000u + k;
		o.rtti = k == 5 ? rttiFade : rttiShape;
		o.name = k % 2 ? name2 : name1;
		o.bound[3] = 50.0f;
		o.baseType = 32;
		o.isNode = k == 5;
		o.isFadeNode = k == 5;
		o.isGeometry = k != 5;
		o.firstClock = 100;
		rec.objects.push_back(o);
	}

	std::ofstream out(argv[1], std::ios::binary | std::ios::trunc);
	if (!out || !Write(out, rec)) {
		std::printf("FAIL: cannot write %s\n", argv[1]);
		return 1;
	}
	out.close();
	std::printf("wrote %s: %zu frames, %zu candidates, %zu lamps, %zu hi-z blobs, %zu objects, %zu string bytes\n",
		argv[1], rec.frames.size(), rec.candidates.size(), rec.lamps.size(), rec.hiZ.size(), rec.objects.size(), rec.strings.size());
	return 0;
}
