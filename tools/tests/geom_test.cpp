// Standalone check of CBRO's view-plane and sphere-ray tests (same formulas as Occlusion.cpp / HiZ.cpp).
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <random>

struct View { float v[4]; };  // x0, x1, y0, y1 in NDC
constexpr float kScaleX = 1.3242f, kScaleY = 2.1187f;

bool OutsideByPlanes(float x, float y, float z, float r, const View& a_view)
{
	const auto outsidePlane = [&](float a_scale, float a_coord, float a_edge, bool a_upper) {
		const float side = a_scale * a_coord - a_edge * z;
		const float reach = r * std::sqrt(a_scale * a_scale + a_edge * a_edge);
		return a_upper ? side > reach : side < -reach;
	};
	return z + r < 0.0f ||
	       outsidePlane(kScaleX, x, a_view.v[0], false) || outsidePlane(kScaleX, x, a_view.v[1], true) ||
	       outsidePlane(kScaleY, y, a_view.v[2], false) || outsidePlane(kScaleY, y, a_view.v[3], true);
}

// Brute force: does any point of the sphere project inside the view (in front of the eye)?
bool VisibleBrute(float cx, float cy, float cz, float r, const View& a_view, std::mt19937& rng)
{
	std::uniform_real_distribution<float> u(-1.0f, 1.0f);
	for (int i = 0; i < 20000; ++i) {
		float px, py, pz;
		do {
			px = u(rng); py = u(rng); pz = u(rng);
		} while (px * px + py * py + pz * pz > 1.0f);
		px = cx + px * r; py = cy + py * r; pz = cz + pz * r;
		if (pz <= 1e-3f) continue;
		const float nx = kScaleX * px / pz, ny = kScaleY * py / pz;
		if (nx >= a_view.v[0] && nx <= a_view.v[1] && ny >= a_view.v[2] && ny <= a_view.v[3]) return true;
	}
	return false;
}

struct SphereRays
{
	float centerX, centerY, depth, radius, texelsPerTanX, texelsPerTanY, axisX, axisY;
	bool Misses(float a_x0, float a_y0, float a_x1, float a_y1) const
	{
		const float du = (std::clamp(centerX, a_x0, a_x1) - centerX) / texelsPerTanX;
		const float dv = (std::clamp(centerY, a_y0, a_y1) - centerY) / texelsPerTanY;
		const float lateral = depth * std::sqrt(du * du + dv * dv);
		const float tu = std::max(std::abs(a_x0 - axisX), std::abs(a_x1 - axisX)) / texelsPerTanX;
		const float tv = std::max(std::abs(a_y0 - axisY), std::abs(a_y1 - axisY)) / texelsPerTanY;
		return lateral > radius * std::sqrt(1.0f + tu * tu + tv * tv);
	}
};

int main()
{
	std::mt19937 rng(1234);
	int wrongOutside = 0, outsideCount = 0, trials = 0;
	std::uniform_real_distribution<float> pos(-3000.0f, 3000.0f), rad(1.0f, 800.0f), edge(-1.3f, 1.3f);
	for (int t = 0; t < 4000; ++t) {
		View view{ { -1.0f, 1.0f, -1.0f, 1.0f } };
		if (t % 2) {  // a turned view box
			float a = edge(rng), b = edge(rng), c = edge(rng), d = edge(rng);
			view = { { std::min(a, b), std::max(a, b) + 0.05f, std::min(c, d), std::max(c, d) + 0.05f } };
		}
		const float x = pos(rng), y = pos(rng), z = pos(rng), r = rad(rng);
		++trials;
		if (OutsideByPlanes(x, y, z, r, view)) {
			++outsideCount;
			if (VisibleBrute(x, y, z, r, view, rng)) {
				++wrongOutside;
				std::printf("WRONG outside: c=(%.0f,%.0f,%.0f) r=%.0f view=[%.2f,%.2f]x[%.2f,%.2f]\n", x, y, z, r, view.v[0], view.v[1], view.v[2], view.v[3]);
			}
		}
	}
	std::printf("plane test: %d trials, %d judged outside, %d of them wrongly (must be 0)\n", trials, outsideCount, wrongOutside);

	// Sphere-ray miss test vs brute force over texel rectangles (320x200 Hi-Z, level-0 texel coords).
	const float W = 320.0f, H = 200.0f;
	int wrongMiss = 0, misses = 0, rects = 0;
	std::uniform_real_distribution<float> dep(60.0f, 5000.0f), lat(-1.0f, 1.0f), fr(0.01f, 0.5f), tex(0.0f, 320.0f), tey(0.0f, 200.0f), sz(0.5f, 40.0f);
	for (int t = 0; t < 20000; ++t) {
		const float z = dep(rng);
		const float r = std::min(z * fr(rng), z - 50.0f);
		if (r <= 0.0f) continue;
		const float x = lat(rng) * z / kScaleX, y = lat(rng) * z / kScaleY;
		SphereRays rays{ (kScaleX * x / z * 0.5f + 0.5f) * W, (0.5f - kScaleY * y / z * 0.5f) * H, z, r, kScaleX * W * 0.5f, kScaleY * H * 0.5f, W * 0.5f, H * 0.5f };
		const float x0 = tex(rng), y0 = tey(rng), x1 = x0 + sz(rng), y1 = y0 + sz(rng);
		++rects;
		if (!rays.Misses(x0, y0, x1, y1)) continue;
		++misses;
		// brute: sample rays through the rectangle; any that hits the sphere makes the "miss" wrong
		bool hit = false;
		for (int i = 0; i <= 24 && !hit; ++i) {
			for (int j = 0; j <= 24 && !hit; ++j) {
				const float px = x0 + (x1 - x0) * i / 24.0f, py = y0 + (y1 - y0) * j / 24.0f;
				const float ndcX = px / W * 2.0f - 1.0f, ndcY = 1.0f - py / H * 2.0f;
				const float dx = ndcX / kScaleX, dy = ndcY / kScaleY, dz = 1.0f;  // ray direction
				const float len = std::sqrt(dx * dx + dy * dy + dz * dz);
				const float ux = dx / len, uy = dy / len, uz = dz / len;
				const float proj = ux * x + uy * y + uz * z;
				const float d2 = x * x + y * y + z * z - proj * proj;
				hit = d2 <= r * r;
			}
		}
		if (hit) {
			++wrongMiss;
			if (wrongMiss < 10) std::printf("WRONG miss: c=(%.0f,%.0f,%.0f) r=%.0f rect=[%.1f,%.1f]x[%.1f,%.1f]\n", x, y, z, r, x0, x1, y0, y1);
		}
	}
	std::printf("ray test: %d rectangles, %d judged missed, %d of them wrongly (must be 0)\n", rects, misses, wrongMiss);
	return wrongOutside || wrongMiss;
}
