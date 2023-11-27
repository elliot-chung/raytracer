#pragma once

#include<vector>

#include <glm/glm.hpp>

#include "../Scene.hpp"

#define VEC3DOT(a, b) (a[0] * b[0] + a[1] * b[1] + a[2] * b[2])
#define VEC2DOT(a, b) (a[0] * b[0] + a[1] * b[1])

#define VEC3CROSS(a, b) { a[1] * b[2] - a[2] * b[1], -(a[0] * b[2] - a[2] * b[0]), a[0] * b[1] - a[1] * b[0] }

#define VEC3MUL(a, b) { a[0] * b[0], a[1] * b[1], a[2] * b[2] }
#define VEC2MUL(a, b) { a[0] * b[0], a[1] * b[1] }

#define VEC3ADD(a, b) { a[0] + b[0], a[1] + b[1], a[2] + b[2] }
#define VEC2ADD(a, b) { a[0] + b[0], a[1] + b[1] }

#define VEC3SUB(a, b) { a[0] - b[0], a[1] - b[1], a[2] - b[2] }
#define VEC2SUB(a, b) { a[0] - b[0], a[1] - b[1] }

#define VEC3MAXDIM(v) ((v.x > v.y) ? ((v.x > v.z) ? 0 : 2) : ((v.y > v.z) ? 1 : 2))

typedef float mvec3[3];
typedef float mvec2[2];



struct RayHit
{
	mvec3 hitPosition;
	float distance;
	mvec2 uv;

	mvec3 normal;
	mvec3 albedo;
	mvec3 roughness;
	mvec3 metal;
	mvec3 ao;
	mvec3 emission;
};

struct Ray
{
	mvec3	origin;
	mvec3	direction;
	int		bounceCount;

	bool	didHit;
	RayHit	hitInfo;
};

struct Color
{
	mvec3	values;
};

class Raytracer
{
public:
	virtual std::vector<float> trace(Scene& s, const std::vector<glm::vec3>& rayDirections, const glm::vec3& origin) {}

	inline void setBounceCount(int count) { bounceCount = count; }
	inline void setMaxDistance(float distance) { maxDistance = distance; }
	inline void setAoIntensity(float intensity) { aoIntensity = intensity; }
protected:
	int bounceCount = 0;
	float maxDistance = 0.0f;

	float aoIntensity = 0.03;
};