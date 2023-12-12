#pragma once

#include<vector>
#include<memory>

#include <glm/glm.hpp>

#include "../Scene.hpp"
#include "../Camera.hpp"
#include "../Material.hpp"


struct RayHit
{
	glm::vec3 hitPosition;
	float distance;
	glm::vec2 uv;

	std::string material;

	glm::mat3 tbnMatrix;

	glm::vec3 normal;
	glm::vec3 albedo;
	glm::vec3 roughness;
	glm::vec3 metal;
	glm::vec3 ao;
	glm::vec3 emission;
};

struct Ray
{
	glm::vec3	origin;
	glm::vec3	direction;
	int			bounceCount;

	bool		didHit;
	RayHit		hitInfo;
};

typedef glm::vec3 Color;

class Raytracer
{
public:
	inline void setBounceCount(int count) { bounceCount = count; }
	inline void setMaxDistance(float distance) { maxDistance = distance; }
	inline void setAOIntensity(float intensity) { aoIntensity = intensity; }
protected:
	int bounceCount = 0;
	float maxDistance = 0.0f;

	float aoIntensity = 0.01;
};