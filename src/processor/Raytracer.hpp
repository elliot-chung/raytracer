#pragma once

#include<vector>
#include<memory>

#include <glm/glm.hpp>

#include "../Scene.hpp"
#include "../Camera.hpp"
#include "../Material.hpp"


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