#pragma once

#include <glm/glm.hpp>

struct RayHit
{
	glm::vec3 hitPosition;
	glm::vec3 normal;
};

struct Ray
{
	glm::vec3 origin;
	glm::vec3 direction;

	bool isIntersecting;
	RayHit hitInfo;
};

class Raytracer
{
public:
	
private:
	
	
};