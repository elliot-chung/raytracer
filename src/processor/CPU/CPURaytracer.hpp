#pragma once

#include "../raytracer.hpp"

#include <memory>
#include <cmath>
#include <math.h>

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

class CPURaytracer : public Raytracer
{
public:
	std::vector<float> raytrace(std::shared_ptr<Scene> s, std::shared_ptr<Camera> c);
private:
	unsigned int randomSeed = 0;
	bool debug = false;

	Color singleTrace(Ray& ray, const Scene::ObjectMap& objects);

	bool intersectsBoundingBox(const Ray& ray, const glm::vec3& minBound, const glm::vec3& maxBound);

	void getIntersectionPoint(Ray& ray, DisplayObject* object);

	void setMaterialData(Ray& ray, Material* material);

	float distToTriangle(const Ray& ray, glm::vec3& v0, glm::vec3& v1, glm::vec3& v2, glm::vec3& barycentricCoords);

	float randomValue(unsigned int& seed);

	float randomValueNormalDistribution(unsigned int& seed);

	glm::vec3 randomUnitVector(unsigned int& seed);

	glm::vec3 randomUnitVectorInHemisphere(unsigned int& seed, const glm::vec3& normal);
	
	inline float min(float a, float b) { return a < b ? a : b; }
	inline float max(float a, float b) { return a > b ? a : b; }
};