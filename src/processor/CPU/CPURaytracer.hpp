#pragma once

#include "../raytracer.hpp"

#include <memory>
#include <cmath>
#include <math.h>

class CPURaytracer : public Raytracer
{
public:
	std::vector<float> trace(std::shared_ptr<Scene> s, std::shared_ptr<Camera> c);
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