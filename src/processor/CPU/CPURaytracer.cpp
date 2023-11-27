#include "CPURaytracer.hpp"
#include <cmath>
#include <math.h>

std::vector<float> CPURaytracer::trace(Scene& scene, const std::vector<glm::vec3>& rayDirections, const glm::vec3& origin)
{
	auto objects = scene.getAllObjects();
	std::vector<float> output(rayDirections.size() * 3, 1.0f);
	int i = 0;
	for (auto direction : rayDirections)
	{
		Ray ray = {};
		ray.origin[0] = origin.x;
		ray.origin[1] = origin.y;
		ray.origin[2] = origin.z;
		ray.direction[0] = direction.x;
		ray.direction[1] = direction.y;
		ray.direction[2] = direction.z;
		ray.bounceCount = bounceCount;

		Color color = singleTrace(ray, objects);

		output[3 * i]     = color.values[0];
		output[3 * i + 1] = color.values[1];
		output[3 * i + 2] = color.values[2];
	}
	return output;
}



Color CPURaytracer::singleTrace(Ray& ray, const std::vector<DisplayObject*>& objects)
{
	for (DisplayObject* object : objects)
	{
		if (!intersectsBoundingBox(ray, object->getMinBound(), object->getMaxBound())) continue; // Skip if ray doesn't intersect BB
		getIntersectionPoint(ray, object);

		Color light = {};       
		light.values[0] = 0.03;
		light.values[1] = 0.03;
		light.values[2] = 0.03;			// slight skylight

		if (ray.didHit)  // Get the ambient light values from the hit location
		{
			mvec3 aoVec = { aoIntensity, aoIntensity, aoIntensity };
			mvec3 ambientColor = VEC3MUL(ray.hitInfo.albedo, ray.hitInfo.ao);
			mvec3 ambient = VEC3MUL(ambientColor, aoVec);
			light.values[0] = ambient[0];
			light.values[1] = ambient[1];
			light.values[2] = ambient[2];
		}

		if (ray.bounceCount == 0 || !ray.didHit) // Return light color if last bounce or if ray full miss
		{
			return light;
		}

		Ray bounceRay = {};
		bounceRay.origin[0] = ray.hitInfo.hitPosition[0];
		bounceRay.origin[1] = ray.hitInfo.hitPosition[1];
		bounceRay.origin[2] = ray.hitInfo.hitPosition[2];
		ray.bounceCount--;
		

		return light;

	}
}

bool CPURaytracer::intersectsBoundingBox(const Ray& ray, const glm::vec3& minBound, const glm::vec3& maxBound)
{
	return true;
}

void CPURaytracer::getIntersectionPoint(Ray& ray, DisplayObject* object)
{
	
	glm::mat4 model = object->getModelMatrix();
	Mesh* mesh = object->getMesh();
	auto vertices = mesh->getVertices();
	auto indices  = mesh->getIndices();
	auto uvCoords = mesh->getUVCoords();

	int closestTriangle = -1;
	glm::vec3 barycentricCoords(0.0f);
	float minDistance = INFINITY;

	// Loop over all the triangles in object
	for (int i = 0; i < mesh->getTriangleCount(); i++)
	{
		int i0 = indices[3 * i];
		int i1 = indices[3 * i + 1];
		int i2 = indices[3 * i + 2];

		glm::vec3 v0( vertices[i0], vertices[i0 + 1], vertices[i0 + 2] );
		glm::vec3 v1( vertices[i1], vertices[i1 + 1], vertices[i1 + 2] );
		glm::vec3 v2( vertices[i2], vertices[i2 + 1], vertices[i2 + 2] );
		v0 = glm::vec3(model * glm::vec4(v0, 1.0f));
		v1 = glm::vec3(model * glm::vec4(v1, 1.0f));
		v2 = glm::vec3(model * glm::vec4(v2, 1.0f));
		
		glm::vec3 bc(0.0f);
		float dist = distToTriangle(ray, v0, v1, v2, bc);
		if (dist < minDistance)
		{
			minDistance = dist;
			closestTriangle = i;
			barycentricCoords = bc;
		}
	}

	// Populate ray with material data from object by using the barycentric coords
	int i0 = indices[3 * closestTriangle];
	int i1 = indices[3 * closestTriangle + 1];
	int i2 = indices[3 * closestTriangle + 2];

	glm::vec3 v0(vertices[i0], vertices[i0 + 1], vertices[i0 + 2]);
	glm::vec3 v1(vertices[i1], vertices[i1 + 1], vertices[i1 + 2]);
	glm::vec3 v2(vertices[i2], vertices[i2 + 1], vertices[i2 + 2]);

	glm::vec2 uv0(uvCoords[i0], uvCoords[i0 + 1]);
	glm::vec2 uv1(uvCoords[i1], uvCoords[i1 + 1]);
	glm::vec2 uv2(uvCoords[i2], uvCoords[i2 + 1]);

	glm::vec3 interpPosition = barycentricCoords.x * v0 + barycentricCoords.y * v1 + barycentricCoords.z * v2;
	glm::vec2 interpUVCoords = barycentricCoords.x * uv0 + barycentricCoords.y * uv1 + barycentricCoords.z * uv2;

	ray.didHit = true;
	ray.hitInfo.hitPosition[0] = interpPosition.x;
	ray.hitInfo.hitPosition[1] = interpPosition.y;
	ray.hitInfo.hitPosition[2] = interpPosition.z;
	ray.hitInfo.distance = minDistance;

	// Return false if no triangles were hit, true otherwise
}

// https://www.pbr-book.org/3ed-2018/Shapes/Triangle_Meshes#fragment-Computeedgefunctioncoefficientsmonoe0monoe1andmonoe2-0
// Some of the code in this function can be moved out into the Ray struct
// Any value that doesn't depend on the triangle vertices can be precomputed to save time
float CPURaytracer::distToTriangle(const Ray& ray, glm::vec3& v0, glm::vec3& v1, glm::vec3& v2, glm::vec3& barycentricCoords)
{
	// Translate vertices by negative ray origin (move ray/triangle structure such that ray starts at origin)
	glm::vec3 v0t = glm::vec3(v0.x - ray.origin[0], v0.y - ray.origin[1], v0.z - ray.origin[2]);
	glm::vec3 v1t = glm::vec3(v1.x - ray.origin[0], v1.y - ray.origin[1], v1.z - ray.origin[2]);
	glm::vec3 v2t = glm::vec3(v2.x - ray.origin[0], v2.y - ray.origin[1], v2.z - ray.origin[2]);

	glm::vec3 d(ray.direction[0], ray.direction[1], ray.direction[2]);

	// Should all be precomputed
	float absDirx = abs(ray.direction[0]);
	float absDiry = abs(ray.direction[1]);
	float absDirz = abs(ray.direction[2]);
	int kz = absDirx > absDiry ? (absDirx > absDirz ? 0 : 2) : (absDiry > absDirz ? 1 : 2);
	int kx = kz == 2 ? 0 : kz + 1;
	int ky = kx == 2 ? 0 : kx + 1;

	// Permute order of values such that the greatest component of direction is in the z axis
	// Follow same permutation for the triangle vertices
	d  = glm::vec3( d[kx],  d[ky],  d[kz]);
	v0t = glm::vec3(v0t[kx], v0t[ky], v0t[kz]);
	v1t = glm::vec3(v1t[kx], v1t[ky], v1t[kz]);
	v2t = glm::vec3(v2t[kx], v2t[ky], v2t[kz]);

	// Precompute this
	float sx = -d.x / d.z;
	float sy = -d.y / d.z;
	float sz = 1.f / d.z;

	// Apply shear transform to x/y values of vertices (z is done after point is confirmed to lie in the triangle)
	v0t.x += sx * v0t.z;
	v0t.y += sy * v0t.z;
	v1t.x += sx * v1t.z;
	v1t.y += sy * v1t.z;
	v2t.x += sx * v2t.z;
	v2t.y += sy * v2t.z;

	// Compute edge coefficients
	float e0 = v1t.x * v2t.y - v1t.y * v2t.x;
	float e1 = v2t.x * v0t.y - v2t.y * v0t.x;
	float e2 = v0t.x * v1t.y - v0t.y * v1t.x;

	if (e0 == 0.0f || e1 == 0.0f || e2 == 0.0f) // double precision recomputation in the unlikely case any edge coefficient is 0
	{
		double p2txp1ty = (double)v2t.x * (double)v1t.y;
		double p2typ1tx = (double)v2t.y * (double)v1t.x;
		e0 = (float)(p2typ1tx - p2txp1ty);
		double p0txp2ty = (double)v0t.x * (double)v2t.y;
		double p0typ2tx = (double)v0t.y * (double)v2t.x;
		e1 = (float)(p0typ2tx - p0txp2ty);
		double p1txp0ty = (double)v1t.x * (double)v0t.y;
		double p1typ0tx = (double)v1t.y * (double)v0t.x;
		e2 = (float)(p1typ0tx - p1txp0ty);
	}

	// Check if point lies in triangle
	if ((e0 < 0 || e1 < 0 || e2 < 0) && (e0 > 0 || e1 > 0 || e2 > 0))
		return INFINITY;
	float det = e0 + e1 + e2;
	if (det == 0)
		return INFINITY;

	// Check point distance from origin
	v0t.z *= sz;
	v1t.z *= sz;
	v2t.z *= sz;
	float tScaled = e0 * v0t.z + e1 * v1t.z + e2 * v2t.z;
	if (det < 0 && (tScaled >= 0 || tScaled < maxDistance * det))
		return INFINITY;
	else if (det > 0 && (tScaled <= 0 || tScaled > maxDistance * det))
		return INFINITY;

	// Calculate barycentric coords and parametric value (distance)
	float invDet = 1 / det;
	float b0 = e0 * invDet;
	float b1 = e1 * invDet;
	float b2 = e2 * invDet;
	float t = tScaled * invDet;

	barycentricCoords.x = b0;
	barycentricCoords.y = b1;
	barycentricCoords.z = b2;
	return t;
}

float CPURaytracer::randomValue(unsigned int& seed)
{
	seed = seed * 747796405 + 2891336453;
	unsigned int result = ((seed >> ((seed >> 28) + 4)) ^ seed) * 277803737;
	result = (result >> 22) ^ result;
	return result / 4294967295.0;
}

float CPURaytracer::randomValueNormalDistribution(unsigned int& seed)
{
	float theta = 2 * 3.1415926 * randomValue(seed);
	float rho = sqrt(-2 * log(randomValue(seed)));
	return rho * cos(theta);
}


