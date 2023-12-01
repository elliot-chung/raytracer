#include "CPURaytracer.hpp"

#include <iostream>
#include <glm/gtx/string_cast.hpp>

std::vector<float> CPURaytracer::trace(std::shared_ptr<Scene> scene, std::shared_ptr<Camera> camera)
{
	Scene::ObjectMap objects = scene->getObjects();
	std::vector<float> output(camera->getPixelCount() * 3, 1.0f);

	glm::vec3 origin = camera->getPosition();

	for (int y = 0; y < camera->getHeight(); y++ )
	{
		for (int x = 0; x < camera->getWidth(); x++)
		{
			glm::vec3 direction = camera->getRayDirection(x, y);
			Ray ray = {};
			ray.origin = origin;
			ray.direction = direction;
			ray.bounceCount = bounceCount;

			Color color = glm::vec3(0.0f);
			// Color color = singleTrace(ray, objects);

			int i = y * camera->getWidth() + x;

			output[3 * i] = color.r;
			output[3 * i + 1] = color.g;
			output[3 * i + 2] = color.b;
			i++;
		}
	}
	return output;
}

Color CPURaytracer::singleTrace(Ray& ray, const Scene::ObjectMap& objects)
{
	for (auto objPair : objects)
	{
		DisplayObject* object = objPair.second;
		if (!intersectsBoundingBox(ray, object->getMinBound(), object->getMaxBound())) continue; // Skip if ray doesn't intersect BB
		getIntersectionPoint(ray, object);
	}

	Color outgoingLight(0.1, 0.1, 0.1);       // Slight skylight

	if (!ray.didHit)  return outgoingLight;	

	// Set material data using closest hit object
	Material* material = Material::getMaterial(ray.hitInfo.material);
	setMaterialData(ray, material);


	glm::vec3 aoVec(aoIntensity);
	outgoingLight = ray.hitInfo.albedo * ray.hitInfo.ao * aoVec; // Ambient occlusion

	if (ray.bounceCount == 0) // Return light color if last bounce 
	{
		return outgoingLight;
	}

	// Create bounce ray according to material data
	Ray bounceRay = {};
	bounceRay.origin = ray.hitInfo.hitPosition;
	bounceRay.bounceCount = ray.bounceCount - 1;
	


	// Color incomingLight = singleTrace(bounceRay, objects);
		
	
	return outgoingLight;
}

// Credit https://tavianator.com/2022/ray_box_boundary.html
bool CPURaytracer::intersectsBoundingBox(const Ray& ray, const glm::vec3& minBound, const glm::vec3& maxBound)
{
	float tmin = 0.0, tmax = ray.didHit ? ray.hitInfo.distance : INFINITY;

	glm::vec3 dirInv = 1.0f / ray.direction;

	for (int d = 0; d < 3; ++d) {
		bool sign = signbit(ray.direction[d]);
		float bmin = sign ? maxBound[d] : minBound[d];
		float bmax = sign ? minBound[d] : maxBound[d];

		float dmin = (bmin - ray.origin[d]) * dirInv[d];
		float dmax = (bmax - ray.origin[d]) * dirInv[d];

		tmin = max(dmin, tmin);
		tmax = min(dmax, tmax);
	}

	return tmin < tmax;
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
	float minDistance = ray.didHit ? ray.hitInfo.distance : INFINITY;

	// Loop over all the triangles in object
	for (int i = 0; i < mesh->getTriangleCount(); i++)
	{
		int i0 = indices[3 * i] * 3;
		int i1 = indices[3 * i + 1] * 3;
		int i2 = indices[3 * i + 2] * 3;

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

	if (closestTriangle == -1) return;   // Ray missed or only hit further objects, exit early


	// Calculate interpolated position and uv coords
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

	// Update ray info
	ray.didHit = true;
	ray.hitInfo.hitPosition = interpPosition;
	ray.hitInfo.distance = minDistance;
	ray.hitInfo.uv = interpUVCoords;
	ray.hitInfo.material = object->getMaterialName();
}

void CPURaytracer::setMaterialData(Ray& ray, Material* material)
{
	ray.hitInfo.normal = material->getNormal(ray.hitInfo.uv.x, ray.hitInfo.uv.y);
	ray.hitInfo.albedo = glm::vec3(material->getAlbedo(ray.hitInfo.uv.x, ray.hitInfo.uv.y));
	ray.hitInfo.roughness = glm::vec3(material->getRoughness(ray.hitInfo.uv.x, ray.hitInfo.uv.y));
	ray.hitInfo.metal = glm::vec3(material->getMetal(ray.hitInfo.uv.x, ray.hitInfo.uv.y));
	ray.hitInfo.ao = material->getAmbientOcclusion(ray.hitInfo.uv.x, ray.hitInfo.uv.y);
	ray.hitInfo.emission = material->getEmissionColor(ray.hitInfo.uv.x, ray.hitInfo.uv.y) * material->getEmissionStrength();
}

// https://www.pbr-book.org/3ed-2018/Shapes/Triangle_Meshes#fragment-Computeedgefunctioncoefficientsmonoe0monoe1andmonoe2-0
// Some of the code in this function can be moved out into the Ray struct
// Any value that doesn't depend on the triangle vertices can be precomputed to save time
float CPURaytracer::distToTriangle(const Ray& ray, glm::vec3& v0, glm::vec3& v1, glm::vec3& v2, glm::vec3& barycentricCoords)
{
	// Translate vertices by negative ray origin (move ray/triangle structure such that ray starts at origin)
	glm::vec3 v0t = v0 - ray.origin;
	glm::vec3 v1t = v1 - ray.origin;
	glm::vec3 v2t = v2 - ray.origin;

	glm::vec3 d(ray.direction);

	// Should all be precomputed
	float absDirx = abs(ray.direction[0]);
	float absDiry = abs(ray.direction[1]);
	float absDirz = abs(ray.direction[2]);
	int kz = absDirx > absDiry ? (absDirx > absDirz ? 0 : 2) : (absDiry > absDirz ? 1 : 2);
	int kx = kz == 2 ? 0 : kz + 1;
	int ky = kx == 2 ? 0 : kx + 1;

	// Permute order of values such that the greatest component of direction is in the z axis
	// Follow same permutation for the triangle vertices
	d   = glm::vec3(  d[kx],   d[ky],   d[kz]);
	v0t = glm::vec3(v0t[kx], v0t[ky], v0t[kz]);
	v1t = glm::vec3(v1t[kx], v1t[ky], v1t[kz]);
	v2t = glm::vec3(v2t[kx], v2t[ky], v2t[kz]);

	// Precomputable
	float sz = 1.f / d.z;
	float sx = -d.x * sz;
	float sy = -d.y * sz;
	

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


