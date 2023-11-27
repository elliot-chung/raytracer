#pragma once

#include <glm/glm.hpp>

#include <vector>
#include <string>

class Mesh
{
public:
	Mesh(std::vector<float>& vertices, std::vector<int>& indices, std::vector<float>& uvCoords) : minBound(INFINITY), maxBound(-INFINITY)
	{
		this->vertices		= vertices;
		this->indices	    = indices;
		this->uvCoords		= uvCoords;
		this->triangleCount = indices.size() / 3;
		calcBounds();
	}

	inline glm::vec3 getMinBound() { return minBound; }
	inline glm::vec3 getMaxBound() { return maxBound; }

	inline std::vector<float> getVertices() { return vertices; }
	inline std::vector<float> getUVCoords() { return uvCoords; }
	inline std::vector<int>	  getIndices()  { return indices; }
	
	inline int getTriangleCount() { return triangleCount; }

private:
	std::vector<float>	vertices;
	std::vector<float>  uvCoords;
	std::vector<int>	indices;

	glm::vec3 minBound; // In local coordinates
	glm::vec3 maxBound; 

	int triangleCount = 0;

	void calcBounds()
	{
		for (int i = 0; i < triangleCount; i++)
		{
			float x = vertices[i];
			float y = vertices[i + 1];
			float z = vertices[i + 2];

			if (minBound.x > x) minBound.x = x;
			if (minBound.y > y) minBound.y = y;
			if (minBound.z > z) minBound.z = z;

			if (maxBound.x < x) maxBound.x = x;
			if (maxBound.y < y) maxBound.y = y;
			if (maxBound.z < z) maxBound.z = z;
		}
	}
};