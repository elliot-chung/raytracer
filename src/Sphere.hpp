#pragma once

#include "DisplayObject.hpp"

#define SUBDIVX 16
#define SUBDIVY 12
#define PI 3.14159f

class Sphere : public DisplayObject
{
public:
	Sphere(glm::vec3 position = DEFAULT_POSITION, glm::quat rotation = DEFAULT_ROTATION, glm::vec3 scale = DEFAULT_SCALE) : DisplayObject(position, rotation, scale)
	{
		static std::vector<float> vertices;
		static std::vector<float> normals;
 		// Create the vertices/normals if they don't exist
		if (vertices.empty())
		{
			int vertCount = 3 * (SUBDIVX * (SUBDIVY + 1));
			vertices.reserve(vertCount);
			normals.reserve(vertCount);

			// Create the bottom vertices
			for (int i = 0; i < SUBDIVX; i++)
			{
				vertices.push_back(0.0f);
				vertices.push_back(-0.5f);
				vertices.push_back(0.0f);

				normals.push_back(0.0f);
				normals.push_back(-1.0f);
				normals.push_back(0.0f);
			}

			// Create the middle vertices
			for (int i = 1; i < SUBDIVY; i++)
			{
				for (int j = 0; j <= SUBDIVX; j++)
				{
					float theta = (float)i / SUBDIVY * PI;
					float phi = (float)j / SUBDIVX * 2 * PI;

					float x = 0.5f * sinf(theta) * cosf(phi);
					float y = -0.5f * cosf(theta);
					float z = 0.5f * sinf(theta) * sinf(phi);

					vertices.push_back(x);
					vertices.push_back(y);
					vertices.push_back(z);

					normals.push_back(x * 2);
					normals.push_back(y * 2);
					normals.push_back(z * 2);
				}
			}

			// Create the top vertices
			for (int i = 0; i < SUBDIVX; i++) 
			{
				vertices.push_back(0.0f);
				vertices.push_back(0.5f);
				vertices.push_back(0.0f);

				normals.push_back(0.0f);
				normals.push_back(1.0f); 
				normals.push_back(0.0f); 
			}
		}

		static std::vector<int> indices;
		// Create the indices if they don't exist
		if (indices.empty())
		{
			int polarTriangles = 2 * SUBDIVX;
			int quadTriangles = 2 * SUBDIVX * (SUBDIVY - 2); 
			indices.reserve(3 * (polarTriangles + quadTriangles)); 

			// Create the top triangle fan
			for (int i = 0; i < SUBDIVX; i++)
			{
				indices.push_back(i); 
				
				indices.push_back(i + SUBDIVX + 1); 
				indices.push_back(i + SUBDIVX); 
			}

			// Create the middle triangle strip
			for (int i = 0; i < SUBDIVY - 2; i++)
			{
				for (int j = 0; j < SUBDIVX; j++)
				{
					indices.push_back(SUBDIVX + (SUBDIVX + 1) * i + j);
					
					indices.push_back(SUBDIVX + (SUBDIVX + 1) * (i + 1) + (j + 1));
					indices.push_back(SUBDIVX + (SUBDIVX + 1) * (i + 1) + j);

					indices.push_back(SUBDIVX + (SUBDIVX + 1) * i + j);
					
					indices.push_back(SUBDIVX + (SUBDIVX + 1) * i + (j + 1));
					indices.push_back(SUBDIVX + (SUBDIVX + 1) * (i + 1) + (j + 1));
				}
			}
			int bottomStart = SUBDIVX + (SUBDIVX + 1) * (SUBDIVY - 2); 
			// Create the bottom triangle fan
			for (int i = 0; i < SUBDIVX; i++)
			{
				indices.push_back(bottomStart + i); 
				
				indices.push_back(bottomStart + i + 1); 
				indices.push_back(bottomStart + i + 1 + SUBDIVX);
				
			}
		}

		static std::vector<float> uvCoords;
		// Create the UV coordinates if they don't exist
		if (uvCoords.empty())
		{
			uvCoords.reserve(2 * vertices.size() / 3); 
			
			// Create the top UV coordinates
			for (int i = 0; i < SUBDIVX; i++)
			{
				uvCoords.push_back((float) (2 * i + 1) / (2 * SUBDIVX));
				uvCoords.push_back(0.0f); 
			}

			// Create the middle UV coordinates
			for (int i = 1; i < SUBDIVY; i++)
			{
				for (int j = 0; j <= SUBDIVX; j++) 
				{
					uvCoords.push_back((float) j / SUBDIVX); 
					uvCoords.push_back((float) i / SUBDIVY); 
				}
			}

			// Create the bottom UV coordinates
			for (int i = 0; i < SUBDIVX; i++)
			{
				uvCoords.push_back((float) (2 * i + 1) / (2 * SUBDIVX)); 
				uvCoords.push_back(1.0f);  
			}
		}
	

		static Mesh sMesh(vertices, indices, uvCoords, normals);  
		meshes[0] = std::pair(&sMesh, 0); 
		updateCompositeBounds();
	}
};
