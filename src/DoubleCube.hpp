#pragma once

#include "DisplayObject.hpp"

// Debuggin object for multiple meshes
class DoubleCube : public DisplayObject
{
public:
	DoubleCube(glm::vec3 position = DEFAULT_POSITION, glm::quat rotation = DEFAULT_ROTATION, glm::vec3 scale = DEFAULT_SCALE) : DisplayObject(position, rotation, scale)
	{
		static std::vector<float> vertices = {
			-0.5f, -0.5f, -0.5f,
			 0.5f, -0.5f, -0.5f,
			 0.5f,  0.5f, -0.5f,
			 0.5f,  0.5f, -0.5f,
			-0.5f,  0.5f, -0.5f,
			-0.5f, -0.5f, -0.5f,

			 0.5f, -0.5f,  0.5f,
			-0.5f, -0.5f,  0.5f,
			-0.5f,  0.5f,  0.5f,
			-0.5f,  0.5f,  0.5f,
			 0.5f,  0.5f,  0.5f,
			 0.5f, -0.5f,  0.5f,

			-0.5f, -0.5f,  0.5f,
			-0.5f, -0.5f, -0.5f,
			-0.5f,  0.5f, -0.5f,
			-0.5f,  0.5f, -0.5f,
			-0.5f,  0.5f,  0.5f,
			-0.5f, -0.5f,  0.5f,

			 0.5f, -0.5f, -0.5f,
			 0.5f, -0.5f,  0.5f,
			 0.5f,  0.5f,  0.5f,
			 0.5f,  0.5f,  0.5f,
			 0.5f,  0.5f, -0.5f,
			 0.5f, -0.5f, -0.5f,

			 0.5f, -0.5f, -0.5f,
			-0.5f, -0.5f, -0.5f,
			-0.5f, -0.5f,  0.5f,
			-0.5f, -0.5f,  0.5f,
			 0.5f, -0.5f,  0.5f,
			 0.5f, -0.5f, -0.5f,

			 0.5f,  0.5f,  0.5f,
			-0.5f,  0.5f,  0.5f,
			-0.5f,  0.5f, -0.5f,
			-0.5f,  0.5f, -0.5f,
			 0.5f,  0.5f, -0.5f,
			 0.5f,  0.5f,  0.5f
		};

		static std::vector<float> vertices2 = {
			-0.5f, 0.5f, -0.5f,
			 0.5f, 0.5f, -0.5f,
			 0.5f, 1.5f, -0.5f,
			 0.5f, 1.5f, -0.5f,
			-0.5f, 1.5f, -0.5f,
			-0.5f, 0.5f, -0.5f,

			 0.5f, 0.5f,  0.5f,
			-0.5f, 0.5f,  0.5f,
			-0.5f, 1.5f,  0.5f,
			-0.5f, 1.5f,  0.5f,
			 0.5f, 1.5f,  0.5f,
			 0.5f, 0.5f,  0.5f,

			-0.5f, 0.5f,  0.5f,
			-0.5f, 0.5f, -0.5f,
			-0.5f, 1.5f, -0.5f,
			-0.5f, 1.5f, -0.5f,
			-0.5f, 1.5f,  0.5f,
			-0.5f, 0.5f,  0.5f,

			 0.5f, 0.5f, -0.5f,
			 0.5f, 0.5f,  0.5f,
			 0.5f, 1.5f,  0.5f,
			 0.5f, 1.5f,  0.5f,
			 0.5f, 1.5f, -0.5f,
			 0.5f, 0.5f, -0.5f,

			 0.5f, 0.5f, -0.5f,
			-0.5f, 0.5f, -0.5f,
			-0.5f, 0.5f,  0.5f,
			-0.5f, 0.5f,  0.5f,
			 0.5f, 0.5f,  0.5f,	
			 0.5f, 0.5f, -0.5f,

			 0.5f, 1.5f,  0.5f,
			-0.5f, 1.5f,  0.5f,
			-0.5f, 1.5f, -0.5f,
			-0.5f, 1.5f, -0.5f,
			 0.5f, 1.5f, -0.5f,
			 0.5f, 1.5f,  0.5f
		};

		static std::vector<int> indices = {
			0,   1,  2,
			3,   4,  5,
			6,   7,  8,
			9,  10, 11,
			12, 13, 14,
			15, 16, 17,
			18, 19, 20,
			21, 22, 23,
			24, 25, 26,
			27, 28, 29,
			30, 31, 32,
			33, 34, 35
		};

		static std::vector<float> uvCoords = {
			0.0f, 0.0f,
			0.0f, 1.0f,
			1.0f, 1.0f,
			1.0f, 1.0f,
			1.0f, 0.0f,
			0.0f, 0.0f,

			0.0f, 0.0f,
			0.0f, 1.0f,
			1.0f, 1.0f,
			1.0f, 1.0f,
			1.0f, 0.0f,
			0.0f, 0.0f,

			0.0f, 0.0f,
			0.0f, 1.0f,
			1.0f, 1.0f,
			1.0f, 1.0f,
			1.0f, 0.0f,
			0.0f, 0.0f,

			0.0f, 0.0f,
			0.0f, 1.0f,
			1.0f, 1.0f,
			1.0f, 1.0f,
			1.0f, 0.0f,
			0.0f, 0.0f,

			0.0f, 0.0f,
			0.0f, 1.0f,
			1.0f, 1.0f,
			1.0f, 1.0f,
			1.0f, 0.0f,
			0.0f, 0.0f,

			0.0f, 0.0f,
			0.0f, 1.0f,
			1.0f, 1.0f,
			1.0f, 1.0f,
			1.0f, 0.0f,
			0.0f, 0.0f,
		};

		static Mesh sMesh(vertices, indices, uvCoords);

		static Mesh sMesh2(vertices2, indices, uvCoords);

		meshes.push_back(std::pair(&sMesh, 0));
		meshes.push_back(std::pair(&sMesh2, 1));

		isComposite = true;

		updateCompositeBounds();  
	}
};