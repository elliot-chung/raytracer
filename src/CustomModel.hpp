#pragma once

#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>
#include "DisplayObject.hpp"

class CustomModel;
typedef std::unordered_map<std::string, std::pair<CustomModel*, int>> CustomModelMap; // <path, <model, ref-count>>

class CustomModel : public DisplayObject
{
public:
	CustomModel(std::string path,
				glm::vec3 position = DEFAULT_POSITION,
				glm::quat rotation = DEFAULT_ROTATION,
				glm::vec3 scale = DEFAULT_SCALE) : DisplayObject(position, rotation, scale)
	{
		this->path = path;

		if (customModels.find(path) != customModels.end())
		{
			customModels[path].second++; 

			copyHostLLData(customModels[path].first);

			return;
		}

		customModels[path] = std::pair<CustomModel*, int>(this, 1);

	}

	~CustomModel()
	{
		if (--customModels[path].second == 0)
		{
			for (auto material : materials)
			{
				delete material;
			}

			materials.clear(); 

			for (auto meshPair : meshes)
			{
				delete meshPair.first;
			}

			meshes.clear(); 
			customModels.erase(path);
		}
	}


private:
	static CustomModelMap customModels;

	std::string path;

	void loadModel(std::string path)
	{
		Assimp::Importer importer;
		const aiScene* scene = importer.ReadFile(path, aiProcess_Triangulate | aiProcess_FlipUVs);

		if (!scene || scene->mFlags & AI_SCENE_FLAGS_INCOMPLETE || !scene->mRootNode)
		{
			printf("ERROR::ASSIMP::%s", importer.GetErrorString());
			return;
		}

		
	}

	void processAllMeshes(const aiScene *scene)
	{
		for (unsigned int i = 0; i < scene->mNumMeshes; i++)
		{
			aiMesh* mesh = scene->mMeshes[i];

			meshes.push_back(std::pair(processMesh(mesh), mesh->mMaterialIndex));
		}
	}

	Mesh* processMesh(aiMesh *mesh)
	{
		

		// return new Mesh();
	}

	void processMaterials(const aiScene *scene)
	{
		for (unsigned int i = 0; i < scene->mNumMaterials; i++)
		{
			aiMaterial* material = scene->mMaterials[i];
		}
	}
};

CustomModelMap CustomModel::customModels = {};