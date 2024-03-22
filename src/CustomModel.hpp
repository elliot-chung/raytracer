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
		glm::vec3 scale = DEFAULT_SCALE); 

	~CustomModel();


private:
	static CustomModelMap customModels;

	std::string path;

	void loadModel(std::string path);

	void processAllMaterials(const aiScene* scene);

	Material* processMaterial(const aiMaterial* material);

	void processAllMeshes(const aiScene* scene);

	Mesh* processMesh(aiMesh* mesh);
};