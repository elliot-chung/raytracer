#include "CustomModel.hpp"

CustomModel::CustomModel(std::string path,
						 glm::vec3 position,
						 glm::quat rotation,
						 glm::vec3 scale) : DisplayObject(position, rotation, scale)
{
	this->path = path;

	if (customModels.find(path) != customModels.end())
	{
		customModels[path].second++;

		copyHostLLData(customModels[path].first);

		return;
	}

	customModels[path] = std::pair<CustomModel*, int>(this, 1);

	loadModel(path);
}

CustomModel::~CustomModel()
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

void CustomModel::loadModel(std::string path)
{
	Assimp::Importer importer;
	const aiScene* scene = importer.ReadFile(path, aiProcess_Triangulate | aiProcess_GenUVCoords | aiProcess_FlipUVs);

	if (!scene || scene->mFlags & AI_SCENE_FLAGS_INCOMPLETE || !scene->mRootNode)
	{
		printf("ERROR::ASSIMP::%s", importer.GetErrorString());
		return;
	}

	processAllMaterials(scene);
	processAllMeshes(scene);
	updateCompositeBounds();
}

void CustomModel::processAllMaterials(const aiScene* scene) 
{
	for (unsigned int i = 0; i < scene->mNumMaterials; i++)
	{
		aiMaterial* material = scene->mMaterials[i];

		materials.push_back(processMaterial(material));
	}
} 

bool setOutputTexture(const aiMaterial* material, aiTextureType type, std::string directory, Material* output)
{
    aiString filename;
	int count = material->GetTextureCount(type); 

	if (count)
	{
		material->GetTexture(type, 0, &filename); 
		std::string texturePath = directory + "/" + filename.C_Str();  
		std::replace(texturePath.begin(), texturePath.end(), '\\', '/'); 
		printf("Texture path: %s\n", texturePath.c_str());
		switch (type)
		{
			case aiTextureType_BASE_COLOR: return output->setAlbedoTexture(texturePath.c_str());  
			case aiTextureType_DIFFUSE: return output->setAlbedoTexture(texturePath.c_str());  
			case aiTextureType_NORMALS: return output->setNormalTexture(texturePath.c_str());  
			case aiTextureType_NORMAL_CAMERA: return output->setNormalTexture(texturePath.c_str());  
			case aiTextureType_EMISSIVE: return output->setEmissionTexture(texturePath.c_str()); 
			case aiTextureType_EMISSION_COLOR: return output->setEmissionTexture(texturePath.c_str()); 
			case aiTextureType_METALNESS: return output->setMetalTexture(texturePath.c_str()); 
			case aiTextureType_DIFFUSE_ROUGHNESS: return output->setRoughnessTexture(texturePath.c_str());  
			case aiTextureType_AMBIENT_OCCLUSION: return output->setAmbientOcclusionTexture(texturePath.c_str());  
		}
	}
	return false;
}

Material* CustomModel::processMaterial(const aiMaterial* material)
{
	aiString name = material->GetName();
	if (name.length == 0) name = aiString("No Material Name Found");
	Material* output = new Material(name.C_Str());


	int si = path.find_last_of('/');
    if (path.substr(si - 6, 6) == "source") si = si - 7; 
	std::string directory = path.substr(0, si); 

	if (!setOutputTexture(material, aiTextureType_BASE_COLOR, directory, output)) // Try setting texture using PBR albedo 
		setOutputTexture(material, aiTextureType_DIFFUSE, directory, output);     // Fallback to legacy diffuse texture

	if (!setOutputTexture(material, aiTextureType_NORMAL_CAMERA, directory, output))		// Try setting texture using PBR normal map
		setOutputTexture(material, aiTextureType_NORMALS, directory, output);				// Fallback to legacy normal map)

	if (!setOutputTexture(material, aiTextureType_EMISSION_COLOR, directory, output))		// Try setting texture using PBR emission map
		setOutputTexture(material, aiTextureType_EMISSIVE, directory, output);				// Fallback to legacy emission map)

	setOutputTexture(material, aiTextureType_METALNESS, directory, output);			// Set texture using PBR metalness map
	setOutputTexture(material, aiTextureType_DIFFUSE_ROUGHNESS, directory, output);	// Set texture using PBR roughness map
	setOutputTexture(material, aiTextureType_AMBIENT_OCCLUSION, directory, output);	// Set texture using PBR ambient occlusion map

	output->sendToGPU();

	return output;
}

void CustomModel::processAllMeshes(const aiScene* scene)
{
	for (unsigned int i = 0; i < scene->mNumMeshes; i++)
	{
		aiMesh* mesh = scene->mMeshes[i];

		meshes.push_back(std::pair(processMesh(mesh), mesh->mMaterialIndex));
	}
	if (scene->mNumMeshes > 0) this->isComposite = true;
	updateCompositeBounds();
}

Mesh* CustomModel::processMesh(aiMesh* mesh)
{
	std::vector<float> vertices;
	std::vector<float> uvCoords;
	std::vector<float> normals;

	for (unsigned int i = 0; i < mesh->mNumVertices; i++)
	{
		vertices.push_back(mesh->mVertices[i].x);
		vertices.push_back(mesh->mVertices[i].y);
		vertices.push_back(mesh->mVertices[i].z);

		if (mesh->mTextureCoords[0])
		{
			uvCoords.push_back(mesh->mTextureCoords[0][i].x);
			uvCoords.push_back(mesh->mTextureCoords[0][i].y);
		}
		else
		{
			uvCoords.push_back(0.0f);
			uvCoords.push_back(0.0f);
		}

		if (mesh->mNormals)
		{
			normals.push_back(mesh->mNormals[i].x);
			normals.push_back(mesh->mNormals[i].y);
			normals.push_back(mesh->mNormals[i].z);
		}
	}

	std::vector<int> indices;
	for (unsigned int i = 0; i < mesh->mNumFaces; i++)
	{
		aiFace face = mesh->mFaces[i];
		if (face.mNumIndices != 3) 
		{
			printf("WARNING::ASSIMP::FACE_HAS_%d_INDICES\n", face.mNumIndices);
			continue;
		}
		for (unsigned int j = 0; j < face.mNumIndices; j++)
		{
			indices.push_back(face.mIndices[j]);
		}
	}

	return new Mesh(vertices, indices, uvCoords, normals);
}

CustomModelMap CustomModel::customModels = {};