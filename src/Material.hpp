#pragma once

#include <glm/glm.hpp>

#include <unordered_map>
#include <string>
#include <cstring>

#include "ImGui/imgui.h"
#include "cuda_runtime.h"
#include "helper_cuda.h"
#include "global.hpp"

class Texture;
typedef std::unordered_map<std::string, std::pair<int, Texture*>> TextureMap; // Format <path-name, <ref-count, texture-pointer>>

class Texture
{
public:
	Texture(char const* path);
	~Texture();

	glm::vec4 sampleTexture(float x, float y);

	inline int getWidth() { return width; }
	inline int getHeight() { return height; }
	inline int size() { return dataSize; }
	inline float* getData() { return data; }
	inline bool getLoadSuccess() { return loadSuccess; }
	inline cudaTextureObject_t* getGPUTexture() { return &gpuTexture; }
	inline std::string getPath() { return path; }

	static inline Texture* getTexture(char const* path) { return textureMap[path].second; }

private:
	std::string path = "";

	int width;
	int height;
	int nChannels;
	int dataSize;

	bool loadSuccess;

	float* data;

	cudaTextureObject_t gpuTexture = 0;

	static TextureMap textureMap;
};

// -------------------------------------------------------------------------------------------------------- //
struct GPUMaterial
{
	const float4 NORMAL = make_float4(0.0f, 0.0f, 1.0f, 0.0f);
	const float3 AO = make_float3(1.0f, 1.0f, 1.0f);

	float4 albedo;
	float3 roughness;
	float3 metal;
	float3 emissionColor;
	float emissionStrength;

	cudaTextureObject_t normalTexture = 0;
	cudaTextureObject_t albedoTexture = 0;
	cudaTextureObject_t roughnessTexture = 0;
	cudaTextureObject_t metalTexture = 0;
	cudaTextureObject_t ambientOcclusionTexture = 0;
	cudaTextureObject_t emissionTexture = 0;

	__device__  float4 getNormal(float x, float y);
	__device__  float4 getAlbedo(float x, float y);
	__device__  float getRoughness(float x, float y);
	__device__  float getMetal(float x, float y);
	__device__  float3 getAmbientOcclusion(float x, float y);
	__device__  float4 getEmission(float x, float y);
};

class Material;
typedef std::unordered_map<std::string, Material*> MaterialMap;

class Material
{
public:
	Material(char const* name, glm::vec4 albedo = glm::vec4(0.9f, 0.9f, 0.9f, 1.0f), float roughness = 0.5f, float metal = 0.0f, glm::vec3 emissionColor = {}, float emission = 0);
	~Material();
	Material(const Material&) = delete;

	inline std::string getName() { return materialName; }
	inline bool getSuccess() { return success; }

	glm::vec3 getNormal(float x, float y);
	glm::vec4 getAlbedo(float x, float y);
	float getRoughness(float x, float y);
	float getMetal(float x, float y);
	glm::vec3 getAmbientOcclusion(float x, float y);
	glm::vec3 getEmissionColor(float x, float y);
	inline float getEmissionStrength() { return emissionStrength; }

	bool setNormalTexture(const char* path);
	bool setAlbedoTexture(const char* path);
	bool setRoughnessTexture(const char* path);
	bool setMetalTexture(const char* path);
	bool setAmbientOcclusionTexture(const char* path);
	bool setEmissionTexture(const char* path);

	void sendToGPU();

	inline static Material* getMaterial(std::string& name) { return materialMap[name]; }
	inline static GPUMaterial* getGPUMaterial(std::string& name) { return materialMap[name]->gpuMaterial; }
	inline static std::vector<Material*> getAllMaterials() { std::vector<Material*> materials; for (auto& m : materialMap) materials.push_back(m.second); return materials; }

	inline GPUMaterial* getGPUMaterial() { return gpuMaterial; }

	static void displayMaterialGUI(ImGuiIO& io)
	{
		ImGui::Begin("Materials");

		for (auto& m : materialMap)
		{
			if (ImGui::TreeNode(m.first.c_str()))
			{
				if (ImGui::Button("Send to GPU"))
				{
					m.second->sendToGPU();
				}

				ImGui::Separator();

				ImGui::ColorEdit3("Albedo", &m.second->simpleAlbedo.r);
				ImGui::DragFloat("Roughness", &m.second->simpleRoughness.r, 0.01f, 0.0f, 1.0f);
				ImGui::DragFloat("Metal", &m.second->simpleMetal.r, 0.01f, 0.0f, 1.0f);
				ImGui::ColorEdit3("Emission Color", &m.second->simpleEmissionColor.r);
				ImGui::DragFloat("Emission Strength", &m.second->emissionStrength, 0.01f, 0.0f, 1000000.0f);

				ImGui::Separator();
				
				static char albedoTexturePath[256] = "";
				if (m.second->albedoTexture && strlen(albedoTexturePath) == 0) strncpy(albedoTexturePath, m.second->albedoTexture->getPath().c_str(), 256);
				ImGui::InputTextWithHint("##Albedo Texture Path", "Albedo Texture Path", albedoTexturePath, 256);
				ImGui::SameLine();
				if (ImGui::Button("Set Texture") && m.second->albedoTexture->getPath() != albedoTexturePath)
				{
					m.second->setAlbedoTexture(albedoTexturePath);
				}

				static char normalTexturePath[256] = "";
				if (m.second->normalTexture && strlen(normalTexturePath) == 0) strncpy(normalTexturePath, m.second->normalTexture->getPath().c_str(), 256);
				ImGui::InputTextWithHint("##Normal Texture Path", "Normal Texture Path", normalTexturePath, 256);
				ImGui::SameLine();
				if (ImGui::Button("Set Texture") && m.second->normalTexture->getPath() != normalTexturePath)
				{
					m.second->setNormalTexture(normalTexturePath);
				}

				static char roughnessTexturePath[256] = "";
				if (m.second->roughnessTexture && strlen(roughnessTexturePath) == 0) strncpy(roughnessTexturePath, m.second->roughnessTexture->getPath().c_str(), 256);
				ImGui::InputTextWithHint("##Texture Path", "Roughness Texture Path", roughnessTexturePath, 256);
				ImGui::SameLine();
				if (ImGui::Button("Set Texture") && m.second->roughnessTexture->getPath() != roughnessTexturePath) 
				{
					m.second->setRoughnessTexture(roughnessTexturePath);
				}

				static char metalTexturePath[256] = "";
				if (m.second->metalTexture && strlen(metalTexturePath) == 0) strncpy(metalTexturePath, m.second->metalTexture->getPath().c_str(), 256);
				ImGui::InputTextWithHint("##Metal Texture Path", "Metal Texture Path", metalTexturePath, 256);
				ImGui::SameLine();
				if (ImGui::Button("Set Texture") && m.second->metalTexture->getPath() != metalTexturePath)
				{
					m.second->setMetalTexture(metalTexturePath);
				}

				static char ambientOcclusionTexturePath[256] = "";
				if (m.second->ambientOcclusionTexture && strlen(ambientOcclusionTexturePath) == 0) strncpy(ambientOcclusionTexturePath, m.second->ambientOcclusionTexture->getPath().c_str(), 256);
				ImGui::InputTextWithHint("##Ambient Occlusion Texture Path", "Ambient Occlusion Texture Path", ambientOcclusionTexturePath, 256);
				ImGui::SameLine();
				if (ImGui::Button("Set Texture") && m.second->ambientOcclusionTexture->getPath() != ambientOcclusionTexturePath) 
				{
					m.second->setAmbientOcclusionTexture(ambientOcclusionTexturePath);
				}

				static char emissionTexturePath[256] = "";
				if (m.second->emissionTexture && strlen(emissionTexturePath) == 0) strncpy(emissionTexturePath, m.second->emissionTexture->getPath().c_str(), 256);
				ImGui::InputTextWithHint("##Emission Texture Path", "Emission Texture Path", emissionTexturePath, 256);
				ImGui::SameLine();
				if (ImGui::Button("Set Texture") && m.second->emissionTexture->getPath() != emissionTexturePath) 
				{
					m.second->setEmissionTexture(emissionTexturePath);
				}
				

				ImGui::TreePop();
			}
		}
		ImGui::Separator();
		static char newMaterialName[256] = "";
		ImGui::InputTextWithHint("##New Material Name", "New Material Name", newMaterialName, IM_ARRAYSIZE(newMaterialName));
		ImGui::SameLine();
		static int success = 0;
		if (ImGui::Button("Add Material"))
		{
			if (strlen(newMaterialName) == 0)
				success = 2;
			else
			{
				Material* m = new Material(newMaterialName);
				success = (int)!m->getSuccess();
				if (success == 0)
				{
					newMaterialName[0] = '\0';
				}
			}
		}

		switch (success)
        {
		case 1:
			ImGui::Text("Material name already exists!"); 
			break;
		case 2:
			ImGui::Text("Material name cannot be empty!");
			break;
		}

		ImGui::End();
	}

protected:
	const glm::vec3 NORMAL = glm::vec3(0.0f, 0.0f, 1.0f);
	const glm::vec3 AO = glm::vec3(1.0f);

	std::string materialName;

	glm::vec4 simpleAlbedo;
	glm::vec3 simpleRoughness;
	glm::vec3 simpleMetal;
	glm::vec3 simpleEmissionColor;

	Texture* normalTexture = 0;
	Texture* albedoTexture = 0;
	Texture* roughnessTexture = 0;
	Texture* metalTexture = 0;
	Texture* ambientOcclusionTexture = 0;
	Texture* emissionTexture = 0;

	GPUMaterial* gpuMaterial = 0;

	float emissionStrength;

	bool success = true;

	static MaterialMap materialMap;
};



