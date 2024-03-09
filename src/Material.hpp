#pragma once

#include <glm/glm.hpp>

#include <unordered_map>
#include <string>

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

	static inline Texture* getTexture(char const* path) { return textureMap[path].second; }

private:
	std::string path;

	int width;
	int height;
	int nChannels;
	int dataSize;

	bool loadSuccess;

	float* data;

	static TextureMap textureMap;
};


// -------------------------------------------------------------------------------------------------------- //
struct GPUMaterial
{
	const float3 NORMAL = make_float3(0.0f, 0.0f, 1.0f);
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

	__device__  float3 getNormal(float x, float y);
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
	Material(char const* name, glm::vec4 albedo = {}, float roughness = 0.0f, float metal = 0.0f, glm::vec3 emissionColor = {}, float emission = 0);
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

	bool setNormalTexture(const char *path);
	bool setAlbedoTexture(const char* path);
	bool setRoughnessTexture(const char* path);
	bool setMetalTexture(const char* path);
	bool setAmbientOcclusionTexture(const char* path);
	bool setEmissionTexture(const char* path);

	void sendToGPU();

	inline static Material* getMaterial(std::string& name) { return materialMap[name]; }
	inline static GPUMaterial* getGPUMaterial(std::string& name) { return materialMap[name]->gpuMaterial; }

	inline GPUMaterial* getGPUMaterial() { return gpuMaterial; }

private:
	const glm::vec3 NORMAL = glm::vec3(0.0f, 0.0f, 1.0f);
	const glm::vec3 AO	   = glm::vec3(1.0f);
	
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



