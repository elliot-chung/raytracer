#pragma once

#include <glm/glm.hpp>

#include <unordered_map>
#include <string>

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
class Material;
typedef std::unordered_map<std::string, Material*> MaterialMap;

class Material
{
public:
	Material(char const* name, glm::vec4 albedo = {}, float roughness = 0.0f, float metal = 0.0f, glm::vec3 emissionColor = {}, float emission = 0);
	~Material();

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

	inline static Material* getMaterial(std::string& name) { return materialMap[name]; }

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

	float emissionStrength;

	bool success = true;

	static MaterialMap materialMap;
};



