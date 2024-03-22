#include "Scene.hpp"

GPUObjectDataVector Scene::getGPUObjectDataVector()
{
	GPUObjectData* objectDataArray = new GPUObjectData[objectMap.size()];
	int i = 0;
	for (auto& objPair : objectMap)
	{
		DisplayObject* obj = objPair.second;
		GPUObjectData data = {};
		glm::mat4 mm = obj->getModelMatrix();

		mat4transfer(data.modelMatrix, mm);
		mat4transfer(data.inverseModelMatrix, glm::inverse(mm));
		data.llData = obj->getGPUData();

		objectDataArray[i++] = data;
	}
	GPUObjectData* objectDataArrayDev;
	checkCudaErrors(cudaMalloc((void**)&objectDataArrayDev, sizeof(GPUObjectData)* objectMap.size()));
	checkCudaErrors(cudaMemcpy(objectDataArrayDev, objectDataArray, sizeof(GPUObjectData)* objectMap.size(), cudaMemcpyHostToDevice));
	delete[] objectDataArray;

	GPUObjectDataVector output;
	output.data = objectDataArrayDev;
	output.size = objectMap.size();

	return output;
}