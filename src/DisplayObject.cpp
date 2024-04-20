#include "DisplayObject.hpp"

DisplayObject::DisplayObject(glm::vec3 position, glm::quat rotation, glm::vec3 scale)
{
	Position = position;
	Rotation = rotation;
	Euler = glm::degrees(glm::eulerAngles(Rotation));
	Scale = scale;
}

DisplayObject::~DisplayObject()
{
	if (gpuMeshes) checkCudaErrors(cudaFree(gpuMeshes));
	if (gpuMaterialIndices) checkCudaErrors(cudaFree(gpuMaterialIndices));
	if (gpuMaterials) checkCudaErrors(cudaFree(gpuMaterials));
	if (gpuData) checkCudaErrors(cudaFree(gpuData));
}

glm::mat4 DisplayObject::getModelMatrix()
{
	glm::mat4 model = glm::mat4(1.0f);

	model = glm::translate(model, Position);
	model = model * glm::mat4_cast(Rotation);
	model = glm::scale(model, Scale); 

	return model;
}

void DisplayObject::sendToGPU()
{
	LLGPUObjectData* hostData = new LLGPUObjectData();

	hostData->minCompositeBounds = compositeMinBounds;
	hostData->maxCompositeBounds = compositeMaxBounds;

	hostData->meshCount = meshes.size();

	hostData->isComposite = isComposite;

	GPUMeshData** meshesHost = new GPUMeshData * [meshes.size()];
	int* materialIndicesHost = new int[meshes.size()];
	GPUMaterial** materialsHost = new GPUMaterial * [materials.size()];


	for (int i = 0; i < meshes.size(); i++)
	{
		meshesHost[i] = meshes[i].first->getGPUMeshData();
		materialIndicesHost[i] = meshes[i].second;
	}

	for (int i = 0; i < materials.size(); i++)
	{
		materialsHost[i] = materials[i]->getGPUMaterial();
	}

	if (gpuMeshes == 0) 
		checkCudaErrors(cudaMalloc((void**)&gpuMeshes, sizeof(GPUMeshData*) * meshes.size()));
	if (gpuMaterialIndices== 0) 
		checkCudaErrors(cudaMalloc((void**)&gpuMaterialIndices, sizeof(int) * meshes.size()));
	if (gpuMaterials == 0) 
		checkCudaErrors(cudaMalloc((void**)&gpuMaterials, sizeof(GPUMaterial*) * materials.size()));

	checkCudaErrors(cudaMemcpy(gpuMeshes, meshesHost, sizeof(GPUMeshData*) * meshes.size(), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(gpuMaterialIndices, materialIndicesHost, sizeof(int) * meshes.size(), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(gpuMaterials, materialsHost, sizeof(GPUMaterial*) * materials.size(), cudaMemcpyHostToDevice));

	delete[] meshesHost;
	delete[] materialIndicesHost;
	delete[] materialsHost;

	hostData->meshes = gpuMeshes;
	hostData->materialIndices = gpuMaterialIndices;
	hostData->materials = gpuMaterials;

	if (gpuData == 0) 
		checkCudaErrors(cudaMalloc((void**)&gpuData, sizeof(LLGPUObjectData)));
	checkCudaErrors(cudaMemcpy(gpuData, hostData, sizeof(LLGPUObjectData), cudaMemcpyHostToDevice));

	delete hostData;
}

void DisplayObject::updateCompositeBounds()
{
	if (!isComposite)
	{
		compositeMaxBounds = meshes[0].first->getMaxBound();
		compositeMinBounds = meshes[0].first->getMinBound();
		return;
	}

	compositeMaxBounds = make_float3(-FLT_MAX, -FLT_MAX, -FLT_MAX);
	compositeMinBounds = make_float3(FLT_MAX, FLT_MAX, FLT_MAX);

	for (auto meshPair : meshes)
	{
		Mesh* mesh = meshPair.first;

		float3 maxBounds = mesh->getMaxBound();
		float3 minBounds = mesh->getMinBound();

		compositeMaxBounds = make_float3(fmaxf(maxBounds.x, compositeMaxBounds.x), fmaxf(maxBounds.y, compositeMaxBounds.y), fmaxf(maxBounds.z, compositeMaxBounds.z));
		compositeMinBounds = make_float3(fminf(minBounds.x, compositeMinBounds.x), fminf(minBounds.y, compositeMinBounds.y), fminf(minBounds.z, compositeMinBounds.z));
	}
}

void DisplayObject::copyHostLLData(const DisplayObject* other)
{
	this->compositeMaxBounds = other->compositeMaxBounds;
	this->compositeMinBounds = other->compositeMinBounds;

	this->meshes = other->meshes;
	this->materials = other->materials;
	this->isComposite = other->isComposite;
}

void DisplayObject::updateGUI(ImGuiIO& io)  
{

	if (ImGui::CollapsingHeader("Transform"))
	{
		ImGui::DragFloat3("Position", &Position[0], 0.01f);
		if (ImGui::DragFloat3("Rotation", &Euler[0], 0.1f, -359.9f, 359.9f))
		{
			Rotation = glm::quat(glm::radians(Euler)); 
		}
		ImGui::Text("(% 1.3f, % 1.3f, % 1.3f, % 1.3f)  Quat", Rotation.x, Rotation.y, Rotation.z, Rotation.w);
		ImGui::DragFloat3("Scale", &Scale[0], 0.01f, 0);
		
	}

	if (ImGui::CollapsingHeader("Meshes"))
	{
		for (int i = 0; i < meshes.size(); i++)
		{
			ImGui::Text("Mesh %d Material Index: %d", i, meshes[i].second);
		}

		ImGui::Separator();

		ImGui::Text("Bounds:");
		ImGui::Text("Max: (%f, %f, %f)", compositeMaxBounds.x, compositeMaxBounds.y, compositeMaxBounds.z);
		ImGui::Text("Min: (%f, %f, %f)", compositeMinBounds.x, compositeMinBounds.y, compositeMinBounds.z);

		ImGui::Separator(); 

		ImGui::Text("Is Composite: %s", isComposite ? "True" : "False");
	}



	if (ImGui::CollapsingHeader("Material"))
	{
		static std::vector<Material*> allMaterials = Material::getAllMaterials();

		if (ImGui::Button("Refresh"))
		{
			allMaterials = Material::getAllMaterials();
		}

		for (int i = 0; i < materials.size(); i++)
		{
			if (ImGui::TreeNode((void*)(intptr_t)i, "Material %d", i))
			{
				std::string name = materials[i]->getName();
				if (ImGui::BeginCombo(("##" + name).c_str(), name.c_str(), ImGuiComboFlags_PopupAlignLeft))
				{
					for (int j = 0; j < allMaterials.size(); j++)
					{
						bool selected = (materials[i]->getName() == allMaterials[j]->getName());
						if (ImGui::Selectable(allMaterials[j]->getName().c_str(), selected))
						{
							materials[i] = allMaterials[j];
							sendToGPU(); 
						}

						if (selected)
						{
							ImGui::SetItemDefaultFocus();
						}
					}
					ImGui::EndCombo();
				}
				ImGui::TreePop();
			}

		}
	}
}

DisplayObject* DisplayObject::selectedObject = 0; 