#include "DisplayObject.hpp"

DisplayObject::DisplayObject(glm::vec3 position, glm::quat rotation, glm::vec3 scale)
{
	Position = position;
	Rotation = rotation;
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

	checkCudaErrors(cudaMalloc((void**)&gpuMeshes, sizeof(GPUMeshData*) * meshes.size()));
	checkCudaErrors(cudaMalloc((void**)&gpuMaterialIndices, sizeof(int) * meshes.size()));
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
		static glm::vec3 euler = glm::degrees(glm::eulerAngles(Rotation));
		ImGui::DragFloat3("Position", &Position[0], 0.1f);
		ImGui::DragFloat3("Rotation", &euler[0], 1.0f, -359.9f, 359.9f);
		ImGui::DragFloat3("Scale", &Scale[0], 0.1f, 0);
		if (euler != glm::degrees(glm::eulerAngles(Rotation)))
		{
			Rotation = glm::quat(glm::radians(euler));
		}
	}

	if (ImGui::CollapsingHeader("Meshes"))
	{
		for (int i = 0; i < meshes.size(); i++)
		{
			ImGui::Text("Mesh %d Material Index: %d", i, meshes[i].second);
		}

		ImGui::Text("Composite Bounds:");
		ImGui::Text("Max: (%f, %f, %f)", compositeMaxBounds.x, compositeMaxBounds.y, compositeMaxBounds.z);
		ImGui::Text("Min: (%f, %f, %f)", compositeMinBounds.x, compositeMinBounds.y, compositeMinBounds.z);

		ImGui::Text("Is Composite: %s", isComposite ? "True" : "False");
	}



	if (ImGui::CollapsingHeader("Material"))
	{
		static std::vector<Material*> allMaterials = Material::getAllMaterials();

		if (ImGui::Button("Refresh Full Material List"))
		{
			allMaterials = Material::getAllMaterials();
		}

		for (int i = 0; i < materials.size(); i++)
		{
			if (ImGui::TreeNode((void*)(intptr_t)i, "Material %d", i))
			{
				if (ImGui::BeginCombo(" ", materials[i]->getName().c_str(), ImGuiComboFlags_PopupAlignLeft))
				{
					for (int j = 0; j < allMaterials.size(); j++)
					{
						bool selected = (materials[i]->getName() == allMaterials[j]->getName());
						if (ImGui::Selectable(allMaterials[j]->getName().c_str(), selected))
						{
							materials[i] = allMaterials[j];
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