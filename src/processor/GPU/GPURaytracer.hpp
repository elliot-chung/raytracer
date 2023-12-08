#pragma once

#include <cuda_runtime.h>

#include "../Raytracer.hpp"
#include "helper_cuda.h"

class GPURaytracer : public Raytracer
{
public:
	GPURaytracer(int argc, char **argv);
	~GPURaytracer();

	std::vector<float> trace(std::shared_ptr<Scene> s, std::shared_ptr<Camera> c) override;

private:
}; 