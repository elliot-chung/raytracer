#include "GPURaytracer.hpp"


GPURaytracer::GPURaytracer(int argc, char **argv)
{
	int deviceID = findCudaDevice(argc, (const char**)argv);
	if (deviceID < 0)
	{
		return;
	}
	else
	{
		gpuFound = true;
	}
}

GPURaytracer::~GPURaytracer()
{}

std::vector<float> GPURaytracer::trace(std::shared_ptr<Scene> s, std::shared_ptr<Camera> c)
{
	return std::vector<float>();
}