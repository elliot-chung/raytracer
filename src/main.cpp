#include "Window.hpp"



// Main code
int main(int argc, char** argv)
{ 
    auto backend = GPURaytracer(argc, argv);
    Raytracer* raytracer = &backend;

    if (backend.didFindGPU())
    {
		printf("GPU Found\n");
	}
    else
    {
        printf("GPU Not Found\n");
    }

    // Window window(400, 300, "Raytracer");
    
    // window.renderLoop();

    return 0;
}
