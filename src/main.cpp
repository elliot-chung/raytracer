#include "Window.hpp"


// Main code
int main(int argc, char** argv)
{ 
    Window window(800, 600, "Raytracer");
    
    window.renderLoop();

    return 0;
}
