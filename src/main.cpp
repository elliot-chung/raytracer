#include "Window.hpp"


// Main code
int main(int argc, char** argv)
{ 
    Window window(401, 303, "Raytracer");
    
    window.renderLoop();

    return 0;
}
