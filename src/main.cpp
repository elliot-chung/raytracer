#include "Window.hpp"


// Main code
int main(int argc, char** argv)
{ 
    Window window(400, 300, "Raytracer");
    
    window.renderLoop();

    return 0;
}
