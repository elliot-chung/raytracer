#include "Window.hpp"



// Main code
int main(int, char**)
{ 
    Window window(400, 300, "Raytracer");
    
    window.renderLoop();

    return 0;
}
