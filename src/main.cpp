#include "RenderWindow.hpp"


// Main code
int main(int argc, char** argv)
{ 
    RenderWindow window(800, 600, "Raytracer");
    
    window.renderLoop();

    return 0;
}
