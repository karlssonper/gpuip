#ifndef GPUIP_GL_CONTEXT_H_
#define GPUIP_GL_CONTEXT_H_

#ifdef __APPLE__
extern "C" {
    bool _HasNSGLContext();
}
#else
#  ifdef _WIN32
#    include <windows.h>
#    include <Wingdi.h>
#  else
#    include <GL/glx.h>
#  endif
#endif
#include <GLFW/glfw3.h>
#include <string>

//----------------------------------------------------------------------------//
namespace gpuip {
//----------------------------------------------------------------------------//
class GLContext
{
  public:
    static bool Exists()
    {
#ifdef __APPLE__
        if (_HasNSGLContext()) {
            return true;
        }
#else
#  ifdef _WIN32
        if (wglGetCurrentContext()) {
            return true;
        }
#  else
        if (glXGetCurrentContext()) {
            return true;
        }
#  endif
#endif
        return false;
    }
    
    static bool Create(std::string * err)
    {
        if (!glfwInit()) {
            (*err) += "gpuip could not initiate GLFW";
            return false;
        }
        GLFWwindow * window = glfwCreateWindow(1, 1, "", NULL, NULL);
        if (!window) {
            (*err) += "gpuip could not create window with glfw";
            return false;
        }
        glfwMakeContextCurrent(window);
        return true;
    }

    static void Delete()
    {
        if(glfwGetCurrentContext()) {
            glfwTerminate();
        }
    }
};
//----------------------------------------------------------------------------//
}// end namespace gpuip

#endif
