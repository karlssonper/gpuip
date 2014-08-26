/*
The MIT License (MIT)

Copyright (c) 2014 Per Karlsson

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/

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
