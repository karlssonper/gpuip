gpuip
=====

Gpuip is a C++ cross-platform framework for Image Processing on the GPU architechure. It tries to simplify the image processing pipeline on the GPU and make it more generic across the thre most common environments: OpenCL, CUDA and OpenGL GLSL. It provides a simply interface to copy data from and to the GPU and makes it easy to compile and run GPU kernel code. 

### API
The online API documentation [can be found here.] (http://karlssonper.github.io/gpuip/)

### pygpuip
The gpuip library comes with optional python bindings to the C++ code. The python bindings have I/O operations included with .exr and .png support (and .jpeg, .tiff and .tga if dev libraries are found at build time). Numpy arrays are used to tranfser data to/from the GPU.

### bin/gpuip
If python bindings are available, gpuip comes with an executable program that has both a GUI version for debugging and development of GPU kernels and a command line version to plug into existing pipelines. The progam uses the gpuip specific XML-based file format *.ip to store settings.
```
usage: Framework for Image Processing on the GPU [-h] [-f FILE]
                                                 [-p kernel param value]
                                                 [-i buffer path]
                                                 [-o buffer path] [-v]
                                                 [--timestamp] [--nogui]

optional arguments:
  -h, --help            show this help message and exit
  -f FILE, --file FILE  Image Processing file *.ip
  -p kernel param value, --param kernel param value
                        Change value of a parameter.
  -i buffer path, --inbuffer buffer path
                        Set input image to a buffer
  -o buffer path, --outbuffer buffer path
                        Set output image to a buffer
  -v, --verbose         Outputs information
  --timestamp           Add timestamp in log output
  --nogui               Command line version

```

### Dependencies
* gpuip:
  * [`OpenCL`](https://www.khronos.org/opencl/) *optional*
  * [`CUDA`](https://developer.nvidia.com/cuda-zone) *optional*
  * [`OpenGL`](http://www.opengl.org/) *optional*
    * [`GLFW`] (http://www.glfw.org/) *OpenGL context creation*
    * [`GLEW`](http://glew.sourceforge.net/) *OpenGL extensions*

* pygpuip:
  * [`Boost Python`](http://www.boost.org/) *python bindings*
  * [`Boost Numpy`] (https://github.com/ndarray/Boost.NumPy) *numpy python bindings*
  * [`OpenEXR`] (http://www.openexr.com/) *exr i/o*
  * [`CImg`] (http://cimg.sourceforge.net/) *png, jpeg,t iff, tga i/o*
  * [`libpng`] (http://www.libpng.org/pub/png/libpng.html) *png*
  * [`zlib`] (http://www.zlib.net) *compression used by OpenEXR and libpng*
  
* bin/gpuip
  * [Qt] (http://qt-project.org/) *GUI*
  * [PySide] (http://qt-project.org/wiki/PySide) *Qt python bindings*
  

### Build/Install ###

#### Linux/OSX

There are two bash scripts provided, `build.sh` and `install.sh`. If you want to generate your own Makefiles, use CMake:
```
mkdir build
cd build
cmake ..
make
sudo make install
```

#### Windows
There are two batch scripts provided, `build.bat` and `install.bat`. If you want to generate your own Visual Studio Solution, use CMake:
```
mkdir build
cd build
cmake ..
cmake --build . --config Release
```

If you have admin rights, you can add `--target INSTALL` to the last command to install files


### CMake options

```
BUILD_THIRD_PARTY_LIBS  // Build and link missing third party libraries
BUILD_SHARED_LIB        // Make gpuip a shared library
BUILD_WITH_OPENCL       // Support OpenCL (if found)
BUILD_WITH_CUDA         // Support CUDA (if found)
BUILD_WITH_GLSL         // Support GLSL (if found)
BUILD_PYTHON_BINDINGS   // Build Python bindings
BUILD_TESTS             // Build unit tests
BUILD_DOCS              // Generate Doxygen documenation
```

### Third party libraries
If the CMake option `BUILD_THIRD_PARTY_LIBS` is set to ON, the build will download the source code from the missing libraries and compile. This does not apply for the core libs OpenCL, CUDA and OpenGL since they are not open source. Although supported, it is not recommended to download and build boost from the git repo as it takes a long time to clone the submodules.

Following CMake variables can be set to help CMake find the third party libraries:

```
-DBOOST_ROOT=...
-DZLIB_ROOT=...
-DGLFW_ROOT=...
```

### Examples ###
The following examples are included the `examples` directory (can be run with `bin/gpuip`):

```
- linear interpolation
- box blur
- gaussian blur
- separable gaussian blur
```

### Tests ###

todo, add how to run tests
