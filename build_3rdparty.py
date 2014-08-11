#!/usr/bin/env python
import sys
import os
import shutil

_TARGET = "install" if sys.platform != "win32" else "INSTALL"
_CWD = os.getcwd()

def _cmd(command):
    print command
    if os.system(command):
        sys.exit(1)

def init_update_submodule(name):
    _cmd("git submodule init %s" % name)
    _cmd("git submodule update %s" % name)

def build(name, install_prefix, cmake_extraargs = ""):
    if not os.path.exists("3rdparty/build/%s" % name):
        os.makedirs("3rdparty/build/%s" % name)
    os.chdir("3rdparty/build/%s" % name)
    _cmd("cmake %s/3rdparty/%s %s -DCMAKE_INSTALL_PREFIX=%s" \
         % (_CWD, name, cmake_extraargs, install_prefix))
    # Remove install cache (will install to old CMAKE_INSTALL_PREFIX if not
    _cmd("cmake %s ." % ("-UINSTALL_BIN_DIR "
                         "-UINSTALL_INC_DIR "
                         "-UINSTALL_LIB_DIR "
                         "-UINSTALL_MAN_DIR "
                         "-UINSTALL_PKGCONFIG_DIR"))
    _cmd("cmake --build . --config Release --target %s -- -j8 " % _TARGET)

def build_zlib(install_prefix):
    print "Building ZLIB..."
    init_update_submodule("3rdparty/zlib")
    build("zlib",install_prefix)

def build_openexr(install_prefix):
    print "Building OpenEXR..."
    init_update_submodule("3rdparty/openexr")
    build("openexr/IlmBase", install_prefix)
    os.chdir(_CWD)

    # hack to remove test from build in mac( doesn't compile)
    # remove this once OpenEXR builds in Mac OS
    if sys.platform == "darwin":
        remove_str = "ADD_SUBDIRECTORY ( IlmImfTest )"
        txt = open("3rdparty/openexr/OpenEXR/CMakeLists.txt").read()
        if txt.find("\n" + remove_str) > 0:
            open("3rdparty/openexr/OpenEXR/CMakeLists.txt", "w").write(
                txt.replace(remove_str, "#" + remove_str))
  
    build("openexr/OpenEXR", install_prefix, 
          "-DZLIB_ROOT=%s " % _INSTALL_PREFIX + \
          "-DILMBASE_PACKAGE_PREFIX=%s" % _INSTALL_PREFIX)

def build_boost_python(install_prefix):
    print "Building Boost Python..."
    init_update_submodule("3rdparty/boost")
    os.chdir("3rdparty/boost")
    init_update_submodule("libs/python")
    init_update_submodule("libs/wave")
    init_update_submodule("tools/build")
    init_update_submodule("tools/inspect")
    _cmd("./bootstrap.sh")
    _cmd("./b2 install --with-python --build-dir=%s --prefix=%s"\
              % (_CWD + "/3rdparty/build", install_prefix))

def build_boost_numpy(install_prefix):
    print "Building Boost Numpy..."
    init_update_submodule("3rdparty/Boost.NumPy")
    build("Boost.NumPy",install_prefix)

def build_glew(install_prefix):
    print "Building GLEW..."
    init_update_submodule("3rdparty/glew")
    build("glew", install_prefix)

def build_glfw(install_prefix):
    print "Building GLFW..."
    init_update_submodule("3rdparty/glfw")
    build("glfw", install_prefix)

def build_png(install_prefix):
    print "Building PNG..."
    init_update_submodule("3rdparty/libpng")
    build("libpng", install_prefix)

def build_cimg(install_prefix):
    print "Building CImg..."
    init_update_submodule("3rdparty/CImg")
    if not os.path.exists(_INSTALL_PREFIX + "/include"):
        os.mkdir(_INSTALL_PREFIX + "/include")
    # Header only library
    shutil.copy("3rdparty/CImg/CImg.h", install_prefix + "/include/CImg.h")

def build_3rdparty(args, install_prefix):
    deps = [arg.lower() for arg in args]
    # order matters
    thirdpartylibs = ["zlib", 
                      "openexr", 
                      "boost_python", 
                      "boost_numpy",
                      "glew",
                      "glfw", 
                      "png", 
                      "cimg"]
    for lib in thirdpartylibs:
        if lib in deps:
            os.chdir(_CWD)
            exec("build_%s('%s')" % (lib, install_prefix))

if __name__ == "__main__":
    if sys.argv[1] == "--install-prefix":
        build_3rdparty(sys.argv[3:], os.path.realpath(sys.argv[2]))
    else:
        print "error: usage is \n build_3rdparty --install-prefix path deps"
