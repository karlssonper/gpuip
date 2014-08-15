#!/usr/bin/env python
import sys
import os
import shutil
import glob
import re

# order matters
_3RDPARTYLIBS = [
    "zlib", 
    "openexr", 
    "boost_python", 
    "boost_numpy",
    "glew",
    "glfw", 
    "png", 
    "cimg"
]
_TARGET = "install" if sys.platform != "win32" else "INSTALL"
_CWD = os.getcwd()

def _cmd(command):
    print command
    if os.system(command):
        sys.exit(1)

def _lib_exists(lib, install_prefix):
    prefix, suffixA, suffixB = "", "", ""
    if os.uname()[0] == "Linux":
        prefix, suffixA, suffixB = "lib", "so", "a"
    elif os.uname()[0] == "Darwin":
        prefix, suffixA, suffixB = "lib", "dylib", "a"
    elif os.uname()[0] == "Win32":
        prefix, suffixA, suffixB = "", "dll", "lib"

    files = glob.glob(os.path.realpath(install_prefix) +"/lib/*."+suffixA) + \
            glob.glob(os.path.realpath(install_prefix) +"/lib/*."+suffixB)
    
    found = any(x.count(prefix+lib) > 0 and \
                (x.endswith(suffixA) or x.endswith(suffixB)) for x in files)
    if found:
        print "Found " + prefix + lib + " in " + install_prefix
    return found

def init_update_submodule(name):
    _cmd("git submodule init %s" % name)
    _cmd("git submodule update %s" % name)

def build(name, install_prefix, cmake_extraargs = ""):
    if not os.path.exists(_CWD + "/3rdparty/build/%s" % name):
        os.makedirs(_CWD + "/3rdparty/build/%s" % name)
    os.chdir(_CWD + "/3rdparty/build/%s" % name)
    _cmd("cmake %s/3rdparty/%s %s -DCMAKE_INSTALL_PREFIX=%s" \
         % (_CWD, name, cmake_extraargs, install_prefix))
    
    # Remove install cache (will install to old CMAKE_INSTALL_PREFIX if not
    _cmd("cmake %s ." % (" ".join(["-UINSTALL_%s_DIR" % s \
                                   for s in ["INC","LIB","MAN","PKGCONFIG"]])))
    _cmd("cmake --build . --config Release --target %s -- -j8 " % _TARGET)

def build_zlib(install_prefix):
    if _lib_exists("z", install_prefix):
        return

    print "Building ZLIB..."

    init_update_submodule("3rdparty/zlib")
    build("zlib",install_prefix)

def build_openexr(install_prefix):
    libs = ["Half", "Iex", "IexMath", "IlmImf", "IlmThread", "Imath"]
    if all(_lib_exists(lib, install_prefix) for lib in libs):
        return

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
          "-DZLIB_ROOT=%s " % install_prefix + \
          "-DILMBASE_PACKAGE_PREFIX=%s" % install_prefix)

def build_boost_python(install_prefix):
    if _lib_exists("boost_python", install_prefix):
        return

    print "Building Boost Python..."
    init_update_submodule("3rdparty/boost")
    os.chdir("3rdparty/boost")
    # Only download necessary boost modules (should save some time)
    libs = [ "array", "assert", "bind", "concept_check", "config", "container",
             "conversion", "core", "detail", "foreach", "function", 
             "functional", "graph", "integer", "iterator", "lexical_cast",
             "math", "move", "mpl", "multi_index", "numeric/conversion", 
             "optional", "parameter", "predef", "preprocessor", "property_map",
             "python", "range", "serialization", "smart_ptr", "static_assert", 
             "throw_exception", "tuple", "type_traits", "typeof", "unordered",
             "utility", "wave" ]
    for lib in libs:
       init_update_submodule("libs/" + lib)
    init_update_submodule("tools/build")
    init_update_submodule("tools/inspect")
    _cmd("./bootstrap.sh")
    _cmd("./b2 headers")
    _cmd("./b2 install --with-python --build-dir=%s --prefix=%s"\
              % (_CWD + "/3rdparty/build", install_prefix))

def build_boost_numpy(install_prefix):
    if _lib_exists("boost_numpy", install_prefix):
        return

    print "Building Boost Numpy..."
    init_update_submodule("3rdparty/Boost.NumPy")
    build("Boost.NumPy",install_prefix, "-DBOOST_ROOT=%s" % install_prefix)

def build_glew(install_prefix):
    if _lib_exists("glew", install_prefix):
        return
    
    print "Building GLEW..."
    init_update_submodule("3rdparty/glew")
    build("glew", install_prefix, "-DONLY_LIBS=TRUE")

def build_glfw(install_prefix):
    if _lib_exists("glfw3", install_prefix):
        return

    print "Building GLFW..."
    init_update_submodule("3rdparty/glfw")
    build("glfw", install_prefix)

def build_png(install_prefix):
    if _lib_exists("png", install_prefix):
        return

    print "Building PNG..."
    init_update_submodule("3rdparty/libpng")
    build("libpng", install_prefix)

def build_cimg(install_prefix):
    print "Building CImg..."
    init_update_submodule("3rdparty/CImg")
    if not os.path.exists(install_prefix + "/include"):
        os.mkdir(install_prefix + "/include")
    # Header only library
    shutil.copy("3rdparty/CImg/CImg.h", install_prefix + "/include/CImg.h")

def build_3rdparty(args, install_prefix):
    deps = [arg.lower() for arg in args]
    for lib in _3RDPARTYLIBS:
        if lib in deps:
            os.chdir(_CWD)
            exec("build_%s('%s')" % (lib, install_prefix))

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--install-prefix":
        build_3rdparty(sys.argv[3:], os.path.realpath(sys.argv[2]))
    else:
        print "usage:\nbuild_3rdparty --install-prefix path lib1 lib2 lib3\n"
        print "available third party libs:"
        for lib in _3RDPARTYLIBS:
            print "  " + lib
