FIND_PACKAGE(PackageHandleStandardArgs)

FIND_LIBRARY(
		Boost_NUMPY_LIBRARY
		NAMES boost_numpy
		PATHS 
		/usr/local/lib/
		/usr/lib64/
		${Boost_LIBRARY_DIRS})

FIND_PATH(Boost_NUMPY_INCLUDE_DIRS
		NAMES
		boost/numpy.hpp
		PATHS
		/usr/local/include
		${Boost_INCLUDE_DIRS})

FIND_PACKAGE_HANDLE_STANDARD_ARGS(BoostNumpy DEFAULT_MSG Boost_NUMPY_LIBRARY Boost_NUMPY_INCLUDE_DIRS)