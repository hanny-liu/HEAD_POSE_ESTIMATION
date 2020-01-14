cmake_minimum_required(VERSION 3.5)
project(head_pose_estimation)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_FLAGS "-std=c++14 -O3")

#OpenCV
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

#pcl
find_package(PCL REQUIRED)
include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})
list(REMOVE_ITEM PCL_LIBRARIES vtkproj4)

include_directories("include")
#include_directories("/usr/include/eigen3")

include("/home/lhw/dlib/dlib/cmake")

#openmg
#set(CMAKE_CXX_STANDARD 11)
#FIND_PACKAGE( OpenMP REQUIRED)
#if(OPENMP_FOUND)
#    message("OPENMP FOUND")
#    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
#    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
#endif()
find_package(PNG QUIET)
if (PNG_FOUND)
set(CMAKE_REQUIRED_LIBRARIES ${PNG_LIBRARY})
CHECK_FUNCTION_EXISTS(png_create_read_struct LIBPNG_IS_GOOD)
endif()
#if (PNG_FOUND AND LIBPNG_IS_GOOD)
#include_directories(${PNG_INCLUDE_DIR})
#set (dlib_needed_libraries ${dlib_needed_libraries} ${PNG_LIBRARY})
#else()

#executable
add_executable(head_pose_estimation head_pose_estimation.cpp )
target_link_libraries(head_pose_estimation ${OpenCV_LIBS} ${PCL_LIBRARIES} dlib::dlib)

