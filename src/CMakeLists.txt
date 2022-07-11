cmake_minimum_required(VERSION 3.9)

set(CMAKE_CONFIGURATION_TYPES Debug Release CACHE TYPE INTERNAL FORCE)

project(face_reconstruction)

# Options
get_filename_component(PROJECT_DIR ../ ABSOLUTE)
set(LIBRARY_DIR "${PROJECT_DIR}/libs")

if(WIN32)
    set(Eigen3_DIR "${LIBRARY_DIR}/eigen/share/eigen3/cmake")
    set(glog_DIR "${LIBRARY_DIR}/glog/lib/cmake/glog")
    set(Ceres_DIR "${LIBRARY_DIR}/ceres/CMake")
    set(OpenCV_DIR "${LIBRARY_DIR}/opencv")
    set(HDF5_DIR "${LIBRARY_DIR}/hdf5/cmake")

    set(Eigen3_INCLUDE_DIR "${LIBRARY_DIR}/eigen/include")
    set(glog_INCLUDE_DIR "${LIBRARY_DIR}/glog/include")
    set(Ceres_INCLUDE_DIR "${LIBRARY_DIR}/ceres/include")
    set(OpenCV_INCLUDE_DIR "${LIBRARY_DIR}/opencv/include")
    set(HDF5_INCLUDE_DIR "${LIBRARY_DIR}/hdf5/include")
    set(GLFW_INCLUDE_DIR "${LIBRARY_DIR}/glfw/include")
    set(GLEW_INCLUDE_DIR "${LIBRARY_DIR}/glew/include")

    set(GLEW_LIBRARIES "${LIBRARY_DIR}/glew/lib/Release/x64/glew32s.lib")
    set(GLFW_LIBRARIES "${LIBRARY_DIR}/glfw/lib-vc2019/glfw3.lib")
    set(OPENGL32_LIBRARY "opengl32.lib")
else()
    set(Eigen3_DIR "${LIBRARY_DIR}/eigen/share/eigen3/cmake")
    set(glog_DIR "${LIBRARY_DIR}/glog/lib/cmake/glog")
    set(Ceres_DIR "${LIBRARY_DIR}/ceres/lib/cmake/Ceres")
    set(OpenCV_DIR "${LIBRARY_DIR}/opencv/lib/cmake/opencv4")
    set(HDF5_DIR "${LIBRARY_DIR}/hdf5/cmake")

    set(Eigen3_INCLUDE_DIR "${LIBRARY_DIR}/eigen/include")
    set(glog_INCLUDE_DIR "${LIBRARY_DIR}/glog/include")
    set(Ceres_INCLUDE_DIR "${LIBRARY_DIR}/ceres/include")
    set(OpenCV_INCLUDE_DIR "${LIBRARY_DIR}/opencv/include")
    set(HDF5_INCLUDE_DIR "${LIBRARY_DIR}/hdf5/include")
endif()


# Set C++ flags
set(CMAKE_CXX_STANDARD 17)

if(WIN32)
    find_package(glog REQUIRED)
    find_package(Ceres REQUIRED)
    find_package(OpenCV REQUIRED)
    find_package(Eigen3 REQUIRED)
    find_package(HDF5 REQUIRED)
    find_package(OpenGL REQUIRED)
else()
    find_package(glog REQUIRED)
    find_package(Ceres REQUIRED)
    find_package(OpenCV REQUIRED)
    find_package(Eigen3 REQUIRED)
    find_package(HDF5 REQUIRED)
    find_package(OpenGL REQUIRED)
    find_package(glfw3 REQUIRED)
    find_package(GLEW REQUIRED)
endif()

add_definitions("-D_DISABLE_EXTENDED_ALIGNED_STORAGE")

if (MSVC)
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /bigobj")
endif()

file(GLOB SOURCE_FILES
     "*.h"
     "*.cpp"
)

add_executable(face_reconstruction main.cpp ${SOURCE_FILES})
#add_compile_options("-fopenmp")
if(WIN32)
    target_link_libraries(face_reconstruction ${CERES_LIBRARIES} ${OpenCV_LIBS} ${HDF5_LIBRARIES} ${GLEW_LIBRARIES} ${GLFW_LIBRARIES} ${OPENGL32_LIBRARY})
    target_include_directories(face_reconstruction PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} ${Eigen3_INCLUDE_DIR} ${glog_INCLUDE_DIR} ${Ceres_INCLUDE_DIR}  ${OpenCV_INCLUDE_DIRS} ${HDF5_INCLUDE_DIR} ${GLFW_INCLUDE_DIR} ${GLEW_INCLUDE_DIR})
else()
    target_link_libraries(face_reconstruction ${CERES_LIBRARIES} ${OpenCV_LIBS} ${HDF5_LIBRARIES} GLEW::glew glfw OpenGL::GL)
    target_include_directories(face_reconstruction PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} ${Eigen3_INCLUDE_DIR} ${glog_INCLUDE_DIR} ${Ceres_INCLUDE_DIR}  ${OpenCV_INCLUDE_DIRS} ${HDF5_INCLUDE_DIR})
endif()