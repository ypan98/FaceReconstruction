cmake_minimum_required(VERSION 3.6)
set(CMAKE_CONFIGURATION_TYPES Debug Release CACHE TYPE INTERNAL FORCE)
project(faceReconstruction)
set(CMAKE_CXX_STANDARD 14)

# Libraries 



# Files
set(HEADER_FILES,
	faceModel
)

set(SOURCE_FILES 
)

add_executable(faceReconstruction main.cpp ${HEADER_FILES} ${SOURCE_FILES} "faceModel.h")
target_link_libraries(faceReconstruction)
#target_include_directories(faceReconstruction PUBLIC)