cmake_minimum_required(VERSION 2.8)
project("CBIR_project")
#set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} /home/keiths/opencv/opencv-2.4.11_build/build)
SET(CMAKE_C_COMPILER mpicc)
SET(CMAKE_CXX_COMPILER mpicxx)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wno-literal-suffix -g -fopenmp")
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x -stdlib=libc++") -Wliteral-suffix

find_package(OpenCV REQUIRED)
#find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})
add_executable(cbir main.cpp lib.cpp mydatatype.cpp mcdetect.cpp feature.cpp featureP.cpp regionprops.cpp f_Haralick.cpp f_Geometric.cpp train.cpp)
target_link_libraries(cbir ${OpenCV_LIBS})

install (TARGETS cbir DESTINATION "$HOME/jobs")

#install (FILES "${PROJECT_BINARY_DIR}/TutorialConfig.h" DESTINATION include)
