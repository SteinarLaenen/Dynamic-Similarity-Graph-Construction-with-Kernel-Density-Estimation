cmake_minimum_required(VERSION 3.14)
project(dynamic-kde-clustering)

set(CMAKE_CXX_STANDARD 20)

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread")

# Find and include the STAG library
find_package(stag REQUIRED)
message(STATUS "[dynamic-kde] Found STAG!")
include_directories(${STAG_INCLUDE_DIRS})

set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)
message(STATUS "[dynamic-kde] Found Threads")

# Define the executable for your project
add_executable(dynamic-kde
        src/multithreading/ctpl_stl.h
        src/kde.h
        src/lsh.h
        src/kde.cpp
        src/lsh.cpp
        src/kde_main.cpp
        src/timing.h
        src/timing.cpp
        src/naiveDynamicCKNS.cpp
        src/dynamicExactKDE.cpp
        src/dynamicRandomSamplingKDE.cpp)

target_link_libraries(dynamic-kde stag)

add_executable(dynamic-similarity-graph
        src/simgraph_main.cpp
        src/multithreading/ctpl_stl.h
        src/kde.h
        src/lsh.h
        src/kde.cpp
        src/lsh.cpp
        src/timing.h
        src/timing.cpp
        src/naiveDynamicCKNS.cpp
        src/dynamicExactKDE.cpp
        src/dynamicRandomSamplingKDE.cpp
        src/cluster.h
        src/cluster.cpp
        src/dynamicFCSimGraph.cpp
        src/naiveDynamicCPS.cpp
        src/dynamickNNSimGraph.cpp)

target_link_libraries(dynamic-similarity-graph stag)
