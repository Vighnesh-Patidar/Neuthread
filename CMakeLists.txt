cmake_minimum_required(VERSION 3.28)
project(Neuthread)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(LIBRARY_TYPE SHARED)


include_directories(include)

file(GLOB THREADING_SRC src)
add_library(Neuthread SHARED neuron.cpp
        neuron.cpp
        neuron.h
        Threader.cpp
        Threader.h
        Layer.cpp
        Layer.h
        Network.cpp
        Network.h
        main.cpp)
