cmake_minimum_required(VERSION 3.15)
project(RadarTaggerCpp VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Options
option(USE_SYSTEM_TFLITE "Use system-installed TensorFlow Lite" OFF)

# Find dependencies
find_package(Threads REQUIRED)

# TensorFlow Lite
if(USE_SYSTEM_TFLITE)
    find_package(TensorFlowLite REQUIRED)
else()
    # Download and build TensorFlow Lite from source
    include(FetchContent)
    
    message(STATUS "Downloading TensorFlow Lite...")
    FetchContent_Declare(
        tensorflow
        GIT_REPOSITORY https://github.com/tensorflow/tensorflow.git
        GIT_TAG        v2.14.0
        GIT_SHALLOW    TRUE
        SOURCE_SUBDIR  tensorflow/lite
    )
    
    set(TFLITE_ENABLE_XNNPACK OFF CACHE BOOL "" FORCE)
    set(TFLITE_ENABLE_RUY ON CACHE BOOL "" FORCE)
    
    FetchContent_MakeAvailable(tensorflow)
    
    set(TensorFlowLite_INCLUDE_DIRS 
        ${tensorflow_SOURCE_DIR}
        ${CMAKE_BINARY_DIR}/_deps/tensorflow-build/flatbuffers/include
    )
    set(TensorFlowLite_LIBRARIES tensorflow-lite)
endif()

# JSON library (header-only)
FetchContent_Declare(
    json
    URL https://github.com/nlohmann/json/releases/download/v3.11.2/json.tar.xz
)
FetchContent_MakeAvailable(json)

# Source files
set(SOURCES
    radar_tagger.cpp
    main.cpp
)

set(HEADERS
    radar_tagger.h
)

# Create executable
add_executable(radar_tagger ${SOURCES} ${HEADERS})

# Include directories
target_include_directories(radar_tagger PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${TensorFlowLite_INCLUDE_DIRS}
)

# Link libraries
target_link_libraries(radar_tagger PRIVATE
    ${TensorFlowLite_LIBRARIES}
    nlohmann_json::nlohmann_json
    Threads::Threads
    ${CMAKE_DL_LIBS}
)

# Compiler options
if(MSVC)
    target_compile_options(radar_tagger PRIVATE /W4)
else()
    target_compile_options(radar_tagger PRIVATE -Wall -Wextra -Wpedantic)
endif()

# Install targets
install(TARGETS radar_tagger DESTINATION bin)
install(FILES ${HEADERS} DESTINATION include/radar_tagger)

# Print configuration
message(STATUS "")
message(STATUS "Radar Tagger C++ Configuration:")
message(STATUS "  Version: ${PROJECT_VERSION}")
message(STATUS "  C++ Standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "  Build Type: ${CMAKE_BUILD_TYPE}")
message(STATUS "  TensorFlow Lite: ${TensorFlowLite_LIBRARIES}")
message(STATUS "")
