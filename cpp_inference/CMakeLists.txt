cmake_minimum_required(VERSION 3.16...3.30)
project(RadarTaggerCpp VERSION 1.0.0 LANGUAGES CXX)

# Set CMake policies for compatibility with CMake 4.1+
if(POLICY CMP0135)
    cmake_policy(SET CMP0135 NEW)  # Use extraction timestamp for URL downloads
endif()

if(POLICY CMP0169)
    cmake_policy(SET CMP0169 OLD)  # Allow deprecated FetchContent_Populate for TensorFlow dependencies
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Options
option(USE_SYSTEM_TFLITE "Use system-installed TensorFlow Lite" OFF)
option(USE_SYSTEM_ONNXRUNTIME "Use system-installed ONNX Runtime" OFF)

# Find dependencies
find_package(Threads REQUIRED)

# TensorFlow Lite
if(USE_SYSTEM_TFLITE)
    find_package(TensorFlowLite REQUIRED)
else()
    # Download and build TensorFlow Lite from source
    include(FetchContent)
    
    message(STATUS "Downloading TensorFlow Lite...")
    
    set(TFLITE_ENABLE_XNNPACK OFF CACHE BOOL "" FORCE)
    set(TFLITE_ENABLE_RUY ON CACHE BOOL "" FORCE)
    
    # Fix for MinGW build issues with gemmlowp and abseil
    if(MINGW OR WIN32)
        # Disable building tests and examples
        set(BUILD_TESTING OFF CACHE BOOL "" FORCE)
        # Use single-threaded builds for problematic targets
        set(CMAKE_BUILD_PARALLEL_LEVEL 1)
        # Disable problematic SIMD optimizations
        set(GEMMLOWP_ENABLE_NEON OFF CACHE BOOL "" FORCE)
        
        # Add min/max macros for cpuinfo compatibility
        add_compile_definitions(NOMINMAX)
        
        message(STATUS "MinGW/Windows detected - gemmlowp and cpuinfo patches will be applied")
    endif()
    
    FetchContent_Declare(
        tensorflow
        GIT_REPOSITORY https://github.com/tensorflow/tensorflow.git
        GIT_TAG        v2.14.0
        GIT_SHALLOW    TRUE
        SOURCE_SUBDIR  tensorflow/lite
    )
    
    # Download TensorFlow Lite source (but don't configure yet)
    FetchContent_GetProperties(tensorflow)
    if(NOT tensorflow_POPULATED)
        message(STATUS "Populating TensorFlow Lite source...")
        FetchContent_Populate(tensorflow)
        
        # Patch TensorFlow Lite CMakeLists.txt for MinGW/Windows
        if(MINGW OR WIN32)
            set(TFLITE_CMAKE "${tensorflow_SOURCE_DIR}/tensorflow/lite/CMakeLists.txt")
            if(EXISTS "${TFLITE_CMAKE}")
                file(READ "${TFLITE_CMAKE}" TFLITE_CONTENT)
                
                # Check if already patched
                string(FIND "${TFLITE_CONTENT}" "PATCHED_FOR_MINGW_MAX_MIN" ALREADY_PATCHED)
                
                if(ALREADY_PATCHED EQUAL -1)
                    message(STATUS "Patching TensorFlow Lite CMakeLists.txt for MinGW compatibility...")
                    
                    # Remove problematic max/min macro definitions that contain > and < characters
                    # These cause issues on Windows due to shell redirection operators
                    # Match various forms: with quotes, without quotes, in set(), etc.
                    
                    # Remove from CMAKE_C_FLAGS and CMAKE_CXX_FLAGS
                    string(REGEX REPLACE 
                        "set\\(CMAKE_C_FLAGS[^)]*-Dmax\\(a,b\\)=[^)]*\\)"
                        "# PATCHED_FOR_MINGW_MAX_MIN: Removed from CMAKE_C_FLAGS"
                        TFLITE_CONTENT "${TFLITE_CONTENT}")
                    
                    string(REGEX REPLACE 
                        "set\\(CMAKE_C_FLAGS[^)]*-Dmin\\(a,b\\)=[^)]*\\)"
                        "# PATCHED_FOR_MINGW_MAX_MIN: Removed from CMAKE_C_FLAGS"
                        TFLITE_CONTENT "${TFLITE_CONTENT}")
                    
                    string(REGEX REPLACE 
                        "set\\(CMAKE_CXX_FLAGS[^)]*-Dmax\\(a,b\\)=[^)]*\\)"
                        "# PATCHED_FOR_MINGW_MAX_MIN: Removed from CMAKE_CXX_FLAGS"
                        TFLITE_CONTENT "${TFLITE_CONTENT}")
                    
                    string(REGEX REPLACE 
                        "set\\(CMAKE_CXX_FLAGS[^)]*-Dmin\\(a,b\\)=[^)]*\\)"
                        "# PATCHED_FOR_MINGW_MAX_MIN: Removed from CMAKE_CXX_FLAGS"
                        TFLITE_CONTENT "${TFLITE_CONTENT}")
                    
                    # Remove from add_compile_definitions
                    string(REGEX REPLACE 
                        "add_compile_definitions\\([^)]*max\\(a,b\\)=[^)]*\\)"
                        "# PATCHED_FOR_MINGW_MAX_MIN: Removed problematic max macro"
                        TFLITE_CONTENT "${TFLITE_CONTENT}")
                    
                    string(REGEX REPLACE 
                        "add_compile_definitions\\([^)]*min\\(a,b\\)=[^)]*\\)"
                        "# PATCHED_FOR_MINGW_MAX_MIN: Removed problematic min macro"
                        TFLITE_CONTENT "${TFLITE_CONTENT}")
                    
                    # Remove from target_compile_definitions
                    string(REGEX REPLACE 
                        "target_compile_definitions\\([^)]*max\\(a,b\\)=[^)]*\\)"
                        "# PATCHED_FOR_MINGW_MAX_MIN: Removed problematic max macro"
                        TFLITE_CONTENT "${TFLITE_CONTENT}")
                    
                    string(REGEX REPLACE 
                        "target_compile_definitions\\([^)]*min\\(a,b\\)=[^)]*\\)"
                        "# PATCHED_FOR_MINGW_MAX_MIN: Removed problematic min macro"
                        TFLITE_CONTENT "${TFLITE_CONTENT}")
                    
                    # Remove standalone -D definitions
                    string(REGEX REPLACE 
                        "-Dmax\\(a,b\\)=\\(\\(a\\)>\\(b\\)\\?\\(a\\):\\(b\\)\\)"
                        ""
                        TFLITE_CONTENT "${TFLITE_CONTENT}")
                    
                    string(REGEX REPLACE 
                        "-Dmin\\(a,b\\)=\\(\\(a\\)<\\(b\\)\\?\\(a\\):\\(b\\)\\)"
                        ""
                        TFLITE_CONTENT "${TFLITE_CONTENT}")
                    
                    file(WRITE "${TFLITE_CMAKE}" "${TFLITE_CONTENT}")
                    message(STATUS "  -> Successfully patched TensorFlow Lite CMakeLists.txt")
                else()
                    message(STATUS "TensorFlow Lite CMakeLists.txt already patched, skipping...")
                endif()
            endif()
        endif()
        
        # Now add TensorFlow Lite to the build
        add_subdirectory(${tensorflow_SOURCE_DIR}/tensorflow/lite ${tensorflow_BINARY_DIR})
        
        # Apply gemmlowp patch AFTER TensorFlow Lite subdirectory is added
        # This allows us to patch the downloaded gemmlowp before it's built
        if(MINGW OR WIN32)
            set(GEMMLOWP_DIR "${CMAKE_BINARY_DIR}/_deps/gemmlowp-src")
            if(EXISTS "${GEMMLOWP_DIR}/CMakeLists.txt")
                file(READ "${GEMMLOWP_DIR}/CMakeLists.txt" GEMMLOWP_CONTENT)
                
                # Check if already patched
                string(FIND "${GEMMLOWP_CONTENT}" "DISABLED_FOR_MINGW" ALREADY_PATCHED)
                
                if(ALREADY_PATCHED EQUAL -1)
                    message(STATUS "Patching gemmlowp CMakeLists.txt to disable eight_bit_int_gemm...")
                    
                    # Comment out add_library for eight_bit_int_gemm
                    string(REGEX REPLACE 
                        "add_library\\(eight_bit_int_gemm"
                        "# DISABLED_FOR_MINGW: add_library(eight_bit_int_gemm"
                        GEMMLOWP_CONTENT "${GEMMLOWP_CONTENT}")
                    
                    # Comment out add_executable for eight_bit_int_gemm    
                    string(REGEX REPLACE 
                        "add_executable\\(eight_bit_int_gemm"
                        "# DISABLED_FOR_MINGW: add_executable(eight_bit_int_gemm"
                        GEMMLOWP_CONTENT "${GEMMLOWP_CONTENT}")
                        
                    # Comment out any target_link_libraries for eight_bit_int_gemm
                    string(REGEX REPLACE 
                        "target_link_libraries\\(eight_bit_int_gemm"
                        "# DISABLED_FOR_MINGW: target_link_libraries(eight_bit_int_gemm"
                        GEMMLOWP_CONTENT "${GEMMLOWP_CONTENT}")
                    
                    # Comment out any set_target_properties for eight_bit_int_gemm
                    string(REGEX REPLACE 
                        "set_target_properties\\(eight_bit_int_gemm"
                        "# DISABLED_FOR_MINGW: set_target_properties(eight_bit_int_gemm"
                        GEMMLOWP_CONTENT "${GEMMLOWP_CONTENT}")
                    
                    file(WRITE "${GEMMLOWP_DIR}/CMakeLists.txt" "${GEMMLOWP_CONTENT}")
                    message(STATUS "  -> Successfully patched gemmlowp CMakeLists.txt")
                else()
                    message(STATUS "gemmlowp CMakeLists.txt already patched, skipping...")
                endif()
            else()
                message(STATUS "gemmlowp not found at ${GEMMLOWP_DIR}, checking alternate locations...")
                # Try alternate location
                set(GEMMLOWP_DIR "${CMAKE_BINARY_DIR}/gemmlowp")
                if(EXISTS "${GEMMLOWP_DIR}/CMakeLists.txt")
                    message(STATUS "Found gemmlowp at ${GEMMLOWP_DIR}, applying patch...")
                    file(READ "${GEMMLOWP_DIR}/CMakeLists.txt" GEMMLOWP_CONTENT)
                    
                    string(FIND "${GEMMLOWP_CONTENT}" "DISABLED_FOR_MINGW" ALREADY_PATCHED)
                    if(ALREADY_PATCHED EQUAL -1)
                        string(REGEX REPLACE 
                            "add_library\\(eight_bit_int_gemm"
                            "# DISABLED_FOR_MINGW: add_library(eight_bit_int_gemm"
                            GEMMLOWP_CONTENT "${GEMMLOWP_CONTENT}")
                        string(REGEX REPLACE 
                            "add_executable\\(eight_bit_int_gemm"
                            "# DISABLED_FOR_MINGW: add_executable(eight_bit_int_gemm"
                            GEMMLOWP_CONTENT "${GEMMLOWP_CONTENT}")
                        string(REGEX REPLACE 
                            "target_link_libraries\\(eight_bit_int_gemm"
                            "# DISABLED_FOR_MINGW: target_link_libraries(eight_bit_int_gemm"
                            GEMMLOWP_CONTENT "${GEMMLOWP_CONTENT}")
                        string(REGEX REPLACE 
                            "set_target_properties\\(eight_bit_int_gemm"
                            "# DISABLED_FOR_MINGW: set_target_properties(eight_bit_int_gemm"
                            GEMMLOWP_CONTENT "${GEMMLOWP_CONTENT}")
                        file(WRITE "${GEMMLOWP_DIR}/CMakeLists.txt" "${GEMMLOWP_CONTENT}")
                        message(STATUS "  -> Successfully patched gemmlowp CMakeLists.txt at alternate location")
                    endif()
                endif()
            endif()
        endif()
    endif()
    
    # Post-configuration fix for MinGW: Exclude eight_bit_int_gemm from the build
    # This is a backup in case the target still exists
    if((MINGW OR WIN32) AND TARGET eight_bit_int_gemm)
        message(STATUS "Excluding eight_bit_int_gemm target from MinGW build")
        set_target_properties(eight_bit_int_gemm PROPERTIES 
            EXCLUDE_FROM_ALL TRUE
            EXCLUDE_FROM_DEFAULT_BUILD TRUE
        )
    endif()
    
    set(TensorFlowLite_INCLUDE_DIRS 
        ${tensorflow_SOURCE_DIR}
        ${CMAKE_BINARY_DIR}/_deps/tensorflow-build/flatbuffers/include
    )
    set(TensorFlowLite_LIBRARIES tensorflow-lite)
endif()

# JSON library (header-only)
# Set policy to allow older CMake minimum version in nlohmann/json
set(CMAKE_POLICY_VERSION_MINIMUM 3.5)

FetchContent_Declare(
    json
    URL https://github.com/nlohmann/json/releases/download/v3.11.2/json.tar.xz
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)
FetchContent_MakeAvailable(json)

# ONNX Runtime (for XGBoost and Random Forest)
if(USE_SYSTEM_ONNXRUNTIME)
    find_package(onnxruntime REQUIRED)
else()
    # Download pre-built ONNX Runtime
    message(STATUS "Downloading ONNX Runtime...")
    
    if(WIN32)
        set(ONNXRUNTIME_URL "https://github.com/microsoft/onnxruntime/releases/download/v1.16.3/onnxruntime-win-x64-1.16.3.zip")
        set(ONNXRUNTIME_HASH "")
    elseif(APPLE)
        set(ONNXRUNTIME_URL "https://github.com/microsoft/onnxruntime/releases/download/v1.16.3/onnxruntime-osx-arm64-1.16.3.tgz")
        set(ONNXRUNTIME_HASH "")
    else()
        set(ONNXRUNTIME_URL "https://github.com/microsoft/onnxruntime/releases/download/v1.16.3/onnxruntime-linux-x64-1.16.3.tgz")
        set(ONNXRUNTIME_HASH "")
    endif()
    
    FetchContent_Declare(
        onnxruntime
        URL ${ONNXRUNTIME_URL}
        DOWNLOAD_EXTRACT_TIMESTAMP TRUE
    )
    FetchContent_MakeAvailable(onnxruntime)
    
    set(ONNXRUNTIME_INCLUDE_DIRS ${onnxruntime_SOURCE_DIR}/include)
    set(ONNXRUNTIME_LIBRARIES ${onnxruntime_SOURCE_DIR}/lib/libonnxruntime.so)
    
    if(WIN32)
        set(ONNXRUNTIME_LIBRARIES ${onnxruntime_SOURCE_DIR}/lib/onnxruntime.dll)
    elseif(APPLE)
        set(ONNXRUNTIME_LIBRARIES ${onnxruntime_SOURCE_DIR}/lib/libonnxruntime.dylib)
    endif()
endif()

# Source files
set(SOURCES
    radar_tagger.cpp
    main.cpp
)

set(HEADERS
    radar_tagger.h
)

set(SOURCES_MULTIOUTPUT
    radar_tagger_multioutput.cpp
    main_multioutput.cpp
)

set(HEADERS_MULTIOUTPUT
    radar_tagger_multioutput.h
)

# Create executables
add_executable(radar_tagger ${SOURCES} ${HEADERS})
add_executable(radar_tagger_multioutput ${SOURCES_MULTIOUTPUT} ${HEADERS_MULTIOUTPUT})

# Include directories for both executables
target_include_directories(radar_tagger PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${TensorFlowLite_INCLUDE_DIRS}
)

target_include_directories(radar_tagger_multioutput PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${TensorFlowLite_INCLUDE_DIRS}
    ${ONNXRUNTIME_INCLUDE_DIRS}
)

# Link libraries for both executables
target_link_libraries(radar_tagger PRIVATE
    ${TensorFlowLite_LIBRARIES}
    nlohmann_json::nlohmann_json
    Threads::Threads
    ${CMAKE_DL_LIBS}
)

target_link_libraries(radar_tagger_multioutput PRIVATE
    ${TensorFlowLite_LIBRARIES}
    ${ONNXRUNTIME_LIBRARIES}
    nlohmann_json::nlohmann_json
    Threads::Threads
    ${CMAKE_DL_LIBS}
)

# Compiler options for both executables
if(MSVC)
    target_compile_options(radar_tagger PRIVATE /W4)
    target_compile_options(radar_tagger_multioutput PRIVATE /W4)
else()
    target_compile_options(radar_tagger PRIVATE -Wall -Wextra -Wpedantic)
    target_compile_options(radar_tagger_multioutput PRIVATE -Wall -Wextra -Wpedantic)
    
    # Add big-obj flag for MinGW
    if(MINGW OR WIN32)
        target_compile_options(radar_tagger PRIVATE -Wa,-mbig-obj)
        target_compile_options(radar_tagger_multioutput PRIVATE -Wa,-mbig-obj)
    endif()
endif()

# Install targets
install(TARGETS radar_tagger radar_tagger_multioutput DESTINATION bin)
install(FILES ${HEADERS} ${HEADERS_MULTIOUTPUT} DESTINATION include/radar_tagger)

# Print configuration
message(STATUS "")
message(STATUS "Radar Tagger C++ Configuration:")
message(STATUS "  Version: ${PROJECT_VERSION}")
message(STATUS "  C++ Standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "  Build Type: ${CMAKE_BUILD_TYPE}")
message(STATUS "  TensorFlow Lite: ${TensorFlowLite_LIBRARIES}")
message(STATUS "  ONNX Runtime: ${ONNXRUNTIME_LIBRARIES}")
message(STATUS "")
