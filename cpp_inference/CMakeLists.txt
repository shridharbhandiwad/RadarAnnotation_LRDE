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
        PATCH_COMMAND  ""
    )
    
    # Download TensorFlow Lite source (but don't configure yet)
    FetchContent_GetProperties(tensorflow)
    if(NOT tensorflow_POPULATED)
        message(STATUS "Populating TensorFlow Lite source...")
        FetchContent_Populate(tensorflow)
        
        # Patch gemmlowp before TensorFlow Lite configures it
        if(MINGW OR WIN32)
            message(STATUS "Applying MinGW/Windows patches to TensorFlow Lite dependencies...")
            
            # Find and patch gemmlowp if it gets downloaded by TensorFlow Lite
            file(GLOB_RECURSE GEMMLOWP_CMAKE_FILES 
                "${tensorflow_SOURCE_DIR}/*gemmlowp*/CMakeLists.txt"
                "${CMAKE_BINARY_DIR}/*gemmlowp*/CMakeLists.txt"
            )
            
            foreach(GEMMLOWP_CMAKE IN LISTS GEMMLOWP_CMAKE_FILES)
                if(EXISTS "${GEMMLOWP_CMAKE}")
                    file(READ "${GEMMLOWP_CMAKE}" GEMMLOWP_CONTENT)
                    string(FIND "${GEMMLOWP_CONTENT}" "DISABLED_FOR_MINGW" ALREADY_PATCHED)
                    
                    if(ALREADY_PATCHED EQUAL -1)
                        message(STATUS "Pre-patching gemmlowp at: ${GEMMLOWP_CMAKE}")
                        string(REGEX REPLACE 
                            "add_library\\(eight_bit_int_gemm"
                            "# DISABLED_FOR_MINGW: add_library(eight_bit_int_gemm"
                            GEMMLOWP_CONTENT "${GEMMLOWP_CONTENT}")
                        string(REGEX REPLACE 
                            "add_executable\\(eight_bit_int_gemm"
                            "# DISABLED_FOR_MINGW: add_executable(eight_bit_int_gemm"
                            GEMMLOWP_CONTENT "${GEMMLOWP_CONTENT}")
                        file(WRITE "${GEMMLOWP_CMAKE}" "${GEMMLOWP_CONTENT}")
                    endif()
                endif()
            endforeach()
        endif()
        
        # Patch TensorFlow Lite CMakeLists.txt for MinGW/Windows
        if(MINGW OR WIN32)
            set(TFLITE_CMAKE "${tensorflow_SOURCE_DIR}/tensorflow/lite/CMakeLists.txt")
            if(EXISTS "${TFLITE_CMAKE}")
                file(READ "${TFLITE_CMAKE}" TFLITE_CONTENT)
                
                # Check if already patched
                string(FIND "${TFLITE_CONTENT}" "PATCHED_FOR_MINGW_MAX_MIN" ALREADY_PATCHED_MAX_MIN)
                string(FIND "${TFLITE_CONTENT}" "PATCHED_FOR_MINGW_GEMMLOWP" ALREADY_PATCHED_GEMMLOWP)
                
                if(ALREADY_PATCHED_MAX_MIN EQUAL -1)
                    message(STATUS "Patching TensorFlow Lite CMakeLists.txt for MinGW compatibility (max/min macros)...")
                    
                    # Remove problematic max/min macro definitions that contain > and < characters
                    # These cause issues on Windows due to shell redirection operators in cmd.exe
                    # The flags must be removed BEFORE the project() command to prevent C compiler test failures
                    
                    # Strategy: Remove any line containing these macros entirely
                    # This is more aggressive but necessary for Windows MinGW builds
                    
                    # Remove entire lines containing max macro definitions
                    string(REGEX REPLACE 
                        "[^\n]*-Dmax\\(a,b\\)=[^\n]*\n"
                        "# PATCHED_FOR_MINGW_MAX_MIN: Line removed (contained problematic max macro)\n"
                        TFLITE_CONTENT "${TFLITE_CONTENT}")
                    
                    # Remove entire lines containing min macro definitions  
                    string(REGEX REPLACE 
                        "[^\n]*-Dmin\\(a,b\\)=[^\n]*\n"
                        "# PATCHED_FOR_MINGW_MAX_MIN: Line removed (contained problematic min macro)\n"
                        TFLITE_CONTENT "${TFLITE_CONTENT}")
                    
                    # Also handle multi-line set() commands - remove max macro from CMAKE_C_FLAGS
                    string(REGEX REPLACE 
                        "(set\\(CMAKE_C_FLAGS[^)]*)-Dmax\\(a,b\\)=\\(\\(a\\)>\\(b\\)\\?\\(a\\):\\(b\\)\\)([^)]*\\))"
                        "\\1\\2 # PATCHED_FOR_MINGW_MAX_MIN"
                        TFLITE_CONTENT "${TFLITE_CONTENT}")
                    
                    # Remove min macro from CMAKE_C_FLAGS
                    string(REGEX REPLACE 
                        "(set\\(CMAKE_C_FLAGS[^)]*)-Dmin\\(a,b\\)=\\(\\(a\\)<\\(b\\)\\?\\(a\\):\\(b\\)\\)([^)]*\\))"
                        "\\1\\2 # PATCHED_FOR_MINGW_MAX_MIN"
                        TFLITE_CONTENT "${TFLITE_CONTENT}")
                    
                    # Remove max macro from CMAKE_CXX_FLAGS
                    string(REGEX REPLACE 
                        "(set\\(CMAKE_CXX_FLAGS[^)]*)-Dmax\\(a,b\\)=\\(\\(a\\)>\\(b\\)\\?\\(a\\):\\(b\\)\\)([^)]*\\))"
                        "\\1\\2 # PATCHED_FOR_MINGW_MAX_MIN"
                        TFLITE_CONTENT "${TFLITE_CONTENT}")
                    
                    # Remove min macro from CMAKE_CXX_FLAGS
                    string(REGEX REPLACE 
                        "(set\\(CMAKE_CXX_FLAGS[^)]*)-Dmin\\(a,b\\)=\\(\\(a\\)<\\(b\\)\\?\\(a\\):\\(b\\)\\)([^)]*\\))"
                        "\\1\\2 # PATCHED_FOR_MINGW_MAX_MIN"
                        TFLITE_CONTENT "${TFLITE_CONTENT}")
                    
                    # Handle string(APPEND CMAKE_C_FLAGS ...) variants
                    string(REGEX REPLACE 
                        "(string\\([Aa][Pp][Pp][Ee][Nn][Dd][^)]*CMAKE_C_FLAGS[^)]*)-Dmax\\(a,b\\)=\\(\\(a\\)>\\(b\\)\\?\\(a\\):\\(b\\)\\)([^)]*\\))"
                        "\\1\\2 # PATCHED_FOR_MINGW_MAX_MIN"
                        TFLITE_CONTENT "${TFLITE_CONTENT}")
                    
                    string(REGEX REPLACE 
                        "(string\\([Aa][Pp][Pp][Ee][Nn][Dd][^)]*CMAKE_C_FLAGS[^)]*)-Dmin\\(a,b\\)=\\(\\(a\\)<\\(b\\)\\?\\(a\\):\\(b\\)\\)([^)]*\\))"
                        "\\1\\2 # PATCHED_FOR_MINGW_MAX_MIN"
                        TFLITE_CONTENT "${TFLITE_CONTENT}")
                    
                    # Handle string(APPEND CMAKE_CXX_FLAGS ...) variants
                    string(REGEX REPLACE 
                        "(string\\([Aa][Pp][Pp][Ee][Nn][Dd][^)]*CMAKE_CXX_FLAGS[^)]*)-Dmax\\(a,b\\)=\\(\\(a\\)>\\(b\\)\\?\\(a\\):\\(b\\)\\)([^)]*\\))"
                        "\\1\\2 # PATCHED_FOR_MINGW_MAX_MIN"
                        TFLITE_CONTENT "${TFLITE_CONTENT}")
                    
                    string(REGEX REPLACE 
                        "(string\\([Aa][Pp][Pp][Ee][Nn][Dd][^)]*CMAKE_CXX_FLAGS[^)]*)-Dmin\\(a,b\\)=\\(\\(a\\)<\\(b\\)\\?\\(a\\):\\(b\\)\\)([^)]*\\))"
                        "\\1\\2 # PATCHED_FOR_MINGW_MAX_MIN"
                        TFLITE_CONTENT "${TFLITE_CONTENT}")
                    
                    # Remove from add_compile_definitions() - max
                    string(REGEX REPLACE 
                        "(add_compile_definitions\\([^)]*)(max\\(a,b\\)=\\(\\(a\\)>\\(b\\)\\?\\(a\\):\\(b\\)\\))([^)]*\\))"
                        "\\1\\3 # PATCHED_FOR_MINGW_MAX_MIN"
                        TFLITE_CONTENT "${TFLITE_CONTENT}")
                    
                    # Remove from add_compile_definitions() - min
                    string(REGEX REPLACE 
                        "(add_compile_definitions\\([^)]*)(min\\(a,b\\)=\\(\\(a\\)<\\(b\\)\\?\\(a\\):\\(b\\)\\))([^)]*\\))"
                        "\\1\\3 # PATCHED_FOR_MINGW_MAX_MIN"
                        TFLITE_CONTENT "${TFLITE_CONTENT}")
                    
                    # Remove from target_compile_definitions() - max
                    string(REGEX REPLACE 
                        "(target_compile_definitions\\([^)]*)(max\\(a,b\\)=\\(\\(a\\)>\\(b\\)\\?\\(a\\):\\(b\\)\\))([^)]*\\))"
                        "\\1\\3 # PATCHED_FOR_MINGW_MAX_MIN"
                        TFLITE_CONTENT "${TFLITE_CONTENT}")
                    
                    # Remove from target_compile_definitions() - min  
                    string(REGEX REPLACE 
                        "(target_compile_definitions\\([^)]*)(min\\(a,b\\)=\\(\\(a\\)<\\(b\\)\\?\\(a\\):\\(b\\)\\))([^)]*\\))"
                        "\\1\\3 # PATCHED_FOR_MINGW_MAX_MIN"
                        TFLITE_CONTENT "${TFLITE_CONTENT}")
                    
                    file(WRITE "${TFLITE_CMAKE}" "${TFLITE_CONTENT}")
                    message(STATUS "  -> Successfully patched TensorFlow Lite CMakeLists.txt (max/min macros)")
                else()
                    message(STATUS "TensorFlow Lite CMakeLists.txt already patched (max/min), skipping...")
                endif()
                
                # Now inject gemmlowp patching code into TensorFlow Lite CMakeLists.txt
                if(ALREADY_PATCHED_GEMMLOWP EQUAL -1)
                    file(READ "${TFLITE_CMAKE}" TFLITE_CONTENT)
                    
                    message(STATUS "Injecting gemmlowp patching code into TensorFlow Lite CMakeLists.txt...")
                    
                    # Find where gemmlowp is populated and inject patching code right after FetchContent_Populate
                    # The injection will patch gemmlowp CMakeLists.txt before add_subdirectory processes it
                    set(GEMMLOWP_PATCH_CODE "\n# PATCHED_FOR_MINGW_GEMMLOWP: Disable eight_bit_int_gemm for MinGW\nif(MINGW OR WIN32)\n  FetchContent_GetProperties(gemmlowp)\n  if(gemmlowp_POPULATED)\n    set(GEMMLOWP_CMAKE \"\${gemmlowp_SOURCE_DIR}/CMakeLists.txt\")\n    if(EXISTS \"\${GEMMLOWP_CMAKE}\")\n      file(READ \"\${GEMMLOWP_CMAKE}\" GEMMLOWP_CONTENT)\n      string(FIND \"\${GEMMLOWP_CONTENT}\" \"DISABLED_FOR_MINGW\" GEMMLOWP_ALREADY_PATCHED)\n      if(GEMMLOWP_ALREADY_PATCHED EQUAL -1)\n        message(STATUS \"Patching gemmlowp to disable eight_bit_int_gemm...\")\n        string(REGEX REPLACE \"add_library\\\\(eight_bit_int_gemm\" \"# DISABLED_FOR_MINGW: add_library(eight_bit_int_gemm\" GEMMLOWP_CONTENT \"\${GEMMLOWP_CONTENT}\")\n        string(REGEX REPLACE \"add_executable\\\\(eight_bit_int_gemm\" \"# DISABLED_FOR_MINGW: add_executable(eight_bit_int_gemm\" GEMMLOWP_CONTENT \"\${GEMMLOWP_CONTENT}\")\n        string(REGEX REPLACE \"target_link_libraries\\\\(eight_bit_int_gemm\" \"# DISABLED_FOR_MINGW: target_link_libraries(eight_bit_int_gemm\" GEMMLOWP_CONTENT \"\${GEMMLOWP_CONTENT}\")\n        string(REGEX REPLACE \"set_target_properties\\\\(eight_bit_int_gemm\" \"# DISABLED_FOR_MINGW: set_target_properties(eight_bit_int_gemm\" GEMMLOWP_CONTENT \"\${GEMMLOWP_CONTENT}\")\n        file(WRITE \"\${GEMMLOWP_CMAKE}\" \"\${GEMMLOWP_CONTENT}\")\n        message(STATUS \"  -> gemmlowp patched successfully\")\n      endif()\n    endif()\n  endif()\nendif()\n")
                    
                    # Find a safe injection point - after "find_package(gemmlowp REQUIRED)" 
                    # This is outside any macro definitions and after gemmlowp is discovered
                    string(FIND "${TFLITE_CONTENT}" "find_package(gemmlowp REQUIRED)" FINDPKG_POS)
                    if(NOT FINDPKG_POS EQUAL -1)
                        # Find the end of this line by extracting substring from found position
                        string(LENGTH "${TFLITE_CONTENT}" CONTENT_LENGTH)
                        math(EXPR REMAINING_LENGTH "${CONTENT_LENGTH} - ${FINDPKG_POS}")
                        string(SUBSTRING "${TFLITE_CONTENT}" ${FINDPKG_POS} ${REMAINING_LENGTH} REMAINING_CONTENT)
                        string(FIND "${REMAINING_CONTENT}" "\n" NEWLINE_OFFSET)
                        if(NOT NEWLINE_OFFSET EQUAL -1)
                            math(EXPR NEWLINE_AFTER "${FINDPKG_POS} + ${NEWLINE_OFFSET}")
                            math(EXPR INSERT_POS "${NEWLINE_AFTER} + 1")
                            string(SUBSTRING "${TFLITE_CONTENT}" 0 ${INSERT_POS} BEFORE_PART)
                            string(SUBSTRING "${TFLITE_CONTENT}" ${INSERT_POS} -1 AFTER_PART)
                            set(TFLITE_CONTENT "${BEFORE_PART}${GEMMLOWP_PATCH_CODE}${AFTER_PART}")
                            file(WRITE "${TFLITE_CMAKE}" "${TFLITE_CONTENT}")
                            message(STATUS "  -> Successfully injected gemmlowp patching code after find_package(gemmlowp)")
                        endif()
                    else()
                        # Fallback: try to find "# Find TensorFlow Lite dependencies." and inject after it
                        string(FIND "${TFLITE_CONTENT}" "# Find TensorFlow Lite dependencies." DEPS_COMMENT_POS)
                        if(NOT DEPS_COMMENT_POS EQUAL -1)
                            string(LENGTH "${TFLITE_CONTENT}" CONTENT_LENGTH)
                            math(EXPR REMAINING_LENGTH "${CONTENT_LENGTH} - ${DEPS_COMMENT_POS}")
                            string(SUBSTRING "${TFLITE_CONTENT}" ${DEPS_COMMENT_POS} ${REMAINING_LENGTH} REMAINING_CONTENT)
                            string(FIND "${REMAINING_CONTENT}" "\n" NEWLINE_OFFSET)
                            if(NOT NEWLINE_OFFSET EQUAL -1)
                                math(EXPR NEWLINE_AFTER "${DEPS_COMMENT_POS} + ${NEWLINE_OFFSET}")
                                math(EXPR INSERT_POS "${NEWLINE_AFTER} + 1")
                                string(SUBSTRING "${TFLITE_CONTENT}" 0 ${INSERT_POS} BEFORE_PART)
                                string(SUBSTRING "${TFLITE_CONTENT}" ${INSERT_POS} -1 AFTER_PART)
                                set(TFLITE_CONTENT "${BEFORE_PART}${GEMMLOWP_PATCH_CODE}${AFTER_PART}")
                                file(WRITE "${TFLITE_CMAKE}" "${TFLITE_CONTENT}")
                                message(STATUS "  -> Successfully injected gemmlowp patching code after dependencies comment")
                            endif()
                        else()
                            message(WARNING "Could not find suitable injection point for gemmlowp patch in TensorFlow Lite CMakeLists.txt")
                        endif()
                    endif()
                else()
                    message(STATUS "TensorFlow Lite CMakeLists.txt already has gemmlowp patch, skipping...")
                endif()
            endif()
        endif()
        
        # Now add TensorFlow Lite to the build
        add_subdirectory(${tensorflow_SOURCE_DIR}/tensorflow/lite ${tensorflow_BINARY_DIR})
    endif()
    
    # Post-configuration backup: Remove eight_bit_int_gemm from the build if it still exists
    # This is a backup in case the patching didn't work
    if(MINGW OR WIN32)
        # Method 1: Try to exclude the target
        if(TARGET eight_bit_int_gemm)
            message(STATUS "Excluding eight_bit_int_gemm target from MinGW build (method 1)")
            set_target_properties(eight_bit_int_gemm PROPERTIES 
                EXCLUDE_FROM_ALL TRUE
                EXCLUDE_FROM_DEFAULT_BUILD TRUE
            )
        endif()
        
        # Method 2: Override the add_library/add_executable commands temporarily
        # This creates a macro that disables eight_bit_int_gemm creation
        # Note: cmake_language(DEFER ...) requires CMake 3.19+
        if(CMAKE_VERSION VERSION_GREATER_EQUAL "3.19")
            macro(disable_eight_bit_int_gemm)
                if(TARGET eight_bit_int_gemm)
                    set_property(TARGET eight_bit_int_gemm PROPERTY EXCLUDE_FROM_ALL TRUE)
                    message(STATUS "Disabled eight_bit_int_gemm target (method 2)")
                endif()
            endmacro()
            
            # Run the macro after all targets are defined
            cmake_language(DEFER CALL disable_eight_bit_int_gemm)
        else()
            message(STATUS "Skipping deferred eight_bit_int_gemm fix (requires CMake 3.19+, you have ${CMAKE_VERSION})")
        endif()
        
        # Method 3: Modify the Makefile generation to skip this target
        set_directory_properties(PROPERTIES 
            ADDITIONAL_MAKE_CLEAN_FILES "${CMAKE_BINARY_DIR}/_deps/gemmlowp-build/libeight_bit_int_gemm.a"
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
    
    if(WIN32)
        # On Windows, link against the import library (.lib), not the DLL
        set(ONNXRUNTIME_LIBRARIES ${onnxruntime_SOURCE_DIR}/lib/onnxruntime.lib)
        # Store DLL path for runtime deployment
        set(ONNXRUNTIME_DLL ${onnxruntime_SOURCE_DIR}/lib/onnxruntime.dll)
    elseif(APPLE)
        set(ONNXRUNTIME_LIBRARIES ${onnxruntime_SOURCE_DIR}/lib/libonnxruntime.dylib)
    else()
        set(ONNXRUNTIME_LIBRARIES ${onnxruntime_SOURCE_DIR}/lib/libonnxruntime.so)
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

# On Windows, copy ONNX Runtime DLL to build directory for runtime
if(WIN32 AND DEFINED ONNXRUNTIME_DLL)
    if(EXISTS ${ONNXRUNTIME_DLL})
        add_custom_command(TARGET radar_tagger_multioutput POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                ${ONNXRUNTIME_DLL}
                $<TARGET_FILE_DIR:radar_tagger_multioutput>
            COMMENT "Copying ONNX Runtime DLL to build directory"
        )
        install(FILES ${ONNXRUNTIME_DLL} DESTINATION bin)
    endif()
endif()

# Print configuration
message(STATUS "")
message(STATUS "Radar Tagger C++ Configuration:")
message(STATUS "  Version: ${PROJECT_VERSION}")
message(STATUS "  C++ Standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "  Build Type: ${CMAKE_BUILD_TYPE}")
message(STATUS "  TensorFlow Lite: ${TensorFlowLite_LIBRARIES}")
message(STATUS "  ONNX Runtime: ${ONNXRUNTIME_LIBRARIES}")
if(WIN32 AND DEFINED ONNXRUNTIME_DLL)
    message(STATUS "  ONNX Runtime DLL: ${ONNXRUNTIME_DLL}")
endif()
message(STATUS "")
