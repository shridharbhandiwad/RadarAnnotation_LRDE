# patch_cpuinfo.cmake
# Patches cpuinfo library to add max/min macro definitions for Windows/MinGW builds
#
# This script can be used in two ways:
# 1. Included in CMakeLists.txt after TensorFlow Lite is populated
# 2. Run manually: cmake -P patch_cpuinfo.cmake

message(STATUS "===== Patching cpuinfo for Windows/MinGW =====")

# Check if we're on Windows/MinGW
if(NOT (WIN32 OR MINGW))
    message(STATUS "Not a Windows/MinGW build, skipping cpuinfo patch")
    return()
endif()

# Function to find cpuinfo directory
function(find_cpuinfo_dir OUTPUT_VAR)
    set(POSSIBLE_PATHS
        "${CMAKE_BINARY_DIR}/_deps/cpuinfo-src"
        "${CMAKE_BINARY_DIR}/cpuinfo"
        "${CMAKE_SOURCE_DIR}/build/_deps/cpuinfo-src"
        "${CMAKE_SOURCE_DIR}/build/cpuinfo"
    )
    
    foreach(PATH ${POSSIBLE_PATHS})
        if(EXISTS "${PATH}/CMakeLists.txt")
            set(${OUTPUT_VAR} "${PATH}" PARENT_SCOPE)
            message(STATUS "Found cpuinfo at: ${PATH}")
            return()
        endif()
    endforeach()
    
    set(${OUTPUT_VAR} "" PARENT_SCOPE)
    message(WARNING "cpuinfo directory not found in any expected location")
endfunction()

# Find cpuinfo directory
find_cpuinfo_dir(CPUINFO_DIR)

if(NOT CPUINFO_DIR)
    message(FATAL_ERROR "Could not locate cpuinfo directory. Make sure TensorFlow Lite has been populated.")
endif()

set(CPUINFO_CMAKE "${CPUINFO_DIR}/CMakeLists.txt")

if(NOT EXISTS "${CPUINFO_CMAKE}")
    message(FATAL_ERROR "cpuinfo CMakeLists.txt not found at: ${CPUINFO_CMAKE}")
endif()

# Read cpuinfo CMakeLists.txt
file(READ "${CPUINFO_CMAKE}" CPUINFO_CONTENT)

# Check if already patched
string(FIND "${CPUINFO_CONTENT}" "PATCHED_MAX_MIN_MACROS" ALREADY_PATCHED)

if(NOT ALREADY_PATCHED EQUAL -1)
    message(STATUS "cpuinfo CMakeLists.txt already patched, skipping...")
    return()
endif()

message(STATUS "Patching cpuinfo CMakeLists.txt...")

# Find the project() command
string(FIND "${CPUINFO_CONTENT}" "project(" PROJECT_POS)

if(PROJECT_POS EQUAL -1)
    # Try alternative: find CMakeLists header
    string(FIND "${CPUINFO_CONTENT}" "cmake_minimum_required" CMAKE_MIN_POS)
    if(NOT CMAKE_MIN_POS EQUAL -1)
        # Find the end of the line
        string(LENGTH "${CPUINFO_CONTENT}" CONTENT_LENGTH)
        math(EXPR REMAINING_LENGTH "${CONTENT_LENGTH} - ${CMAKE_MIN_POS}")
        string(SUBSTRING "${CPUINFO_CONTENT}" ${CMAKE_MIN_POS} ${REMAINING_LENGTH} REMAINING_CONTENT)
        string(FIND "${REMAINING_CONTENT}" "\n" NEWLINE_OFFSET)
        if(NOT NEWLINE_OFFSET EQUAL -1)
            math(EXPR NEWLINE_AFTER "${CMAKE_MIN_POS} + ${NEWLINE_OFFSET}")
            math(EXPR INSERT_POS "${NEWLINE_AFTER} + 1")
            set(PROJECT_POS ${INSERT_POS})
        endif()
    endif()
endif()

if(PROJECT_POS EQUAL -1)
    message(FATAL_ERROR "Could not find suitable insertion point in cpuinfo CMakeLists.txt")
endif()

# Find the end of the project() line or the line after cmake_minimum_required
string(LENGTH "${CPUINFO_CONTENT}" CONTENT_LENGTH)
math(EXPR REMAINING_LENGTH "${CONTENT_LENGTH} - ${PROJECT_POS}")
string(SUBSTRING "${CPUINFO_CONTENT}" ${PROJECT_POS} ${REMAINING_LENGTH} REMAINING_CONTENT)
string(FIND "${REMAINING_CONTENT}" "\n" NEWLINE_OFFSET)

if(NEWLINE_OFFSET EQUAL -1)
    message(FATAL_ERROR "Could not find newline after insertion point")
endif()

math(EXPR NEWLINE_AFTER "${PROJECT_POS} + ${NEWLINE_OFFSET}")
math(EXPR INSERT_POS "${NEWLINE_AFTER} + 1")

# Create the patch content
set(PATCH_CONTENT "\n# PATCHED_MAX_MIN_MACROS: Add max/min macros for Windows compatibility\n")
set(PATCH_CONTENT "${PATCH_CONTENT}# This patch fixes the implicit declaration error for max/min functions\n")
set(PATCH_CONTENT "${PATCH_CONTENT}# in cpuinfo's x86/windows/init.c file\n")
set(PATCH_CONTENT "${PATCH_CONTENT}if(WIN32 OR MINGW)\n")
set(PATCH_CONTENT "${PATCH_CONTENT}    add_compile_definitions(max(a,b)=((a)>(b)?(a):(b)))\n")
set(PATCH_CONTENT "${PATCH_CONTENT}    add_compile_definitions(min(a,b)=((a)<(b)?(a):(b)))\n")
set(PATCH_CONTENT "${PATCH_CONTENT}    message(STATUS \"cpuinfo: Added max/min macro definitions for Windows/MinGW\")\n")
set(PATCH_CONTENT "${PATCH_CONTENT}endif()\n\n")

# Split the content and insert the patch
string(SUBSTRING "${CPUINFO_CONTENT}" 0 ${INSERT_POS} BEFORE_PART)
string(SUBSTRING "${CPUINFO_CONTENT}" ${INSERT_POS} -1 AFTER_PART)
set(CPUINFO_CONTENT "${BEFORE_PART}${PATCH_CONTENT}${AFTER_PART}")

# Write the patched content back
file(WRITE "${CPUINFO_CMAKE}" "${CPUINFO_CONTENT}")

message(STATUS "Successfully patched cpuinfo CMakeLists.txt")
message(STATUS "  Added max/min macro definitions for Windows/MinGW compatibility")
message(STATUS "===== cpuinfo patch complete =====")

# Optional: Also patch the source file directly as a backup
set(INIT_C_FILE "${CPUINFO_DIR}/src/x86/windows/init.c")
if(EXISTS "${INIT_C_FILE}")
    file(READ "${INIT_C_FILE}" INIT_C_CONTENT)
    string(FIND "${INIT_C_CONTENT}" "PATCHED_MAX_MIN_SOURCE" INIT_C_PATCHED)
    
    if(INIT_C_PATCHED EQUAL -1)
        message(STATUS "Also patching init.c source file directly...")
        
        # Find the first #include
        string(FIND "${INIT_C_CONTENT}" "#include" FIRST_INCLUDE_POS)
        if(NOT FIRST_INCLUDE_POS EQUAL -1)
            # Add the macro definitions before the first include
            set(SOURCE_PATCH "/* PATCHED_MAX_MIN_SOURCE: Add max/min macros for Windows */\n")
            set(SOURCE_PATCH "${SOURCE_PATCH}#ifndef max\n")
            set(SOURCE_PATCH "${SOURCE_PATCH}#define max(a,b) (((a) > (b)) ? (a) : (b))\n")
            set(SOURCE_PATCH "${SOURCE_PATCH}#endif\n")
            set(SOURCE_PATCH "${SOURCE_PATCH}#ifndef min\n")
            set(SOURCE_PATCH "${SOURCE_PATCH}#define min(a,b) (((a) < (b)) ? (a) : (b))\n")
            set(SOURCE_PATCH "${SOURCE_PATCH}#endif\n\n")
            
            string(SUBSTRING "${INIT_C_CONTENT}" 0 ${FIRST_INCLUDE_POS} BEFORE_INCLUDE)
            string(SUBSTRING "${INIT_C_CONTENT}" ${FIRST_INCLUDE_POS} -1 AFTER_INCLUDE)
            set(INIT_C_CONTENT "${BEFORE_INCLUDE}${SOURCE_PATCH}${AFTER_INCLUDE}")
            
            file(WRITE "${INIT_C_FILE}" "${INIT_C_CONTENT}")
            message(STATUS "  Successfully patched init.c source file")
        endif()
    endif()
endif()
