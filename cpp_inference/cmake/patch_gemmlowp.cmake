# Patch gemmlowp CMakeLists.txt to disable eight_bit_int_gemm on MinGW
# This script is called after gemmlowp is downloaded but before it's configured

if(MINGW OR WIN32)
    set(GEMMLOWP_DIR "${CMAKE_BINARY_DIR}/gemmlowp")
    set(GEMMLOWP_CMAKE "${GEMMLOWP_DIR}/CMakeLists.txt")
    
    if(EXISTS "${GEMMLOWP_CMAKE}")
        message(STATUS "Patching gemmlowp CMakeLists.txt for MinGW compatibility...")
        
        file(READ "${GEMMLOWP_CMAKE}" GEMMLOWP_CONTENT)
        
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
        
        file(WRITE "${GEMMLOWP_CMAKE}" "${GEMMLOWP_CONTENT}")
        message(STATUS "  -> Successfully patched gemmlowp CMakeLists.txt")
    else()
        message(WARNING "gemmlowp CMakeLists.txt not found at: ${GEMMLOWP_CMAKE}")
    endif()
endif()
