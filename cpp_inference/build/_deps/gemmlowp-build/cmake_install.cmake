# Install script for directory: /workspace/cpp_inference/build/gemmlowp/contrib

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "/usr/local")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "Release")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Install shared libraries without execute permission?
if(NOT DEFINED CMAKE_INSTALL_SO_NO_EXE)
  set(CMAKE_INSTALL_SO_NO_EXE "1")
endif()

# Is this installation the result of a crosscompile?
if(NOT DEFINED CMAKE_CROSSCOMPILING)
  set(CMAKE_CROSSCOMPILING "FALSE")
endif()

# Set default install directory permissions.
if(NOT DEFINED CMAKE_OBJDUMP)
  set(CMAKE_OBJDUMP "/usr/bin/objdump")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/gemmlowp/eight_bit_int_gemm" TYPE FILE FILES "/workspace/cpp_inference/build/gemmlowp/eight_bit_int_gemm/eight_bit_int_gemm.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/gemmlowp/meta" TYPE FILE FILES
    "/workspace/cpp_inference/build/gemmlowp/meta/base.h"
    "/workspace/cpp_inference/build/gemmlowp/meta/legacy_multi_thread_common.h"
    "/workspace/cpp_inference/build/gemmlowp/meta/legacy_multi_thread_gemm.h"
    "/workspace/cpp_inference/build/gemmlowp/meta/legacy_multi_thread_gemv.h"
    "/workspace/cpp_inference/build/gemmlowp/meta/legacy_operations_common.h"
    "/workspace/cpp_inference/build/gemmlowp/meta/legacy_single_thread_gemm.h"
    "/workspace/cpp_inference/build/gemmlowp/meta/multi_thread_common.h"
    "/workspace/cpp_inference/build/gemmlowp/meta/multi_thread_gemm.h"
    "/workspace/cpp_inference/build/gemmlowp/meta/multi_thread_transform.h"
    "/workspace/cpp_inference/build/gemmlowp/meta/quantized_mul_kernels.h"
    "/workspace/cpp_inference/build/gemmlowp/meta/quantized_mul_kernels_arm_32.h"
    "/workspace/cpp_inference/build/gemmlowp/meta/quantized_mul_kernels_arm_64.h"
    "/workspace/cpp_inference/build/gemmlowp/meta/single_thread_gemm.h"
    "/workspace/cpp_inference/build/gemmlowp/meta/single_thread_transform.h"
    "/workspace/cpp_inference/build/gemmlowp/meta/streams.h"
    "/workspace/cpp_inference/build/gemmlowp/meta/streams_arm_32.h"
    "/workspace/cpp_inference/build/gemmlowp/meta/streams_arm_64.h"
    "/workspace/cpp_inference/build/gemmlowp/meta/transform_kernels.h"
    "/workspace/cpp_inference/build/gemmlowp/meta/transform_kernels_arm_32.h"
    "/workspace/cpp_inference/build/gemmlowp/meta/transform_kernels_arm_64.h"
    )
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/gemmlowp/public" TYPE FILE FILES
    "/workspace/cpp_inference/build/gemmlowp/public/bit_depth.h"
    "/workspace/cpp_inference/build/gemmlowp/public/gemmlowp.h"
    "/workspace/cpp_inference/build/gemmlowp/public/map.h"
    "/workspace/cpp_inference/build/gemmlowp/public/output_stages.h"
    )
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/gemmlowp/profiling" TYPE FILE FILES
    "/workspace/cpp_inference/build/gemmlowp/profiling/instrumentation.h"
    "/workspace/cpp_inference/build/gemmlowp/profiling/profiler.h"
    "/workspace/cpp_inference/build/gemmlowp/profiling/pthread_everywhere.h"
    )
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/gemmlowp/internal" TYPE FILE FILES
    "/workspace/cpp_inference/build/gemmlowp/internal/allocator.h"
    "/workspace/cpp_inference/build/gemmlowp/internal/block_params.h"
    "/workspace/cpp_inference/build/gemmlowp/internal/common.h"
    "/workspace/cpp_inference/build/gemmlowp/internal/compute.h"
    "/workspace/cpp_inference/build/gemmlowp/internal/detect_platform.h"
    "/workspace/cpp_inference/build/gemmlowp/internal/dispatch_gemm_shape.h"
    "/workspace/cpp_inference/build/gemmlowp/internal/kernel.h"
    "/workspace/cpp_inference/build/gemmlowp/internal/kernel_avx.h"
    "/workspace/cpp_inference/build/gemmlowp/internal/kernel_default.h"
    "/workspace/cpp_inference/build/gemmlowp/internal/kernel_msa.h"
    "/workspace/cpp_inference/build/gemmlowp/internal/kernel_neon.h"
    "/workspace/cpp_inference/build/gemmlowp/internal/kernel_reference.h"
    "/workspace/cpp_inference/build/gemmlowp/internal/kernel_sse.h"
    "/workspace/cpp_inference/build/gemmlowp/internal/multi_thread_gemm.h"
    "/workspace/cpp_inference/build/gemmlowp/internal/output.h"
    "/workspace/cpp_inference/build/gemmlowp/internal/output_avx.h"
    "/workspace/cpp_inference/build/gemmlowp/internal/output_msa.h"
    "/workspace/cpp_inference/build/gemmlowp/internal/output_neon.h"
    "/workspace/cpp_inference/build/gemmlowp/internal/output_sse.h"
    "/workspace/cpp_inference/build/gemmlowp/internal/pack.h"
    "/workspace/cpp_inference/build/gemmlowp/internal/pack_avx.h"
    "/workspace/cpp_inference/build/gemmlowp/internal/pack_msa.h"
    "/workspace/cpp_inference/build/gemmlowp/internal/pack_neon.h"
    "/workspace/cpp_inference/build/gemmlowp/internal/pack_sse.h"
    "/workspace/cpp_inference/build/gemmlowp/internal/platform.h"
    "/workspace/cpp_inference/build/gemmlowp/internal/simd_wrappers.h"
    "/workspace/cpp_inference/build/gemmlowp/internal/simd_wrappers_common_neon_sse.h"
    "/workspace/cpp_inference/build/gemmlowp/internal/simd_wrappers_msa.h"
    "/workspace/cpp_inference/build/gemmlowp/internal/simd_wrappers_neon.h"
    "/workspace/cpp_inference/build/gemmlowp/internal/simd_wrappers_sse.h"
    "/workspace/cpp_inference/build/gemmlowp/internal/single_thread_gemm.h"
    "/workspace/cpp_inference/build/gemmlowp/internal/unpack.h"
    )
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/gemmlowp/fixedpoint" TYPE FILE FILES
    "/workspace/cpp_inference/build/gemmlowp/fixedpoint/fixedpoint.h"
    "/workspace/cpp_inference/build/gemmlowp/fixedpoint/fixedpoint_avx.h"
    "/workspace/cpp_inference/build/gemmlowp/fixedpoint/fixedpoint_msa.h"
    "/workspace/cpp_inference/build/gemmlowp/fixedpoint/fixedpoint_neon.h"
    "/workspace/cpp_inference/build/gemmlowp/fixedpoint/fixedpoint_sse.h"
    "/workspace/cpp_inference/build/gemmlowp/fixedpoint/fixedpoint_wasmsimd.h"
    )
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE STATIC_LIBRARY FILES "/workspace/cpp_inference/build/_deps/gemmlowp-build/libeight_bit_int_gemm.a")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/cmake/gemmlowp/gemmlowp-config.cmake")
    file(DIFFERENT _cmake_export_file_changed FILES
         "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/cmake/gemmlowp/gemmlowp-config.cmake"
         "/workspace/cpp_inference/build/_deps/gemmlowp-build/CMakeFiles/Export/bdd242ce7a57c2e75f6ccd2dc66d07f4/gemmlowp-config.cmake")
    if(_cmake_export_file_changed)
      file(GLOB _cmake_old_config_files "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/cmake/gemmlowp/gemmlowp-config-*.cmake")
      if(_cmake_old_config_files)
        string(REPLACE ";" ", " _cmake_old_config_files_text "${_cmake_old_config_files}")
        message(STATUS "Old export file \"$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/cmake/gemmlowp/gemmlowp-config.cmake\" will be replaced.  Removing files [${_cmake_old_config_files_text}].")
        unset(_cmake_old_config_files_text)
        file(REMOVE ${_cmake_old_config_files})
      endif()
      unset(_cmake_old_config_files)
    endif()
    unset(_cmake_export_file_changed)
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/cmake/gemmlowp" TYPE FILE FILES "/workspace/cpp_inference/build/_deps/gemmlowp-build/CMakeFiles/Export/bdd242ce7a57c2e75f6ccd2dc66d07f4/gemmlowp-config.cmake")
  if(CMAKE_INSTALL_CONFIG_NAME MATCHES "^([Rr][Ee][Ll][Ee][Aa][Ss][Ee])$")
    file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/cmake/gemmlowp" TYPE FILE FILES "/workspace/cpp_inference/build/_deps/gemmlowp-build/CMakeFiles/Export/bdd242ce7a57c2e75f6ccd2dc66d07f4/gemmlowp-config-release.cmake")
  endif()
endif()

