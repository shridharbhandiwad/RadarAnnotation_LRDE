# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file Copyright.txt or https://cmake.org/licensing for details.

cmake_minimum_required(VERSION 3.5)

file(MAKE_DIRECTORY
  "/workspace/cpp_inference/build/flatbuffers"
  "/workspace/cpp_inference/build/flatbuffers-flatc/src/flatbuffers-flatc-build"
  "/workspace/cpp_inference/build/flatbuffers-flatc"
  "/workspace/cpp_inference/build/flatbuffers-flatc/tmp"
  "/workspace/cpp_inference/build/flatbuffers-flatc/src/flatbuffers-flatc-stamp"
  "/workspace/cpp_inference/build/flatbuffers-flatc/src"
  "/workspace/cpp_inference/build/flatbuffers-flatc/src/flatbuffers-flatc-stamp"
)

set(configSubDirs )
foreach(subDir IN LISTS configSubDirs)
    file(MAKE_DIRECTORY "/workspace/cpp_inference/build/flatbuffers-flatc/src/flatbuffers-flatc-stamp/${subDir}")
endforeach()
if(cfgdir)
  file(MAKE_DIRECTORY "/workspace/cpp_inference/build/flatbuffers-flatc/src/flatbuffers-flatc-stamp${cfgdir}") # cfgdir has leading slash
endif()
