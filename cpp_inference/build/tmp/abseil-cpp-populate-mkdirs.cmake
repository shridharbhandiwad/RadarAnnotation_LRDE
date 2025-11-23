# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file Copyright.txt or https://cmake.org/licensing for details.

cmake_minimum_required(VERSION 3.5)

file(MAKE_DIRECTORY
  "/workspace/cpp_inference/build/abseil-cpp"
  "/workspace/cpp_inference/build/_deps/abseil-cpp-build"
  "/workspace/cpp_inference/build"
  "/workspace/cpp_inference/build/tmp"
  "/workspace/cpp_inference/build/src/abseil-cpp-populate-stamp"
  "/workspace/cpp_inference/build/src"
  "/workspace/cpp_inference/build/src/abseil-cpp-populate-stamp"
)

set(configSubDirs )
foreach(subDir IN LISTS configSubDirs)
    file(MAKE_DIRECTORY "/workspace/cpp_inference/build/src/abseil-cpp-populate-stamp/${subDir}")
endforeach()
if(cfgdir)
  file(MAKE_DIRECTORY "/workspace/cpp_inference/build/src/abseil-cpp-populate-stamp${cfgdir}") # cfgdir has leading slash
endif()
