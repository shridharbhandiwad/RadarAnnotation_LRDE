# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file Copyright.txt or https://cmake.org/licensing for details.

cmake_minimum_required(VERSION 3.5)

file(MAKE_DIRECTORY
  "/workspace/cpp_inference/build/cpuinfo"
  "/workspace/cpp_inference/build/_deps/cpuinfo-build"
  "/workspace/cpp_inference/build/_deps/cpuinfo-subbuild/cpuinfo-populate-prefix"
  "/workspace/cpp_inference/build/_deps/cpuinfo-subbuild/cpuinfo-populate-prefix/tmp"
  "/workspace/cpp_inference/build/_deps/cpuinfo-subbuild/cpuinfo-populate-prefix/src/cpuinfo-populate-stamp"
  "/workspace/cpp_inference/build/_deps/cpuinfo-subbuild/cpuinfo-populate-prefix/src"
  "/workspace/cpp_inference/build/_deps/cpuinfo-subbuild/cpuinfo-populate-prefix/src/cpuinfo-populate-stamp"
)

set(configSubDirs )
foreach(subDir IN LISTS configSubDirs)
    file(MAKE_DIRECTORY "/workspace/cpp_inference/build/_deps/cpuinfo-subbuild/cpuinfo-populate-prefix/src/cpuinfo-populate-stamp/${subDir}")
endforeach()
if(cfgdir)
  file(MAKE_DIRECTORY "/workspace/cpp_inference/build/_deps/cpuinfo-subbuild/cpuinfo-populate-prefix/src/cpuinfo-populate-stamp${cfgdir}") # cfgdir has leading slash
endif()
