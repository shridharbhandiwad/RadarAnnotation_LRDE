# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file Copyright.txt or https://cmake.org/licensing for details.

cmake_minimum_required(VERSION 3.5)

file(MAKE_DIRECTORY
  "/workspace/cpp_inference/build/FXdiv-source"
  "/workspace/cpp_inference/build/FXdiv"
  "/workspace/cpp_inference/build/FXdiv-download/fxdiv-prefix"
  "/workspace/cpp_inference/build/FXdiv-download/fxdiv-prefix/tmp"
  "/workspace/cpp_inference/build/FXdiv-download/fxdiv-prefix/src/fxdiv-stamp"
  "/workspace/cpp_inference/build/FXdiv-download/fxdiv-prefix/src"
  "/workspace/cpp_inference/build/FXdiv-download/fxdiv-prefix/src/fxdiv-stamp"
)

set(configSubDirs )
foreach(subDir IN LISTS configSubDirs)
    file(MAKE_DIRECTORY "/workspace/cpp_inference/build/FXdiv-download/fxdiv-prefix/src/fxdiv-stamp/${subDir}")
endforeach()
if(cfgdir)
  file(MAKE_DIRECTORY "/workspace/cpp_inference/build/FXdiv-download/fxdiv-prefix/src/fxdiv-stamp${cfgdir}") # cfgdir has leading slash
endif()
