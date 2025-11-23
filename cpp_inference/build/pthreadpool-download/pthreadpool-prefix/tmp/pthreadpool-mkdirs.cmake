# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file Copyright.txt or https://cmake.org/licensing for details.

cmake_minimum_required(VERSION 3.5)

file(MAKE_DIRECTORY
  "/workspace/cpp_inference/build/pthreadpool-source"
  "/workspace/cpp_inference/build/pthreadpool"
  "/workspace/cpp_inference/build/pthreadpool-download/pthreadpool-prefix"
  "/workspace/cpp_inference/build/pthreadpool-download/pthreadpool-prefix/tmp"
  "/workspace/cpp_inference/build/pthreadpool-download/pthreadpool-prefix/src/pthreadpool-stamp"
  "/workspace/cpp_inference/build/pthreadpool-download/pthreadpool-prefix/src"
  "/workspace/cpp_inference/build/pthreadpool-download/pthreadpool-prefix/src/pthreadpool-stamp"
)

set(configSubDirs )
foreach(subDir IN LISTS configSubDirs)
    file(MAKE_DIRECTORY "/workspace/cpp_inference/build/pthreadpool-download/pthreadpool-prefix/src/pthreadpool-stamp/${subDir}")
endforeach()
if(cfgdir)
  file(MAKE_DIRECTORY "/workspace/cpp_inference/build/pthreadpool-download/pthreadpool-prefix/src/pthreadpool-stamp${cfgdir}") # cfgdir has leading slash
endif()
