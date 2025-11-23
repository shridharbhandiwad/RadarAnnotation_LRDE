# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file Copyright.txt or https://cmake.org/licensing for details.

cmake_minimum_required(VERSION 3.5)

file(MAKE_DIRECTORY
  "/workspace/cpp_inference/build/_deps/tensorflow-src"
  "/workspace/cpp_inference/build/_deps/tensorflow-build"
  "/workspace/cpp_inference/build/_deps/tensorflow-subbuild/tensorflow-populate-prefix"
  "/workspace/cpp_inference/build/_deps/tensorflow-subbuild/tensorflow-populate-prefix/tmp"
  "/workspace/cpp_inference/build/_deps/tensorflow-subbuild/tensorflow-populate-prefix/src/tensorflow-populate-stamp"
  "/workspace/cpp_inference/build/_deps/tensorflow-subbuild/tensorflow-populate-prefix/src"
  "/workspace/cpp_inference/build/_deps/tensorflow-subbuild/tensorflow-populate-prefix/src/tensorflow-populate-stamp"
)

set(configSubDirs )
foreach(subDir IN LISTS configSubDirs)
    file(MAKE_DIRECTORY "/workspace/cpp_inference/build/_deps/tensorflow-subbuild/tensorflow-populate-prefix/src/tensorflow-populate-stamp/${subDir}")
endforeach()
if(cfgdir)
  file(MAKE_DIRECTORY "/workspace/cpp_inference/build/_deps/tensorflow-subbuild/tensorflow-populate-prefix/src/tensorflow-populate-stamp${cfgdir}") # cfgdir has leading slash
endif()
