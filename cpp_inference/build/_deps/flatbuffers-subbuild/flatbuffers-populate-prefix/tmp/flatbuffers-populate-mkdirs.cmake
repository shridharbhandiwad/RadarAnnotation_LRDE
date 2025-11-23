# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file Copyright.txt or https://cmake.org/licensing for details.

cmake_minimum_required(VERSION 3.5)

file(MAKE_DIRECTORY
  "/workspace/cpp_inference/build/flatbuffers"
  "/workspace/cpp_inference/build/_deps/flatbuffers-build"
  "/workspace/cpp_inference/build/_deps/flatbuffers-subbuild/flatbuffers-populate-prefix"
  "/workspace/cpp_inference/build/_deps/flatbuffers-subbuild/flatbuffers-populate-prefix/tmp"
  "/workspace/cpp_inference/build/_deps/flatbuffers-subbuild/flatbuffers-populate-prefix/src/flatbuffers-populate-stamp"
  "/workspace/cpp_inference/build/_deps/flatbuffers-subbuild/flatbuffers-populate-prefix/src"
  "/workspace/cpp_inference/build/_deps/flatbuffers-subbuild/flatbuffers-populate-prefix/src/flatbuffers-populate-stamp"
)

set(configSubDirs )
foreach(subDir IN LISTS configSubDirs)
    file(MAKE_DIRECTORY "/workspace/cpp_inference/build/_deps/flatbuffers-subbuild/flatbuffers-populate-prefix/src/flatbuffers-populate-stamp/${subDir}")
endforeach()
if(cfgdir)
  file(MAKE_DIRECTORY "/workspace/cpp_inference/build/_deps/flatbuffers-subbuild/flatbuffers-populate-prefix/src/flatbuffers-populate-stamp${cfgdir}") # cfgdir has leading slash
endif()
