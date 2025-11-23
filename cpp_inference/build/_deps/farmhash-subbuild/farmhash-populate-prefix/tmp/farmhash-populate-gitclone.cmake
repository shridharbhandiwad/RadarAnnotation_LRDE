# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file Copyright.txt or https://cmake.org/licensing for details.

cmake_minimum_required(VERSION 3.5)

if(EXISTS "/workspace/cpp_inference/build/_deps/farmhash-subbuild/farmhash-populate-prefix/src/farmhash-populate-stamp/farmhash-populate-gitclone-lastrun.txt" AND EXISTS "/workspace/cpp_inference/build/_deps/farmhash-subbuild/farmhash-populate-prefix/src/farmhash-populate-stamp/farmhash-populate-gitinfo.txt" AND
  "/workspace/cpp_inference/build/_deps/farmhash-subbuild/farmhash-populate-prefix/src/farmhash-populate-stamp/farmhash-populate-gitclone-lastrun.txt" IS_NEWER_THAN "/workspace/cpp_inference/build/_deps/farmhash-subbuild/farmhash-populate-prefix/src/farmhash-populate-stamp/farmhash-populate-gitinfo.txt")
  message(STATUS
    "Avoiding repeated git clone, stamp file is up to date: "
    "'/workspace/cpp_inference/build/_deps/farmhash-subbuild/farmhash-populate-prefix/src/farmhash-populate-stamp/farmhash-populate-gitclone-lastrun.txt'"
  )
  return()
endif()

execute_process(
  COMMAND ${CMAKE_COMMAND} -E rm -rf "/workspace/cpp_inference/build/farmhash"
  RESULT_VARIABLE error_code
)
if(error_code)
  message(FATAL_ERROR "Failed to remove directory: '/workspace/cpp_inference/build/farmhash'")
endif()

# try the clone 3 times in case there is an odd git clone issue
set(error_code 1)
set(number_of_tries 0)
while(error_code AND number_of_tries LESS 3)
  execute_process(
    COMMAND "/usr/bin/git"
            clone --no-checkout --progress --config "advice.detachedHead=false" "https://github.com/google/farmhash" "farmhash"
    WORKING_DIRECTORY "/workspace/cpp_inference/build"
    RESULT_VARIABLE error_code
  )
  math(EXPR number_of_tries "${number_of_tries} + 1")
endwhile()
if(number_of_tries GREATER 1)
  message(STATUS "Had to git clone more than once: ${number_of_tries} times.")
endif()
if(error_code)
  message(FATAL_ERROR "Failed to clone repository: 'https://github.com/google/farmhash'")
endif()

execute_process(
  COMMAND "/usr/bin/git"
          checkout "0d859a811870d10f53a594927d0d0b97573ad06d" --
  WORKING_DIRECTORY "/workspace/cpp_inference/build/farmhash"
  RESULT_VARIABLE error_code
)
if(error_code)
  message(FATAL_ERROR "Failed to checkout tag: '0d859a811870d10f53a594927d0d0b97573ad06d'")
endif()

set(init_submodules TRUE)
if(init_submodules)
  execute_process(
    COMMAND "/usr/bin/git" 
            submodule update --recursive --init 
    WORKING_DIRECTORY "/workspace/cpp_inference/build/farmhash"
    RESULT_VARIABLE error_code
  )
endif()
if(error_code)
  message(FATAL_ERROR "Failed to update submodules in: '/workspace/cpp_inference/build/farmhash'")
endif()

# Complete success, update the script-last-run stamp file:
#
execute_process(
  COMMAND ${CMAKE_COMMAND} -E copy "/workspace/cpp_inference/build/_deps/farmhash-subbuild/farmhash-populate-prefix/src/farmhash-populate-stamp/farmhash-populate-gitinfo.txt" "/workspace/cpp_inference/build/_deps/farmhash-subbuild/farmhash-populate-prefix/src/farmhash-populate-stamp/farmhash-populate-gitclone-lastrun.txt"
  RESULT_VARIABLE error_code
)
if(error_code)
  message(FATAL_ERROR "Failed to copy script-last-run stamp file: '/workspace/cpp_inference/build/_deps/farmhash-subbuild/farmhash-populate-prefix/src/farmhash-populate-stamp/farmhash-populate-gitclone-lastrun.txt'")
endif()
