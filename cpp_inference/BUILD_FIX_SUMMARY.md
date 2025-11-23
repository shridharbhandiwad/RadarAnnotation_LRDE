# C++ Build Fix Summary

## Problem
The CMake configuration was failing with the following errors:
```
CMake Error: Running 'nmake' '-?' failed with: no such file or directory
CMake Error: CMAKE_CXX_COMPILER not set, after EnableLanguage
```

Additional error during compiler detection:
```
/usr/bin/ld: cannot find -lstdc++: No such file or directory
```

## Root Causes

1. **Missing C++ Standard Library**: The system did not have `libstdc++-dev` installed, which is required for linking C++ programs.

2. **Wrong Compiler Selection**: The default `/usr/bin/c++` symlink pointed to Clang, which had issues finding the C++ standard library even after installation.

## Solution

### 1. Installed Required Packages
```bash
sudo apt-get update
sudo apt-get install -y build-essential g++ libstdc++-10-dev
```

### 2. Explicitly Use g++ Compiler
Modified CMake configuration to explicitly use g++:
```bash
cmake .. -DCMAKE_CXX_COMPILER=/usr/bin/g++ -DCMAKE_BUILD_TYPE=Release
```

### 3. Updated build.sh Script
Updated the `build.sh` script to always use g++ to prevent this issue in the future.

## Build Results

✓ **Successfully built both executables:**
- `radar_tagger` (4.0M)
- `radar_tagger_multioutput` (4.1M)

Both are fully linked ELF 64-bit executables ready to run.

## Usage

To build the project, simply run:
```bash
cd cpp_inference
./build.sh
```

Or manually:
```bash
cd cpp_inference
rm -rf build
mkdir build
cd build
cmake .. -DCMAKE_CXX_COMPILER=/usr/bin/g++ -DCMAKE_BUILD_TYPE=Release
cmake --build . --config Release --parallel $(nproc)
```

## Date Fixed
2025-11-23
