# Windows Build Alternatives

## Overview

If the standard CMake build fails on Windows, this document provides alternative approaches to build or use the Radar Tagger C++ inference engine.

---

## Alternative 1: Use WSL2 (Recommended)

Windows Subsystem for Linux 2 provides a complete Linux environment on Windows, avoiding most Windows-specific build issues.

### Setup

```powershell
# Install WSL2
wsl --install

# Restart your computer

# Update to Ubuntu
wsl --install -d Ubuntu

# Launch Ubuntu
wsl
```

### Build in WSL2

```bash
# Inside WSL2
cd /mnt/c/path/to/your/project/cpp_inference

# Install dependencies
sudo apt update
sudo apt install -y build-essential cmake git python3

# Build
chmod +x build.sh
./build.sh
```

### Advantages
- ✅ No Windows-specific build issues
- ✅ Faster compilation
- ✅ Better dependency management
- ✅ Can use Linux binaries

### Disadvantages
- ❌ Requires WSL2 setup
- ❌ Binaries are Linux format (use in WSL2 only)

---

## Alternative 2: Use MSVC Instead of MinGW

Microsoft Visual C++ (MSVC) is often more reliable than MinGW on Windows.

### Prerequisites

1. **Install Visual Studio 2019 or 2022**
   - Download from: https://visualstudio.microsoft.com/
   - Select "Desktop development with C++"

2. **Open Visual Studio Developer Command Prompt**
   - Search for "x64 Native Tools Command Prompt for VS"

### Build with MSVC

```batch
cd cpp_inference

REM Clean build
rmdir /s /q build
mkdir build
cd build

REM Configure with MSVC
cmake -G "Visual Studio 16 2019" -A x64 -DCMAKE_BUILD_TYPE=Release ..

REM Build
cmake --build . --config Release

REM Or use the helper script
cd ..
build_windows_robust.bat msvc
```

### Advantages
- ✅ Better Windows compatibility
- ✅ Official Microsoft toolchain
- ✅ Better debugging support

### Disadvantages
- ❌ Larger download (Visual Studio)
- ❌ May still have TensorFlow Lite issues

---

## Alternative 3: Use vcpkg for Dependencies

vcpkg is Microsoft's C++ package manager, which can simplify dependency management.

### Setup vcpkg

```powershell
# Clone vcpkg
cd C:\
git clone https://github.com/Microsoft/vcpkg.git
cd vcpkg
.\bootstrap-vcpkg.bat

# Add to PATH
setx PATH "%PATH%;C:\vcpkg"
```

### Modified CMakeLists.txt

Create a new file `CMakeLists_vcpkg.txt`:

```cmake
cmake_minimum_required(VERSION 3.16)
project(RadarTaggerCpp VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Use vcpkg packages
find_package(nlohmann_json CONFIG REQUIRED)
find_package(unofficial-onnxruntime CONFIG REQUIRED)

# Note: TensorFlow Lite not available in vcpkg
# You'll need to handle it separately or use ONNX only

# Simplified build without TensorFlow Lite
set(SOURCES_MULTIOUTPUT
    radar_tagger_multioutput.cpp
    main_multioutput.cpp
)

add_executable(radar_tagger_multioutput ${SOURCES_MULTIOUTPUT})

target_include_directories(radar_tagger_multioutput PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(radar_tagger_multioutput PRIVATE
    nlohmann_json::nlohmann_json
    unofficial::onnxruntime::onnxruntime
)
```

### Build with vcpkg

```batch
cd cpp_inference

REM Install dependencies
vcpkg install nlohmann-json:x64-windows
vcpkg install onnxruntime:x64-windows

REM Build
mkdir build
cd build
cmake -DCMAKE_TOOLCHAIN_FILE=C:/vcpkg/scripts/buildsystems/vcpkg.cmake ..
cmake --build . --config Release
```

### Advantages
- ✅ Simplified dependency management
- ✅ Pre-built binaries
- ✅ Better Windows support

### Disadvantages
- ❌ TensorFlow Lite not available in vcpkg
- ❌ Requires vcpkg setup
- ❌ Large dependency downloads

---

## Alternative 4: Docker Container

Run the build in a Docker container with a known-good environment.

### Prerequisites

1. Install Docker Desktop for Windows: https://www.docker.com/products/docker-desktop

### Create Dockerfile

Create `cpp_inference/Dockerfile.windows`:

```dockerfile
FROM ubuntu:22.04

# Install build tools
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    python3 \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /workspace

# Copy project files
COPY . /workspace/

# Build
RUN cd cpp_inference && \
    mkdir build && \
    cd build && \
    cmake -DCMAKE_BUILD_TYPE=Release .. && \
    cmake --build . --config Release

CMD ["/bin/bash"]
```

### Build with Docker

```powershell
# Build the Docker image
docker build -t radar-tagger-cpp -f cpp_inference/Dockerfile.windows .

# Run the build
docker run --rm -v ${PWD}/cpp_inference/build:/workspace/cpp_inference/build radar-tagger-cpp

# Extract binaries
# Binaries will be in cpp_inference/build/
```

### Advantages
- ✅ Isolated build environment
- ✅ Reproducible builds
- ✅ No Windows-specific issues

### Disadvantages
- ❌ Requires Docker Desktop
- ❌ Linux binaries (need WSL2 to run)
- ❌ Slower than native builds

---

## Alternative 5: Pre-built Binaries

### Option A: Build on Linux and Use in WSL2

1. Build on a Linux machine (or in CI/CD)
2. Copy binaries to Windows
3. Run in WSL2

### Option B: Request from Maintainer

Contact the project maintainer to provide pre-built Windows binaries.

### Option C: Use GitHub Actions

Set up a GitHub Actions workflow to build on Windows:

Create `.github/workflows/build-windows.yml`:

```yaml
name: Build Windows

on: [push, pull_request]

jobs:
  build:
    runs-on: windows-latest
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Setup CMake
      uses: lukka/get-cmake@latest
    
    - name: Setup MSVC
      uses: ilammy/msvc-dev-cmd@v1
    
    - name: Build
      run: |
        cd cpp_inference
        mkdir build
        cd build
        cmake -G "Visual Studio 17 2022" -A x64 ..
        cmake --build . --config Release
    
    - name: Upload artifacts
      uses: actions/upload-artifact@v3
      with:
        name: windows-binaries
        path: cpp_inference/build/Release/*.exe
```

Then download the artifacts from GitHub Actions.

---

## Alternative 6: Use Only ONNX Runtime (Simplified)

If TensorFlow Lite is causing all the issues, you can build a version that uses only ONNX Runtime.

### Simplified CMakeLists.txt

Create `CMakeLists_onnx_only.txt`:

```cmake
cmake_minimum_required(VERSION 3.16)
project(RadarTaggerCpp VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Threads REQUIRED)

# JSON library
include(FetchContent)
FetchContent_Declare(
    json
    URL https://github.com/nlohmann/json/releases/download/v3.11.2/json.tar.xz
)
FetchContent_MakeAvailable(json)

# ONNX Runtime
if(WIN32)
    set(ONNXRUNTIME_URL "https://github.com/microsoft/onnxruntime/releases/download/v1.16.3/onnxruntime-win-x64-1.16.3.zip")
else()
    set(ONNXRUNTIME_URL "https://github.com/microsoft/onnxruntime/releases/download/v1.16.3/onnxruntime-linux-x64-1.16.3.tgz")
endif()

FetchContent_Declare(
    onnxruntime
    URL ${ONNXRUNTIME_URL}
)
FetchContent_MakeAvailable(onnxruntime)

set(ONNXRUNTIME_INCLUDE_DIRS ${onnxruntime_SOURCE_DIR}/include)

if(WIN32)
    set(ONNXRUNTIME_LIBRARIES ${onnxruntime_SOURCE_DIR}/lib/onnxruntime.lib)
    set(ONNXRUNTIME_DLL ${onnxruntime_SOURCE_DIR}/lib/onnxruntime.dll)
else()
    set(ONNXRUNTIME_LIBRARIES ${onnxruntime_SOURCE_DIR}/lib/libonnxruntime.so)
endif()

# Build only the multioutput version (which uses ONNX)
add_executable(radar_tagger_multioutput 
    radar_tagger_multioutput.cpp
    main_multioutput.cpp
)

target_include_directories(radar_tagger_multioutput PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${ONNXRUNTIME_INCLUDE_DIRS}
)

target_link_libraries(radar_tagger_multioutput PRIVATE
    ${ONNXRUNTIME_LIBRARIES}
    nlohmann_json::nlohmann_json
    Threads::Threads
)

# Copy DLL on Windows
if(WIN32 AND EXISTS ${ONNXRUNTIME_DLL})
    add_custom_command(TARGET radar_tagger_multioutput POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${ONNXRUNTIME_DLL}
            $<TARGET_FILE_DIR:radar_tagger_multioutput>
    )
endif()

message(STATUS "ONNX-only build configured")
message(STATUS "  ONNX Runtime: ${ONNXRUNTIME_LIBRARIES}")
```

### Build ONNX-only version

```batch
cd cpp_inference
mkdir build_onnx
cd build_onnx

cmake -G "MinGW Makefiles" -DCMAKE_BUILD_TYPE=Release -C ../CMakeLists_onnx_only.txt ..
cmake --build . --config Release
```

### Advantages
- ✅ Much simpler build
- ✅ No TensorFlow Lite issues
- ✅ Faster configuration

### Disadvantages
- ❌ Cannot use TensorFlow Lite models
- ❌ Only multioutput executable

---

## Alternative 7: Use Conan Package Manager

Conan is another C++ package manager that may handle dependencies better.

### Install Conan

```powershell
pip install conan
```

### Create conanfile.txt

```ini
[requires]
nlohmann_json/3.11.2

[generators]
CMakeDeps
CMakeToolchain

[options]

[imports]
```

### Build with Conan

```batch
cd cpp_inference

REM Install dependencies
conan install . --build=missing -s build_type=Release

REM Build
cmake --preset conan-default
cmake --build --preset conan-release
```

---

## Recommendation

**For most users:**
1. Try **Alternative 1 (WSL2)** - Most reliable
2. If WSL2 not available, try **Alternative 2 (MSVC)** with the robust build script
3. If still failing, use **Alternative 6 (ONNX-only)** for a simplified build

**For developers:**
- Use **Alternative 4 (Docker)** for reproducible builds
- Set up **Alternative 5 (GitHub Actions)** for continuous integration

**For production:**
- Use **Alternative 5 (Pre-built binaries)** for deployment

---

## Getting Help

If all alternatives fail:

1. **Open an issue** with:
   - Your Windows version
   - Compiler version (`g++ --version` or `cl`)
   - CMake version (`cmake --version`)
   - Full error log from `build/cmake_config_output.txt`

2. **Check existing documentation**:
   - `WINDOWS_CMAKE_VERSION_FIX.md`
   - `WINDOWS_MINGW_BUILD_FIX.md`
   - `START_HERE.md`

3. **Contact maintainer** for support

---

## Summary

| Alternative | Difficulty | Reliability | Speed | Notes |
|------------|-----------|-------------|-------|-------|
| WSL2 | Easy | ⭐⭐⭐⭐⭐ | Fast | **Recommended** |
| MSVC | Easy | ⭐⭐⭐⭐ | Medium | Good alternative |
| vcpkg | Medium | ⭐⭐⭐ | Medium | No TF Lite |
| Docker | Medium | ⭐⭐⭐⭐⭐ | Slow | For advanced users |
| Pre-built | Easy | ⭐⭐⭐⭐⭐ | Instant | Best for end users |
| ONNX-only | Easy | ⭐⭐⭐⭐ | Fast | Limited functionality |
| Conan | Hard | ⭐⭐⭐ | Medium | Experimental |

---

**Last Updated:** November 25, 2025
