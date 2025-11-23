# CMake "if" Macro Override Error - Fixed

## Problem

During CMake configuration on Windows MinGW, the build was failing with:

```
CMake Error at build/_deps/tensorflow-src/tensorflow/lite/CMakeLists.txt:128 (macro):
  Built-in flow control command "if" cannot be overridden.
```

## Root Cause

The error was caused by our gemmlowp patching code being injected at an unsafe location in the TensorFlow Lite CMakeLists.txt file. The injection logic was trying to insert the code before the first occurrence of `populate_tflite_source_vars`, which appears at line 128 as a macro definition:

```cmake
macro(populate_tflite_source_vars RELATIVE_DIR SOURCES_VAR)
  populate_source_vars(
    "${TFLITE_SOURCE_DIR}/${RELATIVE_DIR}" ${SOURCES_VAR} ${ARGN}
  )
endmacro()
```

When our patching code (which contains `if` statements) was injected inside this macro definition, CMake interpreted it as trying to redefine the built-in `if` command, which is not allowed.

## Solution

The fix changes the injection logic to find a safe location that is:
1. **Outside** of any macro/function definitions
2. **After** the gemmlowp dependency is discovered

The updated code now injects after `find_package(gemmlowp REQUIRED)` (around line 151 in the TensorFlow Lite CMakeLists.txt), which is a safe location in the global scope, well after all macro definitions.

### Changes Made

In `/workspace/cpp_inference/CMakeLists.txt` (lines 182-212):

```cmake
# Find a safe injection point - after "find_package(gemmlowp REQUIRED)" 
# This is outside any macro definitions and after gemmlowp is discovered
string(FIND "${TFLITE_CONTENT}" "find_package(gemmlowp REQUIRED)" FINDPKG_POS)
if(NOT FINDPKG_POS EQUAL -1)
    # Find the end of this line
    string(FIND "${TFLITE_CONTENT}" "\n" NEWLINE_AFTER ${FINDPKG_POS})
    if(NOT NEWLINE_AFTER EQUAL -1)
        math(EXPR INSERT_POS "${NEWLINE_AFTER} + 1")
        string(SUBSTRING "${TFLITE_CONTENT}" 0 ${INSERT_POS} BEFORE_PART)
        string(SUBSTRING "${TFLITE_CONTENT}" ${INSERT_POS} -1 AFTER_PART)
        set(TFLITE_CONTENT "${BEFORE_PART}${GEMMLOWP_PATCH_CODE}${AFTER_PART}")
        file(WRITE "${TFLITE_CMAKE}" "${TFLITE_CONTENT}")
        message(STATUS "  -> Successfully injected gemmlowp patching code after find_package(gemmlowp)")
    endif()
else()
    # Fallback: try to find "# Find TensorFlow Lite dependencies." and inject after it
    string(FIND "${TFLITE_CONTENT}" "# Find TensorFlow Lite dependencies." DEPS_COMMENT_POS)
    # ... fallback logic ...
endif()
```

## Previous Approach (Incorrect)

The old code was trying to inject before `populate_tflite_source_vars` or after `FetchContent_MakeAvailable(gemmlowp)` (which doesn't exist in TensorFlow Lite v2.14.0's CMakeLists.txt). This caused the injection to happen inside macro definitions.

## Testing

To test this fix on Windows MinGW:

```bash
cd cpp_inference
rm -rf build
mkdir build
cd build
cmake ..
```

The CMake configuration should now succeed without the "if cannot be overridden" error.

## Related Files

- `/workspace/cpp_inference/CMakeLists.txt` - Main CMake configuration (fixed)
- `/workspace/cpp_inference/cmake/patch_gemmlowp.cmake` - Standalone gemmlowp patch (still useful as a reference)

## Notes

- This fix is specific to TensorFlow Lite v2.14.0's CMakeLists.txt structure
- The gemmlowp patching is still performed dynamically during CMake configuration
- The patch disables the `eight_bit_int_gemm` target which causes issues on MinGW builds
