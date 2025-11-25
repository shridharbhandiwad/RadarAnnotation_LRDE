#!/usr/bin/env python3
"""
Script to patch TensorFlow Lite dependencies for MinGW builds.
Fixes gemmlowp and cpuinfo compilation issues.
"""

import os
import re
import sys
from pathlib import Path

def patch_gemmlowp(build_dir):
    """Patch gemmlowp CMakeLists.txt to disable eight_bit_int_gemm target."""
    
    # Try multiple possible locations
    possible_paths = [
        build_dir / "_deps" / "gemmlowp-src" / "CMakeLists.txt",
        build_dir / "gemmlowp" / "CMakeLists.txt",
        build_dir / "_deps" / "gemmlowp" / "CMakeLists.txt",
    ]
    
    gemmlowp_cmake = None
    for path in possible_paths:
        if path.exists():
            gemmlowp_cmake = path
            print(f"Found gemmlowp at: {path}")
            break
    
    if not gemmlowp_cmake:
        print("WARNING: gemmlowp CMakeLists.txt not found. It may not be downloaded yet.")
        print("Please run this script again after CMake configuration completes.")
        return False
    
    # Read the file
    with open(gemmlowp_cmake, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Check if already patched
    if "DISABLED_FOR_MINGW" in content:
        print("gemmlowp already patched, skipping...")
        return True
    
    print("Patching gemmlowp CMakeLists.txt...")
    
    # Apply patches
    content = re.sub(
        r'add_library\(eight_bit_int_gemm',
        '# DISABLED_FOR_MINGW: add_library(eight_bit_int_gemm',
        content
    )
    content = re.sub(
        r'add_executable\(eight_bit_int_gemm',
        '# DISABLED_FOR_MINGW: add_executable(eight_bit_int_gemm',
        content
    )
    content = re.sub(
        r'target_link_libraries\(eight_bit_int_gemm',
        '# DISABLED_FOR_MINGW: target_link_libraries(eight_bit_int_gemm',
        content
    )
    content = re.sub(
        r'set_target_properties\(eight_bit_int_gemm',
        '# DISABLED_FOR_MINGW: set_target_properties(eight_bit_int_gemm',
        content
    )
    
    # Write back
    with open(gemmlowp_cmake, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print("  -> Successfully patched gemmlowp")
    return True


def patch_cpuinfo(build_dir):
    """Patch cpuinfo source files to add missing max/min functions."""
    
    # Try to find cpuinfo source file
    possible_paths = [
        build_dir / "_deps" / "cpuinfo-src" / "src" / "x86" / "windows" / "init.c",
        build_dir / "cpuinfo" / "src" / "x86" / "windows" / "init.c",
    ]
    
    cpuinfo_file = None
    for path in possible_paths:
        if path.exists():
            cpuinfo_file = path
            print(f"Found cpuinfo file at: {path}")
            break
    
    if not cpuinfo_file:
        print("WARNING: cpuinfo source file not found. It may not be downloaded yet.")
        return False
    
    # Read the file
    with open(cpuinfo_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Check if already patched
    if "PATCHED_FOR_MINGW" in content:
        print("cpuinfo already patched, skipping...")
        return True
    
    print("Patching cpuinfo source file...")
    
    # Add max/min macros at the beginning after includes
    patch = """
/* PATCHED_FOR_MINGW: Add missing max/min macros */
#ifndef max
#define max(a, b) (((a) > (b)) ? (a) : (b))
#endif
#ifndef min
#define min(a, b) (((a) < (b)) ? (a) : (b))
#endif
/* END PATCHED_FOR_MINGW */
"""
    
    # Find a good place to insert (after #include statements)
    lines = content.split('\n')
    insert_pos = 0
    for i, line in enumerate(lines):
        if line.strip().startswith('#include'):
            insert_pos = i + 1
    
    # Insert the patch
    lines.insert(insert_pos, patch)
    content = '\n'.join(lines)
    
    # Write back
    with open(cpuinfo_file, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print("  -> Successfully patched cpuinfo")
    return True


def main():
    """Main function to apply all patches."""
    
    print("=" * 80)
    print("Radar Tagger Dependency Patcher for MinGW")
    print("=" * 80)
    print()
    
    # Find build directory
    script_dir = Path(__file__).parent
    build_dir = script_dir / "build"
    
    if not build_dir.exists():
        print("ERROR: Build directory not found!")
        print("Please run CMake configuration first:")
        print("  mkdir build && cd build")
        print("  cmake -G \"MinGW Makefiles\" -DCMAKE_BUILD_TYPE=Release ..")
        return 1
    
    print(f"Build directory: {build_dir}")
    print()
    
    # Apply patches
    success = True
    
    print("Patching gemmlowp...")
    if not patch_gemmlowp(build_dir):
        success = False
    print()
    
    print("Patching cpuinfo...")
    if not patch_cpuinfo(build_dir):
        success = False
    print()
    
    if success:
        print("=" * 80)
        print("All patches applied successfully!")
        print("=" * 80)
        print()
        print("You can now continue building:")
        print("  cd build")
        print("  mingw32-make -j4")
        return 0
    else:
        print("=" * 80)
        print("Some patches could not be applied.")
        print("=" * 80)
        print()
        print("This is normal if CMake hasn't finished downloading dependencies yet.")
        print("Run this script again after CMake configuration completes.")
        return 0


if __name__ == "__main__":
    sys.exit(main())
