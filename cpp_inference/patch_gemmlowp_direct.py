#!/usr/bin/env python3
"""
Direct patcher for gemmlowp CMakeLists.txt to fix MinGW compilation issues.
This script searches for and patches gemmlowp CMakeLists.txt files before CMake processes them.
"""

import os
import sys
import glob
import re

def find_gemmlowp_cmake_files(search_dirs):
    """Find all gemmlowp CMakeLists.txt files."""
    cmake_files = []
    seen = set()
    
    for search_dir in search_dirs:
        if not os.path.exists(search_dir):
            continue
        
        # Method 1: Search for gemmlowp CMakeLists.txt
        pattern = os.path.join(search_dir, '**', '*gemmlowp*', 'CMakeLists.txt')
        for f in glob.glob(pattern, recursive=True):
            real_path = os.path.realpath(f)
            if real_path not in seen:
                cmake_files.append(f)
                seen.add(real_path)
        
        # Method 2: Search for CMakeLists.txt that contain eight_bit_int_gemm
        # This catches gemmlowp even if the directory isn't named "gemmlowp"
        pattern = os.path.join(search_dir, '**', 'CMakeLists.txt')
        for f in glob.glob(pattern, recursive=True):
            real_path = os.path.realpath(f)
            if real_path in seen:
                continue
            
            # Skip subbuild files
            if 'subbuild' in f.lower():
                continue
            
            try:
                with open(f, 'r', encoding='utf-8') as fh:
                    content = fh.read()
                    if 'eight_bit_int_gemm' in content:
                        cmake_files.append(f)
                        seen.add(real_path)
            except:
                pass
    
    return cmake_files

def patch_gemmlowp_cmake(cmake_file):
    """Patch a gemmlowp CMakeLists.txt file to disable eight_bit_int_gemm."""
    try:
        with open(cmake_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Check if already patched
        if 'DISABLED_FOR_MINGW' in content:
            print(f"  Already patched: {cmake_file}")
            return False
        
        # Patch: Comment out eight_bit_int_gemm target definitions
        original_content = content
        
        # Disable add_library(eight_bit_int_gemm ...)
        content = re.sub(
            r'add_library\(eight_bit_int_gemm',
            r'# DISABLED_FOR_MINGW: add_library(eight_bit_int_gemm',
            content
        )
        
        # Disable add_executable(eight_bit_int_gemm ...)
        content = re.sub(
            r'add_executable\(eight_bit_int_gemm',
            r'# DISABLED_FOR_MINGW: add_executable(eight_bit_int_gemm',
            content
        )
        
        # Disable target_link_libraries(eight_bit_int_gemm ...)
        content = re.sub(
            r'target_link_libraries\(eight_bit_int_gemm',
            r'# DISABLED_FOR_MINGW: target_link_libraries(eight_bit_int_gemm',
            content
        )
        
        # Disable set_target_properties(eight_bit_int_gemm ...)
        content = re.sub(
            r'set_target_properties\(eight_bit_int_gemm',
            r'# DISABLED_FOR_MINGW: set_target_properties(eight_bit_int_gemm',
            content
        )
        
        # Disable target_compile_options(eight_bit_int_gemm ...)
        content = re.sub(
            r'target_compile_options\(eight_bit_int_gemm',
            r'# DISABLED_FOR_MINGW: target_compile_options(eight_bit_int_gemm',
            content
        )
        
        # Disable target_include_directories(eight_bit_int_gemm ...)
        content = re.sub(
            r'target_include_directories\(eight_bit_int_gemm',
            r'# DISABLED_FOR_MINGW: target_include_directories(eight_bit_int_gemm',
            content
        )
        
        if content != original_content:
            with open(cmake_file, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"  ✓ Patched: {cmake_file}")
            return True
        else:
            print(f"  No changes needed: {cmake_file}")
            return False
    
    except Exception as e:
        print(f"  ✗ Error patching {cmake_file}: {e}", file=sys.stderr)
        return False

def main():
    """Main entry point."""
    build_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Search in build directory and common locations
    search_dirs = [
        os.path.join(build_dir, 'build'),
        os.path.join(build_dir, 'build', '_deps'),
        os.path.join(build_dir, 'build', 'gemmlowp'),
        build_dir,
    ]
    
    # Also search in TensorFlow source if it exists
    tensorflow_src = os.path.join(build_dir, 'build', '_deps', 'tensorflow-src')
    if os.path.exists(tensorflow_src):
        search_dirs.append(tensorflow_src)
    
    print("Searching for gemmlowp CMakeLists.txt files...")
    cmake_files = find_gemmlowp_cmake_files(search_dirs)
    
    if not cmake_files:
        print("No gemmlowp CMakeLists.txt files found.")
        print("This is normal if gemmlowp hasn't been downloaded yet.")
        return 0
    
    print(f"Found {len(cmake_files)} gemmlowp CMakeLists.txt file(s):")
    for f in cmake_files:
        print(f"  - {f}")
    
    print("\nPatching gemmlowp files...")
    patched_count = 0
    for cmake_file in cmake_files:
        if patch_gemmlowp_cmake(cmake_file):
            patched_count += 1
    
    print(f"\nPatched {patched_count} file(s).")
    return 0

if __name__ == '__main__':
    sys.exit(main())
