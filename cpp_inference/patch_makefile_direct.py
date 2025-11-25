#!/usr/bin/env python3
"""
Direct Makefile patcher for gemmlowp eight_bit_int_gemm target.
This patches the generated Makefile to skip building the problematic target.
Use this as a last resort if CMakeLists.txt patching doesn't work.
"""

import os
import sys
import re

def patch_makefile(makefile_path):
    """Patch a Makefile to skip the eight_bit_int_gemm target."""
    try:
        with open(makefile_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Check if already patched
        if 'PATCHED_SKIP_EIGHT_BIT_INT_GEMM' in content:
            print(f"  Already patched: {makefile_path}")
            return False
        
        original_content = content
        
        # Find the eight_bit_int_gemm target and make it a no-op
        # Strategy: Replace the target with a dummy echo command
        
        # Pattern 1: Target definition like "eight_bit_int_gemm:"
        content = re.sub(
            r'(\neight_bit_int_gemm:)',
            r'\n# PATCHED_SKIP_EIGHT_BIT_INT_GEMM\neight_bit_int_gemm:\n\t@echo "Skipping eight_bit_int_gemm (patched for MinGW)"',
            content
        )
        
        # Pattern 2: Dependencies on eight_bit_int_gemm in 'all' target
        content = re.sub(
            r'(\ball:.*?)eight_bit_int_gemm',
            r'\1# eight_bit_int_gemm (disabled)',
            content
        )
        
        # Pattern 3: References in gemmlowp-build/all
        content = re.sub(
            r'(_deps/gemmlowp-build/.*eight_bit_int_gemm)',
            r'# \1 (disabled)',
            content
        )
        
        if content != original_content:
            # Backup original
            backup_path = makefile_path + '.bak'
            if not os.path.exists(backup_path):
                with open(backup_path, 'w', encoding='utf-8') as f:
                    f.write(original_content)
                print(f"  Created backup: {backup_path}")
            
            with open(makefile_path, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"  ✓ Patched: {makefile_path}")
            return True
        else:
            print(f"  No eight_bit_int_gemm references found: {makefile_path}")
            return False
    
    except Exception as e:
        print(f"  ✗ Error patching {makefile_path}: {e}", file=sys.stderr)
        return False

def patch_build_make(build_make_path):
    """Patch the build.make file for eight_bit_int_gemm to skip compilation."""
    try:
        with open(build_make_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        if 'PATCHED_SKIP_EIGHT_BIT_INT_GEMM' in content:
            print(f"  Already patched: {build_make_path}")
            return False
        
        original_content = content
        
        # Replace the entire build rule with a no-op
        # Find the object file build rule
        content = re.sub(
            r'(eight_bit_int_gemm\.dir.*?\.o:.*?\n)(.*?\n)*?(\t.*?\n)*',
            r'\1# PATCHED_SKIP_EIGHT_BIT_INT_GEMM\n\t@echo "Skipping eight_bit_int_gemm compilation (patched)"\n',
            content,
            flags=re.MULTILINE
        )
        
        if content != original_content:
            backup_path = build_make_path + '.bak'
            if not os.path.exists(backup_path):
                with open(backup_path, 'w', encoding='utf-8') as f:
                    f.write(original_content)
            
            with open(build_make_path, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"  ✓ Patched: {build_make_path}")
            return True
        
        return False
    
    except Exception as e:
        print(f"  ✗ Error patching {build_make_path}: {e}", file=sys.stderr)
        return False

def main():
    """Main entry point."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    build_dir = os.path.join(script_dir, 'build')
    
    if not os.path.exists(build_dir):
        print("Build directory not found. Run CMake first.")
        return 1
    
    print("Patching Makefiles to skip eight_bit_int_gemm...")
    print()
    
    patched_count = 0
    
    # Patch main Makefile
    main_makefile = os.path.join(build_dir, 'Makefile')
    if os.path.exists(main_makefile):
        print(f"Checking: {main_makefile}")
        if patch_makefile(main_makefile):
            patched_count += 1
    
    # Patch gemmlowp-build Makefile
    gemmlowp_makefile = os.path.join(build_dir, '_deps', 'gemmlowp-build', 'Makefile')
    if os.path.exists(gemmlowp_makefile):
        print(f"Checking: {gemmlowp_makefile}")
        if patch_makefile(gemmlowp_makefile):
            patched_count += 1
    
    # Patch build.make for eight_bit_int_gemm
    build_make = os.path.join(
        build_dir, '_deps', 'gemmlowp-build', 
        'CMakeFiles', 'eight_bit_int_gemm.dir', 'build.make'
    )
    if os.path.exists(build_make):
        print(f"Checking: {build_make}")
        if patch_build_make(build_make):
            patched_count += 1
    
    print()
    if patched_count > 0:
        print(f"✓ Patched {patched_count} file(s).")
        print()
        print("Now run: cmake --build . --config Release")
    else:
        print("No files needed patching (already patched or target not found).")
    
    return 0

if __name__ == '__main__':
    sys.exit(main())
