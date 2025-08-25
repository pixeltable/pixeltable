#!/usr/bin/env python3
"""
Clean and regenerate Mintlify SDK documentation.
This script ensures a clean slate before regeneration.
"""

import shutil
import sys
from pathlib import Path
import json

def clean_sdk_directory(sdk_dir: Path):
    """Clean the SDK directory before regeneration."""
    if sdk_dir.exists():
        print(f"🧹 Cleaning SDK directory: {sdk_dir}")
        # Remove all subdirectories and files
        for item in sdk_dir.iterdir():
            if item.is_dir():
                print(f"   Removing directory: {item.name}")
                shutil.rmtree(item)
            elif item.is_file() and item.suffix == '.mdx':
                print(f"   Removing file: {item.name}")
                item.unlink()
        print("✅ SDK directory cleaned")
    else:
        print(f"📁 SDK directory doesn't exist, will be created: {sdk_dir}")

def restore_docs_json_from_backup(backup_dir: Path, docs_json_path: Path):
    """Restore docs.json from the latest backup with nested structure."""
    # Find the backup with nested structure (should be the 155140 one)
    backup_file = backup_dir / "docs_20250818_155140.json"
    
    if backup_file.exists():
        print(f"📋 Restoring docs.json from backup: {backup_file}")
        shutil.copy2(backup_file, docs_json_path)
        print("✅ docs.json restored with nested structure")
        return True
    else:
        print(f"❌ Backup file not found: {backup_file}")
        return False

def main():
    """Main execution."""
    script_dir = Path(__file__).parent
    
    # Define paths
    sdk_dir = script_dir.parent / "mintlify" / "docs" / "sdk" / "latest"
    docs_json_path = script_dir.parent / "mintlify" / "docs.json"
    backup_dir = script_dir / "docsjson_bak"
    
    print("=" * 60)
    print("🚀 Clean and Regenerate Mintlify SDK Documentation")
    print("=" * 60)
    
    # Step 1: Clean SDK directory
    clean_sdk_directory(sdk_dir)
    
    # Step 2: Restore docs.json from backup
    if not restore_docs_json_from_backup(backup_dir, docs_json_path):
        print("⚠️  Could not restore docs.json, continuing with current version")
    
    # Step 3: Run the main generator
    print("\n" + "=" * 60)
    print("📚 Running documentation generator...")
    print("=" * 60)
    
    # Import and run the main generator
    from mintlifier import MintlifierGenerator
    
    generator = MintlifierGenerator()
    generator.run()
    
    print("\n" + "=" * 60)
    print("✅ Clean regeneration complete!")
    print("=" * 60)

if __name__ == "__main__":
    main()