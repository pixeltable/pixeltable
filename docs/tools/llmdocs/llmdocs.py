#!/usr/bin/env python3
"""
LLMDocs - Generate LLM-optimized documentation for Pixeltable

This tool generates three key files that help LLMs understand Pixeltable:
1. llm_map.jsonld - Complete public API reference
2. llm_dev_patterns.jsonld - Developer patterns from notebooks
3. llm_quick_reference.md - Human/LLM readable guide

Usage:
    python llmdocs.py
"""

import sys
from pathlib import Path

from config import config
from llm_api_map_gen import LLMApiMapGenerator
from llm_dev_pattern_gen import NotebookPatternExtractor
from llm_quick_ref_gen import generate_quick_reference
from opml_reader import OPMLReader


def get_project_root():
    """Get the Pixeltable project root directory."""
    return Path(__file__).parent.parent.parent.parent


def main():
    """Generate complete LLM documentation suite."""

    print("=== Pixeltable LLM Documentation Generator ===\n")

    # Get paths from config
    project_root = get_project_root()
    opml_path = project_root / config["public_api_opml"].lstrip("/")
    notebooks_dir = project_root / config["notebooks_dir"].lstrip("/")
    output_dir = project_root / config["output_dir"].lstrip("/")

    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Generate LLM map from OPML
    print("1. Generating llm_map.jsonld from OPML...")
    llm_map_gen = LLMApiMapGenerator(
        Path(__file__).parent,
        version=config["version"]
    )

    # Load and process OPML to build the map
    opml_reader = OPMLReader(opml_path)
    opml_reader.load()
    all_pages = opml_reader.get_all_pages()

    # Process each page through the LLM map generator
    for page in all_pages:
        if page.item_type == 'module':
            llm_map_gen.add_module(page.module_path, page.children)
        elif page.item_type == 'class':
            llm_map_gen.add_class(page.module_path, page.children)
        elif page.item_type == 'func':
            llm_map_gen.add_function(page.module_path)
        elif page.item_type == 'type':
            llm_map_gen.add_type(page.module_path)

    # Save the map
    llm_map_path = output_dir / config["llm_map_file"]
    llm_map_gen.save(llm_map_path, flatten=False)
    print(f"   ✓ Generated {llm_map_path.name}")

    # Step 2: Extract patterns from notebooks
    print("\n2. Generating llm_dev_patterns.jsonld from notebooks...")
    extractor = NotebookPatternExtractor(
        opml_path=str(opml_path),
        notebooks_dir=str(notebooks_dir)
    )
    patterns_path = output_dir / config["llm_patterns_file"]
    extractor.save_patterns(str(patterns_path))
    print(f"   ✓ Generated {patterns_path.name}")

    # Step 3: Generate quick reference
    print("\n3. Generating llm_quick_reference.md...")
    generate_quick_reference(output_dir)
    reference_path = output_dir / config["llm_reference_file"]
    print(f"   ✓ Generated {reference_path.name}")

    # Summary
    print(f"\n✅ Complete! LLM docs generated in {output_dir.relative_to(project_root)}")
    print("\nFiles created:")
    print(f"  - {config['llm_map_file']:<30} # Public API reference")
    print(f"  - {config['llm_patterns_file']:<30} # Developer patterns from notebooks")
    print(f"  - {config['llm_reference_file']:<30} # Guide to using the files")
    print("\nLLMs can now use these files to understand Pixeltable completely.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
