#!/usr/bin/env python3
"""
Generate Mintlify SDK documentation from OPML structure.

This is the main entry point that coordinates the documentation generation process.
"""

import json
import sys
from pathlib import Path
from typing import Dict

from opml_reader import OPMLReader
from page_module import ModulePageGenerator
from page_class import ClassPageGenerator
from page_function import FunctionPageGenerator
from page_method import MethodPageGenerator
from page_type import TypePageGenerator
from docsjson_updater import DocsJsonUpdater
from page_llmmap import LLMMapGenerator
from public_api_generator import PublicAPIGenerator


class MintlifierGenerator:
    """Main coordinator for Mintlify documentation generation."""
    
    def __init__(self):
        """Initialize the generator."""
        self.script_dir = Path(__file__).parent
        self.config = None
        self.opml_reader = None
        self.module_gen = None
        self.class_gen = None
        self.function_gen = None
        self.method_gen = None
        self.docs_updater = None
        self.llm_map_gen = None
        
    def _count_pages(self, structure: Dict) -> int:
        """Count the number of pages in the generated structure."""
        count = 0
        if 'groups' in structure:
            for group in structure['groups']:
                count += self._count_pages_in_group(group)
        return count
    
    def _count_pages_in_group(self, group: Dict) -> int:
        """Recursively count pages in a group."""
        count = 0
        if 'pages' in group:
            for page in group['pages']:
                if isinstance(page, str):
                    count += 1
                elif isinstance(page, dict):
                    count += self._count_pages_in_group(page)
        return count
    
    def load_config(self) -> Dict:
        """Load configuration from mintlifier_config.json."""
        config_path = self.script_dir / "mintlifier_config.json"
        
        if not config_path.exists():
            print(f"❌ Config file not found: {config_path}")
            print(f"   Please create mintlifier_config.json with:")
            print(f"   - docs_json_path: path to docs.json")
            print(f"   - sdk_output_dir: where to output SDK docs")
            print(f"   - sdk_tab: name of SDK tab")
            sys.exit(1)
        
        with open(config_path) as f:
            self.config = json.load(f)
        
        print(f"📋 Loaded config: {self.config}")
        return self.config
    
    def run(self):
        """Run the complete documentation generation process."""
        print("🚀 Starting Mintlify SDK documentation generation...")
        print("=" * 60)
        
        # Load configuration
        self.load_config()
        
        # Resolve paths relative to script directory
        docs_json_path = self.script_dir / self.config['docs_json_path']
        sdk_output_dir = self.script_dir / self.config['sdk_output_dir']
        sdk_tab = self.config['sdk_tab']
        
        # Clean the output directory contents first
        import shutil
        if sdk_output_dir.exists():
            for item in sdk_output_dir.iterdir():
                if item.is_dir():
                    shutil.rmtree(item)
                else:
                    item.unlink()
        
        # Initialize components
        opml_path = self.script_dir / "mintlifier.opml"
        
        if not opml_path.exists():
            print(f"❌ OPML file not found: {opml_path}")
            sys.exit(1)
        
        print(f"\n📖 Loading OPML structure from: {opml_path}")
        self.opml_reader = OPMLReader(opml_path)
        
        print(f"📁 Output directory: {sdk_output_dir}")
        # Determine version from output path (e.g., "latest" or "v0.4.9")
        version = "main"  # Default to main for latest
        if "latest" in str(sdk_output_dir):
            version = "main"
        elif "v" in str(sdk_output_dir):
            # Extract version from path like /sdk/v0.4.9/
            path_parts = str(sdk_output_dir).split('/')
            for part in path_parts:
                if part.startswith('v') and '.' in part:
                    version = part  # Use the version tag like "v0.4.9"
                    break
        
        # Initialize page generators with error display setting
        show_errors = self.config.get('show_errors', True)  # Default to showing errors
        self.module_gen = ModulePageGenerator(sdk_output_dir, version=version, show_errors=show_errors)
        self.class_gen = ClassPageGenerator(sdk_output_dir, version=version, show_errors=show_errors)
        self.function_gen = FunctionPageGenerator(sdk_output_dir, version=version, show_errors=show_errors)
        self.method_gen = MethodPageGenerator(sdk_output_dir, version=version, show_errors=show_errors)
        self.type_gen = TypePageGenerator(sdk_output_dir, version=version, show_errors=show_errors)
        
        # Initialize LLM map generator
        self.llm_map_gen = LLMMapGenerator(sdk_output_dir, version=version)
        
        print(f"📝 Docs.json path: {docs_json_path}")
        self.docs_updater = DocsJsonUpdater(docs_json_path, sdk_tab)
        
        # Load OPML structure
        print("\n" + "=" * 60)
        print("📊 Processing OPML structure...")
        tab_structure = self.opml_reader.load()
        
        if not tab_structure:
            print("❌ No SDK tab found in OPML")
            sys.exit(1)
        
        print(f"✅ Found tab: {tab_structure.name}")
        print(f"   Groups: {len(tab_structure.groups)}")
        
        all_pages = self.opml_reader.get_all_pages()
        print(f"   Total pages to generate: {len(all_pages)}")
        
        # Generate documentation pages
        print("\n" + "=" * 60)
        print("📄 Generating documentation pages...")
        
        generated_count = 0
        navigation_pages = {}
        
        for page in all_pages:
            # Route to appropriate generator based on type
            result = None
            llm_entry = None
            
            if page.item_type == 'module':
                # Pass children list if specified for this module
                result = self.module_gen.generate_page(page.module_path, page.parent_groups, page.item_type, page.children)
                # Add to LLM map
                llm_entry = self.llm_map_gen.add_module(page.module_path, page.children)
            elif page.item_type == 'class':
                # Pass children list if specified for this class
                result = self.class_gen.generate_page(page.module_path, page.parent_groups, page.item_type, page.children)
                # Add to LLM map
                llm_entry = self.llm_map_gen.add_class(page.module_path, page.children)
            elif page.item_type == 'func':
                result = self.function_gen.generate_page(page.module_path, page.parent_groups, page.item_type)
                # Add to LLM map
                llm_entry = self.llm_map_gen.add_function(page.module_path)
            elif page.item_type == 'method':
                result = self.method_gen.generate_page(page.module_path, page.parent_groups, page.item_type)
                # Methods are added as part of their parent class
            elif page.item_type == 'type':
                # Skip generating documentation pages for types
                # They're just type markers used in signatures
                result = None
                # Still add to LLM map for signature purposes
                llm_entry = self.llm_map_gen.add_type(page.module_path)
            else:
                print(f"⚠️ Unknown item type: {page.item_type}")
            
            if result:
                generated_count += 1
                navigation_pages[page.module_path] = result
            
            if llm_entry:
                if llm_entry.get("error"):
                    print(f"   ⚠️ LLM map error for {page.module_path}: {llm_entry.get('error')}")
                else:
                    print(f"   ✓ Added to LLM map: {page.module_path}")
                self.llm_map_gen.add_to_map(llm_entry)
        
        print(f"\n✅ Generated: {generated_count} pages")
        
        # Update docs.json navigation
        print("\n" + "=" * 60)
        print("📝 Updating docs.json navigation...")
        
        self.docs_updater.load()
        
        # Build navigation structure with actual generated pages
        try:
            navigation_structure = self._build_navigation_structure(tab_structure, navigation_pages)
        except Exception as e:
            print(f"❌ Error building navigation structure: {e}")
            import traceback
            traceback.print_exc()
            return
        
        # Validate structure
        warnings = self.docs_updater.validate_structure(navigation_structure)
        if warnings:
            print("⚠️  Validation warnings:")
            for warning in warnings:
                print(f"   - {warning}")
        
        # Update and save
        try:
            self.docs_updater.update_navigation(navigation_structure)
            self.docs_updater.save()
        except Exception as e:
            print(f"❌ Error updating navigation: {e}")
            import traceback
            traceback.print_exc()
            return
        
        # Save LLM map
        print("\n📊 Generating LLM map...")
        try:
            self.llm_map_gen.save()
        except Exception as e:
            print(f"⚠️  Error saving LLM map: {e}")
        
        # Generate public API listing
        print("\n📋 Generating public API listing...")
        try:
            # Save to mintlifier directory, not SDK output directory
            mintlifier_dir = self.script_dir
            public_api_gen = PublicAPIGenerator(mintlifier_dir, version=version)
            public_api_gen.generate(all_pages)
        except Exception as e:
            print(f"⚠️  Error generating public API listing: {e}")
        
        # Summary
        print("\n" + "=" * 60)
        print("✨ Documentation generation complete!")
        print(f"   📄 Generated {generated_count} documentation pages")
        print(f"   📁 Output directory: {sdk_output_dir}")
        print(f"   📝 Updated navigation in: {docs_json_path}")
        print(f"   📊 LLM map saved to: mintlifier/llm_map.jsonld")
        
    
    def _build_navigation_structure(self, tab_structure, navigation_pages: Dict) -> Dict:
        """Build navigation structure for docs.json."""
        
        def process_group(group, base_path=''):
            """Process a group recursively."""
            group_path = self._sanitize_path(group.name)
            full_path = f"{base_path}/{group_path}" if base_path else group_path
            
            result = {
                'group': group.name,
                'pages': []
            }
            
            # Add direct pages for this group
            for page in group.pages:
                # Get the navigation structure for this page
                if page.module_path in navigation_pages:
                    nav_item = navigation_pages[page.module_path]
                    
                    # Handle different return types from generators
                    if isinstance(nav_item, str):
                        # Simple page path
                        result['pages'].append(nav_item)
                    elif isinstance(nav_item, dict):
                        # Complex structure with page and optional children
                        if 'page' in nav_item and 'pages' in nav_item:
                            # This is a page with children (like a module or class)
                            # Add it as a nested structure
                            result['pages'].append(nav_item)
                        elif 'group' in nav_item:
                            # This is a group
                            result['pages'].append(nav_item)
                        else:
                            # Just a page
                            result['pages'].append(nav_item.get('page', nav_item))
                    elif isinstance(nav_item, list):
                        # Multiple pages
                        result['pages'].extend(nav_item)
            
            # Process subgroups and add them to the pages array (Mintlify style)
            if group.subgroups:
                for subgroup in group.subgroups:
                    # Add the subgroup as a nested group in the pages array
                    nested_group = process_group(subgroup, full_path)
                    result['pages'].append(nested_group)
            
            return result
        
        # Create a single top-level group "Pixeltable SDK" with all content nested inside
        all_pages = []
        
        # Process each group and add to the pages array
        for group in tab_structure.groups:
            processed_group = process_group(group)
            all_pages.append(processed_group)
        
        # Create version dropdowns structure
        return {
            'tab': tab_structure.name,
            'dropdowns': [
                {
                    'dropdown': 'latest',
                    'icon': 'rocket',
                    'groups': [
                        {
                            'group': 'SDK Reference',
                            'pages': all_pages
                        }
                    ]
                },
                {
                    'dropdown': 'v0.4.9',
                    'icon': 'tag',
                    'pages': ['docs/sdk/v0.4.9/index']
                },
                {
                    'dropdown': 'v0.4.8', 
                    'icon': 'tag',
                    'pages': ['docs/sdk/v0.4.8/index']
                }
            ]
        }
    
    def _sanitize_path(self, text: str) -> str:
        """Convert text to valid file path."""
        return text.lower().replace(' ', '-').replace('/', '-')


def main():
    """Main entry point."""
    generator = MintlifierGenerator()
    
    try:
        generator.run()
    except KeyboardInterrupt:
        print("\n\n⚠️  Generation interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()