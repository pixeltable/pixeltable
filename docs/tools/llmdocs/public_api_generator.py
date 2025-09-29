"""
Generate pixeltable_public_api.json from OPML structure.

This creates a complete listing of all public Pixeltable APIs,
categorized by type (functions, classes, methods, types).
"""

import json
from pathlib import Path
from typing import List
from datetime import datetime
from dataclasses import dataclass
import importlib
import inspect


@dataclass
class PublicAPIEntry:
    """Represents a public API entry."""
    full_path: str
    category: str  # 'function', 'class', 'method', 'type'
    module: str
    name: str
    children: List[str] = None


class PublicAPIGenerator:
    """Generate complete public API listing for LLM context."""

    def __init__(self, output_dir: Path, version: str = "main"):
        """Initialize with output directory and version."""
        self.output_dir = output_dir
        self.version = version
        self.api_entries = {
            "functions": [],
            "classes": [],
            "methods": [],
            "types": []  # Keep tracking types for signature purposes
        }
        # Core types that are needed for signatures (even without doc pages)
        self.core_types = [
            "pixeltable.Image",
            "pixeltable.Video",
            "pixeltable.Audio",
            "pixeltable.Document",
            "pixeltable.Array",
            "pixeltable.Json",
            "pixeltable.String",
            "pixeltable.Int",
            "pixeltable.Float",
            "pixeltable.Bool",
            "pixeltable.Timestamp"
        ]

    def process_opml_pages(self, pages):
        """Process all pages from OPML structure."""
        for page in pages:
            if page.item_type == 'func':
                self.api_entries["functions"].append(page.module_path)

            elif page.item_type == 'type':
                # Explicitly marked as type in OPML
                self.api_entries["types"].append(page.module_path)

            elif page.item_type == 'class':
                self.api_entries["classes"].append(page.module_path)

                # If children are specified, add them as methods
                if page.children:
                    for child in page.children:
                        method_path = f"{page.module_path}.{child}"
                        self.api_entries["methods"].append(method_path)
                else:
                    # Try to introspect for all public methods
                    self._add_class_methods(page.module_path)

            elif page.item_type == 'module':
                # Process module contents
                if page.children:
                    # Specific children listed
                    for child in page.children:
                        child_path = f"{page.module_path}.{child}"
                        # Try to determine if it's a function or class
                        self._categorize_module_child(child_path)
                else:
                    # Process all public members
                    self._process_module(page.module_path)

            elif page.item_type == 'method':
                self.api_entries["methods"].append(page.module_path)

    def _categorize_module_child(self, path: str):
        """Determine if a module member is a function, class, or type."""
        try:
            parts = path.split('.')
            module_path = '.'.join(parts[:-1])
            item_name = parts[-1]

            module = importlib.import_module(module_path)
            if hasattr(module, item_name):
                obj = getattr(module, item_name)

                if inspect.isclass(obj):
                    # Check if it's a type (like Image, Video)
                    if self._is_pixeltable_type(obj):
                        self.api_entries["types"].append(path)
                    else:
                        self.api_entries["classes"].append(path)
                        self._add_class_methods(path)
                elif inspect.isfunction(obj) or inspect.isbuiltin(obj):
                    self.api_entries["functions"].append(path)
        except (ImportError, AttributeError):
            # Default to function if we can't introspect
            self.api_entries["functions"].append(path)

    def _is_pixeltable_type(self, cls):
        """Check if a class is a Pixeltable type (Image, Video, etc.)."""
        # Simple heuristic: types are typically in type_system module
        # and don't have many public methods
        module_name = cls.__module__ if hasattr(cls, '__module__') else ''

        # Known Pixeltable types
        type_names = ['Image', 'Video', 'Audio', 'Document', 'String', 'Int',
                      'Float', 'Bool', 'Json', 'Array', 'Timestamp', 'Date']

        return (cls.__name__ in type_names or
                'type_system' in module_name or
                'types' in module_name.lower())

    def _add_class_methods(self, class_path: str):
        """Add public methods of a class."""
        try:
            parts = class_path.split('.')
            module_path = '.'.join(parts[:-1])
            class_name = parts[-1]

            module = importlib.import_module(module_path)
            if hasattr(module, class_name):
                cls = getattr(module, class_name)
                if inspect.isclass(cls):
                    for name, obj in inspect.getmembers(cls):
                        if (
                            (not name.startswith('_') or name in ['__init__', '__call__'])
                            and (inspect.ismethod(obj) or inspect.isfunction(obj))
                            and name not in ['__init__']
                        ):
                            method_path = f"{class_path}.{name}"
                            if method_path not in self.api_entries["methods"]:
                                self.api_entries["methods"].append(method_path)
        except (ImportError, AttributeError):
            pass  # Silently skip if we can't introspect

    def _process_module(self, module_path: str):
        """Process all public members of a module."""
        try:
            module = importlib.import_module(module_path)
            for name in dir(module):
                if not name.startswith('_'):
                    full_path = f"{module_path}.{name}"
                    self._categorize_module_child(full_path)
        except ImportError:
            pass  # Silently skip if we can't import

    def generate(self, pages) -> Path:
        """Generate the public API JSON file."""
        # Process all pages
        self.process_opml_pages(pages)

        # Always include core types for signature purposes
        self.api_entries["types"].extend(self.core_types)

        # Remove duplicates and sort
        for category in self.api_entries:
            self.api_entries[category] = sorted(set(self.api_entries[category]))

        # Create the complete API structure
        api_json = {
            "version": self.version,
            "generated": datetime.now().isoformat(),
            "counts": {
                "functions": len(self.api_entries["functions"]),
                "classes": len(self.api_entries["classes"]),
                "methods": len(self.api_entries["methods"]),
                "types": len(self.api_entries["types"]),
                "total": sum(len(v) for v in self.api_entries.values())
            },
            **self.api_entries
        }

        # Save to file
        output_path = self.output_dir / 'pixeltable_public_api.json'
        with open(output_path, 'w', encoding="utf-8") as f:
            json.dump(api_json, f, indent=2)

        print("\n📋 Generated public API listing:")
        print(f"   Functions: {len(self.api_entries['functions'])}")
        print(f"   Classes: {len(self.api_entries['classes'])}")
        print(f"   Methods: {len(self.api_entries['methods'])}")
        print(f"   Types: {len(self.api_entries['types'])}")
        print(f"   Total: {api_json['counts']['total']}")
        print(f"   Saved to: {output_path}")

        return output_path


if __name__ == "__main__":
    from opml_reader import OPMLReader

    # Load OPML
    opml_path = Path(__file__).parent.parent / 'public_api.opml'
    opml_reader = OPMLReader(opml_path)
    tab_structure = opml_reader.load()
    all_pages = opml_reader.get_all_pages()

    # Generate API listing
    generator = PublicAPIGenerator(Path(__file__).parent, version='main')

    # Process pages and generate output
    output_path = generator.generate(all_pages)
    print(f"\n✅ Generated {output_path}")
