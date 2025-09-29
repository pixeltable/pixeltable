"""
OPML reader for Mintlifier documentation generator.

Parses the mintlifier.opml file and provides structured access to the documentation hierarchy.
"""

import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime
import shutil


@dataclass
class PageItem:
    """Represents a documentation page."""

    module_path: str  # e.g., "pixeltable.create_table"
    parent_groups: List[str]  # Hierarchy of parent groups
    item_type: str = 'page'  # "page" or "module"
    children: Optional[List[str]] = None  # For modules/classes, list of specific children to document

    @property
    def name(self) -> str:
        """Get the last part of the module path."""
        return self.module_path.split('.')[-1]


@dataclass
class GroupItem:
    """Represents a documentation group."""

    name: str
    pages: List[PageItem]
    subgroups: List['GroupItem']


@dataclass
class TabItem:
    """Represents a documentation tab."""

    name: str
    groups: List[GroupItem]


class OPMLReader:
    """Reads and parses OPML documentation structure."""

    def __init__(self, opml_path: Path, backup_dir: Path | None = None):
        """Initialize with path to OPML file."""
        self.opml_path = opml_path
        self.backup_dir = backup_dir or (Path(__file__).parent / 'opml_bak')
        self.tree = None
        self.root = None
        self.structure = None

    def load(self) -> TabItem:
        """Load and parse the OPML file."""
        # Create timestamped backup
        self._backup_file()

        # Parse OPML
        self.tree = ET.parse(self.opml_path)
        self.root = self.tree.getroot()

        # Process structure
        self.structure = self._process_root()
        return self.structure

    def _backup_file(self):
        """Create timestamped backup of OPML file."""
        self.backup_dir.mkdir(exist_ok=True, parents=True)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_name = f'public_api_{timestamp}.opml'
        backup_path = self.backup_dir / backup_name

        shutil.copy2(self.opml_path, backup_path)
        print(f'📋 Created OPML backup: {backup_path}')

    def _process_root(self) -> Optional[TabItem]:
        """Process the root OPML structure to find SDK tab."""
        for outline in self.root.iter('outline'):
            text = outline.get('text', '')
            if text.startswith('tab|'):
                _, tab_name = text.split('|', 1)
                groups = self._process_groups(outline, [])
                return TabItem(name=tab_name, groups=groups)
        return None

    def _process_groups(self, parent_element, parent_hierarchy: List[str]) -> List[GroupItem]:
        """Process group elements recursively."""
        groups = []

        for child in parent_element:
            text = child.get('text', '')

            if text.startswith('group|'):
                _, group_name = text.split('|', 1)
                new_hierarchy = [*parent_hierarchy, group_name]

                # Process pages and subgroups
                pages = []
                subgroups = []

                for subchild in child:
                    subtext = subchild.get('text', '')

                    # Check for any item type (page, module, func, class, method)
                    if '|' in subtext and not subtext.startswith('group|'):
                        item_type, module_path = subtext.split('|', 1)

                        # Map old 'page' to the actual type if needed
                        if item_type == 'page':
                            # Default to 'func' for backwards compatibility
                            item_type = 'func'

                        # For modules, collect direct function children and process class children separately
                        children = None
                        class_pages = []  # Store class pages to add after module
                        if item_type == 'module':
                            children = []
                            for grandchild in subchild:
                                grandtext = grandchild.get('text', '')
                                if '|' in grandtext:
                                    child_type, child_path = grandtext.split('|', 1)

                                    if child_type == 'class':
                                        # Process class as a separate page with its methods
                                        class_children = []
                                        for class_child in grandchild:
                                            class_child_text = class_child.get('text', '')
                                            if '|' in class_child_text:
                                                _, method_path = class_child_text.split('|', 1)
                                                method_name = method_path.split('.')[-1]
                                                class_children.append(method_name)

                                        # Store class pages to add after module
                                        class_pages.append(
                                            PageItem(
                                                module_path=child_path,
                                                parent_groups=new_hierarchy,
                                                item_type='class',
                                                children=class_children,
                                            )
                                        )
                                    elif child_type == 'func':
                                        # Add function name to module's children list
                                        child_name = child_path.split('.')[-1]
                                        children.append(child_name)
                        elif item_type == 'class':
                            # For standalone classes, collect their method children
                            children = []
                            for grandchild in subchild:
                                grandtext = grandchild.get('text', '')
                                if '|' in grandtext:
                                    _, child_path = grandtext.split('|', 1)
                                    # Extract just the method name, not the full path
                                    child_name = child_path.split('.')[-1]
                                    children.append(child_name)

                        # Add module page first
                        pages.append(
                            PageItem(
                                module_path=module_path,
                                parent_groups=new_hierarchy,
                                item_type=item_type,
                                children=children,
                            )
                        )

                        # Then add any class pages that were found
                        if item_type == 'module' and class_pages:
                            pages.extend(class_pages)
                    elif subtext.startswith('group|'):
                        # Process nested group directly
                        # We need to process this single group element, not pass it to _process_groups
                        # which expects to iterate over multiple elements
                        _, subgroup_name = subtext.split('|', 1)
                        subgroup_hierarchy = [*new_hierarchy, subgroup_name]

                        # Process the subgroup's children
                        subgroup_pages = []
                        nested_subgroups = []

                        for nested_child in subchild:
                            nested_text = nested_child.get('text', '')

                            # Check for any item type (page, module, func, class, method)
                            if '|' in nested_text and not nested_text.startswith('group|'):
                                item_type, module_path = nested_text.split('|', 1)

                                # Map old 'page' to the actual type if needed
                                if item_type == 'page':
                                    item_type = 'func'

                                # For modules and classes, collect their children
                                children = None
                                if item_type in ['module', 'class']:
                                    children = []
                                    for grandchild in nested_child:
                                        grandtext = grandchild.get('text', '')
                                        if '|' in grandtext:
                                            _, child_path = grandtext.split('|', 1)
                                            # Extract just the item name, not the full path
                                            child_name = child_path.split('.')[-1]
                                            children.append(child_name)

                                subgroup_pages.append(
                                    PageItem(
                                        module_path=module_path,
                                        parent_groups=subgroup_hierarchy,
                                        item_type=item_type,
                                        children=children,
                                    )
                                )
                            elif nested_text.startswith('group|'):
                                # For deeper nesting, we would need to recurse
                                # For now, this handles 2 levels which is what we have
                                pass

                        subgroups.append(
                            GroupItem(name=subgroup_name, pages=subgroup_pages, subgroups=nested_subgroups)
                        )

                groups.append(GroupItem(name=group_name, pages=pages, subgroups=subgroups))

        return groups

    def get_all_pages(self) -> List[PageItem]:
        """Get flat list of all pages in the structure."""
        if not self.structure:
            return []

        pages = []

        def collect_pages(groups: List[GroupItem]):
            for group in groups:
                pages.extend(group.pages)
                if group.subgroups:
                    collect_pages(group.subgroups)

        collect_pages(self.structure.groups)
        return pages

    def get_navigation_structure(self) -> Dict:
        """Get navigation structure for docs.json."""
        if not self.structure:
            return {}

        def process_group(group: GroupItem, base_path: str) -> Dict:
            group_path = self._sanitize_path(group.name)
            full_path = f'{base_path}/{group_path}' if base_path else group_path

            result = {
                'group': group.name,
                'pages': [f'docs/sdk/latest/{full_path}/{page.name}' for page in group.pages],
            }

            # Add subgroups if they exist
            if group.subgroups:
                result['groups'] = [process_group(subgroup, full_path) for subgroup in group.subgroups]

            return result

        return {'tab': self.structure.name, 'groups': [process_group(group, '') for group in self.structure.groups]}

    def _sanitize_path(self, text: str) -> str:
        """Convert text to valid file path."""
        return text.lower().replace(' ', '-').replace('/', '-')
