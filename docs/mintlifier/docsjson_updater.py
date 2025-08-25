"""
Docs.json updater for Mintlifier documentation.

Updates the Mintlify navigation structure with generated SDK documentation.
"""

import json
from pathlib import Path
from typing import Dict, List
from datetime import datetime
import shutil


class DocsJsonUpdater:
    """Updates docs.json with SDK navigation structure."""
    
    def __init__(self, docs_json_path: Path, sdk_tab_name: str):
        """Initialize with path to docs.json and SDK tab name."""
        self.docs_json_path = docs_json_path
        self.sdk_tab_name = sdk_tab_name
        self.docs_config = None
    
    def load(self):
        """Load and backup docs.json."""
        # Create timestamped backup
        self._backup_file()
        
        # Load docs.json
        with open(self.docs_json_path) as f:
            self.docs_config = json.load(f)
            
        print(f"📋 Loaded docs.json with {len(self.docs_config.get('navigation', {}).get('tabs', []))} tabs")
    
    def _backup_file(self):
        """Create timestamped backup of docs.json."""
        # Create backup in mintlifier directory, not mintlify
        backup_dir = Path(__file__).parent / "docsjson_bak"
        backup_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_name = f"docs_{timestamp}.json"
        backup_path = backup_dir / backup_name
        
        shutil.copy2(self.docs_json_path, backup_path)
        print(f"📋 Created docs.json backup: {backup_path}")
    
    def update_navigation(self, navigation_structure: Dict):
        """Update navigation with SDK documentation structure."""
        if not self.docs_config:
            raise ValueError("docs.json not loaded. Call load() first.")
        
        # Ensure navigation structure exists
        if 'navigation' not in self.docs_config:
            self.docs_config['navigation'] = {'tabs': []}
        if 'tabs' not in self.docs_config['navigation']:
            self.docs_config['navigation']['tabs'] = []
        
        tabs = self.docs_config['navigation']['tabs']
        
        # Find existing SDK or API Reference tab
        sdk_tab_index = None
        for i, tab in enumerate(tabs):
            if tab.get('tab') == self.sdk_tab_name:
                sdk_tab_index = i
                break
            elif tab.get('tab') == 'API Reference' and tab.get('href'):
                # Found the old external link tab
                sdk_tab_index = i
                break
        
        # Update or add SDK tab
        if sdk_tab_index is not None:
            print(f"📝 Replacing tab at index {sdk_tab_index}: {tabs[sdk_tab_index].get('tab', 'Unknown')}")
            tabs[sdk_tab_index] = navigation_structure
        else:
            print(f"📝 Adding new tab: {self.sdk_tab_name}")
            tabs.append(navigation_structure)
    
    def save(self):
        """Save updated docs.json."""
        if not self.docs_config:
            raise ValueError("No configuration to save")
        
        # Write with proper formatting
        with open(self.docs_json_path, 'w') as f:
            json.dump(self.docs_config, f, indent=2)
        
        print(f"✅ Updated {self.docs_json_path}")
    
    def validate_structure(self, navigation_structure: Dict) -> List[str]:
        """Validate navigation structure and return any warnings."""
        warnings = []
        
        # Check for required fields
        if 'tab' not in navigation_structure:
            warnings.append("Missing 'tab' field in navigation structure")
        
        # Check if it has either groups or dropdowns
        if 'groups' not in navigation_structure and 'dropdowns' not in navigation_structure:
            warnings.append("Missing 'groups' or 'dropdowns' field in navigation structure")
        
        # Check for empty groups
        if navigation_structure.get('groups'):
            for group in navigation_structure['groups']:
                if not group.get('pages') and not group.get('groups'):
                    warnings.append(f"Empty group: {group.get('group', 'Unknown')}")
        
        # Check for duplicate page paths
        all_pages = []
        
        def collect_pages(items):
            for item in items:
                if 'pages' in item:
                    for page in item['pages']:
                        # Skip nested groups (they're dicts, not strings)
                        if isinstance(page, str):
                            all_pages.append(page)
                        elif isinstance(page, dict) and 'pages' in page:
                            # Recursively collect from nested group
                            collect_pages([page])
                if 'groups' in item:
                    collect_pages(item['groups'])
        
        # Collect from groups or dropdowns
        if navigation_structure.get('groups'):
            collect_pages(navigation_structure['groups'])
        elif navigation_structure.get('dropdowns'):
            # Process dropdowns
            for dropdown in navigation_structure['dropdowns']:
                if dropdown.get('pages'):
                    for page in dropdown['pages']:
                        if isinstance(page, str):
                            all_pages.append(page)
                if dropdown.get('groups'):
                    collect_pages(dropdown['groups'])
        
        seen = set()
        for page in all_pages:
            if page in seen:
                warnings.append(f"Duplicate page path: {page}")
            seen.add(page)
        
        return warnings