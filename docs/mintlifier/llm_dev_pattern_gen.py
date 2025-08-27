#!/usr/bin/env python3
"""
Extract patterns from Jupyter notebooks for LLM consumption.

This script walks through notebooks and extracts:
1. Code patterns that use Pixeltable API
2. Explanatory text that provides context
3. Common workflows and use cases
4. Links patterns to the public API from OPML
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Any, Optional
import re
from datetime import datetime
import nbformat
import xml.etree.ElementTree as ET


class NotebookPatternExtractor:
    def __init__(self, opml_path: str, notebooks_dir: str, public_api: set = None):
        """
        Initialize the pattern extractor.
        
        Args:
            opml_path: Path to OPML file (source of truth for public API)
            notebooks_dir: Directory containing Jupyter notebooks
            public_api: Set of public API functions/classes (loaded from OPML)
        """
        self.opml_path = Path(opml_path)
        self.notebooks_dir = Path(notebooks_dir)
        self.public_api = public_api or self._load_public_api_from_opml()
        self.patterns = []
        
    def _load_public_api_from_opml(self) -> set:
        """Load public API identifiers from OPML."""
        public_api = set()
        
        try:
            tree = ET.parse(self.opml_path)
            root = tree.getroot()
            
            # Walk the OPML structure to extract API elements
            for outline in root.iter('outline'):
                # Extract module, class, function names
                if 'text' in outline.attrib:
                    api_element = outline.attrib['text']
                    # Clean up the element name
                    if ' - ' in api_element:
                        api_element = api_element.split(' - ')[0]
                    public_api.add(api_element.strip())
                    
        except Exception as e:
            print(f"Warning: Could not load OPML: {e}")
            
        return public_api
    
    def extract_patterns_from_notebook(self, notebook_path: Path) -> Dict[str, Any]:
        """
        Extract patterns from a single notebook.
        
        Returns:
            Dictionary containing extracted patterns and metadata
        """
        with open(notebook_path, 'r', encoding='utf-8') as f:
            nb = nbformat.read(f, as_version=4)
            
        patterns = {
            'notebook': str(notebook_path.relative_to(self.notebooks_dir)),
            'title': self._extract_title(nb),
            'description': self._extract_description(nb),
            'workflows': [],
            'api_usage': {},
            'key_concepts': [],
            'code_snippets': []
        }
        
        current_workflow = None
        current_explanation = []
        
        for cell in nb.cells:
            if cell.cell_type == 'markdown':
                # Look for workflow headers
                if self._is_workflow_header(cell.source):
                    if current_workflow:
                        patterns['workflows'].append(current_workflow)
                    current_workflow = {
                        'title': self._extract_header_text(cell.source),
                        'explanation': [],
                        'code_blocks': []
                    }
                    current_explanation = []
                else:
                    # Accumulate explanation text
                    if current_workflow:
                        current_workflow['explanation'].append(cell.source)
                    else:
                        current_explanation.append(cell.source)
                        
            elif cell.cell_type == 'code':
                code = cell.source
                
                # Extract API usage
                api_calls = self._extract_api_calls(code)
                for api_call in api_calls:
                    if api_call not in patterns['api_usage']:
                        patterns['api_usage'][api_call] = []
                    patterns['api_usage'][api_call].append({
                        'code': code,
                        'context': '\n'.join(current_explanation[-2:]) if current_explanation else ''
                    })
                
                # Add to current workflow if exists
                if current_workflow:
                    current_workflow['code_blocks'].append({
                        'code': code,
                        'output': self._extract_output(cell),
                        'explanation': '\n'.join(current_explanation)
                    })
                    current_explanation = []
                else:
                    # Standalone code snippet
                    patterns['code_snippets'].append({
                        'code': code,
                        'context': '\n'.join(current_explanation[-2:]) if current_explanation else '',
                        'output': self._extract_output(cell)
                    })
        
        # Don't forget the last workflow
        if current_workflow:
            patterns['workflows'].append(current_workflow)
            
        # Extract key concepts from the patterns
        patterns['key_concepts'] = self._extract_key_concepts(patterns)
        
        return patterns
    
    def _extract_title(self, nb: nbformat.NotebookNode) -> str:
        """Extract notebook title from first markdown cell."""
        for cell in nb.cells:
            if cell.cell_type == 'markdown':
                lines = cell.source.split('\n')
                for line in lines:
                    if line.startswith('#'):
                        return line.strip('#').strip()
        return "Untitled"
    
    def _extract_description(self, nb: nbformat.NotebookNode) -> str:
        """Extract description from early markdown cells."""
        description_lines = []
        for cell in nb.cells[:5]:  # Look at first 5 cells
            if cell.cell_type == 'markdown':
                # Skip headers
                lines = [l for l in cell.source.split('\n') 
                        if not l.startswith('#') and l.strip()]
                description_lines.extend(lines[:3])  # Take first 3 non-header lines
                if len(description_lines) >= 3:
                    break
        return ' '.join(description_lines)
    
    def _is_workflow_header(self, text: str) -> bool:
        """Check if markdown text is a workflow section header."""
        patterns = [
            r'^#{2,3}\s+(step|workflow|example|task|tutorial)',
            r'^#{2,3}\s+\d+\.',  # Numbered sections
            r'^#{2,3}\s+(creating|building|implementing|using)',
        ]
        for pattern in patterns:
            if re.search(pattern, text.lower(), re.MULTILINE):
                return True
        return False
    
    def _extract_header_text(self, text: str) -> str:
        """Extract clean header text."""
        lines = text.split('\n')
        for line in lines:
            if line.startswith('#'):
                return line.strip('#').strip()
        return text.split('\n')[0]
    
    def _extract_api_calls(self, code: str) -> List[str]:
        """Extract Pixeltable API calls from code."""
        api_calls = []
        
        # Common patterns
        patterns = [
            r'pxt\.(\w+)',  # pxt.create_table, etc.
            r'pixeltable\.(\w+)',  # pixeltable.create_table
            r'from pixeltable import (\w+)',  # imports
            r'(\w+)\.add_computed_column',  # table operations
            r'(\w+)\.insert',
            r'(\w+)\.select',
            r'@pxt\.udf',  # decorators
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, code)
            api_calls.extend(matches)
            
        # Filter to only public API
        return [call for call in api_calls if self._is_public_api(call)]
    
    def _is_public_api(self, api_element: str) -> bool:
        """Check if an API element is in the public API."""
        # Check direct match
        if api_element in self.public_api:
            return True
        
        # Check with common prefixes
        for prefix in ['pixeltable.', 'pxt.', 'functions.', 'iterators.']:
            if f"{prefix}{api_element}" in self.public_api:
                return True
                
        return False
    
    def _extract_output(self, cell) -> Optional[str]:
        """Extract relevant output from a code cell."""
        if not hasattr(cell, 'outputs') or not cell.outputs:
            return None
            
        for output in cell.outputs:
            if output.output_type == 'execute_result':
                if 'text/plain' in output.data:
                    return output.data['text/plain']
            elif output.output_type == 'stream':
                return output.text
                
        return None
    
    def _extract_key_concepts(self, patterns: Dict) -> List[str]:
        """Extract key concepts from the patterns."""
        concepts = set()
        
        # Look for common Pixeltable concepts
        concept_keywords = {
            'computed column': ['add_computed_column', 'computed', 'automatic'],
            'embedding': ['embedding', 'vector', 'similarity'],
            'udf': ['@pxt.udf', 'user-defined function', 'custom function'],
            'iterator': ['iterator', 'frame', 'chunk', 'split'],
            'multimodal': ['image', 'video', 'audio', 'document'],
            'incremental': ['incremental', 'recompute', 'update'],
            'tool calling': ['tool', 'invoke_tools', 'agent'],
        }
        
        # Check in code and explanations
        all_text = json.dumps(patterns).lower()
        for concept, keywords in concept_keywords.items():
            for keyword in keywords:
                if keyword in all_text:
                    concepts.add(concept)
                    break
                    
        return list(concepts)
    
    def extract_all_patterns(self) -> Dict[str, Any]:
        """Extract patterns from all notebooks."""
        all_patterns = {
            'generated': datetime.now().isoformat(),
            'notebooks': []
        }
        
        notebook_files = list(self.notebooks_dir.rglob('*.ipynb'))
        print(f"Found {len(notebook_files)} notebooks")
        
        for nb_path in notebook_files:
            # Skip checkpoint notebooks
            if '.ipynb_checkpoints' in str(nb_path):
                continue
                
            print(f"Processing {nb_path.name}...")
            try:
                patterns = self.extract_patterns_from_notebook(nb_path)
                all_patterns['notebooks'].append(patterns)
            except Exception as e:
                print(f"  Error processing {nb_path}: {e}")
                
        return all_patterns
    
    def save_patterns(self, output_path: str):
        """Save extracted patterns to JSON file."""
        patterns = self.extract_all_patterns()
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(patterns, f, indent=2)
            
        print(f"Saved patterns to {output_path}")
        
        # Also create a summary
        self._create_summary(patterns, output_path.parent / 'llm_patterns_summary.md')
        
    def _create_summary(self, patterns: Dict, summary_path: Path):
        """Create a markdown summary of extracted patterns."""
        with open(summary_path, 'w') as f:
            f.write("# Pixeltable Pattern Library Summary\n\n")
            f.write(f"Generated: {patterns['generated']}\n")
            f.write(f"Total notebooks: {len(patterns['notebooks'])}\n\n")
            
            f.write("## Notebooks by Category\n\n")
            
            # Group by directory
            categories = {}
            for nb in patterns['notebooks']:
                category = Path(nb['notebook']).parent.name
                if category not in categories:
                    categories[category] = []
                categories[category].append(nb)
            
            for category, notebooks in sorted(categories.items()):
                f.write(f"### {category.title()}\n\n")
                for nb in notebooks:
                    f.write(f"- **{nb['title']}** ({nb['notebook']})\n")
                    f.write(f"  - Concepts: {', '.join(nb['key_concepts'])}\n")
                    f.write(f"  - Workflows: {len(nb['workflows'])}\n")
                    f.write(f"  - API calls: {len(nb['api_usage'])}\n")
                f.write("\n")
            
            # Most used API calls
            f.write("## Most Used API Calls\n\n")
            api_counts = {}
            for nb in patterns['notebooks']:
                for api_call in nb['api_usage']:
                    api_counts[api_call] = api_counts.get(api_call, 0) + 1
            
            for api_call, count in sorted(api_counts.items(), key=lambda x: -x[1])[:20]:
                f.write(f"- `{api_call}`: {count} notebooks\n")
                
        print(f"Created summary at {summary_path}")


if __name__ == "__main__":
    # Use the OPML from mintlifier as source of truth
    extractor = NotebookPatternExtractor(
        opml_path="mintlifier.opml",
        notebooks_dir="../../notebooks"
    )
    
    extractor.save_patterns("llm_patterns.json")