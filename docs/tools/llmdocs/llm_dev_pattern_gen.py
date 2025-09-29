#!/usr/bin/env python3
"""
Extract patterns from Jupyter notebooks for LLM consumption.

This script walks through notebooks and extracts:
1. Code patterns that use Pixeltable API
2. Explanatory text that provides context
3. Common developer patterns and use cases
4. Links patterns to the public API from OPML
"""

import json
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
            print(f'Warning: Could not load OPML: {e}')

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
            'developer_patterns': [],
            'key_concepts': [],
            'code_snippets': [],
        }

        current_pattern = None
        current_explanation = []

        for cell in nb.cells:
            if cell.cell_type == 'markdown':
                # Look for pattern headers
                if self._is_pattern_header(cell.source):
                    if current_pattern:
                        patterns['developer_patterns'].append(current_pattern)
                    current_pattern = {
                        'title': self._extract_header_text(cell.source),
                        'explanation': [],
                        'code_blocks': [],
                    }
                    current_explanation = []
                # Accumulate explanation text
                elif current_pattern:
                    current_pattern['explanation'].append(cell.source)
                else:
                    current_explanation.append(cell.source)

            elif cell.cell_type == 'code':
                code = cell.source

                # Add to current pattern if exists
                if current_pattern:
                    current_pattern['code_blocks'].append(
                        {
                            'code': code,
                            'output': self._extract_output(cell),
                            'explanation': '\n'.join(current_explanation),
                        }
                    )
                    current_explanation = []
                else:
                    # Standalone code snippet
                    patterns['code_snippets'].append(
                        {
                            'code': code,
                            'context': '\n'.join(current_explanation[-2:]) if current_explanation else '',
                            'output': self._extract_output(cell),
                        }
                    )

        # Don't forget the last pattern
        if current_pattern:
            patterns['developer_patterns'].append(current_pattern)

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
        return 'Untitled'

    def _extract_description(self, nb: nbformat.NotebookNode) -> str:
        """Extract description from early markdown cells, removing badge links."""
        description_lines = []
        for cell in nb.cells[:5]:  # Look at first 5 cells
            if cell.cell_type == 'markdown':
                # Skip headers and badge links
                lines = []
                for line in cell.source.split('\n'):
                    # Skip headers and Kaggle/Colab badges
                    if line.startswith('#'):
                        continue
                    if '[![Kaggle]' in line or '[![Colab]' in line:
                        continue
                    if line.strip():
                        lines.append(line)
                description_lines.extend(lines[:3])  # Take first 3 non-header lines
                if len(description_lines) >= 3:
                    break
        return ' '.join(description_lines)

    def _is_pattern_header(self, text: str) -> bool:
        """Check if markdown text is a pattern section header."""
        patterns = [
            r'^#{2,3}\s+(step|pattern|example|task|tutorial)',
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
        all_patterns = {'generated': datetime.now().isoformat(), 'notebooks': []}

        notebook_files = list(self.notebooks_dir.rglob('*.ipynb'))
        print(f'Found {len(notebook_files)} notebooks')

        for nb_path in notebook_files:
            # Skip checkpoint notebooks
            if '.ipynb_checkpoints' in str(nb_path):
                continue

            print(f'Processing {nb_path.name}...')
            try:
                patterns = self.extract_patterns_from_notebook(nb_path)
                all_patterns['notebooks'].append(patterns)
            except Exception as e:
                print(f'  Error processing {nb_path}: {e}')

        return all_patterns

    def save_patterns(self, output_path: str):
        """Save extracted patterns to JSON-LD file."""
        patterns = self.extract_all_patterns()

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert to JSON-LD format with proper structure
        jsonld_patterns = self._convert_to_jsonld(patterns)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(jsonld_patterns, f, indent=2)

        print(f'Saved developer patterns to {output_path}')

    def _convert_to_jsonld(self, patterns: Dict) -> Dict:
        """Convert patterns to JSON-LD format with GitHub links."""
        base_github_url = 'https://github.com/pixeltable/pixeltable/blob/main/docs/notebooks'

        return {
            '@context': {
                '@vocab': 'https://schema.org/',
                'pxt': 'https://pixeltable.com/ontology#',
                'patterns': 'pxt:developerPatterns',
                'concepts': 'pxt:keyConcepts',
            },
            '@type': 'DataCatalog',
            'name': 'Pixeltable Developer Patterns',
            'description': 'Curated examples demonstrating Pixeltable capabilities',
            'dateModified': patterns['generated'],
            'dataset': [
                {
                    '@type': 'Dataset',
                    '@id': f'pxt:pattern/{i}',
                    'name': nb['title'],
                    'description': nb['description'],
                    'url': f'{base_github_url}/{nb["notebook"]}',
                    'keywords': nb['key_concepts'],
                    'hasPart': [
                        {
                            '@type': 'HowTo',
                            'name': pattern['title'],
                            'description': ' '.join(pattern['explanation'][:2]) if pattern['explanation'] else '',
                            'step': [
                                {
                                    '@type': 'HowToStep',
                                    'position': j,
                                    'text': block.get('explanation', ''),
                                    'codeSample': {
                                        '@type': 'SoftwareSourceCode',
                                        'programmingLanguage': 'Python',
                                        'text': block['code'],
                                    },
                                }
                                for j, block in enumerate(pattern['code_blocks'], 1)
                            ],
                        }
                        for pattern in nb['developer_patterns']
                    ],
                }
                for i, nb in enumerate(patterns['notebooks'], 1)
            ],
        }

    def _old_create_summary(self, patterns: Dict, summary_path: Path):
        """Create a markdown summary of extracted patterns for LLM consumption."""
        with open(summary_path, 'w') as f:
            f.write('# Pixeltable Developer Patterns\n\n')
            f.write(
                f'> A curated collection of {len(patterns["notebooks"])} working examples demonstrating Pixeltable capabilities\n\n'
            )

            f.write('## Key Concepts Covered\n\n')

            # Collect all concepts with counts
            concept_counts = {}
            for nb in patterns['notebooks']:
                for concept in nb['key_concepts']:
                    concept_counts[concept] = concept_counts.get(concept, 0) + 1

            for concept, count in sorted(concept_counts.items(), key=lambda x: -x[1]):
                f.write(f'- **{concept}** ({count} examples)\n')
            f.write('\n')

            f.write('## Pattern Categories\n\n')

            # Group by category with better descriptions
            category_descriptions = {
                '': 'Getting Started',
                'feature-guides': 'Core Features',
                'fundamentals': 'Fundamentals',
                'integrations': 'AI Model Integrations',
                'use-cases': 'Real-World Applications',
            }

            categories = {}
            for nb in patterns['notebooks']:
                category = Path(nb['notebook']).parent.name
                if category not in categories:
                    categories[category] = []
                categories[category].append(nb)

            for category, notebooks in sorted(categories.items()):
                category_name = category_descriptions.get(category, category.title())
                f.write(f'### {category_name}\n\n')
                for nb in notebooks:
                    f.write(f'**{nb["title"]}**\n')
                    if nb['description']:
                        # First 150 chars of description
                        desc = nb['description'][:150] + ('...' if len(nb['description']) > 150 else '')
                        f.write(f'  {desc}\n')
                    if nb['key_concepts']:
                        f.write(f'  *Demonstrates: {", ".join(nb["key_concepts"])}*\n')
                    if nb['developer_patterns']:
                        f.write(f'  *{len(nb["developer_patterns"])} code patterns*\n')
                    f.write('\n')
                f.write('\n')

        print(f'Created summary at {summary_path}')


if __name__ == '__main__':
    # Use the OPML from mintlifier as source of truth
    extractor = NotebookPatternExtractor(opml_path='mintlifier.opml', notebooks_dir='../notebooks')

    # Save to llm_output directory with correct name
    output_dir = Path('llm_output')
    output_dir.mkdir(exist_ok=True)
    extractor.save_patterns(output_dir / 'llm_dev_patterns.jsonld')
