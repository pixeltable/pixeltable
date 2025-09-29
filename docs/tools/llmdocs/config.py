"""
LLMDocs configuration file
"""

config = {
    # Paths relative to project root
    'public_api_opml': '/docs/tools/public_api.opml',
    'notebooks_dir': '/docs/notebooks',
    'output_dir': '/docs/tools/llmdocs/llm_output',
    # Output file names
    'llm_map_file': 'llm_map.jsonld',
    'llm_patterns_file': 'llm_dev_patterns.jsonld',
    'llm_reference_file': 'llm_quick_reference.md',
    # Generation settings
    'version': 'main',
    'show_progress': True,
}
