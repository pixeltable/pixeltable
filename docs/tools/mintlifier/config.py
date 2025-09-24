"""
Mintlifier configuration file
"""

config = {
    # Paths relative to project root
    "opml_path": "/docs/tools/public_api.opml",
    "docs_json_path": "/docs/mintlify/docs.json",
    "sdk_output_dir": "/docs/mintlify/docs/sdk/latest",
    "opml_backup_dir": "/docs/tools/mintlifier/opml_bak",
    "docs_json_backup_dir": "/docs/tools/mintlifier/docsjson_bak",

    # GitHub repository information for source links
    "github_repo": "pixeltable/pixeltable",
    "github_package_path": "pixeltable",

    # Mintlify SDK tab name
    "sdk_tab": "Pixeltable SDK",

    # Display errors in generated docs (can be overridden with --no-errors flag)
    "show_errors": True,

    # Internal/runtime classes injected into modules that should never be documented
    "internal_blacklist": [
        "Env",
        "TempStore",
        "MediaStore",
        "Catalog",
        "StorageBackend",
        "TransactionalDirectory",
        "Config",
        "Client",
        "Server",
        "StorageManager",
        "CacheManager",
        "FileCache",
        "S3Client",
        "TableVersionHandle",
        "SchemaObject",
        "Path",
        "Dir",
        "TableVersion",
        "ColumnVersion",
        "FunctionVersion"
    ]
}