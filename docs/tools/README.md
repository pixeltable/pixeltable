# Pixeltable Documentation Tools

## 1. Generating SDK docs for Mintlify

Generate API documentation from Python docstrings for the Mintlify documentation site.

### Setup
```bash
conda activate pxt
cd docs/tools/mintlifier
```

### Usage
```bash
python mintlifier.py              # Generate docs (uses config for error display)
python mintlifier.py --no-errors  # Hide import, missing documentation and other errors in generated docs
```

**Options:**
- `--no-errors`: Suppress import, missing documentation and other error messages in generated documentation (overrides config)

### Output
Generated MDX files are written to `docs/mintlify/docs/sdk/latest/`

### Preview
To preview the generated documentation:
```bash
cd ../../mintlify  # Navigate to docs/mintlify
mintlify dev       # Start local preview server
```

### Configuration
The documentation structure is defined in `public_api.opml`, which whitelists the modules, classes, and functions to include in the generated docs.

Edit `mintlifier/config.py`:
- `show_errors`: Display import errors in generated docs (default: true)
- `sdk_output_dir`: Output directory for generated MDX files
- `docs_json_path`: Path to Mintlify navigation file
- `internal_blacklist`: List of internal classes to exclude from documentation

## 2. Generating LLM docs

TODO: Instructions go here

## 3. Generating Mintlify MDX files for notebooks

TODO: Instructions go here