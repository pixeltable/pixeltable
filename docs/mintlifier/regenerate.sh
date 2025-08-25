#!/bin/bash
# Regenerate Mintlify SDK documentation

echo "🔄 Regenerating Mintlify SDK Documentation"
echo "=========================================="

# Get the script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Backup current docs.json
echo "📋 Backing up current docs.json..."
cp ../mintlify/docs.json ../mintlify/docs.json.bak.$(date +%Y%m%d_%H%M%S)

# Activate conda environment if available
if command -v conda &> /dev/null; then
    echo "🐍 Activating conda environment 'pxt'..."
    eval "$(conda shell.bash hook)"
    conda activate pxt
fi

# Run the generator
echo "🚀 Running mintlifier..."
python mintlifier.py

# Check exit code
if [ $? -eq 0 ]; then
    echo "✅ Documentation regenerated successfully!"
    echo ""
    echo "Next steps:"
    echo "1. cd ../mintlify"
    echo "2. npx mintlify dev"
    echo "3. Open http://localhost:3000 to preview"
else
    echo "❌ Error during regeneration"
    exit 1
fi