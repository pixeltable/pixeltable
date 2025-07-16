#!/bin/bash

# Pixeltable Docs v2 Setup Script
echo "🚀 Setting up Pixeltable Docs v2..."

# Check if we're in the right directory
if [ ! -f "docs2/mint.json" ]; then
    echo "❌ Please run this script from the pixeltable repo root"
    exit 1
fi

# Install Node dependencies
echo "📦 Installing dependencies..."
cd docs2
npm install

echo ""
echo "✅ Setup complete!"
echo ""
echo "🔥 Next steps:"
echo "  1. Start local development: cd docs2 && npm run dev"
echo "  2. Visit http://localhost:3000"
echo "  3. Set up Netlify deployment"
echo ""
echo "📚 What's been created:"
echo "  ✅ Modern Mintlify documentation structure"
echo "  ✅ Responsive, fast, searchable docs"
echo "  ✅ Get Started section with quickstart"
echo "  ✅ Placeholder sections for examples"
echo "  ✅ GitHub Actions for build testing"
echo "  ✅ Netlify configuration for deploy previews"
echo ""
echo "🎯 Ready for Friday demo!"