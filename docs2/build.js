#!/usr/bin/env node

// Simple static site generator for Mintlify-style docs
const fs = require('fs');
const path = require('path');

console.log('🔨 Building static docs...');

// Create output directory
const outDir = 'out';
if (!fs.existsSync(outDir)) {
  fs.mkdirSync(outDir, { recursive: true });
}

// Copy all necessary files
const filesToCopy = [
  'mint.json',
  'favicon.svg'
];

const dirsToCopy = [
  'docs',
  'logo'
];

// Copy files
filesToCopy.forEach(file => {
  if (fs.existsSync(file)) {
    fs.copyFileSync(file, path.join(outDir, file));
    console.log(`✅ Copied ${file}`);
  }
});

// Copy directories
dirsToCopy.forEach(dir => {
  if (fs.existsSync(dir)) {
    copyDir(dir, path.join(outDir, dir));
    console.log(`✅ Copied ${dir}/`);
  }
});

// Create a basic index.html that loads Mintlify
const indexHtml = `<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Pixeltable Documentation</title>
    <script src="https://unpkg.com/mintlify@latest/dist/mintlify.js"></script>
    <link rel="stylesheet" href="https://unpkg.com/mintlify@latest/dist/mintlify.css">
</head>
<body>
    <div id="mintlify-root"></div>
    <script>
        // Initialize Mintlify with our config
        Mintlify.init({
            configPath: './mint.json'
        });
    </script>
</body>
</html>`;

fs.writeFileSync(path.join(outDir, 'index.html'), indexHtml);
console.log('✅ Created index.html');

console.log('🎉 Build complete!');

function copyDir(src, dest) {
  if (!fs.existsSync(dest)) {
    fs.mkdirSync(dest, { recursive: true });
  }
  
  const files = fs.readdirSync(src);
  files.forEach(file => {
    const srcPath = path.join(src, file);
    const destPath = path.join(dest, file);
    
    if (fs.statSync(srcPath).isDirectory()) {
      copyDir(srcPath, destPath);
    } else {
      fs.copyFileSync(srcPath, destPath);
    }
  });
}