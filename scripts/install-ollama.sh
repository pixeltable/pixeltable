#!/bin/bash -e

# Installs ollama and starts the server with nohup (so that the server outlives this script)

# Usage: install-ollama.sh

curl -fsSL https://ollama.com/install.sh -o /tmp/install-ollama.sh

sh /tmp/install-ollama.sh

nohup ollama serve > /tmp/ollama-serve.log 2>&1 &
