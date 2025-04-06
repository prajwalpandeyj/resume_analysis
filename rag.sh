#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

echo "Installing Ollama..."
curl -fsSL https://ollama.com/install.sh | sh

echo "Updating package list..."
sudo apt update

echo "Installing Python3..."
sudo apt install python3 -y

echo "Installing pip for Python3..."
sudo apt install python3-pip -y

echo "Updating PATH in .bashrc..."
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc

echo "Pulling Ollama models..."
ollama pull qwen2.5:32b
ollama pull nomic-embed-text

echo "Installing Python dependencies..."
pip install -r requirements.txt

echo "Launching Streamlit app..."
streamlit run main.py
