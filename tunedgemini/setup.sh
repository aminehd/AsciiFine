#!/bin/bash
# Simple setup script for tunedgemini

set -e

# Check if Poetry is installed
if ! command -v poetry &> /dev/null; then
    echo "Poetry is not installed. Please install it first:"
    echo "curl -sSL https://install.python-poetry.org | python3 -"
    exit 1
fi

echo "Installing dependencies with Poetry..."
poetry install

echo "Setting up environment..."
poetry run tunedgemini config --create-env

echo ""
echo "Setup complete!"
echo ""
echo "To activate the virtual environment, run:"
echo "  poetry shell"
echo ""
echo "To run the configuration check again:"
echo "  tunedgemini config"
echo ""
echo "To create a .env file with your GCP settings:"
echo "  tunedgemini config --create-env --force"
echo ""
echo "For more information, see the README.md file." 