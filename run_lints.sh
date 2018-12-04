#!/bin/bash
set -e

echo "Running pylint..."
pylint --output-format=colorized pycrumbs/ tests/

echo -e "\n\nRunning flake8..."
flake8 --max-line-length=120 pycrumbs/ tests/

echo -e "\n\nAll lints passed successfully."
