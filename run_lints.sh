#!/bin/bash
set -e

echo "Running pylint..."
pylint --output-format=colorized pycrumbs/ tests/ examples/

echo -e "\n\nRunning flake8..."
flake8 --max-line-length=120 pycrumbs/ tests/ examples/

echo -e "\n\nAll lints passed successfully."
