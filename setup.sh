#!/bin/bash

set -e
virtualenv -p python3 venv/
source venv/bin/activate
pip install --no-cache-dir -r requirements.txt
if [ ${USER} = 'teamcity' ]; then
  pip install -r test_requirements.txt
fi
python setup.py develop

echo -e "\n\nSuccess"
