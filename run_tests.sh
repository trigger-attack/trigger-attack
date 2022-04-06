#!/bin/sh
python -W ignore -m unittest discover -s tests -p "*_test.py" -v
python -W ignore -m unittest discover -s tests/losses -p "*_test.py" -v
python -W ignore -m unittest discover -s tests/preprocessing -p "*_test.py" -v