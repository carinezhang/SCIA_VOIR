#!/bin/bash

python3 -m venv ./VOIR_zhang_d
source VOIR_zhang_d/bin/activate
pip install imutils opencv-python
python3 edge.py
rm -fr ./VOIR_zhang_d
