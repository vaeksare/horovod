#!/bin/bash
pip uninstall -y horovod
rm -r ./build
python setup.py clean
python setup.py install
