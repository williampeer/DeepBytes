#!/usr/bin/env bash

# Assuming that Python 2.7 is installed and the default version, this script is created for Ubuntu 14.04 LTS,
#   and sets up Theano, along with imagemagick, which should be sufficient for executing the experimental suite on the
#   CPU.
sudo apt-get update
sudo apt-get install python-numpy python-scipy python-dev python-pip python-nose g++ libopenblas-dev git
sudo pip install Theano

# Testing Theano/BLAS:
# python `python -c "import os, theano; print os.path.dirname(theano.__file__)"`/misc/check_blas.py
# Optionally, you may go into python by typing "python" in the terminal, and try to import theano by issuing the
#   following command: "import theano"

# Updating Theano:
sudo pip install --upgrade --no-deps theano
# Quoting the online guide fount at: http://deeplearning.net/software/theano/install_ubuntu.html
#   "If you want to also installed NumPy/SciPy with pip instead of the system package, you can run this:"
# sudo pip install --upgrade theano

sudo apt-get install imagemagick

echo Installer complete! Attempting to run a model demo..

python demo.py