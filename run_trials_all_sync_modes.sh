#!/usr/bin/env bash
sudo echo sudo test

echo Run mode 1
echo theano-cache clear
theano-cache clear
echo python ExperimentExecutionm1.py
python ExperimentExecutionm1.py

echo Run mode 2
theano-cache clear
python ExperimentExecutionm2.py

echo Run mode 3
theano-cache clear
python ExperimentExecutionm3.py

echo Run mode 4
theano-cache clear
python ExperimentExecutionm4.py

sudo poweroff