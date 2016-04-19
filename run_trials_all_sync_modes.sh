#!/usr/bin/env bash
sudo echo testing sudo

cd saved_data
counters_reset_copy.py
cd ..

echo Run mode 1
theano-cache clear
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