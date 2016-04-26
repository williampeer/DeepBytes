#!/usr/bin/env bash

echo Run mode 1
echo theano-cache clear
theano-cache clear
echo python ExperimentExecutionm1.py
python ExperimentExecutionm1.py

echo Run mode 2
echo theano-cache clear
theano-cache clear
echo python ExperimentExecutionm2.py
python ExperimentExecutionm2.py

echo Run mode 3
echo theano-cache clear
theano-cache clear
echo python ExperimentExecutionm3.py
python ExperimentExecutionm3.py

echo Run mode 4
echo theano-cache clear
theano-cache clear
echo python ExperimentExecutionm4.py
python ExperimentExecutionm4.py

echo Run mode 5
echo theano-cache clear
theano-cache clear
echo python ExperimentExecutionm5.py
python ExperimentExecutionm5.py

echo Run mode 6
echo theano-cache clear
theano-cache clear
echo python ExperimentExecutionm6.py
python ExperimentExecutionm6.py

sleep 5m  # while syncing
sudo poweroff