#!/usr/bin/env bash

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

echo Run mode 5
echo theano-cache clear
theano-cache clear
echo python ExperimentExecutionm5.py
python ExperimentExecutionm5.py

sleep 5m  # while syncing
poweroff