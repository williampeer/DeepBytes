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

echo Run turnover rate experiments
echo theano-cache clear
theano-cache clear
echo python ExperimentExecutionm.py
python ExperimentExecutionm.py

#touch /home/williapb/Dropbox/log.txt
#cat saved_data/log.txt > /home/williapb/Dropbox/log.txt

sleep 5m  # while syncing the log-file.
poweroff