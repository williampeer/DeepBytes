#!/usr/bin/env bash
sudo echo testing sudo
COUNTER=0
while [  $COUNTER -lt 2 ]; do
	echo Run \#$COUNTER
	echo theano-cache clear
	theano-cache clear
	echo python ExperimentExecution.py
	python ExperimentExecution.py
	let COUNTER=COUNTER+1
	done
sudo poweroff