#!/usr/bin/env bash

echo theano-cache clear
theano-cache clear

echo python EEFailedSuite.py
python EEFailedSuite.py

echo theano-cache clear
theano-cache clear

echo python EEHetero.py
python EEHetero.py

echo sleep 5m
sleep 5m
echo sudo poweroff
sudo poweroff