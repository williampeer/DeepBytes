#!/usr/bin/env bash

echo theano-cache clear
theano-cache clear

echo python EE.py
python EE.py

echo sleep 1m
sleep 1m

echo sudo poweroff
sudo poweroff
