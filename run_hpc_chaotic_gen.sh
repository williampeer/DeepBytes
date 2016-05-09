#!/usr/bin/env bash

echo theano-cache clear
theano-cache clear

echo python EE.py
python EE.py

echo sleep 5m
sleep 5m
echo poweroff
poweroff