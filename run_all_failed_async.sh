#!/usr/bin/env bash

echo theano-cache clear
theano-cache clear
echo python EEDGWs.py
python EEDGWs.py

echo theano-cache clear
theano-cache clear
echo python EEAllFailedAsync.py
python EEAllFailedAsync.py

echo theano-cache clear
theano-cache clear
echo python EEHeteroAsyncDgw25.py
python EEHeteroAsyncDgw25.py