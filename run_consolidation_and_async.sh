#!/usr/bin/env bash

echo theano-cache clear
theano-cache clear
echo python EE.py
python EE.py

echo theano-cache clear
theano-cache clear
echo python NeocorticalMemoryConsolidation.py
python NeocorticalMemoryConsolidation.py

sudo poweroff