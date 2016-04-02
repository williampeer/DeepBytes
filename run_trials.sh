COUNTER=0
while [  $COUNTER -lt 75 ]; do
	echo Run \# $COUNTER
	echo theano-cache clear
	theano-cache clear
	echo python ExperimentExecution.py
	python ExperimentExecution.py
	let COUNTER=COUNTER+1 
	done
