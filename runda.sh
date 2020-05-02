#!/bin/sh
rm checkpoints/exp*
python amazon/dacentvat.py -s books -t dvd -a 0.01 -b 0 -c 0
rm checkpoints/exp*
python amazon/dacentvat.py -s books -t electronics -a 0.01 -b 0 -c 0
rm checkpoints/exp*
python amazon/dacentvat.py -s books -t kitchen -a 0.01 -b 0 -c 0 
rm checkpoints/exp*
python amazon/dacentvat.py -s dvd -t books -a 0.01 -b 0 -c 0
rm checkpoints/exp*
python amazon/dacentvat.py -s dvd -t electronics -a 0.01 -b 0 -c 0
rm checkpoints/exp*
python amazon/dacentvat.py -s dvd -t kitchen -a 0.01 -b 0 -c 0
rm checkpoints/exp*
python amazon/dacentvat.py -s electronics -t books -a 0.01 -b 0 -c 0
rm checkpoints/exp*
python amazon/dacentvat.py -s electronics -t dvd -a 0.01 -b 0 -c 0
rm checkpoints/exp*
python amazon/dacentvat.py -s electronics -t kitchen -a 0.01 -b 0 -c 0
rm checkpoints/exp*
python amazon/dacentvat.py -s kitchen -t books -a 0.01 -b 0 -c 0
rm checkpoints/exp*
python amazon/dacentvat.py -s kitchen -t dvd -a 0.01 -b 0 -c 0
rm checkpoints/exp*
python amazon/dacentvat.py -s kitchen -t electronics -a 0.01 -b 0 -c 0
rm checkpoints/exp*
