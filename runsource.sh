#!/bin/sh
rm checkpoints/exp*
python amazon/sourceonly.py -s books -t dvd
rm checkpoints/exp*
python amazon/sourceonly.py -s books -t electronics
rm checkpoints/exp*
python amazon/sourceonly.py -s books -t kitchen
rm checkpoints/exp*
python amazon/sourceonly.py -s dvd -t books
rm checkpoints/exp*
python amazon/sourceonly.py -s dvd -t electronics
rm checkpoints/exp*
python amazon/sourceonly.py -s dvd -t kitchen
rm checkpoints/exp*
python amazon/sourceonly.py -s electronics -t books
rm checkpoints/exp*
python amazon/sourceonly.py -s electronics -t dvd
rm checkpoints/exp*
python amazon/sourceonly.py -s electronics -t kitchen
rm checkpoints/exp*
python amazon/sourceonly.py -s kitchen -t books
rm checkpoints/exp*
python amazon/sourceonly.py -s kitchen -t dvd
rm checkpoints/exp*
python amazon/sourceonly.py -s kithcen -t electronics
rm checkpoints/exp*
