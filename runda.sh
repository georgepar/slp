#!/bin/sh
rm checkpoints/da2/0/*/exp*
python amazon/da.py -s books -t dvd -a 0.01 -i 0 > da2_bd.out
rm checkpoints/da2/1/*/exp*
python amazon/da.py -s books -t electronics -a 0.01 -i 1 > da2_be.out
rm checkpoints/da2/2/*/exp*
python amazon/da.py -s books -t kitchen -a 0.01 -i 2 > da2_bk.out
rm checkpoints/da2/3/*/exp*
python amazon/da.py -s dvd -t books -a 0.01 -i 3 > da2_db.out
rm checkpoints/da2/4/*/exp*
python amazon/da.py -s dvd -t electronics -a 0.01 -i 4 > da2_de.out
rm checkpoints/da2/5/*/exp*
python amazon/da.py -s dvd -t kitchen -a 0.01 -i 5 > da2_dk.out
rm checkpoints/da2/6/*/exp*
python amazon/da.py -s electronics -t books -a 0.01 -i 6 > da2_eb.out
rm checkpoints/da2/7/*/exp*
python amazon/da.py -s electronics -t dvd -a 0.01 -i 7 > da2_ed.out
rm checkpoints/da2/8/*/exp*
python amazon/da.py -s electronics -t kitchen -a 0.01 -i 8 > da2_ek.out
rm checkpoints/da2/9/*/exp*
python amazon/da.py -s kitchen -t books -a 0.01 -i 9 > da2_kb.out
rm checkpoints/da2/10/*/exp*
python amazon/da.py -s kitchen -t dvd -a 0.01 -i 10 > da2_kd.out
rm checkpoints/da2/11/*/exp*
python amazon/da.py -s kitchen -t electronics -a 0.01 -i 11 > da2_ke.out
