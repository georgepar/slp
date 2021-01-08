import sys

sum = 0
n = 0
for line in sys.stdin:
    line = float(line)
    sum += line
    n += 1

print(float(sum) / n)
