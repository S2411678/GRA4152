#Excercise 2.10
import sys
cost = float(sys.argv[1])
usage = float(sys.argv[2])
price = float(sys.argv[3])
power = float(sys.argv[4])
year = float(sys.argv[5])
resale = float(sys.argv[6])

print(f"Total cost of owning a car for {year} years is {cost + (usage/power)*price*year - resale}")