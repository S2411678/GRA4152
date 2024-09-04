""" import numpy as np
a = np.array ([1,2,3,4]) 
b = [3, 0, 0, -1]
c = [True, False, False, True]

print(a[b])
print(a[c])

x=5
y=2
print(f"x!=y is {x!=y}")
print(f"x<y is {x<y}")

print ("A","B","C",sep ="-")


from datetime import datetime
a = datetime (2024,8,12).strftime("%d%m%y")
print(a)

print(bool(2))
print(bool((1,2)))
print(bool([3,4]))
print(bool("hello")) """


#Excercise 2.3
"""x=float(input("x="))
print("square of x is ",x*x)
print("cube of x is ",x*x*x)
print("fourth power of x is ",x**4) """


#Excercise 2.10
""" cost = float(input("The cost of a new car:"))
usage = float(input("The estimated miles driven per year:"))
price = float(input("The estimated gas price:"))
power = float(input("The efficiency in miles per gallon:"))
year = float(input("The number of years in use:"))
resale = float(input("The estimated resale value:"))"""

""" import sys
cost = float(sys.argv[1])
usage = float(sys.argv[2])
price = float(sys.argv[3])
power = float(sys.argv[4])
year = float(sys.argv[5])
resale = float(sys.argv[6]) """

#print(f"Total cost of owning a car for {year} years is {cost + (usage/power)*price*year - resale}")


#Excercise 3.12
""" X={"A":4,"B":3,"C":2,"D":1}
adjust={"+":0.3,"-":-0.3}
x=input("Enter your letter grade:").upper()

if x=="F":
    print("The numeric value is",0)
elif X.get(x[0])==None:
    print("This is not a valid grade")
elif len(x)==1:
    print("The numeric value is",X[x])
elif len(x)==2:
    if x[1] == "+":
        print("The numeric value is",min(X[x[0]]+adjust[x[1]],4))
    if x[1] == "-":
        print("The numeric value is",X[x[0]]+adjust[x[1]])
else:
    print("This is not a valid grade") """


""" import math
print(dir(math)) """


#Excercise 4.15
n = int(input("input n:"))

""" def fibonacci(n):
    if n <= 2:
        return 1
    return(fibonacci(n-1)+fibonacci(n-2))
print (fibonacci(n)) """

#Excercise 4.17
""" def is_prime(n):
    if n==1:
        return False
    if n==2:
        return True
    if n%2==0:
        return False
    for i in range (3,int(n**0.5+1),2):
        if n%i==0:
            return False
    return True 

prime_list=[]
for a in range (n+1):
    if is_prime(a):
        prime_list.append(a)  

print(prime_list)
     """