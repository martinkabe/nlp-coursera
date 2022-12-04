import random

numbers = [1,2,3,4,5,6,7,8,9]

def odd_even(a, b):
    return 'even' if a%b==0 else 'odd'

def rand():
    return [random.randint(1, 101) for x in range(len(numbers))]

# print(rand())
# print(odd_even(9,5))

res = list(map(odd_even, numbers, rand()))
print(res)