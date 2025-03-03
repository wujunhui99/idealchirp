import math

# p = 1-math.exp(-1024 * 1024 / 20000)/2
# print(p )
# r = -1024 * 1024 / 20000
# print(r)

r  = 2*(1- 0.92 **(1/127))
print(math.log(r))