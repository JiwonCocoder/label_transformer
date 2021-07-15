def print_param4(a, b, c):
    print(a,b,c)
p = ['a', 'b', 'c']
print_param4(*p)

p2 = {'c': '1', 'a':'2', 'b': '3'}
print_param4(**p2)