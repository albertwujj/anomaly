import random
out = open('data.txt', 'w')

dim = 100
[[out.write(str(random.uniform(-1000, 1000)) + (" " if i < dim - 1 else "\n")) for i in range(dim)] for _ in range(5000)]