import random

max = 1000000
count = 0
for i in range(max):
    rand = random.randint(0, 1)
    if rand < 0.5:
        count += 1

print(count/max)

