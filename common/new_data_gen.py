with open('./data.csv', 'r') as data:
    lines = data.readlines()

sum_array = [0 for _ in range(80)]

res_array = [0 for i in range(80)]

for line in lines[1:]:
    line = line[:-1]
    spl = line.split(',')
    for i in range(len(spl[5:])):
        res_array[i]+=float(spl[i])