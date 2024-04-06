def open_csv(csv_path):
    with open(csv_path, 'r') as file:
        lines = file.readlines()
    
    res = {}
    for line in lines[1:]:
        parced_line = line.split(',')
        res[parced_line[0]] = parced_line[1:]
    return res


bench = open_csv('GPU_benchmarks_v7.csv')
mounths_data = open_csv('dynamyc_m_data.csv')

out_dyn = open('./data.csv', 'w+')

out_dyn.write('Name,G3D,G2D,Price,TDP,')
out_dyn.write(','.join([f'm_{i}' for i in range(80)]))

s = 0
for gpu_name in mounths_data.keys():
    lit = gpu_name
    lit = lit.replace('NVIDIA ', '')
    lit = lit.replace('Intel ', '')
    lit = lit.replace('AMD ', '')
    lit = lit.replace(' Graphics', '')
    lit = lit.replace(' Series', '')
    lit = lit.replace(' Series', '')
    if lit in bench:
        if '' not in bench[lit]:
            out_dyn.write(','.join([gpu_name] + bench[lit][:2]+ [str(1 / float(bench[lit][3]))] + [str(1/float(bench[lit][4]))] + mounths_data[gpu_name]) + '\n')
            out_dyn.write(','.join([gpu_name] + mounths_data[gpu_name]) + '\n')
            
print(s)