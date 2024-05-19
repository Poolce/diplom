




def open_csv(csv_path):
    with open(csv_path, 'r') as file:
        lines = file.readlines()
    
    res = {}
    for line in lines[1:]:
        parced_line = line.split(',')
        res[parced_line[0]] = parced_line[1:]
    return res

def open_score(csv_path):
    with open(csv_path, 'r') as file:
        lines = file.readlines()
    
    res = {}
    for line in lines[1:]:
        parced_line = line.split(',')
        res[f"{parced_line[0]} {parced_line[1]}"] = parced_line[4]
    return res


main_data = open_csv("data.csv")
scores_data = open_score("GPU_scores_graphicsAPIs.csv")

out_dyn = open('./ndata.csv', 'w+')

out_dyn.write('Name,Opencl,G3D,G2D,Price,TDP,')
out_dyn.write(','.join([f'm_{i}' for i in range(80)]))
out_dyn.write('\n')

s = 0
print(main_data)

for gpu_name in main_data.keys():
    for score_gpu in scores_data.keys():
        if score_gpu == gpu_name:
            out_dyn.write(f"{gpu_name},{scores_data[gpu_name]},{','.join(main_data[gpu_name])}")
            s+=1
print(s)