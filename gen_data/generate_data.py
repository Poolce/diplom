import requests
import re
from dataclasses import dataclass
import os.path
import json


@dataclass
class GPU:
    name: str
    metrics: dict

def get_html(link, year, id):
    res = requests.get(link)
    with open(f'./html/{year}_{id}.html', 'w+') as file:
        file.write(res.text)

def get_gpu_by_link22_24(year, id, gl_month):
    with open(f'./html/{year}_{id}.html', 'r') as f:
        text = f.read()
    s1 = re.split(r'<div class="substats_col_left col_header">ALL VIDEO CARDS</div>', text)[1]
    s2 = re.split(r'<div class="substats_col_left">Other</div>', s1)[0]
    lines = s2.split('\n')
    month_pat = re.compile(r'<div class="substats_col_month col_header">(.*)</div>')
    name_pat = re.compile(r'<span>(.*)</span></div></div>')
    metric_pat = re.compile(r'<div class="substats_col_month">(.*)</div>')

    res_gpu_list = []

    mounths = []
    mounth_id = 0

    last_gpu = None

    for line in lines:
        month = re.findall(month_pat, line)
        name = re.findall(name_pat, line)
        metric = re.findall(metric_pat, line)

        if month:
            mounths.append(f'{year}_{month[0]}')

        if name:
            if last_gpu:
                res_gpu_list.append(last_gpu)
            last_gpu = GPU(name[0], {})
            mounth_id = 0

        if metric:
            m = metric[0]
            if m == '-':
                last_gpu.metrics.update({gl_month+mounth_id : 0})
            else:
                last_gpu.metrics.update({gl_month+mounth_id  : float(m[:-1])})
            mounth_id+=1

    return res_gpu_list

def get_gpu_by_link17_21(year, id, gl_month):
    with open(f'./html/{year}_{id}.html', 'r') as f:
        text = f.read()
    s1 = re.split(r'<div class="substats_col_left col_header">ALL VIDEO CARDS</div>', text)[1]
    s2 = re.split(r'<div class="substats_col_left">Other</div>', s1)[0]
    lines = s2.split('\n')
    month_pat = re.compile(r'<div class="substats_col_month col_header">(.*)</div>')
    name_pat = re.compile(r'<div class="substats_col_left"><img src=".*" border="0"/> (.*)</div>')
    metric_pat = re.compile(r'<div class="substats_col_month">(.*)</div>')

    res_gpu_list = []

    mounths = []
    mounth_id = 0

    last_gpu = None
    for line in lines:
        month = re.findall(month_pat, line)
        name = re.findall(name_pat, line)
        metric = re.findall(metric_pat, line)

        if month:
            mounths.append(f'{year}_{month[0]}')

        if name:
            if last_gpu:
                res_gpu_list.append(last_gpu)
            last_gpu = GPU(name[0], {})
            mounth_id = 0

        if metric:
            m = metric[0]
            if m == '-':
                last_gpu.metrics.update({gl_month+mounth_id : 0})
            else:
                last_gpu.metrics.update({gl_month+mounth_id  : float(m[:-1])})
            mounth_id+=1


    return res_gpu_list

res = {}
links22_24 = [
    ('https://web.archive.org/web/20240309230030/https://store.steampowered.com/hwsurvey/videocard/', 24, 1),
    ('https://web.archive.org/web/20231122180740/https://store.steampowered.com/hwsurvey/videocard/', 23, 3),
    ('https://web.archive.org/web/20230727191151/https://store.steampowered.com/hwsurvey/videocard/', 23, 2),
    ('https://web.archive.org/web/20230316172337/https://store.steampowered.com/hwsurvey/videocard/', 23, 1),
    ('https://web.archive.org/web/20221124122359/https://store.steampowered.com/hwsurvey/videocard/', 22, 3),
    ('https://web.archive.org/web/20220713085535/https://store.steampowered.com/hwsurvey/videocard/', 22, 2),
    ('https://web.archive.org/web/20220309034826/https://store.steampowered.com/hwsurvey/videocard/', 22, 1),
]

links17_21 = [
    ('https://web.archive.org/web/20211105090247/https://store.steampowered.com/hwsurvey/videocard/', 21, 3),
    ('https://web.archive.org/web/20210706033055/https://store.steampowered.com/hwsurvey/videocard/', 21, 2),
    ('https://web.archive.org/web/20210323230749/https://store.steampowered.com/hwsurvey/videocard/', 21, 1),
    ('https://web.archive.org/web/20201127094650/https://store.steampowered.com/hwsurvey/videocard/', 20, 3),
    ('https://web.archive.org/web/20200712230514/https://store.steampowered.com/hwsurvey/videocard/', 20, 2),
    ('https://web.archive.org/web/20200303224140/https://store.steampowered.com/hwsurvey/videocard/', 20, 1),
    ('https://web.archive.org/web/20191106170426/https://store.steampowered.com/hwsurvey/videocard/', 19, 3),
    ('https://web.archive.org/web/20190725181556/https://store.steampowered.com/hwsurvey/videocard/', 19, 2),
    ('https://web.archive.org/web/20190305143013/https://store.steampowered.com/hwsurvey/videocard/', 19, 1),
    ('https://web.archive.org/web/20181115143041/https://store.steampowered.com/hwsurvey/videocard/', 18, 3),
    ('https://web.archive.org/web/20180710005013/https://store.steampowered.com/hwsurvey/videocard/', 18, 2),
    ('https://web.archive.org/web/20180310085647/https://store.steampowered.com/hwsurvey/videocard/', 18, 1),
    ('https://web.archive.org/web/20171129124710/https://store.steampowered.com/hwsurvey/videocard/', 17, 3),
    # ('https://web.archive.org/web/20170724122821/https://store.steampowered.com/hwsurvey/videocard/', 17, 2),
    # ('https://web.archive.org/web/20170321151210/https://store.steampowered.com/hwsurvey/videocard/', 17, 1),
]


# for l, y, id in links:
#     get_html(l, y, id)


gl_month = 0


links17_21.reverse()

for l, y, id in links17_21:
    print(y, id)
    s = get_gpu_by_link17_21(y, id, gl_month)
    gl_month += 4
    for gpu in s:
        if gpu.name in res:
            res[gpu.name].update(gpu.metrics)
        else:
            res[gpu.name] = gpu.metrics

links22_24.reverse()

for l, y, id in links22_24:
    print(y, id)
    s = get_gpu_by_link22_24(y, id, gl_month)
    gl_month += 4
    for gpu in s:
        if gpu.name in res:
            res[gpu.name].update(gpu.metrics)
        else:
            res[gpu.name] = gpu.metrics


max_len_gpu = {}

for i in res.values():
    if len(i.values()) > len(max_len_gpu.values()):
        max_len_gpu = i

print(max_len_gpu)
# for key in res:
#     print(f"{key}\t\t\t{res[key]}")

with open('./data.json', 'w') as f:
    json.dump(res, f)


with open('./data.csv', 'w') as f:
    f.write('name,')
    n = [f'm_{i}' for i in range(80)]
    f.write(','.join(n) + '\n')
    for gpu in res.keys():
        strs = []
        strs.append(gpu)
        for i in range(80):
            if i in res[gpu]:
                strs.append(str(res[gpu][i]))
            else:
                strs.append('0')
        print(','.join(strs))
        f.write(','.join(strs) + '\n')
