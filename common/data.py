from dataclasses import dataclass, asdict
from typing import List
from math import log
from os.path import join, abspath
from os import getcwd
import pandas as pd

import numpy as np
DIM = 0


@dataclass
class Metrics:
    Opencl: float
    Vulkan: float
    G3: float
    G2: float
    # reciprocal value 1/val
    RP: float
    RTDP: float


@dataclass   
class Stats:
    stats_list : List


class GPU:
    name: str
    stat: list
    metrics: Metrics

    def __init__(self, _name, _stat, _metr:Metrics) -> None:
        self.name = _name
        self.stat = _stat
        self.metrics = _metr
        self._2d_metrics = self.get_2d_metrics()
        self._3d_metrics = self.get_3d_metrics()

    def get_metrics(self) -> dict:
        return asdict(self.metrics)
    
    def get_3d_metrics(self) -> dict:
        res = {}
        i = 0
        for keya, vala in asdict(self.metrics).items():
            i+=1
            j=0
            for keyb, valb in asdict(self.metrics).items():
                j+=1
                k = 0
                for keyc, valc in asdict(self.metrics).items():
                    k+=1
                    if j >= i and k >= j and k >= i:
                        res[f"{keya}*{keyb}*{keyc}"] = vala*valb*valc
        return res
    
    def get_4d_metrics(self) -> dict:
        res = {}
        i = 0
        for keya, vala in asdict(self.metrics).items():
            i+=1
            j=0
            for keyb, valb in asdict(self.metrics).items():
                j+=1
                k = 0
                for keyc, valc in asdict(self.metrics).items():
                    k+=1
                    m = 0
                    for keyd, vald in asdict(self.metrics).items():
                        m+=1
                        if j >= i and k >= j and m >= k:
                            res[f"{keya}*{keyb}*{keyc}*{keyd}"] = vala*valb*valc*vald
        return res
    
    def get_5d_metrics(self) -> dict:
        res = {}
        i = 0
        for keya, vala in asdict(self.metrics).items():
            i+=1
            j=0
            for keyb, valb in asdict(self.metrics).items():
                j+=1
                k = 0
                for keyc, valc in asdict(self.metrics).items():
                    k+=1
                    m = 0
                    for keyd, vald in asdict(self.metrics).items():
                        m+=1
                        n=0
                        for keye, vale in asdict(self.metrics).items():
                            n+=1
                            if j >= i and k >= j and m >= k:
                                res[f"{keya}*{keyb}*{keyc}*{keyd}*{keye}"] = vala*valb*valc*vald*vale
        return res
    
    
    def get_2d_metrics(self) -> dict:
        res = {}
        i = 0
        for keya, vala in asdict(self.metrics).items():
            i+=1
            j = 0
            for keyb, valb in asdict(self.metrics).items():
                j+=1
                if j <= i:
                    res[f"{keya}*{keyb}"] = vala*valb
        return res
    
    def get_stats(self) -> dict:
        return {i : self.stat[i] for i, _ in enumerate(self.stat)}

    def get_name(self):
        return self.name
    
    def get_united_metrics(self) -> dict:
        s = {}
        s.update(asdict(self.metrics))
        if DIM > 1:
            s.update(self.get_2d_metrics())
        if DIM > 2:
            s.update(self.get_3d_metrics())
        if DIM > 4:
            s.update(self.get_5d_metrics())
            
        return s
    
    def get_Jval(self) -> float:
        stats_arr = self.stat
        res = 0
        start_flag = False
        n = 0
        for i in range(len(stats_arr)):
            if stats_arr[i] != 0:
                start_flag = True
            if not start_flag:
                continue
            if i == 0: continue
            if stats_arr[i] and stats_arr[i-1]:
                res += log(1 + ((stats_arr[i]-stats_arr[i-1])/stats_arr[i-1]))
            n+=1
        return res/n + 2




def GetData(zero_num, dim):
    global DIM
    DIM = dim
    hh = pd.read_csv(join(abspath(join(__file__, '..')), 'ndata.csv')).iloc[:, :7]
    hh.to_latex("data.tex",index=False, longtable=True)
    
    with open(join(abspath(join(__file__, '..')), 'ndata.csv'), 'r') as data:
        lines = data.readlines()
    
    res = []
    for line in lines[1:]:
        line = line[:-1]
        spl = line.split(',')
        if spl.count("0") < zero_num:
            res.append(line)
    print(f"len 1 = {len(lines)} len = {len(res)}")
    lines = res
    res_gpu = []
    sum_array = [0 for _ in range(80)]
    for line in lines:
        spl = line.split(',')
        metr = Metrics(float(spl[1]), float(spl[2]), float(spl[3]), float(spl[4]), 1/float(spl[5]), 1/float(spl[6]))
        stat = [float(i) for i in spl[7:]]
        res_gpu.append(GPU(spl[0], stat, metr))
    return res_gpu

def GetTest(zero_num, dim):
    global DIM
    DIM = dim
    with open(join(abspath(join(__file__, '..')), 'ndata.csv'), 'r') as data:
        lines = data.readlines()
    res = []
    for line in lines[1:]:
        line = line[:-1]
        spl = line.split(',')
        res.append(line)
    print(f"TEST len 1 = {len(lines)} len = {len(res)}")
    
    lines = res
    res_gpu = []
    for line in lines:
        spl = line.split(',')
        metr = Metrics(float(spl[1]), float(spl[2]), float(spl[3]), float(spl[4]), 1/float(spl[5]), 1/float(spl[6]))
        stat = [float(i) for i in spl[7:]]
        res_gpu.append(GPU(spl[0], stat, metr))
    return res_gpu