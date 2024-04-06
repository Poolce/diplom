from dataclasses import dataclass, asdict
from typing import List
from math import log

@dataclass
class Metrics:
    G3Dmark: float
    G2Dmark: float
    # reciprocal value 1/val
    recip_price: float
    recip_TDP: float


@dataclass
class Metrics2D:
    _2D_G3Dmark2: float
    _2D_G2Dmark2: float
    _2D_recip_price2: float
    _2D_recip_TDP2: float
    _2D_G3Dmark_G2Dmark:float
    _2D_G3Dmark_recip_price:float
    _2D_G3Dmark_recip_TDP:float
    _2D_G2Dmark2_recip_price:float
    _2D_G2Dmark2_recip_TDP:float
    _2D_recip_price_recip_TDP:float

@dataclass
class Metrics3D:
    _3D_G3G2price: float
    _3D_G3G2TDP: float
    _3D_G3priceTDP: float
    _3D_G2priceTDP: float

@dataclass
class Metrics4D:
    _4D_ALL_CHARACTERS: float


@dataclass   
class Stats:
    stats_list : List


class GPU:
    name: str
    stat: List
    metrics: Metrics
    metrics2D: Metrics2D
    metrics3D: Metrics3D

    def __init__(self, _name, _stat, _metr:Metrics) -> None:
        self.name = _name
        self.stat = _stat
        self.metrics = _metr
        self.metrics2D = Metrics2D(_metr.G3Dmark**2, _metr.G2Dmark**2, _metr.recip_price**2, _metr.recip_TDP**2,
                                   _metr.G3Dmark*_metr.G2Dmark, _metr.G3Dmark*_metr.recip_price, _metr.G3Dmark*_metr.recip_TDP,
                                   _metr.G2Dmark*_metr.recip_price, _metr.G2Dmark*_metr.recip_TDP,
                                   _metr.recip_price*_metr.recip_TDP)
        
        self.metrics3D = Metrics3D(_metr.G3Dmark*_metr.G2Dmark* _metr.recip_price,
                                   _metr.G3Dmark*_metr.G2Dmark*_metr.recip_TDP,
                                   _metr.G3Dmark* _metr.recip_price*_metr.recip_TDP,
                                   _metr.G2Dmark* _metr.recip_price*_metr.recip_TDP)
        
        self.metrics4D = Metrics4D(_metr.G3Dmark*_metr.G2Dmark* _metr.recip_price*_metr.recip_TDP)

    def get_metrics(self) -> dict:
        return asdict(self.metrics)
    
    def get_2dmetrics(self) -> dict:
        return asdict(self.metrics2D)
    
    def get_stats(self) -> dict:
        return asdict(self.stat)

    def get_name(self):
        return self.name
    
    def get_united_metrics(self) -> dict:
        s = asdict(self.stat)
        s.update(asdict(self.metrics))
        s.update(asdict(self.metrics2D))
        s.update(asdict(self.metrics3D))
        s.update(asdict(self.metrics4D))
        return s
    
    def get_Jval(self) -> float:
        stats_arr = list(asdict(self.stat).values())
        res = 0
        for i in range(len(stats_arr)):
            if i == 0: continue
            res += log(1 + ((stats_arr[i]-stats_arr[i-1])/stats_arr[i-1]))
        return res/len(stats_arr)




def get_data(csv_file_path) -> List[GPU]:
    with open(csv_file_path, 'r') as file:
        lines = file.readlines()
    res = []
    for line in lines[1:]:
        d = line.split(',')
        res.append(
            GPU(d[0],
            Stats(float(d[1]), float(d[2]), float(d[3]),
                  float(d[4]), float(d[5])),
            Metrics(float(d[6]), float(d[7]),
                    1/float(d[8]), 1/float(d[9])))
        )
    return res