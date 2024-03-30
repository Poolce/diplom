from dataclasses import dataclass, asdict
from typing import List


@dataclass
class Metrics:
    G3Dmark: float
    G2Dmark: float
    # reciprocal value 1/val
    recip_price: float
    recip_TDP: float


@dataclass   
class Stats:
    m1: float
    m2: float
    m3: float
    m4: float
    m5: float


class GPU:
    name: str
    stat: Stats
    metrics: Metrics

    def __init__(self, _name, _stat, _metr) -> None:
        self.name = _name
        self.stat = _stat
        self.metrics = _metr

    def get_metrics(self) -> dict:
        return asdict(self.metrics)
    
    def get_stats(self) -> dict:
        return asdict(self.stat)

    def get_name(self):
        return self.name


def get_data(csv_file_path) -> List[GPU]:
    with open(csv_file_path, 'r') as file:
        lines = file.readlines()
    res = []
    for line in lines[1:]:
        d = line.split(',')
        res.append(
            GPU(d[0],
            Stats(float(d[1]), float(d[2]), float(d[3]), float(d[4]), float(d[5])),
            Metrics(float(d[6]), float(d[7]), 1/float(d[8]), 1/float(d[9])))
        )
    return res