import numpy as np
def areaThreshold_by_avg(axis, exp):
    avga = np.average([(s[1] - s[0]) * (s[3] - s[2]) for i,s in axis.items()])
    low = avga/2**exp
    high = avga*2**exp
    return low, high

def areaThreshold_by_havg(axis, exp):
    areas = np.sort([(s[1] - s[0]) * (s[3] - s[2]) for i,s in axis.items()])
    alen = len(areas)
    avga = np.average([areas[i] for i in range(int(alen/2**exp), int(alen*(1-1.0/2**exp)))])
    low = avga / 2 ** exp
    high = avga * 2 ** exp
    return low, high

def areaThreshold_by_top(axis, exp):
    area = np.sort([(s[1] - s[0]) * (s[3] - s[2]) for i, s in axis.items()])[-1]
    return area/2**exp , area*2**exp

if __name__ == '__main__':
    s1= {208.0: [6, 52, 6, 23], 209.0: [17, 61, 19, 51], 210.0: [58, 59, 31, 32]}
    print(areaThreshold_by_top(s1,2))
    print(np.sort([(s[1] - s[0]) * (s[3] - s[2]) for i, s in s1.items()]))

    print(areaThreshold_by_havg(s1, 2))
    print(np.sort([(s[1] - s[0]) * (s[3] - s[2]) for i, s in s1.items()]))