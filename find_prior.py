import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

arrival = pd.read_csv('arrival time matrix.csv',header=None)
arrival = arrival.values
M = arrival.copy()
M_diff = M[:, :-1] - M[:, 1:]
M_zero = M <= 0
M_nzero = ~M_zero
M_valid = M_nzero[:, :-1] & M_nzero[:, 1:]

station = 2
travel_time = M_diff[:,station]
travel_time = travel_time[M_valid[:,station]]
plt.hist(travel_time, bins=100)

