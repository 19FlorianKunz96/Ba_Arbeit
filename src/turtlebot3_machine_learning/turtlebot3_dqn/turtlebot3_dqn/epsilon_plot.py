import csv
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from time import sleep
import numpy
import math

step_counter = numpy.arange(200001)
epsilon_decay=18000
k=1
epsilon_min = 0.05
epsilon_plot = list()
for i in range(len(step_counter)):
    epsilon_plot.append( epsilon_min + (1.0 - epsilon_min) * math.exp(-1.0 * k * step_counter[i] / epsilon_decay))

plt.plot(step_counter,epsilon_plot,color='r', label='k=1')
k=0.5
epsilon_plot.clear()
for i in range(len(step_counter)):
    epsilon_plot.append( epsilon_min + (1.0 - epsilon_min) * math.exp(-1.0 * k * step_counter[i] / epsilon_decay))
plt.plot(step_counter,epsilon_plot,color='b', label='k=0.05')

epsilon_plot.clear()
epsilon_decay =30000
for i in range(len(step_counter)):
    epsilon_plot.append( epsilon_min + (1.0 - epsilon_min) * math.exp(-1.0 * k * step_counter[i] / epsilon_decay))
plt.plot(step_counter,epsilon_plot,color='k', label='k=0.05/double decay')
plt.grid()
plt.legend()
plt.show()