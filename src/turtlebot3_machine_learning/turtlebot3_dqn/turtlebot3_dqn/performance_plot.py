import csv
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from time import sleep
import numpy

def moving_average(data, window_size=100):
    if len(data) < window_size:
        return numpy.array(data)  # noch nicht genug Daten
    return numpy.convolve(data, numpy.ones(window_size)/window_size, mode='same')


path = Path('src/turtlebot3_machine_learning/turtlebot3_dqn/trainings_done/df5db7b6-45d3-4af8-a7fa-a542015383bd_2025-10-28_stage3_rainbow/graph_data.csv')
path2 = Path('src/turtlebot3_machine_learning/turtlebot3_dqn/trainings_done/df5db7b6-45d3-4af8-a7fa-a542015383bd_2025-10-28_stage3_rainbow/loss_data.csv')
data= pd.read_csv(path)
data2=pd.read_csv(path2)

fig = plt.figure(figsize=(10,8))
fig.suptitle('Leistungsprotokoll', fontsize=16)
gs = fig.add_gridspec(2,2)

plot1=fig.add_subplot(gs[0,0]);plot1.grid(True);plot1.set_title('Reward');plot1.set_ylabel('Reward');plot1.set_xlabel('Episode')
plot2=fig.add_subplot(gs[1,:]);plot2.grid(True);plot2.set_title('Loss / Epsilon');plot2.set_ylabel('Loss');plot2.set_xlabel('Steps')
plot3=fig.add_subplot(gs[0,1]);plot3.grid(True);plot3.set_title('Average Max Q-Value');plot3.set_ylabel('Q-Value');plot3.set_xlabel('Episode')

fig.set_constrained_layout(True)

episode = data['Episode'].to_numpy()
q_value= data['Reward'].to_numpy()
reward = data['Q_value'].to_numpy()

step=data2['Episode'].to_numpy()
loss=data2['Loss'].to_numpy()
epsilon=data2['Epsilon'].to_numpy()

plot1.plot(episode,reward,color='r', label='Reward')
plot1.plot(episode, moving_average(reward), color='k', linestyle='--',linewidth=1, label = 'Moving Average 100')
plot1.legend()

plot3.plot(episode,q_value,color='g', label='Max Q-Value')
plot3.plot(episode, moving_average(q_value), color='k', linestyle='--',linewidth=1, label = 'Moving Average 100')
plot3.legend()


ax2=plot2.twinx()
plot2.plot(step,loss,color='b',label='Loss')
plot2.plot(step, moving_average(loss),color='k',linestyle='--', linewidth=1,label='Moving Average 100')
ax2.plot(step,epsilon,color = 'g',label='Epsilon')
plot2.legend(loc='upper center')
ax2.legend()
plt.show()

