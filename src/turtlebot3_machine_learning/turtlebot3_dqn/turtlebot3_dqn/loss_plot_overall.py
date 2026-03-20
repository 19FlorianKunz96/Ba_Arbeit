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

#every loss plotted in world 4

path_default = Path('/home/verwalter/turtlebot3_ws/src/turtlebot3_machine_learning/turtlebot3_dqn/evaluation/Default/2ce27f24-4d36-45bf-be18-84f32621726b_2025-10-22_stage4/loss_data.csv')
path_double_huber = Path('/home/verwalter/turtlebot3_ws/src/turtlebot3_machine_learning/turtlebot3_dqn/evaluation/Rainbow/e8262055-9ed7-4885-a0db-57512ebb1481_2025-10-23_stage4_rainbow/loss_data.csv')
path_fullstack_robotis = Path('/home/verwalter/turtlebot3_ws/src/turtlebot3_machine_learning/turtlebot3_dqn/evaluation/NAV2_TEST/AS5_RobotisReward_AlleKomponenten/f3fa2021-8f56-4e07-92d1-623c857a326b_2026-01-14_stage4_rainbow/loss_data.csv')
path_fullstack_paper = Path('/home/verwalter/turtlebot3_ws/src/turtlebot3_machine_learning/turtlebot3_dqn/evaluation/NAV2_TEST/AS5_RewardPaper_AlleKomponenten/c8afb22e-058d-444b-8e3e-3a218ac2eed4_2025-11-03_stage4_rainbow/loss_data.csv')
data_default= pd.read_csv(path_default)
data_double_huber= pd.read_csv(path_double_huber)
data_fullstack_robotis= pd.read_csv(path_fullstack_robotis)
data_fullstack_paper= pd.read_csv(path_fullstack_paper)

fig = plt.figure(figsize=(10,8))
episode=data_default['Episode'].to_numpy()
loss = data_default['Loss'].to_numpy()
plt.plot(episode,moving_average(loss))

episode=data_double_huber['Episode'].to_numpy()
loss = data_double_huber['Loss'].to_numpy()
plt.plot(episode,moving_average(loss))

episode=data_fullstack_paper['Episode'].to_numpy()
loss = data_fullstack_paper['Loss'].to_numpy()
plt.plot(episode,moving_average(loss))

episode=data_fullstack_robotis['Episode'].to_numpy()
loss = data_fullstack_robotis['Loss'].to_numpy()
plt.plot(episode,moving_average(loss))

plt.show()


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

