#!/usr/bin/env python3
#################################################################################
# Copyright 2019 ROBOTIS CO., LTD.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#################################################################################
#
# Authors: Ryan Shim, Gilbert, ChanHyeong Lee

import collections
import os
import sys
import time
from pathlib import Path

import numpy
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
from std_srvs.srv import Empty
import tensorflow
from tensorflow.keras.layers import Dense
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import RMSprop
#from turtlebot3_dqn.rainbow_agent import Dueling_DQN
from turtlebot3_dqn.per_agent import Dueling_DQN, NoisyDense

from turtlebot3_msgs.srv import Dqn
import matplotlib.pyplot as plt
from shutil import copytree


class DQNTest(Node):

    def __init__(self):
        super().__init__('dqn_test')
#------------------------------------------------Parameters----------------------------------------------------------------------
        self.declare_parameter('load_from_folder','actual')
        self.declare_parameter('load_from_stage',1)
        self.declare_parameter('load_from_episode',50)
        self.declare_parameter('stagex',1)
        self.declare_parameter('max_episodes',100)
        self.declare_parameter('action_space',5)

        self.load_from_folder = self.get_parameter('load_from_folder').get_parameter_value().string_value
        self.load_from_stage = self.get_parameter('load_from_stage').get_parameter_value().integer_value
        self.load_from_episode = self.get_parameter('load_from_episode').get_parameter_value().integer_value
        self.stage = self.get_parameter('stagex').get_parameter_value().integer_value
        self.max_episodes = self.get_parameter('max_episodes').get_parameter_value().integer_value
        self.action_size = self.get_parameter('action_space').get_parameter_value().integer_value
        
        self.test_mode = True
        self.rainbowmode=False

        self.state_size = 28
        #self.action_size = 5

        self.success_counter=0
        self.collission_counter=0
        self.timeout_counter=0

        self.memory = collections.deque(maxlen=1000000)

#-----------------------------------------------Pfade--------------------------------------------------------------------------------
        self.training_dir_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.realpath(__file__))),
            'trainings_done'
        )

        if self.load_from_folder == 'actual':
            model_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.realpath(__file__))),
                'saved_model',
                f'stage{self.load_from_stage:05d}_episode{self.load_from_episode:05d}.h5'
                )
        else:
            training_dir = self.load_from_folder
            model_folder = os.path.join(self.training_dir_path,training_dir)
            model_path = os.path.join(model_folder,f'stage{self.load_from_stage:05d}_episode{self.load_from_episode:05d}.h5')

#---------------------------------------------Model-------------------------------------------------------------------------------------
        def dueling_factory(**_):
            return Dueling_DQN(
                n_actions=self.action_size,
                fc1=512,
                fc2=256,
                fc3=128
            )

        if self.load_from_folder.endswith('rainbow'):
            self.rainbowmode=True
            try:
                self.model.load_model(model_path, compile = False, custom_objects={'Dueling_DQN':Dueling_DQN, 'NoisyDense':NoisyDense})
            except Exception:
                self.model = Dueling_DQN(self.action_size,512,256,128)
                _ = self.model(tensorflow.zeros((1, self.state_size),dtype=tensorflow.float32))
                self.model.load_weights(model_path)


        else:
            self.model = self.build_model()
            loaded_model = load_model(model_path, compile=False, custom_objects={'mse': MeanSquaredError()})
            self.model.set_weights(loaded_model.get_weights())
            
#-------------------------------------------Messages----------------------------------------------------------------------------------------
        self.rl_agent_interface_client = self.create_client(Dqn, 'rl_agent_interface')

        self.action_pub = self.create_publisher(Float32MultiArray, '/get_action', 10)
        self.result_pub = self.create_publisher(Float32MultiArray, 'result', 10)
        self.loss_pub = self.create_publisher(Float32MultiArray,'loss',10)

#---------------------------------------------Run-----------------------------------------------------------------------------------------
        self.run_test()

#-----------------------------------------------Functions----------------------------------------------------------------------------------
    def build_model(self):
        model = Sequential()
        model.add(Dense(
            512, input_shape=(self.state_size,),
            activation='relu',
            kernel_initializer='lecun_uniform'
        ))
        model.add(Dense(256, activation='relu', kernel_initializer='lecun_uniform'))
        model.add(Dense(128, activation='relu', kernel_initializer='lecun_uniform'))
        model.add(Dense(self.action_size, activation='linear', kernel_initializer='lecun_uniform'))
        model.compile(loss=MeanSquaredError(), optimizer=RMSprop(learning_rate=0.00025))
        return model

    def get_action(self, state):
        if self.rainbowmode:
            s=numpy.asarray(state,dtype=numpy.float32).reshape(1,-1)
            q=self.model(s,training=False).numpy()[0]
            return int(numpy.argmax(q))
        else:
            state = numpy.asarray(state)
            q_values = self.model.predict(state.reshape(1, -1), verbose=0)
            return int(numpy.argmax(q_values[0]))
    

    def moving_average(self,data, window_size=10):
        if len(data) < window_size:
            return numpy.array(data)  # noch nicht genug Daten
        return numpy.convolve(data, numpy.ones(window_size)/window_size, mode='same')

    def run_test(self):

        plt.ion()
        score_list= list()
        episode_list=list()

        fig = plt.figure(figsize=(10, 8))
        gs = fig.add_gridspec(2, 2)
        plot1 = fig.add_subplot(gs[0, 0]);plot1.grid(True)
        plot2 = fig.add_subplot(gs[0, 1]);plot2.grid(True)
        plot3 = fig.add_subplot(gs[1, :]);plot3.grid(True)

        fig.set_constrained_layout(True)
        #plt.grid()
        #plt.tight_layout()

        for episode in range(self.max_episodes):
            done = False
            success = False
            collission = False
            timeout = False
            init = True
            score = 0
            local_step = 0
            next_state = []

            time.sleep(1.0)

            while not done:
                local_step += 1
                action = 2 if local_step == 1 else self.get_action(next_state)

                req = Dqn.Request()
                req.action = action
                req.init = init

                while not self.rl_agent_interface_client.wait_for_service(timeout_sec=1.0):
                    self.get_logger().warn(
                        'rl_agent interface service not available, waiting again...')

                future = self.rl_agent_interface_client.call_async(req)
                rclpy.spin_until_future_complete(self, future)

                if future.done() and future.result() is not None:
                    next_state = future.result().state
                    reward = future.result().reward
                    done = future.result().done
                    success = future.result().success
                    collission = future.result().collission
                    timeout = future.result().timeout
                    score += reward
                    init = False
                    if done:
                        score_list.append(score)
                        episode_list.append(episode)
                    if success:
                        self.success_counter+=1
                    if collission:
                        self.collission_counter+=1
                    if timeout:
                        self.timeout_counter+=1


                else:
                    self.get_logger().error(f'Service call failure: {future.exception()}')

                time.sleep(0.2)#vorher 0.01, irgendwann wurde aber rl agent callback mit 60Hz aufgerufen-->schnelle Timeouts

            #Interactive Plot
            plot1.clear()
            plot1.plot(episode_list,score_list,color='r')
            plot1.set_title('Score')
            plot1.set_xlabel('Episode')
            plot1.set_ylabel('Score')
            plot1.grid(True)
            
            plot2.clear()
            bars = plot2.bar(['Successes','Collissions','Timeouts'],[self.success_counter,self.collission_counter,self.timeout_counter],color=['skyblue', 'lightgreen', 'salmon'])
            plot2.bar_label(bars, fmt='%d', label_type='edge', padding=3 )
            plot2.set_title('Evalutation')
            plot2.set_xlabel('Kategorien')
            plot2.set_ylabel('Ergebnis')
            plot2.grid(True)

            plot3.clear()
            plot3.plot(episode_list,self.moving_average(score_list),color='r')
            plot3.set_title('Moving Average 10')
            plot3.set_xlabel('Episode')
            plot3.set_ylabel('Score')
            plot3.grid(True)

            fig.canvas.draw()
            plt.pause(0.1)

        plt.ioff()
        plt.show() 
    
        #SavePlots
        evaluation_path = Path('/home/verwalter/turtlebot3_ws/src/turtlebot3_machine_learning/turtlebot3_dqn/trainings_done')
        real_evaluation_path = Path('/home/verwalter/turtlebot3_ws/src/turtlebot3_machine_learning/turtlebot3_dqn/evaluation')
        os.makedirs(real_evaluation_path,exist_ok=True)
        save_path = os.path.join(evaluation_path,self.load_from_folder)
        copy_path = os.path.join(real_evaluation_path,self.load_from_folder)
        plt.figure()
        bar=plt.bar(['Successes','Collissions','Timeouts'],[self.success_counter,self.collission_counter,self.timeout_counter],color=['skyblue', 'lightgreen', 'salmon'])
        plt.bar_label(bar, fmt='%d', label_type='edge', padding=3)
        plt.title('Evaluation')
        plt.xlabel('Kategorien')
        plt.ylabel('Ergebnis')
        plt.grid()
        #plt.show()
        plt.savefig(os.path.join(save_path,'Evaluation_Score.png'))
        plt.close()
        plt.figure()
        plt.plot(episode_list,score_list)
        plt.title('Rewards')
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.grid()
        plt.savefig(os.path.join(save_path,'Evaluation_Reward.png'))
        plt.close()
        plt.figure()
        plt.plot(episode_list,self.moving_average(score_list))
        plt.title('Moving Average 10')
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.grid()
        plt.savefig(os.path.join(save_path,'Moving_Avg.png'))

        copytree(save_path,copy_path, dirs_exist_ok=True)




def main():
    rclpy.init()
    node = DQNTest()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
