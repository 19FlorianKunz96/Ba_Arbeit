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
import datetime
import json
import math
import os
import random
import sys
import time
import uuid
import json
import glob

import numpy
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
from std_srvs.srv import Empty
import tensorflow
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

from turtlebot3_msgs.srv import Dqn


tensorflow.config.set_visible_devices([], 'GPU')

LOGGING = True
current_time = datetime.datetime.now().strftime('[%mm%dd-%H:%M]')


class DQNMetric(tensorflow.keras.metrics.Metric):

    def __init__(self, name='dqn_metric'):
        super(DQNMetric, self).__init__(name=name)
        self.loss = self.add_weight(name='loss', initializer='zeros')
        self.episode_step = self.add_weight(name='step', initializer='zeros')

    def update_state(self, y_true, y_pred=0, sample_weight=None):
        self.loss.assign_add(y_true)
        self.episode_step.assign_add(1)

    def result(self):
        return self.loss / self.episode_step

    def reset_states(self):
        self.loss.assign(0)
        self.episode_step.assign(0)


class DQNAgent(Node):

    def __init__(self):       #added param stage_boost for taking weights from last stages
        super().__init__('dqn_agent')
        #evtl im parentlaunch die default werte weglassen, dass nicht ein string in integer umgewandelt werden muss ?
        self.declare_parameter('stagex',1)
        self.declare_parameter('max_episodes',100)
        self.declare_parameter('stage_boost',False)
        self.declare_parameter('load_from_folder','actual')
        self.declare_parameter('load_from_stage',1)
        self.declare_parameter('load_from_episode',50)
        self.stage = self.get_parameter('stagex').get_parameter_value().integer_value
        self.train_mode = True
        self.state_size = 26
        self.action_size = 5
        self.max_training_episodes = self.get_parameter('max_episodes').get_parameter_value().integer_value

        self.stage_boost = self.get_parameter('stage_boost').get_parameter_value().bool_value
        self.load_from_folder = self.get_parameter('load_from_folder').get_parameter_value().string_value
        self.load_from_stage = self.get_parameter('load_from_stage').get_parameter_value().integer_value
        self.load_from_episode = self.get_parameter('load_from_episode').get_parameter_value().integer_value
        self.done = False
        self.succeed = False
        self.fail = False
        self.info = ['Adjusted Epsilon, Reward set on -100 if collision']

        self.discount_factor = 0.99
        self.learning_rate = 0.0007
        self.epsilon = 1.0
        self.step_counter = 0
        self.epsilon_decay = 6000 * self.stage
        self.epsilon_min = 0.05
        self.batch_size = 128

        self.replay_memory = collections.deque(maxlen=500000)
        self.min_replay_memory_size = 5000

        self.uuid = uuid.uuid4()
        self.date = datetime.date.today()
        self.training_dir = f'{self.uuid}_{self.date}_stage{self.stage}'
        #give the other nodes the dir
        with open(os.path.join(os.path.dirname(os.path.realpath(__file__)),'temp.json'),'w') as temp:
            json.dump({'folder': self.training_dir},temp)

        self.model = self.create_qnetwork()
        self.target_model = self.create_qnetwork()
        self.update_target_model()
        self.update_target_after = 5000
        self.target_update_after_counter = 0

        #set this to True and adjust the load_episode to continue training on the same stage
        #be careful: higher stages already will have used the weights of lower parameters
        self.load_model = False
        self.load_episode = 0

        self.model_dir_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.realpath(__file__))),
            'saved_model'
        )
        self.training_dir_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.realpath(__file__))),
            'trainings_done'
        )

        self.training_path_comp = os.path.join(self.training_dir_path,self.training_dir)

        self.model_path = os.path.join(

            self.model_dir_path,
            'stage' + str(self.stage) + '_episode' + str(self.load_episode) + '.h5'
        )

        if self.load_model:
            self.model.set_weights(load_model(self.model_path).get_weights())
            with open(os.path.join(
                self.model_dir_path,
                'stage' + str(self.stage) + '_episode' + str(self.load_episode) + '.json'
            )) as outfile:
                param = json.load(outfile)
                self.epsilon = param.get('epsilon')
                self.step_counter = param.get('step_counter')

        #Training the stage with the youngest weights
        #stage_boost has to be set in the terminal
        if self.stage_boost == True:
            if self.load_from_folder == 'actual':
                files = os.listdir(self.model_dir_path)
                stage_files = [f for f in files if f.startswith('stage') and f.endswith('.h5')]#TODO doesnt sort correctly because of the numbers
                sorted_stage_files = sorted(stage_files,key=lambda x:int(x[-8:-5]))
                sorted_stage_files.sort(key= lambda x: int(x[5:10])) 

                if stage_files:
                    self.last_model_path = os.path.join(self.model_dir_path, stage_files[-1])
                    self.model.set_weights(load_model(self.last_model_path).get_weights())
                    self.epsilon = 0.65
                    self.update_target_model()
                    self.info.append('last actual pre-trained model loaded, target model updated')
                    self.get_logger().info('last actual pre-trained model loaded')
                else:
                    self.get_logger().warn('No training data in previous stages')
                    self.info.append('No training data in previous stages')
            else:
                self.file_root = os.path.join(self.training_dir_path,
                                               os.path.join(self.load_from_folder,
                                                f'stage{self.load_from_stage:05d}_episode{self.load_from_episode:05d}.h5'))
                if os.path.exists(self.file_root) == True:
                    self.model.set_weights(load_model(self.file_root).get_weights())
                    self.epsilon = 0.65
                    self.update_target_model()
                    self.info.append('Weights successfully loaded, target model updated')
                    self.get_logger().info('Weights successfully loaded')
                else:
                    self.info.append('File doesnt exist. No weights loaded')
                    self.get_logger().warn('File doesnt exist. No weights loaded')


        self.epsilon_start = self.epsilon
        self.hyperparams = {
            'Stage' : self.stage,
            'Folder Name' : self.training_dir,
            'Learned from older Stages' : self.stage_boost,
            'Learned from Model' : {'Folder' : self.load_from_folder,'Stage':self.load_from_stage, 'Episode': self.load_from_episode},
            'Success' : self.info,
            'State Size' : self.state_size,
            'Action Szie' : self.action_size,
            'Maximum Episodes' : self.max_training_episodes,
            'Discount Factor' : self.discount_factor,
            'Learning Rate' : self.learning_rate,
            'Starting with Epsilon' : self.epsilon_start,
            'Starting Step Counter' : self.step_counter,
            'Epsilon Decay' : self.epsilon_decay,
            'Minimum Epsilon' : self.epsilon_min,
            'Batch Size' : self.batch_size,
            'Replay Memory Max' : self.replay_memory.maxlen,
            'Replay Memory Min' : self.min_replay_memory_size,

        }
        
        if LOGGING:
            tensorboard_file_name = current_time + ' dqn_stage' + str(self.stage) + '_reward'
            home_dir = os.path.expanduser('~')
            dqn_reward_log_dir = os.path.join(
                home_dir, 'turtlebot3_dqn_logs', 'gradient_tape', tensorboard_file_name
            )
            self.dqn_reward_writer = tensorflow.summary.create_file_writer(dqn_reward_log_dir)
            self.dqn_reward_metric = DQNMetric()

        self.rl_agent_interface_client = self.create_client(Dqn, 'rl_agent_interface')
        self.make_environment_client = self.create_client(Empty, 'make_environment')
        self.reset_environment_client = self.create_client(Dqn, 'reset_environment')

        self.action_pub = self.create_publisher(Float32MultiArray, '/get_action', 10)
        self.result_pub = self.create_publisher(Float32MultiArray, 'result', 10)

        self.process()

    def process(self):
        self.env_make()
        time.sleep(1.0)

        episode_num = self.load_episode

        for episode in range(self.load_episode + 1, self.max_training_episodes + 1):
            # if episode ==1:
            #     self.get_logger().info(f'Training with Stage Boost = {self.stage_boost}')
            state = self.reset_environment()
            episode_num += 1
            local_step = 0
            score = 0
            sum_max_q = 0.0

            time.sleep(1.0)

            while True:
                local_step += 1

                #berchnet die q-values für alle actions im aktuellen Schritt. 
                q_values = self.model.predict(state)

                #schaut welcher der größte ist.Nur für den Graphen ??
                sum_max_q += float(numpy.max(q_values))

                #Sucht die Action mit größtem Reward durch Prediction und Argmax
                action = int(self.get_action(state))

                #.step() gibt die action weiter an environment, welches Werte berechnet, wieder zurückgibt und folgende Werte als return liefert
                next_state, reward, done = self.step(action)

                #score rechnet die rewards aus allen schrittenzusammen
                score += reward

                msg = Float32MultiArray()
                msg.data = [float(action), float(score), float(reward)]
                self.action_pub.publish(msg)

                if self.train_mode:
                    #added den Schritt als Sample in die EpisodenCollection
                    self.append_sample((state, action, reward, next_state, done))

                    #
                    self.train_model(done)

                state = next_state

                if done:
                    avg_max_q = sum_max_q / local_step if local_step > 0 else 0.0

                    msg = Float32MultiArray()
                    msg.data = [float(score), float(avg_max_q)]
                    self.result_pub.publish(msg)

                    if LOGGING:
                        self.dqn_reward_metric.update_state(score)
                        with self.dqn_reward_writer.as_default():
                            tensorflow.summary.scalar(
                                'dqn_reward', self.dqn_reward_metric.result(), step=episode_num
                            )
                        self.dqn_reward_metric.reset_states()

                    print(
                        'Episode:', episode,
                        'score:', score,
                        'memory length:', len(self.replay_memory),
                        'epsilon:', self.epsilon)

                    param_keys = ['epsilon', 'step']
                    param_values = [self.epsilon, self.step_counter]
                    param_dictionary = dict(zip(param_keys, param_values))
                    break

                time.sleep(0.01)

            if self.train_mode:
                #default value : every 100 episodes
                if os.path.exists(self.training_path_comp) == False:
                        os.mkdir(self.training_path_comp)
                        with open(os.path.join(self.training_path_comp,'config.json'),'w') as f:
                            json.dump(self.hyperparams,f)


                if episode % 50 == 0:
                    self.model_path = os.path.join(
                        self.model_dir_path, f'stage{self.stage:05d}_episode{episode:05d}.h5')
                        #'stage' + str(self.stage:05d) + '_episode' + str(episode) + '.h5')
                    self.model.save(self.model_path)
                    self.model.save(os.path.join(self.training_path_comp, f'stage{self.stage:05d}_episode{episode:05d}.h5'))
                    with open(os.path.join(self.training_path_comp, f'stage{self.stage:05d}_episode{episode:05d}.json'),'w') as out2:
                        json.dump(param_dictionary,out2)
                    with open(
                        os.path.join(
                            self.model_dir_path,
                            f'stage{self.stage:05d}_episode{episode:05d}.json'
                        ),
                        'w'
                    ) as outfile:
                        json.dump(param_dictionary, outfile)
                

    def env_make(self):
        while not self.make_environment_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().warn(
                'Environment make client failed to connect to the server, try again ...'
            )

        self.make_environment_client.call_async(Empty.Request())

    def reset_environment(self):
        while not self.reset_environment_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().warn(
                'Reset environment client failed to connect to the server, try again ...'
            )

        future = self.reset_environment_client.call_async(Dqn.Request())

        rclpy.spin_until_future_complete(self, future)
        if future.result() is not None:
            state = future.result().state
            state = numpy.reshape(numpy.asarray(state), [1, self.state_size])
        else:
            self.get_logger().error(
                'Exception while calling service: {0}'.format(future.exception()))

        return state

    def get_action(self, state):
        if self.train_mode:
            self.step_counter += 1
            #adjusted for different epsilon values(1.0-->self.epsilon)
            # self.epsilon = self.epsilon_min + (self.epsilon - self.epsilon_min) * math.exp(
            #     -self.epsilon * self.step_counter / self.epsilon_decay)         
            # 
            # --complete new--
            self.epsilon = self.epsilon_min + (self.epsilon_start - self.epsilon_min) * math.exp(-self.step_counter / self.epsilon_decay)              
            lucky = random.random()
            if lucky > (1 - self.epsilon):
                result = random.randint(0, self.action_size - 1)
            else:
                result = numpy.argmax(self.model.predict(state))
        else:
            result = numpy.argmax(self.model.predict(state))

        return result

    def step(self, action):
        req = Dqn.Request()
        req.action = action

        while not self.rl_agent_interface_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('rl_agent interface service not available, waiting again...')

        future = self.rl_agent_interface_client.call_async(req)

        rclpy.spin_until_future_complete(self, future)

        if future.result() is not None:
            next_state = future.result().state
            next_state = numpy.reshape(numpy.asarray(next_state), [1, self.state_size])
            reward = future.result().reward
            done = future.result().done
        else:
            self.get_logger().error(
                'Exception while calling service: {0}'.format(future.exception()))

        return next_state, reward, done

    def create_qnetwork(self):
        model = Sequential()
        model.add(Input(shape=(self.state_size,)))
        model.add(Dense(512, activation='relu'))
        model.add(Dense(256, activation='relu'))
        model.add(Dense(128, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss=MeanSquaredError(), optimizer=Adam(learning_rate=self.learning_rate))
        model.summary()

        return model

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())
        self.target_update_after_counter = 0
        print('*Target model updated*')

    def append_sample(self, transition):
        self.replay_memory.append(transition)

    def train_model(self, terminal):

        #wenn die collection größer ist als min batch size, dann suche dir aus der collection einen random batch mit der grösse batchsize
        if len(self.replay_memory) < self.min_replay_memory_size:
            return
        data_in_mini_batch = random.sample(self.replay_memory, self.batch_size)


        #nur vorwärtspfade, da die normale cost funktion nicht genommen werden kann
        #predicted q-value für den aktuellen state
        current_states = numpy.array([transition[0] for transition in data_in_mini_batch])
        current_states = current_states.squeeze()
        current_qvalues_list = self.model.predict(current_states)

        #predicted werte, aus predicteten werten, da next_states in process() ja aus dem aktuellen state predicted wird.
        #rewards , wenn eine action gemacht wird und danach optimal verhalten wird
        next_states = numpy.array([transition[3] for transition in data_in_mini_batch])
        next_states = next_states.squeeze()
        next_qvalues_list = self.target_model.predict(next_states)

        x_train = []
        y_train = []

        for index, (current_state, action, reward, _, done) in enumerate(data_in_mini_batch):
            current_q_values = current_qvalues_list[index]

            if not done:
                future_reward = numpy.max(next_qvalues_list[index])
                desired_q = reward + self.discount_factor * future_reward
            else:
                desired_q = reward

            current_q_values[action] = desired_q
            x_train.append(current_state)
            y_train.append(current_q_values)

        x_train = numpy.array(x_train)
        y_train = numpy.array(y_train)
        x_train = numpy.reshape(x_train, [len(data_in_mini_batch), self.state_size])
        y_train = numpy.reshape(y_train, [len(data_in_mini_batch), self.action_size])

        self.model.fit(
            tensorflow.convert_to_tensor(x_train, tensorflow.float32),
            tensorflow.convert_to_tensor(y_train, tensorflow.float32),
            batch_size=self.batch_size, verbose=0
        )
        self.target_update_after_counter += 1

        if self.target_update_after_counter > self.update_target_after and terminal:
            self.update_target_model()


def main():
    # if args is None:
    #     args = sys.argv
    # stage_num = int(args[1]) if len(args) > 1 else '1'
    # max_training_episodes = int(args[2]) if len(args) > 2 else '1000'
    # stage_boost = args[3]=='True' if len(args) > 3 else False
    rclpy.init()

    dqn_agent = DQNAgent()
    rclpy.spin(dqn_agent)

    dqn_agent.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
