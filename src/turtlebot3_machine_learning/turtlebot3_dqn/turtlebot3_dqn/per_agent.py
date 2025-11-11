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


        #---------------------------------------------------------------------------------------------------------------------------------
        #                                       Headers/Imports
        #---------------------------------------------------------------------------------------------------------------------------------

#Standart
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
#Ros
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
from std_srvs.srv import Empty
from turtlebot3_msgs.srv import Dqn
#Tensorflow
import tensorflow
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from keras.saving import register_keras_serializable
from tensorflow.keras import initializers
from tensorflow.keras.layers import Layer
#Utils/Own
from turtlebot3_dqn.utils import DQNMetric
from turtlebot3_dqn.utils import Dueling_DQN
from turtlebot3_dqn.utils import PERBuffer
from turtlebot3_dqn.utils import N_Step
from turtlebot3_dqn.utils import NoisyDense
from turtlebot3_dqn.utils import DuelingQRDQN
from turtlebot3_dqn.utils import quantile_huber_loss


tensorflow.config.set_visible_devices([], 'GPU')
LOGGING = True
current_time = datetime.datetime.now().strftime('[%mm%dd-%H:%M]')

class DQNAgent(Node):


    def __init__(self):
        super().__init__('dqn_agent')
        #---------------------------------------------------------------------------------------------------------------------------------
        #                                                  Training Mode
        #---------------------------------------------------------------------------------------------------------------------------------
        self.train_mode = True

        #---------------------------------------------------------------------------------------------------------------------------------
        #                                                    Declare
        #---------------------------------------------------------------------------------------------------------------------------------
        self.declare_parameter('stagex',1)
        self.declare_parameter('max_episodes',100)
        self.declare_parameter('stage_boost',False)
        self.declare_parameter('load_from_folder','actual')
        self.declare_parameter('load_from_stage',1)
        self.declare_parameter('load_from_episode',50)
        self.declare_parameter('action_space',5)

        #---------------------------------------------------------------------------------------------------------------------------------
        #                                        Initial Parameters for the Agent
        #---------------------------------------------------------------------------------------------------------------------------------
        #Agent
        self.state_size = 28
        self.action_size=self.get_parameter('action_space').get_parameter_value().integer_value
        #World
        self.stage = self.get_parameter('stagex').get_parameter_value().integer_value
        self.max_training_episodes = self.get_parameter('max_episodes').get_parameter_value().integer_value
        self.stage_boost = self.get_parameter('stage_boost').get_parameter_value().bool_value
        self.load_from_folder = self.get_parameter('load_from_folder').get_parameter_value().string_value
        self.load_from_stage = self.get_parameter('load_from_stage').get_parameter_value().integer_value
        self.load_from_episode = self.get_parameter('load_from_episode').get_parameter_value().integer_value
        #Eval
        self.done = False
        self.succeed = False
        self.fail = False
        #Others
        self.load_episode=0
        self.epsilon=0.0 #dummy, dass logging nicht crasht
        self.info = ['Trainable in Both Target disabled']

        #---------------------------------------------------------------------------------------------------------------------------------
        #                                              Hyperparameters
        #---------------------------------------------------------------------------------------------------------------------------------
        #Standart
        self.discount_factor = 0.99
        self.learning_rate = 0.001
        self.step_counter = 0
        self.batch_size = 64 if self.distributional_mode else 64
        self.gradient_clipping = True
        #Distributional
        self.distributional_mode=True
        self.num_quantiles = 51
        self.taus = tensorflow.constant((numpy.arange(self.num_quantiles) + 0.5) / self.num_quantiles, dtype=tensorflow.float32)
        self.learningrate_distributional = 1e-3
        self.epsilon_distributional = 1e-8
        #Target Update
        self.tau = 0.005
        self.use_soft_target = True
        #PER/nStep
        self.nsteps=5
        self.mix_one_step = True
        self.mixing_ratio = 0.2 
        self.eps_per = 1e-2
        self.beta_frames=300_000
        self.beta_start = 0.4
        self.alpha = 0.6
        self.max_buffer_size = 500_000
        #Architecture
        self.full_noisy_dense = True
        self.fc1=256
        self.fc2=256
        self.fc3=None

        self.hyperparams = {
            'Stage' : self.stage,
            'Folder Name' : self.training_dir,
            'Learned from older Stages' : self.stage_boost,
            'Learned from Model' : {'Folder' : self.load_from_folder,'Stage':self.load_from_stage, 'Episode': self.load_from_episode},
            'Success' : self.info,
            'State Size' : self.state_size,
            'Action Szie' : self.action_size,
            'Dueling':True,
            'Double DQN': True,
            'Architecture' : [self.fc1,self.fc2,self.fc3],
            'Maximum Episodes' : self.max_training_episodes,
            'Discount Factor' : self.discount_factor,
            'Learning Rate' : self.learning_rate if self.distributional_mode is False else self.learningrate_distributional,
            'NSteps' : self.nsteps,
            'Batch Size' : self.batch_size,
            'Replay Memory Min' : self.min_replay_memory_size,
            'Distributional Mode' : self.distributional_mode,
            'Distributional Epsilon' : self.epsilon_distributional,
            'Number Quantiles' : self.num_quantiles,
            'Soft Target': self.use_soft_target,
            'Tau for Softupdate' : self.tau,
            'Full Noisy': self.full_noisy_dense,
            'Mixing One Step':self.mix_one_step,
            'Mixing Ratio' : self.mixing_ratio,
            'PER': True,
            'Eps PER':self.eps_per,
            'Alpha':self.alpha,
            'MaxBufferSize':self.max_buffer_size,
            'Beta Start': self.beta_start,
            'Beta Frames': self.beta_frames,
            'Gradient Clipping': self.gradient_clipping


            }
        #---------------------------------------------------------------------------------------------------------------------------------
        #                                                        Create Buffers
        #---------------------------------------------------------------------------------------------------------------------------------
        self.nstep_memory = N_Step(self.max_buffer_size,self.nsteps,self.discount_factor)
        self.per = PERBuffer(capacity=self.max_buffer_size, alpha=self.alpha, beta_start=self.beta_start, beta_frames=self.beta_frames, eps=self.eps_per)
        self.min_replay_memory_size = 5000

        #---------------------------------------------------------------------------------------------------------------------------------
        #                                                           Create Models
        #---------------------------------------------------------------------------------------------------------------------------------
        loss = tensorflow.keras.losses.Huber(delta=1.0, name="huber")
        if self.distributional_mode:
            self.model = DuelingQRDQN(self.action_size, fc1=self.fc1, fc2=self.fc2,fc3=self.fc3,num_quantiles=self.num_quantiles,full_noisy=self.full_noisy_dense)
            self.model.compile(optimizer=Adam(learning_rate=self.learningrate_distributional, epsilon=self.epsilon_distributional))
            self.target_model = DuelingQRDQN(self.action_size, fc1=self.fc1, fc2=self.fc2,fc3=self.fc3,num_quantiles=self.num_quantiles,full_noisy=self.full_noisy_dense)
            self.target_model.compile(optimizer=Adam(learning_rate=self.learningrate_distributional, epsilon=self.epsilon_distributional))
        else:
            self.model = Dueling_DQN(self.action_size,fc1=self.fc1,fc2=self.fc2,fc3=self.fc3, full_noisy = self.full_noisy_dense)
            self.model.compile(loss=loss, optimizer=Adam(learning_rate=self.learning_rate))
            self.target_model = Dueling_DQN(self.action_size,fc1=self.fc1,fc2=self.fc2,fc3=self.fc3, full_noisy = self.full_noisy_dense)
            self.target_model.compile(loss=loss, optimizer=Adam(learning_rate=self.learning_rate))

        _ = self.model(tensorflow.zeros((1, self.state_size)))
        _ = self.target_model(tensorflow.zeros((1, self.state_size)))
        self.update_target_model()

        #---------------------------------------------------------------------------------------------------------------------------------
        #                                                        Create the Storing Path
        #---------------------------------------------------------------------------------------------------------------------------------
        self.uuid = uuid.uuid4()
        self.date = datetime.date.today()
        self.training_dir = f'{self.uuid}_{self.date}_stage{self.stage}_rainbow'
        #give the other nodes the dir
        with open(os.path.join(os.path.dirname(os.path.realpath(__file__)),'temp.json'),'w') as temp:
            json.dump({'folder': self.training_dir},temp)

        self.model_dir_path = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))),'saved_model')
        self.training_dir_path = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))),'trainings_done')
        self.training_path_comp = os.path.join(self.training_dir_path,self.training_dir)
        self.model_path = os.path.join(self.model_dir_path,'stage' + str(self.stage) + '_episode' + str(self.load_episode) + '.h5')

        #---------------------------------------------------------------------------------------------------------------------------------
        #                                                           Stage Boost Settings
        #   Parameters have to be set in Launch File
        #---------------------------------------------------------------------------------------------------------------------------------
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

        #---------------------------------------------------------------------------------------------------------------------------------
        #                                                    Logging
        #   TODO: DELETE
        #---------------------------------------------------------------------------------------------------------------------------------

       
        if LOGGING:
            tensorboard_file_name = current_time + ' dqn_stage' + str(self.stage) + '_reward'
            home_dir = os.path.expanduser('~')
            dqn_reward_log_dir = os.path.join(
                home_dir, 'turtlebot3_dqn_logs', 'gradient_tape', tensorboard_file_name
            )
            self.dqn_reward_writer = tensorflow.summary.create_file_writer(dqn_reward_log_dir)
            self.dqn_reward_metric = DQNMetric()

        #---------------------------------------------------------------------------------------------------------------------------------
        #                                               Publishers and Clients
        #---------------------------------------------------------------------------------------------------------------------------------
        self.rl_agent_interface_client = self.create_client(Dqn, 'rl_agent_interface')
        self.make_environment_client = self.create_client(Empty, 'make_environment')
        self.reset_environment_client = self.create_client(Dqn, 'reset_environment')

        self.action_pub = self.create_publisher(Float32MultiArray, '/get_action', 10)
        self.result_pub = self.create_publisher(Float32MultiArray, 'result', 10)
        self.loss_pub = self.create_publisher(Float32MultiArray,'loss',10)


        self.process()

        #---------------------------------------------------------------------------------------------------------------------------------
        #                                                    Class Functions
        #---------------------------------------------------------------------------------------------------------------------------------
    
    def process(self):
        self.env_make()
        time.sleep(1.0)


        episode_num = self.load_episode


        for episode in range(self.load_episode + 1, self.max_training_episodes + 1):
            # if episode ==1:
            #     self.get_logger().info(f'Training with Stage Boost = {self.stage_boost}')
            self.nstep_memory.reset()
            state = self.reset_environment()
            episode_num += 1
            local_step = 0
            score = 0
            sum_max_q = 0.0

            time.sleep(1.0)

            while True:
                local_step += 1
                if self.distributional_mode:
                    q_exp = self.model.q_expectation(state, training=self.train_mode)
                    sum_max_q += float(tensorflow.reduce_max(q_exp).numpy())
                else:
                    q_values = self.model(state, training = self.train_mode)
                    sum_max_q += float(tensorflow.reduce_max(q_values).numpy())

                action = int(self.get_action(state))
                next_state, reward, done = self.step(action)
                score += reward

                msg = Float32MultiArray()
                msg.data = [float(action), float(score), float(reward)]
                self.action_pub.publish(msg)

                if self.train_mode:

                    self.nstep_memory.append(state.squeeze(0),action,reward,next_state.squeeze(0),done)
                    if self.nstep_memory.can_pop():
                        s0, a0, Rn, s_n, done_n, gamma_n = self.nstep_memory.pop()
                        self.per.push(s0,a0,Rn,s_n,done_n, gamma_n)
                    if self.mix_one_step and (random.random()<self.mixing_ratio):
                        self.append_sample((state, action, reward, next_state, done,self.discount_factor))
                    if done:
                        for (s0, a0, Rn, s_n, done_n, gamma_n) in self.nstep_memory.flush():
                            self.per.push(s0, a0, Rn, s_n, done_n, gamma_n)

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
                        'memory length:', self.per.tree.n_entries,
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
            self.get_logger().warn('Environment make client failed to connect to the server, try again ...')

        self.make_environment_client.call_async(Empty.Request())


    def reset_environment(self):
        while not self.reset_environment_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().warn('Reset environment client failed to connect to the server, try again ...')

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
        self.step_counter += 1
        if self.distributional_mode:
            q_exp = self.model.q_expectation(state, training = self.train_mode)
            return int(tensorflow.argmax(q_exp,axis=1).numpy()[0])
        else:
            if self.train_mode:
                q_values= self.model(state,training = True)
            else:
                q_values= self.model(state,training = False)
            return int(tensorflow.argmax(q_values, axis = 1).numpy()[0])


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

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())
        self.target_update_after_counter = 0
        print('*Target model updated*')

    def soft_update_target(self, tau=None):
        if tau is None:
            tau = self.tau
        online_vars = self.model.trainable_variables
        target_vars = self.target_model.trainable_variables
        for ov, tv in zip(online_vars, target_vars):
            tv.assign((1.0 - tau) * tv + tau * ov)

    def append_sample(self, transition):
        s,a,r,s2,d,g = transition
        self.per.push(s.squeeze(0), a, r, s2.squeeze(0), d,g)

    def train_model(self, terminal):

        if self.per.tree.n_entries < self.min_replay_memory_size:
            return
        try:
            idxs, batch, is_w, _ = self.per.sample(self.batch_size)
        except RuntimeError:
            return
        
        filtered = []
        for i, data in enumerate(batch):
            if data is None or any(x is None for x in data):
                continue
            filtered.append((idxs[i], data, is_w[i]))
        if not filtered:
            return

        idxs, batch, is_w = map(list, zip(*filtered))
        
        B=len(batch)

        s, a, r, s2, d, gamma_n = map(numpy.array, zip(*batch))
        s = s.astype(numpy.float32)
        s2 = s2.astype(numpy.float32)
        a  = a.astype(numpy.int32)
        r  = r.astype(numpy.float32)
        d  = d.astype(numpy.float32)
        w  = numpy.asarray(is_w[:B], dtype = numpy.float32)
        gamma_n = gamma_n.astype(numpy.float32)

        s_tf  = tensorflow.convert_to_tensor(s, dtype=tensorflow.float32)
        s2_tf = tensorflow.convert_to_tensor(s2, dtype=tensorflow.float32)
        a_tf  = tensorflow.convert_to_tensor(a, dtype=tensorflow.int32)
        r_tf  = tensorflow.convert_to_tensor(r, dtype=tensorflow.float32)
        d_tf  = tensorflow.convert_to_tensor(d, dtype=tensorflow.float32)
        w_tf  = tensorflow.convert_to_tensor(w, dtype=tensorflow.float32)
        gamma_n_tf = tensorflow.convert_to_tensor(gamma_n, dtype=tensorflow.float32)

        if self.distributional_mode:
            N = self.num_quantiles
            taus = self.taus
            # ---------- Double DQN: Argmax über Erwartungswert des Online-Netzes ----------
            q_next_online_exp = self.model.q_expectation(s2_tf, training=False)              # (B, A)
            best_next_actions = tensorflow.argmax(q_next_online_exp, axis=1, output_type=tensorflow.int32)
            # Target-Quantile aus Target-Netz (ohne Noise)
            q_next_target_all = self.target_model(s2_tf, training=False)                     # (B, A, N)
            idx = tensorflow.stack([tensorflow.range(B, dtype=tensorflow.int32), best_next_actions], axis=1)  # (B,2)
            target_quantiles = tensorflow.gather_nd(q_next_target_all, idx)                  # (B, N)

            y_quantiles = tensorflow.expand_dims(r_tf, axis=1) + \
                        tensorflow.expand_dims(1.0 - d_tf, axis=1) * \
                        tensorflow.expand_dims(gamma_n_tf, axis=1) * target_quantiles           # (B, N)

            with tensorflow.GradientTape() as tape:
                pred_all = self.model(s_tf, training=True)                                   # (B, A, N)
                idx2 = tensorflow.stack([tensorflow.range(B, dtype=tensorflow.int32), a_tf], axis=1)
                pred_quantiles = tensorflow.gather_nd(pred_all, idx2)                        # (B, N)

                loss = quantile_huber_loss(pred_quantiles, y_quantiles, taus, weights=w_tf, kappa=1.0)

            grads = tape.gradient(loss, self.model.trainable_variables)
            grads = [tensorflow.clip_by_norm(g, 10.0) if g is not None else None for g in grads]
            self.model.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

            # PER-Prioritäten: benutze mittleren |TD| über Quantile
            td = tensorflow.abs(y_quantiles - pred_quantiles)                                # (B, N)
            td_err = tensorflow.reduce_mean(td, axis=1).numpy()                              # (B,)
            self.per.update_priorities(idxs, td_err) 

        else:
            q_next_online = self.model(s2_tf, training=False)
            best_next_actions = tensorflow.argmax(q_next_online, axis=1, output_type=tensorflow.int32)

            q_next_target_all = self.target_model(s2_tf, training=False)
            idx = tensorflow.stack([tensorflow.range(B, dtype=tensorflow.int32), best_next_actions], axis=1)
            q_next_target = tensorflow.gather_nd(q_next_target_all, idx)

            y = r_tf + (1.0 - d_tf) * (gamma_n_tf * q_next_target )

            huber = tensorflow.keras.losses.Huber(delta=1.0, reduction=tensorflow.keras.losses.Reduction.NONE)

            with tensorflow.GradientTape() as tape:
                q_pred_all = self.model(s_tf, training=True)                             # (B, A)
                idx2 = tensorflow.stack([tensorflow.range(B, dtype=tensorflow.int32), a_tf], axis=1)
                q_pred = tensorflow.gather_nd(q_pred_all, idx2)                          # (B,)
                td_error = y - q_pred                                                    # (B,)
                per_sample_loss = huber(y, q_pred)                                       # (B,)
                weighted_loss = per_sample_loss * w_tf                                   
                loss = tensorflow.reduce_mean(weighted_loss)

            grads = tape.gradient(loss, self.model.trainable_variables)
            #Gradient Clipping
            if self.gradient_clipping:
                grads = [tensorflow.clip_by_norm(g, 10.0) if g is not None else None for g in grads]

            self.model.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

            # Prioritäten mit |TD-Fehler| aktualisieren
            td_np = numpy.abs(td_error.numpy())
            self.per.update_priorities(idxs, td_np)


        if self.use_soft_target:
            self.soft_update_target()
        else:

            self.target_update_after_counter += 1
            if self.target_update_after_counter > self.update_target_after and terminal:
                self.update_target_model()

        # publish loss (optional wie bisher)
        if self.step_counter % 50 == 0:
            msg = Float32MultiArray()
            msg.data = [float(loss.numpy()), float(self.step_counter), float(self.epsilon)]
            self.loss_pub.publish(msg)


def main():

    rclpy.init()
    dqn_agent = DQNAgent()
    rclpy.spin(dqn_agent)
    dqn_agent.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
