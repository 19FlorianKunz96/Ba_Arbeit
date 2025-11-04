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

'''Double Network / Dueling Network / Action Space 6 / Prioritized Experience Replay'''


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
from keras.saving import register_keras_serializable
from tensorflow.keras import initializers
from tensorflow.keras.layers import Layer


from turtlebot3_msgs.srv import Dqn

@register_keras_serializable(package="Custom", name="NoisyDense")
class NoisyDense(Layer):
    def __init__(self, units, activation=None, sigma0=0.5, **kwargs):
        super().__init__(**kwargs)
        self.units = int(units)
        self.activation = tensorflow.keras.activations.get(activation)
        self.sigma0 = float(sigma0)

    def build(self, input_shape):
        in_features = int(input_shape[-1])
        # Fortunato init
        mu_range = 1.0 / math.sqrt(in_features)  # statt tensorflow.math.sqrt(...)
        sigma_init = self.sigma0 / math.sqrt(in_features)
        self.mu_w = self.add_weight(
            name="mu_w",
            shape=(in_features, self.units),
            initializer=initializers.RandomUniform(minval=-mu_range, maxval=mu_range),
            trainable=True,
        )
        self.mu_b = self.add_weight(
            name="mu_b",
            shape=(self.units,),
            initializer=initializers.RandomUniform(minval=-mu_range, maxval=mu_range),
            trainable=True,
        )
        self.sigma_w = self.add_weight(
            name="sigma_w",
            shape=(in_features, self.units),
            initializer=initializers.Constant(sigma_init),
            trainable=True,
        )
        self.sigma_b = self.add_weight(
            name="sigma_b",
            shape=(self.units,),
            initializer=initializers.Constant(sigma_init),
            trainable=True,
        )
        super().build(input_shape)

    @staticmethod
    def _f(x):
        return tensorflow.sign(x) * tensorflow.sqrt(tensorflow.abs(x))

    def call(self, inputs, training=None):
        if training:
            eps_in = tensorflow.random.normal((tensorflow.shape(inputs)[-1],))
            eps_out = tensorflow.random.normal((self.units,))
            f_in = self._f(eps_in)          # (in,)
            f_out = self._f(eps_out)        # (out,)
            noise_w = tensorflow.tensordot(f_in, f_out, axes=0)  # (in, out)
            w = self.mu_w + self.sigma_w * noise_w
            b = self.mu_b + self.sigma_b * f_out
        else:
            w = self.mu_w
            b = self.mu_b

        y = tensorflow.linalg.matmul(inputs, w) + b
        return self.activation(y) if self.activation is not None else y

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"units": self.units, "activation": tensorflow.keras.activations.serialize(self.activation), "sigma0": self.sigma0})
        return cfg
    

class N_Step:
    def __init__(self, capacity, n , gamma):
        self.buffer = collections.deque(maxlen = capacity)
        self.gamma = gamma
        self.n = n

    def reset(self):
        self.buffer.clear()

    def append(self, s, a, r, s2, done):
        self.buffer.append((s,a,r,s2,done))

    def can_pop(self):
        return len(self.buffer) >= self.n
    
    def pop(self):
        Rn = 0.0
        discount = 1.0
        done_out = False

        s0, a0, _, _,_ = self.buffer[0]
        for i in range(self.n):
            _, _, r_i, _, d_i = self.buffer[i]
            Rn += discount * r_i
            if d_i:
                done_out = True
                s_n = self.buffer[i][3]
                break
            discount *= self.gamma

        if not done_out:
            s_n = self.buffer[self.n - 1][3]

        self.buffer.popleft()
        return s0, a0, Rn, s_n, done_out, discount
    
    def flush(self):
        out = []
        while self.buffer:
            Rn= 0.0
            discount = 1.0
            done_out = False

            s0, a0, _, _, _ =  self.buffer[0]
            last_next = self.buffer[0][3]
            for i in range(len(self.buffer)):
                _, _, r_i, s_next_i, d_i = self.buffer[i]
                Rn+= discount* r_i
                last_next = s_next_i
                if d_i:
                    done_out = True
                    break
                discount *= self.gamma
            self.buffer.popleft()
            out.append((s0,a0,Rn,last_next, done_out, discount))
        return out
    



class SumTree:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.tree = numpy.zeros(2 * capacity - 1, dtype=numpy.float32)
        self.data = [None] * capacity
        self.write = 0
        self.n_entries = 0

    @property
    def total(self):
        return float(self.tree[0])

    def add(self, p: float, data):
        idx = self.write + self.capacity - 1
        self.data[self.write] = data
        self.update(idx, p)
        self.write = (self.write + 1) % self.capacity
        self.n_entries = min(self.n_entries + 1, self.capacity)

    def update(self, idx: int, p: float):
        change = p - self.tree[idx]
        self.tree[idx] = p
        parent = (idx - 1) // 2
        while True:
            self.tree[parent] += change
            if parent == 0:
                break
            parent = (parent - 1) // 2

    def get(self, s: float):
        idx = 0
        while True:
            left = 2 * idx + 1
            right = left + 1
            if left >= len(self.tree):
                break
            if s <= self.tree[left]:
                idx = left
            else:
                s -= self.tree[left]
                idx = right
        data_index = idx - self.capacity + 1
        return idx, self.tree[idx], self.data[data_index]


class PERBuffer:
    def __init__(self, capacity, alpha=0.6, beta_start=0.4, beta_frames=200_000, eps=1e-5):
        self.tree = SumTree(capacity)
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.frame = 1
        self.eps = eps
        self.max_p = 1.0  # neue Transitions sofort wichtig

    def push(self, s, a, Rn, s2, done, gamma):
        p = (self.max_p + self.eps) ** self.alpha
        self.tree.add(p, (s, a, Rn, s2, done,gamma))

    def sample(self, batch_size):

        total = self.tree.total
        # Nichts zu ziehen? Sauber abbrechen.
        if total <= 0.0 or self.tree.n_entries == 0:
            raise RuntimeError("PERBuffer.sample called with empty/zero-total tree")

        segment = total / batch_size
        idxs, batch, priorities = [], [], []
        i = 0
        tries = 0
        max_tries = batch_size * 10  # Schutz vor Endlosschleifen

        while i < batch_size:
            if tries > max_tries:
                break
            s = numpy.random.uniform(segment * i, segment * (i + 1))
            idx, p, data = self.tree.get(s)
            tries += 1

            # Leeres Blatt? Neu ziehen.
            if data is None or p <= 0.0:
                continue

            idxs.append(idx)
            priorities.append(float(p))
            batch.append(data)
            i += 1

        if len(batch) < batch_size:
            # Zu wenig gültige Samples – Caller soll später nochmal versuchen.
            raise RuntimeError(f"PERBuffer.sample got only {len(batch)} valid samples")

        probs = numpy.asarray(priorities, dtype=numpy.float32) / (total + 1e-12)
        probs = numpy.clip(probs, 1e-12, 1.0)  # niemals 0

        beta = min(1.0, self.beta_start + (1.0 - self.beta_start) * (self.frame / self.beta_frames))
        self.frame += 1

        N = max(1, self.tree.n_entries)
        weights = (N * probs) ** (-beta)
        weights = weights / (weights.max() + 1e-12)  # vermeiden von NaN/Inf

        return idxs, batch, weights.astype(numpy.float32), probs.astype(numpy.float32)

    def update_priorities(self, idxs, td_errors):
        td = numpy.asarray(td_errors, dtype=numpy.float64)
        td = numpy.nan_to_num(td, nan=0.0, posinf=1e6, neginf=0.0)
        td = numpy.clip(numpy.abs(td) + self.eps, 1e-12, 1e6)

        ps = (td) ** self.alpha
        self.max_p = max(self.max_p, float(ps.max()))
        for idx, p in zip(idxs, ps):
            self.tree.update(int(idx), float(p))






tensorflow.config.set_visible_devices([], 'GPU')


LOGGING = True
current_time = datetime.datetime.now().strftime('[%mm%dd-%H:%M]')

@register_keras_serializable(package="Custom", name="Dueling_DQN")
class Dueling_DQN(tensorflow.keras.Model):
    def __init__(self,n_actions,fc1,fc2,fc3=None,full_noisy = False, **kwargs):
        super(Dueling_DQN,self).__init__(**kwargs)
        if full_noisy:
            self.dense1=NoisyDense(fc1,activation='relu')
            self.dense2=NoisyDense(fc2,activation='relu')
            self.dense3=None
            if fc3 is not None:
                self.dense3=NoisyDense(fc3,activation='relu')

        else:
            self.dense1=tensorflow.keras.layers.Dense(fc1,activation='relu')
            self.dense2=tensorflow.keras.layers.Dense(fc2,activation='relu')
            self.dense3=None
            if fc3 is not None:
                self.dense3=tensorflow.keras.layers.Dense(fc3,activation='relu')

        self.A= NoisyDense(n_actions,activation=None)
        self.V= NoisyDense(1, activation = None)
        self.n_actions=int(n_actions)
        self.fc1 = int(fc1)
        self.fc2 = int(fc2)
        self.fc3 = None if fc3 is None else int(fc3)


    def call(self,state,training = None):
        x=self.dense1(state, training= training)
        x=self.dense2(x, training = training)
        if self.dense3 is not None:
            x=self.dense3(x,training = training)
        V= self.V(x,training = training)
        A=self.A(x, training = training)


        Q= V + (A - tensorflow.math.reduce_mean(A, axis=1, keepdims=True))
        return Q
   
    def advantage(self,state, training = None):
        x=self.dense1(state, training = training)
        x=self.dense2(x, training = training)
        if self.dense3 is not None:
            x = self.dense3(x, training = training)
        A= self.A(x, training = training)
        return A
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "n_actions": self.n_actions,
            "fc1": self.fc1,
            "fc2": self.fc2,
            "fc3": self.fc3,
        })
        return config
    
    @classmethod
    def from_config(cls, config):
        # Standard reicht meist; hier explizit der Klarheit halber
        return cls(**config)








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
        self.declare_parameter('action_space',5)
        self.stage = self.get_parameter('stagex').get_parameter_value().integer_value
        self.train_mode = True
        self.state_size = 28 #adjusted for new environment
        self.action_size=self.get_parameter('action_space').get_parameter_value().integer_value
        self.max_training_episodes = self.get_parameter('max_episodes').get_parameter_value().integer_value


        self.stage_boost = self.get_parameter('stage_boost').get_parameter_value().bool_value
        self.load_from_folder = self.get_parameter('load_from_folder').get_parameter_value().string_value
        self.load_from_stage = self.get_parameter('load_from_stage').get_parameter_value().integer_value
        self.load_from_episode = self.get_parameter('load_from_episode').get_parameter_value().integer_value
        self.done = False
        self.succeed = False
        self.fail = False
        self.info = ['Dueling,Double, HuberLoss, PER, n_step,NoisyNets in allen Layern, 256-256, action space 5, soft target update, gradient clipping, reward funktion komplett aus repo übernommen, target netz nicht trainable']


        self.discount_factor = 0.99
        self.learning_rate = 0.001
        self.epsilon = 1.0
        self.step_counter = 0
        self.epsilon_decay = 30000 #6000 * self.stage
        self.epsilon_min = 0.05
        self.k=0.5 # for epsilon decay
        self.batch_size = 64
        self.nsteps=5
        self.tau = 0.005
        self.use_soft_target = True
        self.full_noisy_dense = True

        #-----------------------------------PER + N-STEP INIT----------------------------------------------------------------------------------
        self.nstep_memory = N_Step(500000,self.nsteps,self.discount_factor)
        

        self.per = PERBuffer(capacity=500000, alpha=0.6, beta_start=0.4, beta_frames=300_000, eps=1e-5)
        self.min_replay_memory_size = 5000
        #------------------------------------------------------------------------------------------------------------------------------

        self.history=None


        self.uuid = uuid.uuid4()
        self.date = datetime.date.today()
        self.training_dir = f'{self.uuid}_{self.date}_stage{self.stage}_rainbow'
        #give the other nodes the dir
        with open(os.path.join(os.path.dirname(os.path.realpath(__file__)),'temp.json'),'w') as temp:
            json.dump({'folder': self.training_dir},temp)


        #self.model = self.create_qnetwork()
        #self.target_model = self.create_qnetwork()
        self.model = Dueling_DQN(self.action_size,fc1=256,fc2=256, full_noisy = self.full_noisy_dense)
        self.target_model = Dueling_DQN(self.action_size,fc1=256,fc2=256, full_noisy = self.full_noisy_dense)
        loss = tensorflow.keras.losses.Huber(delta=1.0, name="huber")
        self.model.compile(loss=loss, optimizer=Adam(learning_rate=self.learning_rate))
        _ = self.model(tensorflow.zeros((1, self.state_size)))
        self.model.summary()
        self.target_model.compile(loss=loss, optimizer=Adam(learning_rate=self.learning_rate))
        _ = self.target_model(tensorflow.zeros((1, self.state_size)))
        self.target_model.summary()
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
        self.loss_pub = self.create_publisher(Float32MultiArray,'loss',10)


        self.process()


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


                #berchnet die q-values für alle actions im aktuellen Schritt.
                q_values = self.model(state, training = self.train_mode)


                #schaut welcher der größte ist.Nur für den Graphen ??
                sum_max_q += float(tensorflow.reduce_max(q_values).numpy())


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
                    self.nstep_memory.append(state.squeeze(0),action,reward,next_state.squeeze(0),done)

                    if self.nstep_memory.can_pop():
                        s0, a0, Rn, s_n, done_n, gamma_n = self.nstep_memory.pop()
                        self.per.push(s0,a0,Rn,s_n,done_n, gamma_n)

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
            q_values= self.model(state,training = True)
        else:
            q_values= self.model(state,training = False)

        result = int(tensorflow.argmax(q_values, axis = 1).numpy()[0])
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


    #----------------------------------------------adjusted for PER------------------------------------------------------------------------------
    def append_sample(self, transition):
        s,a,r,s2,d,g = transition
        self.per.push(s.squeeze(0), a, r, s2.squeeze(0), d,g)
        #self.replay_memory.append(transition)


    def train_model(self, terminal):


        #wenn die collection größer ist als min batch size, dann suche dir aus der collection einen random batch mit der grösse batchsize
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
