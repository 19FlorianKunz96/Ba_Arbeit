
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
import numpy
import collections
import datetime
import math

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


@register_keras_serializable(package="Custom", name="Dueling_QR_DQN")
class DuelingQRDQN(tensorflow.keras.Model):
    def __init__(self, n_actions, fc1, fc2, fc3=None, num_quantiles=51, full_noisy=True, **kwargs):
        super().__init__(**kwargs)
        self.n_actions = int(n_actions)
        self.N = int(num_quantiles)
        self.full_noisy = full_noisy

        DenseCls = NoisyDense if full_noisy else tensorflow.keras.layers.Dense
        self.dense1 = DenseCls(fc1, activation="relu")
        self.dense2 = DenseCls(fc2, activation="relu")
        self.dense3 = None
        if fc3 is not None:
            self.dense3 = DenseCls(fc3, activation="relu")

        # quantile heads
        # V: (N)  |  A: (A * N)
        self.V = NoisyDense(self.N, activation=None)
        self.A = NoisyDense(self.n_actions * self.N, activation=None)

    def call(self, state, training=None):
        x = self.dense1(state, training=training)
        x = self.dense2(x, training=training)
        if self.dense3 is not None:
            x = self.dense3(x, training=training)

        V = self.V(x, training=training)                                  # (B, N)
        A = self.A(x, training=training)                                  # (B, A*N)
        A = tensorflow.reshape(A, (-1, self.n_actions, self.N))           # (B, A, N)
        V = tensorflow.expand_dims(V, axis=1)                             # (B, 1, N)
        Q = V + (A - tensorflow.reduce_mean(A, axis=1, keepdims=True))    # (B, A, N)
        return Q  # quantiles pro Action

    def q_expectation(self, state, training=None):
        # Erwartungswert über Quantile (für Argmax/Logging)
        Q = self.call(state, training=training)                           # (B, A, N)
        return tensorflow.reduce_mean(Q, axis=-1)                         # (B, A)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"n_actions": self.n_actions, "num_quantiles": self.N})
        return cfg



def quantile_huber_loss(pred, target, taus, weights=None, kappa=1.0):
    """
    pred:   (B, N)   vorhergesagte Quantile für gewählte Aktion
    target: (B, N)   Ziel-Quantile (Double DQN über Target-Net)
    taus:   (N,)     gleichmäßig z. B. (0.5/N, ..., (2N-1)/(2N))
    weights:(B,)     PER-Gewichte (optional)
    """
    # pairwise TD-Fehler u_{j,i} = y_j - q_i
    u = tensorflow.expand_dims(target, axis=2) - tensorflow.expand_dims(pred, axis=1)  # (B, N, N)

    # Huber
    if kappa > 0.0:
        abs_u = tensorflow.abs(u)
        huber = tensorflow.where(abs_u <= kappa, 0.5 * u**2, kappa * (abs_u - 0.5 * kappa))
    else:
        huber = tensorflow.abs(u)

    # |tau - 1_{u<0}|
    tau = tensorflow.reshape(taus, (1, -1, 1))                                         # (1, N, 1)
    indicator = tensorflow.cast(u < 0.0, tensorflow.float32)                           # (B, N, N)
    quantile_weight = tensorflow.abs(tau - indicator)                                  # (B, N, N)

    loss = quantile_weight * huber                                                     # (B, N, N)
    loss = tensorflow.reduce_mean(loss, axis=1)                                        # (B, N)  mittlere j
    loss = tensorflow.reduce_sum(loss, axis=1)                                         # (B,)    sum über i

    if weights is not None:
        loss = loss * weights
    return tensorflow.reduce_mean(loss)
