import tensorflow as tf
import numpy as np
from tensorflow.keras import layers


class Agent:
    states = []
    actions = []
    targets = []

    def __init__(self, num_inputs, num_actions, num_hidden=128, gamma=1, lr=1e-2):
        self.gamma = gamma
        self.lr = lr
        self.num_actions = num_actions
        self.num_inputs = num_inputs

        inputs = layers.Input(shape=(num_inputs,))
        hidden_1 = layers.Dense(num_hidden, activation="relu")(inputs)
        hidden_2 = layers.Dense(num_hidden)(hidden_1)
        output = layers.Dense(num_actions, activation='linear')(hidden_2)

        self.dqn = tf.keras.Model(inputs=inputs, outputs=output)
        self.dqn.compile(tf.optimizers.Adam(self.lr), loss=tf.keras.losses.MeanSquaredError())

    def write_history(self, state, action, target):
        self.states.append(state)
        self.actions.append(action)
        self.targets.append(target)

    def clear_history(self):
        self.states.clear()
        self.actions.clear()
        self.targets.clear()

    def predict(self, states):
        return self.dqn.predict(np.reshape(np.array(states), (-1, self.num_inputs)))

    def retrain(self):
        #Q(s_t, a_t) = r_t + gamma * (max_a'(Q(s_t+1, a'))
        amax_predictoins = np.amax(self.predict([self.states]), axis=1)
        for i in range(len(self.states)-2, -1, -1):
            self.targets[i][self.actions[i]] += self.gamma * amax_predictoins[i+1]

        self.dqn.fit(np.array(self.states), np.array(self.targets), verbose=0)
        self.clear_history()

