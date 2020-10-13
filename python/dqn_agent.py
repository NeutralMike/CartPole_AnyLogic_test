import tensorflow as tf
import numpy as np
from tensorflow.keras import layers


class Agent:
    lr = 1e-3

    class History:
        states = []
        actions = []
        rewards = []
        saved_history = []

        def write(self, state, action, predicted_rewards):
            self.states.append(state)
            self.actions.append(action)
            self.rewards.append(predicted_rewards)

        def save(self):
            self.saved_history.extend(zip(self.states, self. actions, self.rewards))
            self.states = []
            self.actions = []
            self.rewards = []

    def __init__(self, num_inputs, num_actions, num_hidden=128):
        inputs = layers.Input(shape=(num_inputs,))
        common = layers.Dense(num_hidden, activation="relu")(inputs)
        action = layers.Dense(num_actions, activation="softmax")(common)
        output = layers.Dense(1)(common)

        self.dqn = tf.keras.Model(inputs=inputs, outputs=[action, output])
        self.dqn.compile(tf.optimizers.Adam(self.lr), loss=tf.keras.losses.MeanSquaredError())
        self.history = self.History()

    def predict(self, state):
        return self.dqn.predict(np.reshape(np.array(state), (1, -1)))[0][0]

    def retrain(self, gamma):
        for i in range(len(self.history.rewards)-2, -1, -1):
            self.history.rewards[i][self.history.actions[i]] +=  (self.history.rewards[i+1][self.history.actions[i+1]])
        self.dqn.fit(np.array(self.history.states), np.array(self.history.rewards), epochs=1, verbose=0)
        self.history.save()

    def update(self, state, action, predicted_rewards, reward, new_state, done):
        if not done:
            reward = reward + (np.amax(self.predict(new_state)))
        predicted_rewards[action] = reward
        # print(state)
        print(predicted_rewards)
        self.dqn.fit(np.array([state]), np.array([predicted_rewards]), epochs=1, verbose=0)
