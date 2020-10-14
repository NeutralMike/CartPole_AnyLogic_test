import numpy as np
from py4j.java_gateway import JavaGateway
from dqn_agent import Agent

gateway = JavaGateway()
env = gateway.entry_point.getCartPole()

max_episodes = 10000
max_steps_per_episode = 1000
reward_threshold = 195
running_reward = 0

num_inputs = 4
num_actions = 2
agent = Agent(num_inputs, num_actions)

for episode_i in range(1, max_episodes+1):
    env.Reset()
    episode_reward = 0

    for step_i in range(max_steps_per_episode):
        state = np.array(env.getState())
        prediction = agent.predict([state])[0]
        action = np.argmax(prediction)

        if np.random.rand() <= 0.1:
            action = np.random.randint(0, num_actions)

        env.Step(int(action))

        reward = env.getReward()
        episode_reward += reward
        done = env.isDone()

        prediction[action] = reward
        agent.write_history(state, action, prediction)
        if done:
            break

    running_reward = episode_reward * 0.01 + running_reward * .99

    agent.retrain()

    if (episode_i) % 10 == 0:
        print('running reward: {:.2f} at episode {}'.format(running_reward, episode_i))
    if running_reward > reward_threshold:
        print('running reward: {:.2f} at episode {}'.format(running_reward, episode_i))
        break
