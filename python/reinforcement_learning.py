import numpy as np
from py4j.java_gateway import JavaGateway
from dqn_agent import Agent

gateway = JavaGateway()
env = gateway.entry_point.getCartPole()

gamma = 0.1  # Discount factor for past rewards
max_episodes = 1000
max_steps_per_episode = 1000
reward_threshold = 195
running_reward = 0
num_inputs = 4
num_actions = 2
agent= Agent(num_inputs, num_actions)


for episode_i in range(max_episodes):
    env.Reset()
    episode_reward = 0
    for step_i in range(max_steps_per_episode):
        state = list(env.getState())
        # action = np.random.choice(num_actions, p=np.squeeze(agent.predict(state)[0]))
        predicted_rewards = agent.predict(list(env.getState()))
        # print(predicted_rewards)
        # action = np.random.choice(num_actions, p=np.squeeze(predicted_rewards/np.sum(predicted_rewards)))
        action = np.argmax(predicted_rewards)
        if np.random.rand() <= 0.1:
            action = np.random.randint(0, num_actions)
        # print(action)
        env.Step(int(action))
        reward = env.getReward()
        episode_reward += reward
        running_reward = episode_reward * 0.01 + running_reward * .99
        done = env.isDone()
        predicted_rewards[action] = reward
        agent.history.write(state, action, predicted_rewards)
        if done:
            print(step_i)
            break
    agent.retrain((1 - episode_i/max_episodes))
    if episode_i % 10 == 0:
        print('running reward: {:.2f} at episode {}'.format(running_reward, episode_i))
    if running_reward > reward_threshold:
        break
