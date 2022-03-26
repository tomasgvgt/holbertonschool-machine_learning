#!/usr/bin/env python3
"""Perform Q-learning"""
import numpy as np

def epsilon_greedy(Q, state, epsilon):
    """
    Use epsilon-greedy to determine the next action:
    Q is a numpy.ndarray containing the q-table
    state is the current state
    epsilon is the epsilon to use for the calculation
    You should sample p with numpy.random.uniformn to determine if your
        algorithm should explore or exploit
    If exploring, you should pick the next action with numpy.random.randint
        from all possible actions
    Returns: the next action index
    """
    actions = Q.shape[1]
    random = np.random.uniform(0, 1)
    if epsilon >= random:
        action_i = np.random.randint(0, actions)
    else:
        action_i = Q.argmax(axis=1)[state]

    return action_i


def train(env, Q, episodes=5000, max_steps=100, alpha=0.1,
          gamma=0.99, epsilon=1, min_epsilon=0.1, epsilon_decay=0.05):
    """
    Performs Q-learning:

    env is the FrozenLakeEnv instance
    Q is a numpy.ndarray containing the Q-table
    episodes is the total number of episodes to train over
    max_steps is the maximum number of steps per episode
    alpha is the learning rate
    gamma is the discount rate
    epsilon is the initial threshold for epsilon greedy
    min_epsilon is the minimum value that epsilon should decay to
    epsilon_decay is the decay rate for updating epsilon between episodes
    When the agent falls in a hole, the reward should be updated to be -1
    Returns: Q, total_rewards
        Q is the updated Q-table
        total_rewards is a list containing the rewards per episode
    """
    rewards = []
    for episode in range(episodes):
        state = env.reset()
        episode_reward = 0
        done = False
        for step in range(max_steps):
            action = epsilon_greedy(Q, state, epsilon)
            new_state, reward, done, info = env.step(action)
            if reward == 0 and done:
                reward = -1
            Q[state, action] = (Q[state, action] * (1 - alpha) + alpha * (reward + gamma * np.max(Q[new_state, :])))
            state = new_state
            episode_reward += reward
            if done is True:
                break
        epsilon = (min_epsilon + (1 - min_epsilon) * np.exp(-epsilon_decay * episode))
        rewards.append(episode_reward)
    return Q, rewards
