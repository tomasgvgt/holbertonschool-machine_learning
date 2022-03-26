#!/usr/bin/env python3
"""Initialize the q-table"""
import gym
import numpy as np


def q_init(env):
    """
    Initializes the Q-table:
    env is the FrozenLakeEnv instance
    Returns: the Q-table as a numpy.ndarray of zeros
    """
    actions = env.action_space.n
    states = env.observation_space.n
    q_table = np.zeros((states, actions))
    return q_table