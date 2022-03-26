#!/usr/bin/env python3
"""Use epsilon-greedy to determine the next action:"""
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