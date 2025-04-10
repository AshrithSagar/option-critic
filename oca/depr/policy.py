"""
utils/policy.py \n
Policy utils
"""

from abc import ABC, abstractmethod

import numpy as np


class PolicyBaseABC(ABC):
    @abstractmethod
    def select_action(self, state):
        pass

    @abstractmethod
    def update(self, *args, **kwargs):
        pass


class PolicyOverOptions(PolicyBaseABC):
    def __init__(self, num_options, epsilon=0.1):
        self.num_options = num_options
        self.epsilon = epsilon
        self.q_values = {}  # (state, option) -> Q

    def select_action(self, state):
        q_vals = np.array(
            [self.q_values.get((state, o), 0.0) for o in range(self.num_options)]
        )
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.num_options)
        return np.argmax(q_vals)

    def update(self, state, option, target, alpha=0.1):
        key = (state, option)
        self.q_values[key] = self.q_values.get(key, 0.0) + alpha * (
            target - self.q_values.get(key, 0.0)
        )


class IntraOptionPolicy(PolicyBaseABC):
    def __init__(self, num_actions, theta=None):
        self.num_actions = num_actions
        self.theta = theta or {}  # (option, state, action) -> preference score

    def softmax(self, prefs):
        e = np.exp(prefs - np.max(prefs))
        return e / e.sum()

    def select_action(self, state, option):
        prefs = np.array(
            [self.theta.get((option, state, a), 0.0) for a in range(self.num_actions)]
        )
        probs = self.softmax(prefs)
        return np.random.choice(self.num_actions, p=probs)

    def update(self, state, option, action, advantage, alpha=0.1):
        for a in range(self.num_actions):
            key = (option, state, a)
            grad = 1.0 if a == action else 0.0
            probs = self.softmax(
                np.array(
                    [
                        self.theta.get((option, state, i), 0.0)
                        for i in range(self.num_actions)
                    ]
                )
            )
            self.theta[key] = (
                self.theta.get(key, 0.0) + alpha * (grad - probs[a]) * advantage
            )


class EpsilonGreedyPolicy(PolicyBaseABC):
    def __init__(self, num_actions, epsilon=0.1):
        self.num_actions = num_actions
        self.epsilon = epsilon
        self.q_values = {}  # (state, action) -> Q

    def select_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.num_actions)
        q_vals = np.array(
            [self.q_values.get((state, a), 0.0) for a in range(self.num_actions)]
        )
        return np.argmax(q_vals)

    def update(self, state, action, target, alpha=0.1):
        key = (state, action)
        self.q_values[key] = self.q_values.get(key, 0.0) + alpha * (
            target - self.q_values.get(key, 0.0)
        )


class BoltzmannPolicy(PolicyBaseABC):
    def __init__(self, num_actions, tau=1.0):
        self.num_actions = num_actions
        self.tau = tau
        self.q_values = {}  # (state, action) -> Q

    def softmax(self, q_vals):
        e = np.exp(q_vals / self.tau)
        return e / e.sum()

    def select_action(self, state):
        q_vals = np.array(
            [self.q_values.get((state, a), 0.0) for a in range(self.num_actions)]
        )
        probs = self.softmax(q_vals)
        return np.random.choice(self.num_actions, p=probs)

    def update(self, state, action, target, alpha=0.1):
        key = (state, action)
        self.q_values[key] = self.q_values.get(key, 0.0) + alpha * (
            target - self.q_values.get(key, 0.0)
        )
