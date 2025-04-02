"""
agent.py \n
Option-Critic Architecture
"""

import numpy as np


class OptionCriticAgent:
    def __init__(
        self,
        n_states,
        n_actions,
        n_options,
        alpha=0.1,  # learning rate
        beta=0.01,  # policy/termination learning rate
        gamma=0.99,  # discount factor
        epsilon=0.1,  # epsilon for exploration
        temperature=1.0,  # softmax temperature for option policies
    ):
        self.nS = n_states
        self.nA = n_actions
        self.nO = n_options
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.epsilon = epsilon
        self.temperature = temperature

        # Intra-option Q(s, omega, a): shape = [nS, nO, nA]
        self.Q_U = np.zeros((n_states, n_options, n_actions))

        # Option policies pi_omega(a|s): shape = [nO, nS, nA]
        # Start uniform or random
        self.option_policies = np.ones((n_options, n_states, n_actions)) / n_actions

        # Termination functions beta_omega(s): shape = [nO, nS]
        # Start at random or 0.5
        self.beta_omega = 0.5 * np.ones((n_options, n_states))

        # Policy over options pi_Omega(omega|s): shape = [nS, nO]
        # Start uniform or random
        self.option_policy = np.ones((n_states, n_options)) / n_options

    def choose_option(self, state):
        """
        Epsilon-soft selection of an option from pi_Omega(omega|s).
        """
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.nO)
        else:
            # e.g. choose greedily w.r.t. option_policy, or do a softmax
            probs = self.option_policy[state]
            return np.random.choice(self.nO, p=probs)

    def choose_action(self, state, option):
        """
        Sample an action from the current option's policy pi_omega(a|s).
        """
        # Softmax or epsilon-greedy on self.option_policies[option][state]
        # Here we do a simple softmax for demonstration:
        preferences = self.option_policies[option, state]
        # numerical stability
        max_pref = np.max(preferences)
        exp_prefs = np.exp((preferences - max_pref) / self.temperature)
        action_probs = exp_prefs / np.sum(exp_prefs)
        return np.random.choice(self.nA, p=action_probs)

    def update(self, s, o, a, r, s_next, done, next_o=None):
        """
        Perform the core tabular updates for:
            1. Intra-option Q
            2. Option policy improvement
            3. Termination function
            4. Option policy over options
        """
        # 1. Intra-option Q update
        #   delta = r + gamma * [ (1 - beta_omega(o, s_next)) * max_a' Q_U(s_next, o, a')
        #                         + beta_omega(o, s_next) * max_{o'} max_a' Q_U(s_next, o', a') ]
        #         - Q_U(s, o, a)
        q_val = self.Q_U[s, o, a]

        # Max over actions for continuing the same option
        max_Q_U_same_option = np.max(self.Q_U[s_next, o])

        # Max over all options, actions for switching
        max_Q_U_all_options = np.max(self.Q_U[s_next])

        beta_val = self.beta_omega[o, s_next]
        if done:
            target = r  # no future value
        else:
            target = r + self.gamma * (
                (1 - beta_val) * max_Q_U_same_option + beta_val * max_Q_U_all_options
            )

        delta = target - q_val
        self.Q_U[s, o, a] += self.alpha * delta

        # 2. Option policy improvement (tabular approach)
        #   We want to push pi_omega(a|s) toward actions that maximize Q_U(s, o, a).
        #   In a strict tabular sense, you could do an epsilon-greedy or a small gradient step.
        #   Here, let's do a simple approach that nudges probabilities towards the best actions:
        best_a = np.argmax(self.Q_U[s, o])
        for action_idx in range(self.nA):
            if action_idx == best_a:
                self.option_policies[o, s, action_idx] += self.beta * (
                    1.0 - self.option_policies[o, s, action_idx]
                )
            else:
                self.option_policies[o, s, action_idx] += self.beta * (
                    0.0 - self.option_policies[o, s, action_idx]
                )
        # re-normalize
        self.option_policies[o, s] /= np.sum(self.option_policies[o, s] + 1e-8)

        # 3. Termination function update
        #   Typically, we want beta_omega(s) to be high if continuing is less valuable.
        #   A common approach is a gradient step in the direction that lowers termination if Q_U is higher.
        #   For tabular, do a simple rule: if continuing the option is better than switching, lower beta.
        #   If switching is better, raise beta.
        #   i.e. gradient sign:  d/d beta_omega(s) [ Q_U(s, o) - max_{o'} Q_U(s, o') ]
        Q_continue = np.max(self.Q_U[s_next, o]) if not done else 0.0
        Q_switch = np.max(self.Q_U[s_next]) if not done else 0.0
        advantage = Q_switch - Q_continue
        # If advantage > 0 => better to switch => increase beta
        # If advantage < 0 => better to continue => decrease beta
        self.beta_omega[o, s] += self.beta * (1.0 if advantage > 0 else -1.0)
        # clamp to [0, 1]
        self.beta_omega[o, s] = np.clip(self.beta_omega[o, s], 0.0, 1.0)

        # 4. Option-policy over options improvement
        #   We can do an epsilon-greedy or softmax wrt. the value of each option from state s.
        #   A typical approach: define Q_\Omega(s, o) = max_a Q_U(s, o, a).
        #   Then pick the best option with prob 1 - epsilon, or random otherwise.
        Q_omega_values = np.max(self.Q_U[s], axis=1)  # shape = [nO]
        best_o = np.argmax(Q_omega_values)
        for option_idx in range(self.nO):
            if option_idx == best_o:
                self.option_policy[s, option_idx] = (
                    1.0 - self.epsilon + (self.epsilon / self.nO)
                )
            else:
                self.option_policy[s, option_idx] = self.epsilon / self.nO

    def should_terminate(self, state, option):
        """
        Draw from Bernoulli with parameter beta_omega(option, state).
        """
        return np.random.rand() < self.beta_omega[option, state]
