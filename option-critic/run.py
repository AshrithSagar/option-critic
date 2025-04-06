"""
run.py \n
Run the option-critic agent
"""

import gymnasium as gym
from agent import OptionCriticAgent


def main():
    env = gym.make("Taxi-v3", render_mode="human")
    nS = env.observation_space.n
    nA = env.action_space.n

    agent = OptionCriticAgent(n_states=nS, n_actions=nA, n_options=2)

    n_episodes = 500
    max_steps = 200
    for episode in range(n_episodes):
        s, _ = env.reset()
        env.render()
        o = agent.choose_option(s)
        total_reward = 0

        for t in range(max_steps):
            # Choose action from current option's policy
            a = agent.choose_action(s, o)
            s_next, r, done, truncated, info = env.step(a)
            total_reward += r

            agent.update(s, o, a, r, s_next, done)

            # Check whether to terminate the option
            if done:
                break
            if agent.should_terminate(s_next, o):
                # Choose a new option if it terminates
                o = agent.choose_option(s_next)

            s = s_next

        print(f"Episode {episode+1}, total reward = {total_reward}")

    env.close()


if __name__ == "__main__":
    main()
