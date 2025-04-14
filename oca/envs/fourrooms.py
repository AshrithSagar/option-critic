"""
oca/fourrooms.py \n
Fourrooms environment
"""

import logging
from typing import Any

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from gymnasium import spaces

logging.getLogger("matplotlib").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


class FourRoomsEnv(gym.Env):
    """Fourrooms environment"""

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 50,
        "torch": True,
    }

    def __init__(self, **kwargs):
        layout = """\
wwwwwwwwwwwww
w     w     w
w     w     w
w           w
w     w     w
w     w     w
ww wwww     w
w     www www
w     w     w
w     w     w
w           w
w     w     w
wwwwwwwwwwwww
"""
        self.occupancy = np.array(
            [
                list(map(lambda c: 1 if c == "w" else 0, line))
                for line in layout.splitlines()
            ]
        )

        # From any state the agent can perform one of four actions: up, down, left or right
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(np.sum(self.occupancy == 0),)
        )

        self.directions = [
            np.array((-1, 0)),
            np.array((1, 0)),
            np.array((0, -1)),
            np.array((0, 1)),
        ]
        self.rng = np.random.RandomState(1234)

        self.tostate = {}
        statenum = 0
        for i in range(13):
            for j in range(13):
                if self.occupancy[i, j] == 0:
                    self.tostate[(i, j)] = statenum
                    statenum += 1
        self.tocell = {v: k for k, v in self.tostate.items()}

        self.goal = 62  # East doorway
        self.init_states = list(range(self.observation_space.shape[0]))
        self.init_states.remove(self.goal)
        self.ep_steps = 0

        self.render_mode = kwargs.get("render_mode", None)
        self.render_args = kwargs["render_args"] or {
            "show_goal": True,
            "force_update": True,
            "pause_delay": 1e-3,
        }

    def empty_around(self, cell):
        avail = []
        for action in range(self.action_space.n):
            nextcell = tuple(cell + self.directions[action])
            if not self.occupancy[nextcell]:
                avail.append(nextcell)
        return avail

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        state = self.rng.choice(self.init_states)
        self.currentcell = self.tocell[state]
        self.ep_steps = 0
        return self.get_state(state), {}

    def switch_goal(self):
        prev_goal = self.goal
        self.goal = self.rng.choice(self.init_states)
        self.init_states.append(prev_goal)
        self.init_states.remove(self.goal)
        assert prev_goal in self.init_states
        assert self.goal not in self.init_states

    def get_state(self, state):
        s = np.zeros(self.observation_space.shape[0], dtype=np.float32)
        s[state] = 1
        return s

    def render(self):
        if self.render_mode == "human":
            self.render_human()

    def render_human(self):
        current_grid = np.array(self.occupancy)

        # Mark the agent's current position with -1 (or any other identifier)
        current_grid[self.currentcell[0], self.currentcell[1]] = -1

        if self.render_args["show_goal"]:
            # Mark goal's position
            goal_cell = self.tocell[self.goal]
            current_grid[goal_cell[0], goal_cell[1]] = 2

        # If the plot has not been created yet, create it
        if not hasattr(self, "im"):  # Check if the im object exists
            self.fig, self.ax = plt.subplots()
            self.im = self.ax.imshow(current_grid, cmap="gray", interpolation="nearest")
            self.ax.set_title("Fourrooms Environment")
            self.cbar = self.fig.colorbar(self.im, ax=self.ax, label="Grid values")
            plt.ion()
        elif self.render_args["force_update"]:
            # Update the existing plot only if force_update is True
            self.im.set_data(current_grid)
            self.cbar.update_ticks()  # Update the colorbar ticks if necessary

        if self.render_args["force_update"]:
            plt.draw()  # Redraw the updated plot without blocking
            plt.pause(
                self.render_args["pause_delay"]
            )  # Pause briefly to allow the plot to update
            plt.show(block=False)

    def step(self, action):
        """
        The agent can perform one of four actions,
        up, down, left or right, which have a stochastic effect. With probability 2/3, the actions
        cause the agent to move one cell in the corresponding direction, and with probability 1/3,
        the agent moves instead in one of the other three directions, each with 1/9 probability. In
        either case, if the movement would take the agent into a wall then the agent remains in the
        same cell.
        We consider a case in which rewards are zero on all state transitions.
        """
        self.ep_steps += 1

        nextcell = tuple(self.currentcell + self.directions[action])
        if not self.occupancy[nextcell]:
            if self.rng.uniform() < 1 / 3.0:
                empty_cells = self.empty_around(self.currentcell)
                self.currentcell = empty_cells[self.rng.randint(len(empty_cells))]
            else:
                self.currentcell = nextcell

        state = self.tostate[self.currentcell]
        done = state == self.goal
        reward = float(done)

        if not done and self.ep_steps >= 1000:
            truncated = True
            reward = 0.0
        else:
            truncated = False

        return self.get_state(state), reward, done, truncated, {}
