import numpy as np
from copy import deepcopy

from ..multiagentenv import MultiAgentEnv

class DoubleReferenceGame(MultiAgentEnv):
    def __init__(
            self,
            num_buttons = 5,
            seed = None,
            key = None,
            time_limit = 100,
            ood_mode = None,    # None | "train" | "test"
            ood_k = None,
        ):
        self.n_agents = 2
        self.num_buttons = num_buttons
        np.random.seed(seed)
        self.seed = seed
        self.episode_limit = time_limit
        if ood_mode is None:
            self.prob_mass = [1 / num_buttons] * num_buttons
        elif ood_mode == "train":
            self.prob_mass = ([1 / (num_buttons - 1)] * (num_buttons - 1)) + [0.0]
        elif ood_mode == "test":
            assert ood_k is not None
            assert ood_k <= 1.0 and ood_k >= 0.0
            self.prob_mass = (
                ([(1 - ood_k) / (num_buttons - 1)] * (num_buttons - 1)) + [float(ood_k)]
            )

    def step(self, actions):
        """ Returns reward, terminated, info """
        success = (self.on_buttons[1, actions[0]] == 1) and (self.on_buttons[0, actions[1]] == 1)
        self.success_count += int(success)
        self.t += 1
        done = (self.t >= self.episode_limit)
        info = dict(success=self.success_count)
        if self.chose_last_button:
            info["last_button_sr"] = self.success_count
        return [float(success)] * self.n_agents, done, info

    def get_obs(self):
        """ Returns all agent observations in a list """
        return deepcopy([self.on_buttons[a] for a in range(self.n_agents)])

    def get_obs_agent(self, agent_id):
        """ Returns observation for agent_id """
        return deepcopy(self.on_buttons[agent_id])

    def get_obs_size(self):
        """ Returns the shape of the observation """
        return self.num_buttons

    def get_state(self):
        return deepcopy(self.on_buttons.reshape(-1))

    def get_state_size(self):
        """ Returns the shape of the state"""
        return self.num_buttons * self.n_agents

    def get_avail_actions(self):
        return [([1] * self.num_buttons) for _ in range(self.n_agents)]

    def get_avail_agent_actions(self, agent_id):
        """ Returns the available actions for agent_id """
        return [1] * self.num_buttons

    def get_total_actions(self):
        """ Returns the total number of actions an agent could ever take """
        return self.num_buttons

    def reset(self):
        """ Returns initial observations and states"""
        self.on_buttons = np.zeros((self.n_agents, self.num_buttons), dtype=np.float32)
        choice = np.random.multinomial(1, self.prob_mass, size=self.n_agents).argmax(1)
        self.chose_last_button = (
            (choice[0] == self.num_buttons - 1) or (choice[1] == self.num_buttons - 1)
        )
        self.on_buttons[range(self.n_agents), choice] = 1.0
        self.success_count = 0
        self.t = 0
        return self.get_obs(), self.get_state()

    def render(self):
        raise NotImplementedError

    def close(self):
        return

    def seed(self):
        return self.seed

    def save_replay(self):
        raise NotImplementedError

    def get_stats(self):
        return dict()