import logging

from collections import defaultdict, OrderedDict
import gym
from gym import spaces

from robotic_warehouse.utils import MultiAgentActionSpace, MultiAgentObservationSpace

from enum import Enum
import numpy as np
import cv2

from typing import List, Tuple, Optional, Dict

import networkx as nx

_AXIS_Z = 0
_AXIS_Y = 1
_AXIS_X = 2

_COLLISION_LAYERS = 3

_LAYER_AGENTS = 0
_LAYER_SHELFS = 1
_LAYER_OBSTACLES = 2


class _VectorWriter:
    def __init__(self, size: int):
        self.vector = np.zeros(size, dtype=np.float32)
        self.idx = 0

    def write(self, data):
        data_size = len(data)
        self.vector[self.idx : self.idx + data_size] = data
        self.idx += data_size

    def skip(self, bits):
        self.idx += bits


class Action(Enum):
    NOOP = 0
    FORWARD = 1
    LEFT = 2
    RIGHT = 3
    TOGGLE_LOAD = 4


class Direction(Enum):
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3


class RewardType(Enum):
    GLOBAL = 0
    INDIVIDUAL = 1
    TWO_STAGE = 2


class Entity:
    def __init__(self, id_: int, y: int, x: int):
        self.id = id_
        self.prev_x = None
        self.prev_y = None
        self.y = y
        self.x = x


class Agent(Entity):
    counter = 0

    def __init__(self, y: int, x: int, dir_: Direction, msg_bits: int):
        Agent.counter += 1
        super().__init__(Agent.counter, y, x)
        self.dir = dir_
        self.message = np.zeros(msg_bits)
        self.req_action: Optional[Action] = None
        self.carrying_shelf: Optional[Shelf] = None
        self.canceled_action = None
        self.has_delivered = False

    @property
    def collision_layers(self):
        if self.loaded:
            return (_LAYER_AGENTS, _LAYER_SHELFS)
        else:
            return (_LAYER_AGENTS,)

    def req_location(self, grid_size) -> Tuple[int, int]:
        if self.req_action != Action.FORWARD:
            return self.y, self.x
        elif self.dir == Direction.UP:
            return max(0, self.y - 1), self.x
        elif self.dir == Direction.DOWN:
            return min(grid_size[0] - 1, self.y + 1), self.x
        elif self.dir == Direction.LEFT:
            return self.y, max(0, self.x - 1)
        elif self.dir == Direction.RIGHT:
            return self.y, min(grid_size[1] - 1, self.x + 1)

        raise ValueError(
            f"Direction is {self.dir}. Should be one of {[v for v in Direction]}"
        )

    def req_direction(self) -> Direction:
        wraplist = [Direction.UP, Direction.RIGHT, Direction.DOWN, Direction.LEFT]
        if self.req_action == Action.RIGHT:
            return wraplist[(wraplist.index(self.dir) + 1) % len(wraplist)]
        elif self.req_action == Action.LEFT:
            return wraplist[(wraplist.index(self.dir) - 1) % len(wraplist)]
        else:
            return self.dir


class Shelf(Entity):
    counter = 0

    def __init__(self, y, x):
        Shelf.counter += 1
        super().__init__(Shelf.counter, y, x)
        self.remain_time = None

    def dec_remain_time(self):
        self.remain_time -= 1

    def set_remain_time(self, remain_time):
        self.remain_time = remain_time

    def disable_time(self):
        self.remain_time = None

    @property
    def collision_layers(self):
        return (_LAYER_SHELFS,)


class WarehouseToy(gym.Env):

    metadata = {"render.modes": ["human", "rgb_array"]}

    def __init__(
        self,
        n_agents,
        msg_bits=0,
        sensor_range=1,
        max_steps=100,
        reward_type=RewardType.GLOBAL,
        fast_obs=True,
        reward_share_wgt = 1.0,
    ):

        self.grid_size = (
            2 * n_agents,   # y
            n_agents + 1,   # x
        )

        self.n_agents = n_agents
        self.msg_bits = msg_bits
        self.sensor_range = sensor_range
        self.reward_type = reward_type
        self.reward_share_wgt = reward_share_wgt

        self._cur_inactive_steps = None
        self._cur_steps = 0
        self.max_steps = max_steps

        self.grid = np.zeros((_COLLISION_LAYERS, *self.grid_size), dtype=np.int32)

        sa_action_space = [len(Action), *msg_bits * (2,)]
        if len(sa_action_space) == 1:
            sa_action_space = spaces.Discrete(sa_action_space[0])
        else:
            sa_action_space = spaces.MultiDiscrete(sa_action_space)
        self.action_space = spaces.Tuple(tuple(n_agents * [sa_action_space]))

        self.request_queue_size = 1
        self.request_queue = []

        self.agents: List[Agent] = []

        # TODO: doesn't need to be list; it's determined to be just one tile
        # by the way, I changed this to y,x format rather than x,y
        self.goals: List[Tuple[int, int]] = [
            (2 * n_agents - 1, 1)
        ]

        # Used for tests
        self._obs_bits_for_self = 4 + len(Direction)
        self._obs_bits_per_agent = 1 + len(Direction) + self.msg_bits
        self._obs_bits_per_shelf = 2
        _obs_bits_for_obstacles = 1

        self._obs_sensor_locations = (1 + 2 * self.sensor_range) ** 2

        self._obs_length = (
            self._obs_bits_for_self + self._obs_sensor_locations * (
                self._obs_bits_per_agent + self._obs_bits_per_shelf +
                _obs_bits_for_obstacles
            )
        )

        # default values:
        self.fast_obs = None
        self.observation_space = None
        self._use_slow_obs()

        # for performance reasons we
        # can flatten the obs vector
        if fast_obs:
            self._use_fast_obs()

        self.grid[_LAYER_OBSTACLES, :, :] = 1
        self.grid[_LAYER_OBSTACLES, n_agents, :] = 0
        self.grid[_LAYER_OBSTACLES, :, 1] = 0

        self.num_delivers = 0

    def _use_slow_obs(self):
        self.fast_obs = False
        # if self.shelf_time_limit is not None:
            # remain_time_space = spaces.MultiDiscrete([self.shelf_time_limit + 1]),
        # else:
        #     remain_time_space = spaces.MultiDiscrete([1])
        self_space = spaces.Dict(
            OrderedDict({
                "location": spaces.MultiDiscrete(
                    [self.grid_size[1], self.grid_size[0]]
                ),
                "carrying_shelf": spaces.MultiDiscrete([2]),
                "direction": spaces.Discrete(4),
                "on_highway": spaces.MultiDiscrete([2]),
            })
        )

        sensors_space = spaces.Tuple(
            self._obs_sensor_locations
            * (
                spaces.Dict(
                    OrderedDict({
                        "has_agent": spaces.MultiDiscrete([2]),
                        "direction": spaces.Discrete(4),
                        "local_message": spaces.MultiBinary(self.msg_bits),
                        "has_shelf": spaces.MultiDiscrete([2]),
                        "shelf_requested": spaces.MultiDiscrete([2]),
                        "has_obstacle": spaces.MultiDiscrete([2]),
                    })
                ),
            )
        )

        self.observation_space = spaces.Tuple(
            tuple([
                spaces.Dict(
                    OrderedDict({
                        "self": self_space,
                        "sensors": sensors_space,
                    })
                )
                for _ in range(self.n_agents)
            ])
        )

    def _use_fast_obs(self):
        if self.fast_obs:
            return

        self.fast_obs = True
        ma_spaces = []
        for sa_obs in self.observation_space:
            flatdim = spaces.flatdim(sa_obs)
            ma_spaces += [
                spaces.Box(
                    low=-float("inf"),
                    high=float("inf"),
                    shape=(flatdim,),
                    dtype=np.float32,
                )
            ]

        self.observation_space = spaces.Tuple(tuple(ma_spaces))

    def _is_highway(self, y: int, x: int) -> bool:
        return (
            (y == self.n_agents) or (y > self.n_agents and x == 1)
        )
    
    def _is_obstacle_or_wall(self, y, x):
        return (
            (y < 0) or (y >= self.grid_size[0])
            or (x < 0) or (x >= self.grid_size[1])
            or (y != self.n_agents and x != 1)
        )

    def _make_obs(self, agent):

        y_scale, x_scale = self.grid_size[0] - 1, self.grid_size[1] - 1

        min_x = agent.x - self.sensor_range
        max_x = agent.x + self.sensor_range + 1

        min_y = agent.y - self.sensor_range
        max_y = agent.y + self.sensor_range + 1
        # sensors
        if (
            (min_x < 0)
            or (min_y < 0)
            or (max_x > self.grid_size[1])
            or (max_y > self.grid_size[0])
        ):
            padded_agents = np.pad(
                self.grid[_LAYER_AGENTS], self.sensor_range, mode="constant"
            )
            padded_shelfs = np.pad(
                self.grid[_LAYER_SHELFS], self.sensor_range, mode="constant"
            )
            padded_obstacles = np.pad(
                self.grid[_LAYER_OBSTACLES], self.sensor_range, mode="constant",
                constant_values=1
            )
            # + self.sensor_range due to padding
            min_x += self.sensor_range
            max_x += self.sensor_range
            min_y += self.sensor_range
            max_y += self.sensor_range

        else:
            padded_agents = self.grid[_LAYER_AGENTS]
            padded_shelfs = self.grid[_LAYER_SHELFS]
            padded_obstacles = self.grid[_LAYER_OBSTACLES]


        agents = padded_agents[min_y:max_y, min_x:max_x].reshape(-1)
        shelfs = padded_shelfs[min_y:max_y, min_x:max_x].reshape(-1)
        obstacles = padded_obstacles[min_y:max_y, min_x:max_x].reshape(-1)

        if self.fast_obs:
            obs = _VectorWriter(self.observation_space[agent.id - 1].shape[0])

            obs.write([agent.y, agent.x, int(agent.carrying_shelf is not None)])
            direction = np.zeros(4)
            direction[agent.dir.value] = 1.0
            obs.write(direction)
            obs.write([int(self._is_highway(agent.y, agent.x))])


            for i, (id_agent, id_shelf, has_obstacle) in enumerate(zip(agents, shelfs, obstacles)):
                if id_agent == 0:
                    obs.skip(1)
                    obs.write([1.0])
                    obs.skip(3 + self.msg_bits)
                else:
                    obs.write([1.0])
                    direction = np.zeros(4)
                    direction[self.agents[id_agent - 1].dir.value] = 1.0
                    obs.write(direction)
                    if self.msg_bits > 0:
                        obs.write(self.agents[id_agent - 1].message)
                if id_shelf == 0:
                    obs.skip(2)
                else:
                    obs.write(
                        [1.0, int(self.shelfs[id_shelf - 1] in self.request_queue)]
                    )
                obs.write([float(has_obstacle)])

            return obs.vector

        # --- self data
        obs = {}
        obs["self"] = {
            "location": np.array([agent.y, agent.x]),
            "carrying_shelf": [int(agent.carrying_shelf is not None)],
            "direction": agent.dir.value,
            "on_highway": [int(self._is_highway(agent.y, agent.x))],
        }
        # --- sensor data
        obs["sensors"] = tuple({} for _ in range(self._obs_sensor_locations))

        # find neighboring agents
        for i, id_ in enumerate(agents):
            if id_ == 0:
                obs["sensors"][i]["has_agent"] = [0]
                obs["sensors"][i]["direction"] = 0
                obs["sensors"][i]["local_message"] = self.msg_bits * [0]
            else:
                obs["sensors"][i]["has_agent"] = [1]
                obs["sensors"][i]["direction"] = self.agents[id_ - 1].dir.value
                obs["sensors"][i]["local_message"] = self.agents[id_ - 1].message

        # find neighboring shelfs:
        for i, id_ in enumerate(shelfs):
            if id_ == 0:
                obs["sensors"][i]["has_shelf"] = [0]
                obs["sensors"][i]["shelf_requested"] = [0]
            else:
                obs["sensors"][i]["has_shelf"] = [1]
                obs["sensors"][i]["shelf_requested"] = [
                    int(self.shelfs[id_ - 1] in self.request_queue)
                ]

        for i, has_obstacle in enumerate(obstacles):
            obs["sensors"][i]["has_obstacle"] = [int(has_obstacle)]

        return obs

    def _recalc_grid(self):
        self.grid[_LAYER_AGENTS, :] = 0
        self.grid[_LAYER_SHELFS, :] = 0
        for s in self.shelfs:
            self.grid[_LAYER_SHELFS, s.y, s.x] = s.id

        for a in self.agents:
            self.grid[_LAYER_AGENTS, a.y, a.x] = a.id

    def _shelf_at_loc(self, y, x):
        shelf_id = self.grid[_LAYER_SHELFS, y, x]
        return self.shelfs[shelf_id - 1] if shelf_id != 0 else None
    
    def _agent_at_loc(self, y, x):
        agent_id = self.grid[_LAYER_AGENTS, y, x]
        return self.agents[agent_id - 1] if agent_id != 0 else None

    def reset(self):
        Shelf.counter = 0
        Agent.counter = 0
        self._cur_inactive_steps = 0
        self._cur_steps = 0

        # n_xshelf = (self.grid_size[1] - 1) // 3
        # n_yshelf = (self.grid_size[0] - 2) // 9

        # make the shelfs
        self.shelfs = [
            Shelf(y, x)
            for y, x in zip(
                np.indices(self.grid_size)[0].reshape(-1),
                np.indices(self.grid_size)[1].reshape(-1),
            )
            if not self._is_highway(y, x) and not self._is_obstacle_or_wall(y, x)
        ]

        # spawn agents at random locations
        agent_locs = [(self.n_agents, 0)]
        agent_locs += [(self.n_agents, x) for x in range(2, self.n_agents + 1)]
        # and direction
        agent_dirs = np.random.choice([d for d in Direction], size=self.n_agents)

        self.agents = []
        for idx in range(len(agent_locs)):
            self.agents.append(
                Agent(
                    agent_locs[idx][0],
                    agent_locs[idx][1],
                    agent_dirs[idx],
                    self.msg_bits
                )
            )

        self._recalc_grid()

        req_shelf = self._shelf_at_loc(0, 1)
        assert req_shelf is not None
        self.request_queue = [req_shelf]

        self.num_delivers = 0

        return tuple([self._make_obs(agent) for agent in self.agents])

    def step(
        self, actions: List[Action]
    ) -> Tuple[List[np.ndarray], List[float], List[bool], Dict]:
        assert len(actions) == len(self.agents)

        # print(f"{self.grid[0]}")
        # print(f"{self.grid[1]}")
        # print(f"{actions}")
        # for idx, agent in enumerate(self.agents):
        #     print(f"{agent.id, agent.x, agent.y, agent.dir, agent.carrying_shelf}")
        #     if agent.carrying_shelf is not None:
        #         print(f"carrying {agent.carrying_shelf.id}")
        #     print(f"{actions[idx]}")
        # for idx, shelf in enumerate(self.shelfs):
        #     print(f"{shelf.id, shelf.x, shelf.y}")

        for agent, action in zip(self.agents, actions):
            if self.msg_bits > 0:
                agent.req_action = Action(action[0])
                agent.message[:] = action[1:]
            else:
                agent.req_action = Action(action)

        # # stationary agents will certainly stay where they are
        # stationary_agents = [agent for agent in self.agents if agent.action != Action.FORWARD]

        # # forward agents will move only if they avoid collisions
        # forward_agents = [agent for agent in self.agents if agent.action == Action.FORWARD]
        commited_agents = set()

        G = nx.DiGraph()

        for agent in self.agents:
            start = agent.y, agent.x
            target = agent.req_location(self.grid_size)

            if (
                # agent.carrying_shelf
                self.grid[_LAYER_SHELFS, start[0], start[1]]
                and start != target
                and self.grid[_LAYER_SHELFS, target[0], target[1]]
                and not (
                    self.grid[_LAYER_AGENTS, target[0], target[1]]
                    and self.agents[
                        self.grid[_LAYER_AGENTS, target[0], target[1]] - 1
                    ].carrying_shelf
                )
            ):
                # there's a standing shelf at the target location
                # our agent is carrying a shelf so there's no way
                # this movement can succeed. Cancel it.
                # also block agent moving from block to block
                agent.req_action = Action.NOOP
                G.add_edge(start, start)
            # elif self.grid[_LAYER_OBSTACLES, target[0], target[1]] == 1:
            elif self._is_obstacle_or_wall(target[0], target[1]):
                # Cancel action if the agent would bump into an obstacle
                agent.req_action = Action.NOOP
                # assert self.grid[_LAYER_OBSTACLES, start[0], start[1]] == 0
                G.add_edge(start, start)
            else:
                G.add_edge(start, target)

        wcomps = [G.subgraph(c).copy() for c in nx.weakly_connected_components(G)]

        for comp in wcomps:
            try:
                # if we find a cycle in this component we have to
                # commit all nodes in that cycle, and nothing else
                cycle = nx.algorithms.find_cycle(comp)
                if len(cycle) == 2:
                    # we have a situation like this: [A] <-> [B]
                    # which is physically impossible. so skip
                    continue
                for edge in cycle:
                    start_node = edge[0]
                    agent_id = self.grid[_LAYER_AGENTS, start_node[0], start_node[1]]
                    if agent_id > 0:
                        commited_agents.add(agent_id)

            except nx.NetworkXNoCycle:

                longest_path = nx.algorithms.dag_longest_path(comp)
                for y, x in longest_path:
                    agent_id = self.grid[_LAYER_AGENTS, y, x]
                    if agent_id:
                        commited_agents.add(agent_id)

        commited_agents = set([self.agents[id_ - 1] for id_ in commited_agents])
        failed_agents = set(self.agents) - commited_agents

        for agent in failed_agents:
            assert agent.req_action == Action.FORWARD
            agent.req_action = Action.NOOP

        rewards = np.zeros(self.n_agents)

        for agent in self.agents:
            agent.prev_y, agent.prev_x = agent.y, agent.x

            if agent.req_action == Action.FORWARD:
                agent.y, agent.x = agent.req_location(self.grid_size)
                if agent.carrying_shelf:
                    agent.carrying_shelf.y, agent.carrying_shelf.x = agent.y, agent.x
            elif agent.req_action in [Action.LEFT, Action.RIGHT]:
                agent.dir = agent.req_direction()
            elif agent.req_action == Action.TOGGLE_LOAD and not agent.carrying_shelf:
                shelf_id = self.grid[_LAYER_SHELFS, agent.y, agent.x]
                if shelf_id:
                    agent.carrying_shelf = self.shelfs[shelf_id - 1]
            elif agent.req_action == Action.TOGGLE_LOAD and agent.carrying_shelf:
                if not self._is_highway(agent.y, agent.x):
                    agent.carrying_shelf = None
                    if agent.has_delivered and self.reward_type == RewardType.TWO_STAGE:
                        rewards[agent.id - 1] += 0.5
                        rewards[:agent.id - 1] += self.reward_share_wgt * 0.5
                        rewards[agent.id + 1:] += self.reward_share_wgt * 0.5

                    agent.has_delivered = False

        self._recalc_grid()

        shelf_delivered = False
        for y, x in self.goals:
            shelf_id = self.grid[_LAYER_SHELFS, y, x]
            if not shelf_id:
                continue
            shelf = self.shelfs[shelf_id - 1]

            if shelf not in self.request_queue:
                continue
            # a shelf was successfully delivered.
            shelf_delivered = True
            # remove from queue and replace it
            # new_request = np.random.choice(
            #     list(set(self.shelfs) - set(self.request_queue))
            # )
            # self.request_queue[self.request_queue.index(shelf)] = new_request
            # also reward the agents
            if self.reward_type == RewardType.GLOBAL:
                rewards += 1
            elif self.reward_type == RewardType.INDIVIDUAL:
                agent_id = self.grid[_LAYER_AGENTS, y, x]
                rewards[agent_id - 1] += 1
                rewards[:agent_id - 1] += self.reward_share_wgt
                rewards[agent_id + 1:] += self.reward_share_wgt
            elif self.reward_type == RewardType.TWO_STAGE:
                agent_id = self.grid[_LAYER_AGENTS, y, x]
                self.agents[agent_id - 1].has_delivered = True
                rewards[agent_id - 1] += 0.5
                rewards[:agent_id - 1] += self.reward_share_wgt * 0.5
                rewards[agent_id + 1:] += self.reward_share_wgt * 0.5

        if shelf_delivered:
            self.num_delivers += 1
        self._cur_steps += 1

        if shelf_delivered or (self.max_steps and self._cur_steps >= self.max_steps):
            dones = self.n_agents * [True]
        else:
            dones = self.n_agents * [False]

        new_obs = tuple([self._make_obs(agent) for agent in self.agents])
        info = {"num_delivers": self.num_delivers}
        return new_obs, list(rewards), dones, info

    def render(self, mode="human"):
        # if not self.renderer:
        #     from robotic_warehouse.rendering import Viewer

        #     self.renderer = Viewer(self.grid_size)
        # return self.renderer.render(self, return_rgb_array=mode == "rgb_array")
        return None

    def close(self):
        # if self.renderer:
        #     self.renderer.close()
        pass

    def seed(self, seed=None):
        ...

    def draw(self):
        width_config = {
            "tile_width": 30,
            "agent_rad": 10,
            "outline": 2,
            "shelf_width": 26,
        }
        colour_config = {   # BGR
            "agent": (255, 165, 0),
            "agent_carrying": (255, 140, 0),
            "agent_idx": (180, 180, 0),
            "goal": (0, 128, 128),
            "req_shelf": (255, 0, 0),
            "nonreq_shelf": (72, 61, 139),
            "obstacle": (0, 0, 0),
        }
        canvas_height = width_config["outline"] + self.grid_size[0] * \
            (width_config["outline"] + width_config["tile_width"])
        canvas_width = width_config["outline"] + self.grid_size[1] * \
            (width_config["outline"] + width_config["tile_width"])

        canvas = np.zeros((canvas_width, canvas_height, 3), np.uint8)
        canvas[:, :, :] = 255

        for y in range(self.grid_size[0] + 1):
            top_y = y * (width_config["outline"] + width_config["tile_width"])
            bottom_y = top_y + width_config["outline"]
            canvas[:, top_y : bottom_y, :] = 0

        for x in range(self.grid_size[1] + 1):
            left_x = x * (width_config["outline"] + width_config["tile_width"])
            right_x = left_x + width_config["outline"]
            canvas[left_x : right_x, :, :] = 0

        for y in range(self.grid_size[0]):
            for x in range(self.grid_size[1]):
                tile_top_y = width_config["outline"] + y * (width_config["outline"] + \
                    width_config["tile_width"])
                tile_left_x = width_config["outline"] + x * (width_config["outline"] + \
                    width_config["tile_width"])

                # draw goal tiles
                if (y, x) in self.goals:
                    top_left = (tile_top_y, tile_left_x)
                    bottom_right = (tile_top_y + width_config["tile_width"] - 1,
                        tile_left_x + width_config["tile_width"] - 1)
                    cv2.rectangle(canvas, top_left, bottom_right, colour_config["goal"], -1)

                # draw obstacles
                if self.grid[_LAYER_OBSTACLES, y, x] == 1:
                    top_left = (tile_top_y, tile_left_x)
                    bottom_right = (tile_top_y + width_config["tile_width"] - 1,
                        tile_left_x + width_config["tile_width"] - 1)
                    cv2.rectangle(canvas, top_left, bottom_right, colour_config["obstacle"], -1)

                # draw shelves
                shelf_id = self.grid[_LAYER_SHELFS, y, x]
                if shelf_id != 0:
                    shelf = self.shelfs[shelf_id - 1]
                    if shelf in self.request_queue:
                        colour = colour_config["req_shelf"]
                    else:
                        colour = colour_config["nonreq_shelf"]
                    pad = (width_config["tile_width"] - width_config["shelf_width"]) // 2
                    top_left = (tile_top_y + pad, tile_left_x + pad)
                    bottom_right = (tile_top_y + pad + width_config["shelf_width"] - 1,
                        tile_left_x + pad + width_config["shelf_width"] - 1)
                    cv2.rectangle(canvas, top_left, bottom_right, colour, -1)

                # draw agents
                agent_id = self.grid[_LAYER_AGENTS, y, x]
                if agent_id != 0:
                    agent = self.agents[agent_id - 1]
                    if agent.carrying_shelf:
                        colour = colour_config["agent_carrying"]
                    else:
                        colour = colour_config["agent"]
                    center = (tile_top_y + width_config["tile_width"] // 2,
                        tile_left_x + width_config["tile_width"] // 2)
                    cv2.circle(canvas, center, width_config["agent_rad"], colour, -1)

                    # indicate direction facing
                    if agent.dir == Direction.UP:
                        dest = (center[0] - width_config["agent_rad"] + 1, center[1])
                    elif agent.dir == Direction.DOWN:
                        dest = (center[0] + width_config["agent_rad"] - 1, center[1])
                    elif agent.dir == Direction.LEFT:
                        dest = (center[0], center[1] - width_config["agent_rad"] + 1)
                    else:
                        dest = (center[0], center[1] + width_config["agent_rad"] - 1)
                    cv2.line(canvas, center, dest, 1)

        # indicate agent index
        for agent in self.agents:
            bottom_y = width_config["outline"] + agent.y * (width_config["outline"] + \
                width_config["tile_width"])
            # left_x = width_config["outline"] + (agent.x + 1) * (width_config["outline"] + \
            #     width_config["tile_width"])
            left_x = (agent.x + 1) * (width_config["outline"] + \
                width_config["tile_width"])
            bottom_left = (bottom_y, left_x)
            cv2.putText(canvas, str(agent.id - 1), bottom_left, cv2.FONT_HERSHEY_SIMPLEX,
                0.5, colour_config["agent_idx"], 2, cv2.LINE_AA)

        return canvas


if __name__ == "__main__":
    # env = Warehouse(9, 8, 3, 10, 3, 1, 5, None, None, RewardType.GLOBAL)
    env = Warehouse(
        column_height=8,
        shelf_rows=1,
        shelf_columns=3,
        n_agents=4,
        msg_bits=0,
        sensor_range=1,
        request_queue_size=4,
        max_inactivity_steps=None,
        max_steps=500,
        reward_type=RewardType.INDIVIDUAL
    )
    env.reset()

    img = env.draw()
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite("out.png", img)

    # time.sleep(2)
    # env.render()
    # env.step(18 * [Action.LOAD] + 2 * [Action.NOOP])

    for i in range(1000000):
    # for _ in range(1000000):
        # time.sleep(2)
        # env.render()
        actions = env.action_space.sample()
        env.step(actions)
