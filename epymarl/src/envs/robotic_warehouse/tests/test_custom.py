import os
import sys
import pytest
import gym
import numpy as np
from gym import spaces
from itertools import product

TEST_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.abspath(os.path.join(TEST_DIR, os.pardir))
sys.path.insert(0, PROJECT_DIR)

from robotic_warehouse.warehouse_custom import WarehouseCustom, Direction, Action, RewardType

@pytest.fixture
def custom_env_0():
    obstacles = np.array([
        [1, 0, 1, 1, 1],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [1, 0, 1, 1, 1]
    ])
    nonrand_agent_pos = [(1, 3), (1, 4), (2, 4)]
    shelves = [(0, 1), (1, 1), (2, 1)]
    nonrand_req_shelves = [(0, 1)]
    goals = [(3, 1)]

    env = WarehouseCustom(
        obstacles=obstacles,
        nonrand_agent_pos=nonrand_agent_pos,
        goals=goals,
        shelves=shelves,
        nonrand_req_shelves=nonrand_req_shelves,
        max_steps=100,
    )
    env.reset()
    return env

@pytest.fixture
def custom_env_1():
    obstacles = np.array([
        [1, 1, 1, 0, 1],
        [0, 0, 0, 0, 1],
        [1, 1, 0, 0, 0],
        [0, 0, 1, 0, 0]
    ])
    nonrand_agent_pos = [(0, 3), (1, 0)]
    shelves = [(3, 0)]
    nonrand_req_shelves = [(3, 0)]
    goals = [(3, 1)]
    env = WarehouseCustom(
        obstacles=obstacles,
        nonrand_agent_pos=nonrand_agent_pos,
        goals=goals,
        shelves=shelves,
        nonrand_req_shelves=nonrand_req_shelves,
        max_steps=100,
    )
    env.reset()
    return env

def test_shelves(custom_env_0):
    env = custom_env_0
    assert len(env.shelfs) == 3

    expected_shelf_pos = {(0, 1), (1, 1), (2, 1)}
    for _shelf in env.shelfs:
        query = (_shelf.y, _shelf.x)
        assert query in expected_shelf_pos
        expected_shelf_pos.remove(query)

    assert len(env.request_queue) == 1
    assert env.request_queue[0].y == 0
    assert env.request_queue[0].x == 1

def test_obs_walls(custom_env_1):
    env = custom_env_1
    obs = env.reset()
    agent_id = 2
    for sensor_idx in range(9):
        if sensor_idx in [0, 1, 2, 3, 6, 7, 8]:   # this includes out-of-bound walls and in-map walls
            assert obs[agent_id - 1][8 + 8 * sensor_idx + 7] == 1.0
        else:
            assert obs[agent_id - 1][8 + 8 * sensor_idx + 7] == 0.0

    env.agents[1].dir = Direction.RIGHT
    env._recalc_grid()
    obs, _, _, _ = env.step([Action.NOOP, Action.FORWARD])
    for sensor_idx in range(9):
        if sensor_idx in [0, 1, 2, 6, 7]:   # this includes out-of-bound walls and in-map walls
            assert obs[agent_id - 1][8 + 8 * sensor_idx + 7] == 1.0
        else:
            assert obs[agent_id - 1][8 + 8 * sensor_idx + 7] == 0.0

def test_blocked_by_agent(custom_env_0):
    env = custom_env_0
    env.reset()
    agent1 = env.agents[1]
    agent2 = env.agents[2]
    agent1.dir = Direction.DOWN
    agent2.dir = Direction.UP
    env._recalc_grid()

    env.step([Action.NOOP, Action.FORWARD, Action.FORWARD])
    assert agent1.y == 1
    assert agent1.x == 4
    assert agent2.y == 2
    assert agent2.x == 4

def test_blocked_by_walls(custom_env_1):
    env = custom_env_1
    env.reset()

    agent = env.agents[1]
    agent.dir = Direction.LEFT
    env._recalc_grid()

    env.step([Action.NOOP, Action.FORWARD])
    assert (agent.y, agent.x) == (1, 0)
    env.step([Action.NOOP, Action.LEFT])
    assert agent.dir == Direction.DOWN
    env.step([Action.NOOP, Action.FORWARD])
    assert (agent.y, agent.x) == (1, 0)
    env.step([Action.NOOP, Action.LEFT])
    assert agent.dir == Direction.RIGHT
    env.step([Action.NOOP, Action.FORWARD])
    assert (agent.y, agent.x) == (1, 1)
    env.step([Action.NOOP, Action.FORWARD])
    assert (agent.y, agent.x) == (1, 2)
    env.step([Action.NOOP, Action.RIGHT])
    assert agent.dir == Direction.DOWN
    env.step([Action.NOOP, Action.FORWARD])
    assert (agent.y, agent.x) == (2, 2)
    env.step([Action.NOOP, Action.RIGHT])
    assert agent.dir == Direction.LEFT
    env.step([Action.NOOP, Action.FORWARD])
    assert (agent.y, agent.x) == (2, 2)
    env.step([Action.NOOP, Action.RIGHT])
    assert agent.dir == Direction.UP
    env.step([Action.NOOP, Action.RIGHT])
    assert agent.dir == Direction.RIGHT
    env.step([Action.NOOP, Action.FORWARD])
    env.step([Action.NOOP, Action.FORWARD])
    env.step([Action.NOOP, Action.FORWARD])
    assert (agent.y, agent.x) == (2, 4)
    env.step([Action.NOOP, Action.RIGHT])
    assert agent.dir == Direction.DOWN
    env.step([Action.NOOP, Action.FORWARD])
    env.step([Action.NOOP, Action.FORWARD])
    assert (agent.y, agent.x) == (3, 4)

def test_success_scenario(custom_env_0):
    env = custom_env_0
    env.reset()

    env.agents[0].dir = Direction.LEFT
    env.agents[1].dir = Direction.LEFT
    env.agents[2].dir = Direction.LEFT
    env._recalc_grid()

    action_seq = [
        [Action.FORWARD, Action.FORWARD, Action.FORWARD],
        [Action.FORWARD, Action.FORWARD, Action.FORWARD],
        [Action.TOGGLE_LOAD, Action.NOOP, Action.FORWARD],
        [Action.FORWARD, Action.FORWARD, Action.TOGGLE_LOAD],
        [Action.NOOP, Action.RIGHT, Action.FORWARD],
        [Action.NOOP, Action.FORWARD, Action.NOOP],
        [Action.NOOP, Action.TOGGLE_LOAD, Action.NOOP],
        [Action.NOOP, Action.RIGHT, Action.NOOP],
        [Action.NOOP, Action.RIGHT, Action.NOOP],
        [Action.NOOP, Action.FORWARD, Action.NOOP],
        [Action.NOOP, Action.FORWARD, Action.NOOP],
    ]
    last_action = [Action.NOOP, Action.FORWARD, Action.NOOP]

    for action in action_seq:
        _, rwd, done, _ = env.step(action)
        assert rwd == pytest.approx([0.0, 0.0, 0.0])
        assert not any(done)

    _, rwd, done, _ = env.step(last_action)
    assert rwd == pytest.approx([1.0, 1.0, 1.0])
    assert all(done)

def test_random_request_option():
    obstacles = np.zeros((9, 7))
    nonrand_agent_pos = [(0, 0), (0, 1), (0, 2), (0, 3)]
    shelves = list(product([1, 2, 6, 7], [1, 2, 3, 4]))
    goals = [(4, 6)]
    request_queue_size = 4
    env = WarehouseCustom(
        obstacles = obstacles,
        nonrand_agent_pos = nonrand_agent_pos,
        goals = goals,
        shelves = shelves,
        nonrand_req_shelves = None,
        random_request = True,
        request_queue_size = request_queue_size,
        max_steps = 100,
    )

    prev_request_set = None
    is_randomized = False
    for _ in range(1000):
        env.reset()
        new_request_set = set()
        for y, x in shelves:
            shelf = env._shelf_at_loc(y, x)
            if env._is_requested(shelf):
                new_request_set.add((y, x))
        assert len(new_request_set) == request_queue_size
        if prev_request_set is not None and prev_request_set != new_request_set:
            is_randomized = True
        prev_request_set = new_request_set
    assert is_randomized

def test_random_agent_positions():
    obstacles = np.zeros((9, 7))
    shelves = list(product([1, 2, 6, 7], [1, 2, 3, 4]))
    nonrand_req_shelves = [(1, 1)]
    goals = [(4, 6)]
    num_agents = 4
    env = WarehouseCustom(
        obstacles=obstacles,
        nonrand_agent_pos=None,  # No predefined agent positions
        goals=goals,
        shelves=shelves,
        nonrand_req_shelves=nonrand_req_shelves,
        random_agent_pos=True,  # Enable random agent positions
        rand_num_agents=num_agents,
        max_steps=100,
    )

    prev_agent_positions = None
    is_randomized = False
    for _ in range(1000):
        env.reset()
        new_agent_positions = set((agent.y, agent.x) for agent in env.agents)
        assert len(new_agent_positions) == num_agents
        if prev_agent_positions is not None and prev_agent_positions != new_agent_positions:
            is_randomized = True
        prev_agent_positions = new_agent_positions
    assert is_randomized

def test_block_agent_from_shelf_to_shelf():
    obstacles = np.array([
        [0, 0, 0, 0],
    ])
    nonrand_agent_pos = [(0, 0)]
    shelves = [(0, 0), (0, 1), (0, 2)]
    nonrand_req_shelves = [(0, 0)]
    goals = [(0, 3)]

    env = WarehouseCustom(
        obstacles=obstacles,
        nonrand_agent_pos=nonrand_agent_pos,
        goals=goals,
        shelves=shelves,
        nonrand_req_shelves=nonrand_req_shelves,
        max_steps=100,
        block_shelf_to_shelf=True,
    )
    env.reset()
    _agent = env.agents[0]
    _agent.dir = Direction.RIGHT
    env._recalc_grid()

    env.step([Action.FORWARD])
    assert (_agent.y, _agent.x) == (0, 0)
    env.step([Action.FORWARD])
    assert (_agent.y, _agent.x) == (0, 0)

    env = WarehouseCustom(
        obstacles=obstacles,
        nonrand_agent_pos=nonrand_agent_pos,
        goals=goals,
        shelves=shelves,
        nonrand_req_shelves=nonrand_req_shelves,
        max_steps=100,
        block_shelf_to_shelf=False,
    )
    env.reset()
    _agent = env.agents[0]
    _agent.dir = Direction.RIGHT
    env._recalc_grid()

    env.step([Action.FORWARD])
    assert (_agent.y, _agent.x) == (0, 1)
    env.step([Action.FORWARD])
    assert (_agent.y, _agent.x) == (0, 2)

def test_force_request():
    obstacles = np.zeros((9, 7))
    nonrand_agent_pos = [(0, 0), (0, 1), (0, 2), (0, 3)]
    shelves = list(product([1, 2, 6, 7], [1, 2, 3, 4]))
    goals = [(4, 6)]
    request_queue_size = 4
    env = WarehouseCustom(
        obstacles = obstacles,
        nonrand_agent_pos = nonrand_agent_pos,
        goals = goals,
        shelves = shelves,
        nonrand_req_shelves = None,
        random_request = True,
        request_queue_size = request_queue_size,
        max_steps = 100,
    )
    env.reset()

    desired_requests = [
        [(1, 1), (1, 2), (1, 3), (1, 4)],
        [(1, 1), (2, 2), (6, 3), (7, 4)],
        [(1, 1), (2, 1), (6, 1), (7, 1)],
    ]
    for _desired_requests in desired_requests:
        env._force_request(_desired_requests)
        request_set = set()
        for y, x in shelves:
            shelf = env._shelf_at_loc(y, x)
            if env._is_requested(shelf):
                request_set.add((y, x))
        assert set(_desired_requests) == request_set

def test_new_request_upon_deliv():
    obstacles = np.array([
        [0, 0, 0, 0],
    ])
    nonrand_agent_pos = [(0, 2)]
    shelves = [(0, 0), (0, 1), (0, 2)]
    nonrand_req_shelves = [(0, 1), (0, 2)]
    goals = [(0, 3)]

    env = WarehouseCustom(
        obstacles=obstacles,
        nonrand_agent_pos=nonrand_agent_pos,
        goals=goals,
        shelves=shelves,
        nonrand_req_shelves=nonrand_req_shelves,
        max_steps=100,
        block_shelf_to_shelf=True,
        terminate_on_deliv=False,
    )
    env.reset()
    _agent = env.agents[0]
    _agent.dir = Direction.RIGHT
    env._force_request([(0, 1), (0, 2)])
    env._recalc_grid()

    env.step([Action.TOGGLE_LOAD])
    _, reward, done, _ = env.step([Action.FORWARD])
    assert not any(done)
    assert reward == [1.0]
    assert not env._is_requested(env._shelf_at_loc(0, 2))
    assert env._is_requested(env._shelf_at_loc(0, 1))
    assert env._is_requested(env._shelf_at_loc(0, 0))

def test_new_request_upon_deliv_2():
    obstacles = np.zeros((2, 4))
    nonrand_agent_pos = [(0, 2), (1, 2)]
    shelves = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)]
    nonrand_req_shelves = [(0, 1), (0, 2), (1, 1), (1, 2)]
    goals = [(0, 3), (1, 3)]

    env = WarehouseCustom(
        obstacles=obstacles,
        nonrand_agent_pos=nonrand_agent_pos,
        goals=goals,
        shelves=shelves,
        nonrand_req_shelves=nonrand_req_shelves,
        max_steps=100,
        block_shelf_to_shelf=True,
        terminate_on_deliv=False,
    )
    env.reset()
    for _agent in env.agents:
        _agent.dir = Direction.RIGHT
    env._force_request([(0, 1), (0, 2), (1, 1), (1, 2)])
    env._recalc_grid()

    env.step([Action.TOGGLE_LOAD, Action.TOGGLE_LOAD])
    _, reward, done, _ = env.step([Action.FORWARD, Action.FORWARD])
    assert not any(done)
    assert reward[0] == 2.0 and reward[1] == 2.0
    assert not env._is_requested(env._shelf_at_loc(0, 2))
    assert env._is_requested(env._shelf_at_loc(0, 1))
    assert env._is_requested(env._shelf_at_loc(0, 0))
    assert not env._is_requested(env._shelf_at_loc(1, 2))
    assert env._is_requested(env._shelf_at_loc(1, 1))
    assert env._is_requested(env._shelf_at_loc(1, 0))

def test_reset_obstacles():
    obstacles = np.array([
        [0, 0, 0], [0, 0, 0], [1, 1, 1]
    ])
    shelves = [(0, 0)]
    goals = [(0, 2)]
    rand_obst_pos_list = [(0, 1), (1, 0), (1, 1), (1, 2)]
    env = WarehouseCustom(
        obstacles = obstacles,
        shelves = shelves,
        goals = goals,
        add_rand_obst = True,
        rand_obst_pos_list = rand_obst_pos_list,
        rand_obst_num = 2,
        random_agent_pos = True,
        rand_num_agents = 1,
        random_request = True,
        request_queue_size = 1,
    )

    for _ in range(10000):
        env.reset()
        obst_count = 0
        agent = env.agents[0]
        assert not env._is_obstacle_or_wall(agent.y, agent.x)
        for y in [0, 1, 2]:
            for x in [0, 1, 2]:
                if env._is_obstacle_or_wall(y, x):
                    assert y == 2 or (y, x) in rand_obst_pos_list
                    obst_count += 1
        assert obst_count == 5