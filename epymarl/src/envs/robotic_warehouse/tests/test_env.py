import os
import sys
import pytest
import gym
import numpy as np
from gym import spaces

TEST_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.abspath(os.path.join(TEST_DIR, os.pardir))
sys.path.insert(0, PROJECT_DIR)

from robotic_warehouse.warehouse import Warehouse, Direction, Action, RewardType


@pytest.fixture
def env_single_agent():
    env = Warehouse(3, 8, 3, 1, 0, 1, 5, None, None, RewardType.GLOBAL)
    env.reset()
    return env


@pytest.fixture
def env_0():
    env = Warehouse(3, 8, 3, 1, 0, 1, 5, 10, None, RewardType.GLOBAL)
    env.reset()

    env.agents[0].x = 4  # should place it in the middle (empty space)
    env.agents[0].y = 27
    env.agents[0].dir = Direction.DOWN

    env.shelfs[0].x = 4
    env.shelfs[0].y = 27

    env.agents[0].carrying_shelf = env.shelfs[0]

    env.request_queue[0] = env.shelfs[0]
    env._recalc_grid()
    return env


def test_grid_size():
    env = Warehouse(
        shelf_columns=1,
        column_height=3,
        shelf_rows=3,
        n_agents=1,
        msg_bits=0,
        sensor_range=1,
        request_queue_size=5,
        max_inactivity_steps=None,
        max_steps=None,
        reward_type=RewardType.GLOBAL,
    )
    assert env.grid_size == (14, 4)
    env = Warehouse(
        shelf_columns=3,
        column_height=3,
        shelf_rows=3,
        n_agents=1,
        msg_bits=0,
        sensor_range=1,
        request_queue_size=5,
        max_inactivity_steps=None,
        max_steps=None,
        reward_type=RewardType.GLOBAL,
    )
    assert env.grid_size == (14, 10)


def test_action_space_0():
    env = Warehouse(
        shelf_columns=1,
        column_height=3,
        shelf_rows=3,
        n_agents=2,
        msg_bits=0,
        sensor_range=1,
        request_queue_size=5,
        max_inactivity_steps=None,
        max_steps=None,
        reward_type=RewardType.GLOBAL,
    )
    env.reset()
    assert env.action_space == spaces.Tuple(2 * (spaces.Discrete(len(Action)), ))
    env.step(env.action_space.sample())


def test_action_space_1():
    env = Warehouse(
        shelf_columns=1,
        column_height=3,
        shelf_rows=3,
        n_agents=2,
        msg_bits=1,
        sensor_range=1,
        request_queue_size=5,
        max_inactivity_steps=None,
        max_steps=None,
        reward_type=RewardType.GLOBAL,
    )
    env.reset()
    assert env.action_space == spaces.Tuple(2 * (spaces.MultiDiscrete([len(Action), 2]), ))
    env.step(env.action_space.sample())


def test_action_space_2():
    env = Warehouse(
        shelf_columns=1,
        column_height=3,
        shelf_rows=3,
        n_agents=2,
        msg_bits=2,
        sensor_range=1,
        request_queue_size=5,
        max_inactivity_steps=None,
        max_steps=None,
        reward_type=RewardType.GLOBAL,
    )
    env.reset()
    assert env.action_space == spaces.Tuple(2 * (spaces.MultiDiscrete([len(Action), 2, 2]), ))
    env.step(env.action_space.sample())


def test_action_space_3():
    env = Warehouse(
        shelf_columns=1,
        column_height=3,
        shelf_rows=3,
        n_agents=10,
        msg_bits=5,
        sensor_range=1,
        request_queue_size=5,
        max_inactivity_steps=None,
        max_steps=None,
        reward_type=RewardType.GLOBAL,
    )
    env.reset()
    assert env.action_space == spaces.Tuple(10 * (spaces.MultiDiscrete([len(Action), *5 * (2,)]), ))
    env.step(env.action_space.sample())


# def test_action_space_4():
#     env = Warehouse(
#         shelf_columns=1,
#         column_height=3,
#         shelf_rows=3,
#         n_agents=10,
#         msg_bits=5,
#         sensor_range=1,
#         request_queue_size=5,
#         max_inactivity_steps=None,
#         max_steps=None,
#         reward_type=RewardType.GLOBAL,
#     )
#     env.reset()

#     env.agents[0].x = 1
#     env.agents[0].y = 6
#     env.agents[0].dir = 3
#     env.agents[0].carrying_shelf = None

#     env.agents[1].x = 0
#     env.agents[1].y = 10
#     env.agents[1].dir = 3
#     env.agents[1].carrying_shelf = None

#     env.agents[2].x = 1
#     env.agents[2].y = 3
#     env.agents[2].dir = 2
#     env.agents[2].carrying_shelf = None

#     env.agents[3].x = 2
#     env.agents[3].y = 3
#     env.agents[3].dir = 0
#     env.agents[3].carrying_shelf = None

#     env.agents[4].x = 2
#     env.agents[4].y = 4
#     env.agents[4].dir = 0
#     env.agents[4].carrying_shelf = None

#     env.agents[5].x = 3
#     env.agents[5].y = 4
#     env.agents[5].dir = 1
#     env.agents[5].carrying_shelf = None

#     env.agents[6].x = 1
#     env.agents[6].y = 13
#     env.agents[6].dir = 2
#     env.agents[6].carrying_shelf = None

#     env.agents[7].x = 3
#     env.agents[7].y = 3
#     env.agents[7].dir = 1
#     env.agents[7].carrying_shelf = None

#     env.agents[8].x = 0
#     env.agents[8].y = 8
#     env.agents[8].dir = 3
#     env.agents[8].carrying_shelf = None

#     env.agents[9].x = 2
#     env.agents[9].y = 3
#     env.agents[9].dir = 3
#     env.agents[9].carrying_shelf = None

#     env.shelfs[0].x = 1
#     env.shelfs[0].y = 1

#     env.shelfs[1].x = 2
#     env.shelfs[1].y = 1

#     env.shelfs[2].x = 1
#     env.shelfs[2].y = 2

#     env.shelfs[3].x = 2
#     env.shelfs[3].y = 2

#     env.shelfs[4].x = 1
#     env.shelfs[4].y = 3

#     env.shelfs[5].x = 2
#     env.shelfs[5].y = 3

#     env._recalc_grid()

#     actions = (
#         np.array([4, 1, 1, 0, 0, 1]),
#         np.array([2, 0, 1, 1, 0, 1]),
#         np.array([0, 1, 0, 0, 1, 1]),
#         np.array([3, 1, 0, 0, 0, 0]),
#         np.array([0, 0, 0, 0, 1, 0]),
#         np.array([2, 1, 0, 1, 1, 1]),
#         np.array([0, 1, 0, 1, 0, 1]),
#         np.array([0, 0, 1, 0, 1, 0]),
#         np.array([0, 1, 1, 0, 1, 1]),
#         np.array([0, 0, 0, 0, 0, 1])
#     )
#     env.step(actions)


def test_obs_space_0():
    env = Warehouse(
        shelf_columns=1,
        column_height=3,
        shelf_rows=3,
        n_agents=10,
        msg_bits=5,
        sensor_range=1,
        request_queue_size=5,
        max_inactivity_steps=None,
        max_steps=None,
        reward_type=RewardType.GLOBAL,
        fast_obs=False,
    )
    obs = env.reset()
    assert env.observation_space[0]["self"].contains(obs[0]["self"])
    # print(f"*** test_obs_space_0 1: {env.observation_space[0]}")
    # print(f"*** test_obs_space_0 2: {obs[0]}")
    assert env.observation_space[0].contains(obs[0])
    assert env.observation_space.contains(obs)
    nobs, _, _, _ = env.step(env.action_space.sample())
    assert env.observation_space.contains(nobs)


def test_obs_space_1():
    env = Warehouse(
        shelf_columns=1,
        column_height=3,
        shelf_rows=3,
        n_agents=10,
        msg_bits=5,
        sensor_range=1,
        request_queue_size=5,
        max_inactivity_steps=None,
        max_steps=None,
        reward_type=RewardType.GLOBAL,
    )
    obs = env.reset()
    for _ in range(200):
        obs, _, _, _ = env.step(env.action_space.sample())
        assert env.observation_space.contains(obs)


def test_obs_space_2():
    env = Warehouse(
        shelf_columns=1,
        column_height=3,
        shelf_rows=3,
        n_agents=10,
        msg_bits=5,
        sensor_range=1,
        request_queue_size=5,
        max_inactivity_steps=None,
        max_steps=None,
        reward_type=RewardType.GLOBAL,
    )
    obs = env.reset()
    for s, o in zip(env.observation_space, obs):
        assert len(gym.spaces.flatten(s, o)) == env._obs_length


def test_time_limit_0():
    env = Warehouse(
        shelf_columns=1,
        column_height=3,
        shelf_rows=3,
        n_agents=1,
        msg_bits=0,
        sensor_range=1,
        request_queue_size=5,
        max_inactivity_steps=None,
        max_steps=None,
        reward_type=RewardType.GLOBAL,
        shelf_time_limit=10,
        shelf_timeout_penalty=5.0,
        fast_obs=False
    )
    obs = env.reset()
    # print(f"*** test_time_limit_0 -1: {env.observation_space}")
    for _ in range(200):
        obs, _, _, _ = env.step(env.action_space.sample())
        # print(f"*** test_time_limit_0 0: {obs}")
        assert env.observation_space.contains(obs)

    obs = env.reset()
    for req_shelf in obs[0]["requests"]:
        assert req_shelf["remain_time"] == [10]
    for time_passed in range(10):
        obs, rwd, _, _ = env.step([Action.NOOP])
        if time_passed == 9:
            assert rwd[0] == pytest.approx(-25.0)
        else:
            assert rwd[0] == 0.0
        for req_shelf in obs[0]["requests"]:
            assert req_shelf["remain_time"] == [9 - time_passed]
    for time_passed in range(100):
        obs, rwd, _, _ = env.step([Action.NOOP])
        assert rwd[0] == pytest.approx(-25.0)
        for req_shelf in obs[0]["requests"]:
            assert req_shelf["remain_time"] == [0]

def test_time_limit_1():
    env = Warehouse(
        shelf_columns=1,
        column_height=3,
        shelf_rows=3,
        n_agents=1,
        msg_bits=0,
        sensor_range=1,
        request_queue_size=5,
        max_inactivity_steps=None,
        max_steps=None,
        reward_type=RewardType.GLOBAL,
        shelf_time_limit=10,
        shelf_timeout_penalty=5.0,
        fast_obs=False
    )
    obs = env.reset()
    env.agents[0].x = 1  # should place it in the middle (empty space)
    env.agents[0].y = 12
    env.agents[0].dir = Direction.DOWN
    env.request_queue[0].x = 1
    env.request_queue[0].y = 12
    env.agents[0].carrying_shelf = env.request_queue[0]
    env._recalc_grid()

    old_shelf = env.request_queue[0]
    obs, rwd, _, _ = env.step([Action.FORWARD])
    assert env.request_queue[0] is not old_shelf
    assert rwd[0] == pytest.approx(1.0)
    assert obs[0]["requests"][0]["remain_time"] == [10]
    for i in range(1, 5):
        assert obs[0]["requests"][i]["remain_time"] == [9]


def test_inactivity_0(env_0):
    env = env_0
    for i in range(9):
        _, _, done, _ = env.step([Action.NOOP])
        assert done == [False]
    _, _, done, _ = env.step([Action.NOOP])
    assert done == [True]


def test_inactivity_1(env_0):
    env = env_0
    for i in range(4):
        _, _, done, _ = env.step([Action.NOOP])
        assert done == [False]

    _, reward, _, _, = env.step([Action.FORWARD])
    assert reward[0] == pytest.approx(1.0)
    for i in range(9):
        _, _, done, _ = env.step([Action.NOOP])
        assert done == [False]

    _, _, done, _ = env.step([Action.NOOP])
    assert done == [True]


@pytest.mark.parametrize("time_limit,", [1, 100, 200])
def test_time_limit(time_limit):
    env = Warehouse(
        shelf_columns=1,
        column_height=3,
        shelf_rows=3,
        n_agents=10,
        msg_bits=5,
        sensor_range=1,
        request_queue_size=5,
        max_inactivity_steps=None,
        max_steps=time_limit,
        reward_type=RewardType.GLOBAL,
    )
    _ = env.reset()

    for _ in range(time_limit - 1):
        _, _, done, _ = env.step(env.action_space.sample())
        assert done == 10 * [False]

    _, _, done, _ = env.step(env.action_space.sample())
    assert done == 10 * [True]


def test_inactivity_2(env_0):
    env = env_0
    for i in range(9):
        _, _, done, _ = env.step([Action.NOOP])
        assert done == [False]
    _, _, done, _ = env.step([Action.NOOP])
    assert done == [True]
    env.reset()
    for i in range(9):
        _, _, done, _ = env.step([Action.NOOP])
        assert done == [False]
    _, _, done, _ = env.step([Action.NOOP])
    assert done == [True]


def test_fast_obs_0():
    env = Warehouse(3, 8, 3, 2, 0, 1, 5, 10, None, RewardType.GLOBAL, fast_obs=False)
    env.reset()

    slow_obs_space = env.observation_space

    for _ in range(10):
        slow_obs = [env._make_obs(agent) for agent in env.agents]
        env._use_fast_obs()
        fast_obs = [env._make_obs(agent) for agent in env.agents]
        assert len(fast_obs) == 2
        assert len(slow_obs) == 2

        flattened_slow = [spaces.flatten(osp, obs) for osp, obs in zip(slow_obs_space, slow_obs)]

        # print(f"*** test_fast_obs_0 1: {slow_obs[0]}")
        # print(f"*** test_fast_obs_0 2: {flattened_slow}")
        # print(f"*** test_fast_obs_0 3: {fast_obs}")
        for i in range(len(fast_obs)):
            # print(f"*** test_fast_obs_0 4: {i}, {fast_obs[i]}")
            # print(f"*** test_fast_obs_0 5: {flattened_slow[i]}")
            assert list(fast_obs[i]) ==  list(flattened_slow[i])

        env._use_slow_obs()
        env.step(env.action_space.sample())

def test_fast_obs_1():
    env = Warehouse(3, 8, 3, 3, 0, 1, 5, 10, None, RewardType.GLOBAL, fast_obs=False)
    env.reset()

    slow_obs_space = env.observation_space

    for _ in range(10):
        slow_obs = [env._make_obs(agent) for agent in env.agents]
        env._use_fast_obs()
        fast_obs = [env._make_obs(agent) for agent in env.agents]
        assert len(fast_obs) == 3
        assert len(slow_obs) == 3

        flattened_slow = [spaces.flatten(osp, obs) for osp, obs in zip(slow_obs_space, slow_obs)]

        for i in range(len(fast_obs)):
            print(slow_obs[0])
            assert list(fast_obs[i]) ==  list(flattened_slow[i])

        env._use_slow_obs()
        env.step(env.action_space.sample())

def test_fast_obs_2():
    env = Warehouse(3, 8, 3, 3, 2, 1, 5, 10, None, RewardType.GLOBAL, fast_obs=False)
    env.reset()

    slow_obs_space = env.observation_space

    for _ in range(10):
        slow_obs = [env._make_obs(agent) for agent in env.agents]
        env._use_fast_obs()
        fast_obs = [env._make_obs(agent) for agent in env.agents]
        assert len(fast_obs) == 3
        assert len(slow_obs) == 3

        flattened_slow = [spaces.flatten(osp, obs) for osp, obs in zip(slow_obs_space, slow_obs)]

        for i in range(len(fast_obs)):
            print(slow_obs[0])
            assert list(fast_obs[i]) ==  list(flattened_slow[i])

        env._use_slow_obs()
        env.step(env.action_space.sample())

