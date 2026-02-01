import os
import sys
import pytest
import gym
import numpy as np
from gym import spaces

TEST_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.abspath(os.path.join(TEST_DIR, os.pardir))
sys.path.insert(0, PROJECT_DIR)

from robotic_warehouse.warehouse_toy import WarehouseToy, Direction, Action, RewardType

def test_toy_size_2agents():
    env = WarehouseToy(
        n_agents = 2,
        reward_type = RewardType.GLOBAL,
    )
    env.reset()
    assert len(env.goals) == 1
    assert env.goals[0][0] == 3
    assert env.goals[0][1] == 1

    assert env.n_agents == 2
    expected_agent_pos = [(2, 0), (2, 2)]
    for _agent in env.agents:
        query = (_agent.y, _agent.x)
        assert query in expected_agent_pos
        expected_agent_pos.remove(query)

    assert len(env.shelfs) == 2
    assert len(env.request_queue) == 1
    assert env.request_queue[0].y == 0
    assert env.request_queue[0].x == 1
    
    expected_nonreq_shelf_pos = {(1, 1)}
    nonreq_shelfs = set(env.shelfs) - set(env.request_queue)
    for _shelf in nonreq_shelfs:
        query = (_shelf.y, _shelf.x)
        assert query in expected_nonreq_shelf_pos
        expected_nonreq_shelf_pos.remove(query)
    
    assert env.grid_size == (4, 3)
    for y in range(env.grid_size[0]):
        for x in range(env.grid_size[1]):
            if y == 2:
                assert not env._is_obstacle_or_wall(y, x)
                assert env._is_highway(y, x)
            elif x == 1:
                assert not env._is_obstacle_or_wall(y, x)
                if y <= 1:
                    assert not env._is_highway(y, x)
                else:
                    assert env._is_highway(y, x)
            else:
                assert env._is_obstacle_or_wall(y, x)

def test_toy_size_3agents():
    env = WarehouseToy(
        n_agents = 3,
        reward_type = RewardType.GLOBAL,
    )
    env.reset()
    assert len(env.goals) == 1
    assert env.goals[0][0] == 5
    assert env.goals[0][1] == 1

    assert env.n_agents == 3
    expected_agent_pos = [(3, 0), (3, 2), (3, 3)]
    for _agent in env.agents:
        query = (_agent.y, _agent.x)
        assert query in expected_agent_pos
        expected_agent_pos.remove(query)

    assert len(env.shelfs) == 3
    assert len(env.request_queue) == 1
    assert env.request_queue[0].y == 0
    assert env.request_queue[0].x == 1
    
    expected_nonreq_shelf_pos = {(1, 1), (2, 1)}
    nonreq_shelfs = set(env.shelfs) - set(env.request_queue)
    for _shelf in nonreq_shelfs:
        query = (_shelf.y, _shelf.x)
        assert query in expected_nonreq_shelf_pos
        expected_nonreq_shelf_pos.remove(query)
    
    assert env.grid_size == (6, 4)
    for y in range(env.grid_size[0]):
        for x in range(env.grid_size[1]):
            if y == 3:
                assert not env._is_obstacle_or_wall(y, x)
                assert env._is_highway(y, x)
            elif x == 1:
                assert not env._is_obstacle_or_wall(y, x)
                if y <= 2:
                    assert not env._is_highway(y, x)
                else:
                    assert env._is_highway(y, x)
            else:
                assert env._is_obstacle_or_wall(y, x)

def test_toy_size_4agents():
    env = WarehouseToy(
        n_agents = 4,
        reward_type = RewardType.GLOBAL,
    )
    env.reset()
    assert len(env.goals) == 1
    assert env.goals[0][0] == 7
    assert env.goals[0][1] == 1

    assert env.n_agents == 4
    expected_agent_pos = [(4, 0), (4, 2), (4, 3), (4, 4)]
    for _agent in env.agents:
        query = (_agent.y, _agent.x)
        assert query in expected_agent_pos
        expected_agent_pos.remove(query)

    assert len(env.shelfs) == 4
    assert len(env.request_queue) == 1
    assert env.request_queue[0].y == 0
    assert env.request_queue[0].x == 1
    
    expected_nonreq_shelf_pos = {(1, 1), (2, 1), (3, 1)}
    nonreq_shelfs = set(env.shelfs) - set(env.request_queue)
    for _shelf in nonreq_shelfs:
        query = (_shelf.y, _shelf.x)
        assert query in expected_nonreq_shelf_pos
        expected_nonreq_shelf_pos.remove(query)
    
    assert env.grid_size == (8, 5)
    for y in range(env.grid_size[0]):
        for x in range(env.grid_size[1]):
            if y == 4:
                assert not env._is_obstacle_or_wall(y, x)
                assert env._is_highway(y, x)
            elif x == 1:
                assert not env._is_obstacle_or_wall(y, x)
                if y <= 3:
                    assert not env._is_highway(y, x)
                else:
                    assert env._is_highway(y, x)
            else:
                assert env._is_obstacle_or_wall(y, x)


def test_toy_size_100agents():
    env = WarehouseToy(
        n_agents = 100,
        reward_type = RewardType.GLOBAL,
    )
    env.reset()
    assert len(env.goals) == 1
    assert env.goals[0][0] == 199
    assert env.goals[0][1] == 1

    assert env.n_agents == 100
    expected_agent_pos = [(100, 0)] + [(100, x) for x in range(2, 101)]
    for _agent in env.agents:
        query = (_agent.y, _agent.x)
        assert query in expected_agent_pos
        expected_agent_pos.remove(query)

    assert len(env.shelfs) == 100
    assert len(env.request_queue) == 1
    assert env.request_queue[0].y == 0
    assert env.request_queue[0].x == 1
    
    expected_nonreq_shelf_pos = {(y, 1) for y in range(1, 100)}
    nonreq_shelfs = set(env.shelfs) - set(env.request_queue)
    for _shelf in nonreq_shelfs:
        query = (_shelf.y, _shelf.x)
        assert query in expected_nonreq_shelf_pos
        expected_nonreq_shelf_pos.remove(query)
    
    assert env.grid_size == (200, 101)
    for y in range(env.grid_size[0]):
        for x in range(env.grid_size[1]):
            if y == 100:
                assert not env._is_obstacle_or_wall(y, x)
                assert env._is_highway(y, x)
            elif x == 1:
                assert not env._is_obstacle_or_wall(y, x)
                if y <= 99:
                    assert not env._is_highway(y, x)
                else:
                    assert env._is_highway(y, x)
            else:
                assert env._is_obstacle_or_wall(y, x)

def test_toy_obs_walls():
    env = WarehouseToy(
        n_agents = 3,
    )
    obs = env.reset()
    agent_id = env._agent_at_loc(3, 0).id
    for sensor_idx in range(9):
        if sensor_idx in [0, 1, 3, 6, 7]:   # this includes out-of-bound walls and in-map walls
            assert obs[agent_id - 1][8 + 8 * sensor_idx + 7] == 1.0
        else:
            assert obs[agent_id - 1][8 + 8 * sensor_idx + 7] == 0.0

def test_blocked_by_agent():
    env = WarehouseToy(
        n_agents = 3,
    )
    env.reset()
    agent1 = env._agent_at_loc(3, 2)
    agent1.dir = Direction.RIGHT
    agent2 = env._agent_at_loc(3, 3)
    agent2.dir = Direction.LEFT
    env._recalc_grid()
    
    env.step([Action.NOOP, Action.FORWARD, Action.FORWARD])
    assert agent1.y == 3
    assert agent1.x == 2
    assert agent2.y == 3
    assert agent2.x == 3

def test_blocked_by_shelf():
    env = WarehouseToy(
        n_agents = 2,
    )
    env.reset()
    agent = env._agent_at_loc(2, 0)
    agent.dir = Direction.RIGHT
    env._recalc_grid()

    env.step([Action.FORWARD, Action.NOOP])
    env.step([Action.LEFT, Action.NOOP])
    env.step([Action.FORWARD, Action.NOOP])
    env.step([Action.FORWARD, Action.NOOP])
    assert (agent.y, agent.x) == (1, 1) # blocked while not holding shelf
    env.step([Action.TOGGLE_LOAD, Action.NOOP])
    env.step([Action.FORWARD, Action.NOOP])
    assert (agent.y, agent.x) == (1, 1) # blocked while holding shelf


def test_blocked_by_inner_wall():
    env = WarehouseToy(
        n_agents = 2,
    )
    env.reset()
    agent = env._agent_at_loc(2, 0)
    agent.dir = Direction.RIGHT
    env._recalc_grid()

    env.step([Action.FORWARD, Action.NOOP])
    env.step([Action.LEFT, Action.NOOP])
    env.step([Action.FORWARD, Action.NOOP])
    env.step([Action.TOGGLE_LOAD, Action.NOOP])
    env.step([Action.LEFT, Action.NOOP])
    env.step([Action.FORWARD, Action.NOOP])
    assert (agent.y, agent.x) == (1, 1)
    env.step([Action.LEFT, Action.NOOP])
    env.step([Action.LEFT, Action.NOOP])
    env.step([Action.FORWARD, Action.NOOP])
    assert (agent.y, agent.x) == (1, 1)

def test_blocked_by_outer_wall_0():
    env = WarehouseToy(
        n_agents = 2,
    )
    env.reset()
    agent = env._agent_at_loc(2, 0)
    agent.dir = Direction.RIGHT
    env._recalc_grid()

    env.step([Action.FORWARD, Action.NOOP])
    env.step([Action.LEFT, Action.NOOP])
    env.step([Action.FORWARD, Action.NOOP])
    env.step([Action.TOGGLE_LOAD, Action.NOOP])
    env.step([Action.LEFT, Action.NOOP])
    env.step([Action.LEFT, Action.NOOP])
    env.step([Action.FORWARD, Action.NOOP])
    env.step([Action.FORWARD, Action.NOOP])
    env.step([Action.FORWARD, Action.NOOP])
    assert (agent.y, agent.x) == (3, 1)

def test_blocked_by_outer_wall_1():
    env = WarehouseToy(
        n_agents = 2,
    )
    env.reset()
    agent = env._agent_at_loc(2, 0)
    agent.dir = Direction.RIGHT
    env._recalc_grid()

    env.step([Action.FORWARD, Action.NOOP])
    env.step([Action.LEFT, Action.NOOP])
    env.step([Action.FORWARD, Action.NOOP])
    env.step([Action.TOGGLE_LOAD, Action.NOOP])
    env.step([Action.LEFT, Action.NOOP])
    env.step([Action.LEFT, Action.NOOP])
    env.step([Action.FORWARD, Action.NOOP])
    env.step([Action.RIGHT, Action.NOOP])
    env.step([Action.FORWARD, Action.NOOP])
    env.step([Action.FORWARD, Action.NOOP])
    assert (agent.y, agent.x) == (2, 0)

def test_blocked_by_outer_wall_2():
    env = WarehouseToy(
        n_agents = 2,
    )
    env.reset()
    agent = env._agent_at_loc(2, 2)
    agent.dir = Direction.LEFT
    env._recalc_grid()

    env.step([Action.FORWARD, Action.NOOP])
    env.step([Action.RIGHT, Action.NOOP])
    env.step([Action.FORWARD, Action.NOOP])
    env.step([Action.TOGGLE_LOAD, Action.NOOP])
    env.step([Action.LEFT, Action.NOOP])
    env.step([Action.LEFT, Action.NOOP])
    env.step([Action.FORWARD, Action.NOOP])
    env.step([Action.LEFT, Action.NOOP])
    env.step([Action.FORWARD, Action.NOOP])
    env.step([Action.FORWARD, Action.NOOP])
    assert (agent.y, agent.x) == (2, 2)

def test_success_scenario():
    env = WarehouseToy(
        n_agents = 3,
    )
    env.reset()
    agent1 = env._agent_at_loc(3, 0)
    agent1.dir = Direction.RIGHT
    agent2 = env._agent_at_loc(3, 2)
    agent2.dir = Direction.LEFT
    agent3 = env._agent_at_loc(3, 3)
    agent3.dir = Direction.LEFT
    env._recalc_grid()

    action_seq = [
        [Action.NOOP, Action.FORWARD, Action.FORWARD],
        [Action.NOOP, Action.RIGHT, Action.NOOP],
        [Action.NOOP, Action.FORWARD, Action.NOOP],
        [Action.NOOP, Action.TOGGLE_LOAD, Action.NOOP],
        [Action.NOOP, Action.LEFT, Action.NOOP],
        [Action.NOOP, Action.LEFT, Action.NOOP],
        [Action.NOOP, Action.FORWARD, Action.NOOP],
        [Action.NOOP, Action.FORWARD, Action.FORWARD],
        [Action.NOOP, Action.LEFT, Action.RIGHT],
        [Action.NOOP, Action.LEFT, Action.FORWARD],
        [Action.NOOP, Action.FORWARD, Action.FORWARD],
        [Action.NOOP, Action.RIGHT, Action.TOGGLE_LOAD],
        [Action.NOOP, Action.FORWARD, Action.LEFT],
        [Action.NOOP, Action.FORWARD, Action.LEFT],
        [Action.NOOP, Action.NOOP, Action.FORWARD],
        [Action.NOOP, Action.NOOP, Action.FORWARD],
        [Action.NOOP, Action.NOOP, Action.LEFT],
        [Action.FORWARD, Action.NOOP, Action.FORWARD],
        [Action.LEFT, Action.NOOP, Action.NOOP],
        [Action.FORWARD, Action.NOOP, Action.NOOP],
        [Action.FORWARD, Action.NOOP, Action.NOOP],
        [Action.FORWARD, Action.NOOP, Action.NOOP],
        [Action.TOGGLE_LOAD, Action.NOOP, Action.NOOP],
        [Action.LEFT, Action.NOOP, Action.NOOP],
        [Action.LEFT, Action.NOOP, Action.NOOP],
        [Action.FORWARD, Action.NOOP, Action.NOOP],
        [Action.FORWARD, Action.NOOP, Action.NOOP],
        [Action.FORWARD, Action.NOOP, Action.NOOP],
        [Action.FORWARD, Action.NOOP, Action.NOOP],
    ]
    last_action = [Action.FORWARD, Action.NOOP, Action.NOOP]

    for action in action_seq:
        _, rwd, done, _ = env.step(action)
        assert rwd == pytest.approx([0.0, 0.0, 0.0])
        assert not any(done)
    
    _, rwd, done, _ = env.step(last_action)
    assert rwd == pytest.approx([1.0, 1.0, 1.0])
    assert all(done)