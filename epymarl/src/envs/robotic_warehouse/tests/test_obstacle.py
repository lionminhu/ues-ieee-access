import os
import sys
import pytest

TEST_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.abspath(os.path.join(TEST_DIR, os.pardir))
sys.path.insert(0, PROJECT_DIR)

from robotic_warehouse.warehouse import Warehouse, Direction, Action, RewardType


@pytest.fixture
def env_obstacle():
    env = Warehouse(3, 8, 3, 1, 0, 1, 5, None, None, RewardType.GLOBAL, True, [(4, 26)])
    env.reset()
    return env


@pytest.fixture
def env_obstacle_two_agents():
    env = Warehouse(3, 8, 3, 2, 0, 1, 5, None, None, RewardType.GLOBAL, True, [(4, 26)])
    # Obstacle at (4,26)
    env.reset()
    return env


@pytest.fixture
def env_obstacle_three_agents():
    env = Warehouse(3, 8, 3, 3, 0, 1, 5, None, None, RewardType.GLOBAL, True, [(4, 26)])
    env.reset()
    return env


def test_simple_movement_down(env_obstacle):
    env = env_obstacle
    env.reset()

    env.agents[0].x = 4  # should place it in the middle (empty space)
    env.agents[0].y = 24
    env.agents[0].dir = Direction.DOWN
    env._recalc_grid()
    env.step([Action.FORWARD])

    assert env.agents[0].x == 4
    assert env.agents[0].y == 25


def test_obstacle_0(env_obstacle):
    env = env_obstacle
    env.reset()

    env.agents[0].x = 4
    env.agents[0].y = 25
    env.agents[0].dir = Direction.DOWN
    env._recalc_grid()
    env.step([Action.FORWARD])

    assert env.agents[0].x == 4
    assert env.agents[0].y == 25  # blocked by obstacle


def test_obstacle_1(env_obstacle):
    env = env_obstacle
    env.reset()

    env.agents[0].x = 3
    env.agents[0].y = 26
    env.agents[0].dir = Direction.RIGHT
    env._recalc_grid()
    env.step([Action.FORWARD])

    assert env.agents[0].x == 3
    assert env.agents[0].y == 26  # blocked by obstacle


def test_obstacle_2(env_obstacle):
    env = env_obstacle
    env.reset()

    env.agents[0].x = 4
    env.agents[0].y = 27
    env.agents[0].dir = Direction.UP
    env._recalc_grid()
    env.step([Action.FORWARD])

    assert env.agents[0].x == 4
    assert env.agents[0].y == 27  # blocked by obstacle


def test_obstacle_two_agent(env_obstacle_two_agents):
    env = env_obstacle_two_agents
    env.reset()

    env.agents[0].x = 3
    env.agents[0].y = 26
    env.agents[0].dir = Direction.RIGHT

    env.agents[1].x = 4
    env.agents[1].y = 27
    env.agents[1].dir = Direction.UP
    env._recalc_grid()
    env.step([Action.FORWARD, Action.FORWARD])

    assert env.agents[0].x == 3
    assert env.agents[0].y == 26
    assert env.agents[1].x == 4
    assert env.agents[1].y == 27


def test_obstacle_three_agent_0(env_obstacle_three_agents):
    env = env_obstacle_three_agents
    env.reset()

    env.agents[0].x = 3
    env.agents[0].y = 26
    env.agents[0].dir = Direction.RIGHT

    env.agents[1].x = 4
    env.agents[1].y = 27
    env.agents[1].dir = Direction.UP

    env.agents[2].x = 4
    env.agents[2].y = 25
    env.agents[2].dir = Direction.DOWN

    env._recalc_grid()
    env.step([Action.FORWARD, Action.FORWARD, Action.FORWARD])

    assert env.agents[0].x == 3
    assert env.agents[0].y == 26
    assert env.agents[1].x == 4
    assert env.agents[1].y == 27
    assert env.agents[2].x == 4
    assert env.agents[2].y == 25


def test_obstacle_three_agent_1(env_obstacle_three_agents):
    env = env_obstacle_three_agents
    env.reset()

    env.agents[0].x = 4
    env.agents[0].y = 25
    env.agents[0].dir = Direction.DOWN

    env.agents[1].x = 4
    env.agents[1].y = 24
    env.agents[1].dir = Direction.DOWN

    env.agents[2].x = 4
    env.agents[2].y = 23
    env.agents[2].dir = Direction.DOWN

    env._recalc_grid()
    env.step([Action.FORWARD, Action.FORWARD, Action.FORWARD])

    assert env.agents[0].x == 4
    assert env.agents[0].y == 25
    assert env.agents[1].x == 4
    assert env.agents[1].y == 24
    assert env.agents[2].x == 4
    assert env.agents[2].y == 23


def test_observation_0(env_obstacle):
    env = env_obstacle
    env.reset()

    env.agents[0].x = 4
    env.agents[0].y = 25
    env.agents[0].dir = Direction.DOWN

    env._recalc_grid()
    obs, _, _, _ = env.step([Action.NOOP])
    assert obs[0][0] == 4.0
    assert obs[0][1] == 25.0
    for sensor_idx in range(9):
        if sensor_idx == 7:
            assert obs[0][8 + 8 * sensor_idx + 7] == 1.0
        else:
            assert obs[0][8 + 8 * sensor_idx + 7] == 0.0


def test_observation_1(env_obstacle):
    env = env_obstacle
    env.reset()

    env.agents[0].x = 4
    env.agents[0].y = 27
    env.agents[0].dir = Direction.DOWN

    env._recalc_grid()
    obs, _, _, _ = env.step([Action.NOOP])
    assert obs[0][0] == 4.0
    assert obs[0][1] == 27.0
    for sensor_idx in range(9):
        if sensor_idx == 1:
            assert obs[0][8 + 8 * sensor_idx + 7] == 1.0
        else:
            assert obs[0][8 + 8 * sensor_idx + 7] == 0.0