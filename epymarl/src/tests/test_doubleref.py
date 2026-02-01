import os
import sys
import pytest

project_dir = os.path.abspath(os.path.join(os.path.abspath(__file__), os.pardir, os.pardir))
sys.path.insert(0, project_dir)

from envs import REGISTRY
env_fn = REGISTRY["doubleref"]

def test_reset():
    env = env_fn()
    obs, _ = env.reset()
    for _o in obs:
        assert _o.sum() == 1.0

def test_step_done():
    env = env_fn()
    env.reset()
    for _ in range(99):
        _, done, _ = env.step([0, 0])
        assert done is False
    _, done, _ = env.step([0, 0])
    assert done is True

def test_success():
    env = env_fn()
    for _ in range(10):
        obs, _ = env.reset()
        a1 = obs[0].argmax(axis=0)
        a0 = obs[1].argmax(axis=0)
        for _ in range(100):
            reward, _, _ = env.step([a0, a1])
            assert len(reward) == 2
            assert reward[0] == 1.0 and reward[1] == 1.0

def test_failure():
    env = env_fn()
    for _ in range(1000):
        obs, _ = env.reset()
        # Wrong actions
        a1 = (obs[0].argmax(axis=0) + 1) % 5
        a0 = (obs[1].argmax(axis=0) + 1) % 5
        for _ in range(100):
            reward, _, _ = env.step([a0, a1])
            assert reward[0] == 0.0 and reward[1] == 0.0