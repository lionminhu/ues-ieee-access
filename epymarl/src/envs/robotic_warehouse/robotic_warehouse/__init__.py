import gym
from .warehouse import Warehouse, RewardType, Action
from .warehouse_toy import WarehouseToy
from .warehouse_custom import WarehouseCustom
import itertools
import numpy as np
import copy

version_name = "2"
toy_version_name = "2"
custom_version_name = "6"

_sizes = {
    "tiny": (1, 3),
    "small": (2, 3),
    "medium": (2, 5),
    "large": (3, 5),
}

_difficulty = {"-easy": 2, "": 1, "-hard": 0.5}

_perms = itertools.product(_sizes.keys(), _difficulty, range(1, 20),)

for size, diff, agents in _perms:
    # normal tasks
    gym.register(
        id=f"rware-{size}-{agents}ag{diff}-v{version_name}",
        entry_point="robotic_warehouse.warehouse:Warehouse",
        kwargs={
            "column_height": 8,
            "shelf_rows": _sizes[size][0],
            "shelf_columns": _sizes[size][1],
            "n_agents": agents,
            "msg_bits": 0,
            "sensor_range": 1,
            "request_queue_size": int(agents * _difficulty[diff]),
            "max_inactivity_steps": None,
            "max_steps": 500,
            "reward_type": RewardType.INDIVIDUAL,
        },
    )


def full_registration():
    global version_name
    _perms = itertools.product(
        range(1, 5),
        range(3, 10, 2),
        range(1, 20),
        range(1, 20),
        ["indiv", "global", "twostage"],
    )
    _rewards = {
        "indiv": RewardType.INDIVIDUAL,
        "global": RewardType.GLOBAL,
        "twostage": RewardType.TWO_STAGE,
    }

    for rows, cols, agents, req, rew in _perms:
        gym.register(
            id=f"rware-{rows}x{cols}-{agents}ag-{req}req-{rew}-v{version_name}",
            entry_point="robotic_warehouse.warehouse:Warehouse",
            kwargs={
                "column_height": 8,
                "shelf_rows": rows,
                "shelf_columns": cols,
                "n_agents": agents,
                "msg_bits": 0,
                "sensor_range": 1,
                "request_queue_size": req,
                "max_inactivity_steps": None,
                "max_steps": 500,
                "reward_type": _rewards[rew],
            },
        )

full_registration()


obstacles_pos = {
    0: None,
    2: [(4, 8), (5, 6)],
    4: [(4, 8), (5, 6), (4, 4), (5, 2)]
}

_perms = itertools.product(
    [0, 2, 4],
    [2, 4, 6, 10]
)

for num_obst, num_agent in _perms:
    kwargs = {
        "column_height": 8,
        "shelf_rows": 1,
        "shelf_columns": 3,
        "n_agents": num_agent,
        "msg_bits": 0,
        "sensor_range": 1,
        "request_queue_size": num_agent,
        "max_inactivity_steps": None,
        "max_steps": 500,
        "obstacles": obstacles_pos[num_obst]
    }
    gym.register(
        id=f"rware-{num_obst}obst-{num_agent}ag-v{version_name}",
        entry_point="robotic_warehouse.warehouse:Warehouse",
        kwargs={
            **kwargs,
            "reward_type": RewardType.INDIVIDUAL,
        }
    )
    gym.register(
        id=f"rware-{num_obst}obst-{num_agent}ag-twostage-v{version_name}",
        entry_point="robotic_warehouse.warehouse:Warehouse",
        kwargs={
            **kwargs,
            "reward_type": RewardType.TWO_STAGE,
        }
    )

_perms = itertools.product(
    [2, 4, 6],   # num agents
    [5, 10, 20, 40, 60, 80],   # shelf time limit
    [5.0, 1.0, 3.0],     # shelf timeout penalty mantissa
    [0, -1, -2, -3]      # shelf timeout penalty exponent
)

def apply_num_str_format(mantissa, pow):
    num = mantissa * (10 ** pow)
    mantissa_str = str(mantissa).replace(".", "")
    if pow < 0:
        pow_str = f"n{abs(int(pow))}"
    else:
        pow_str = f"{int(pow)}"
    return num, f"{mantissa_str}e{pow_str}"

for num_agents, shelf_time_limit, timeout_pen_mantissa, timeout_pen_exp in _perms:
    timeout_penalty, penalty_str = apply_num_str_format(timeout_pen_mantissa, timeout_pen_exp)
    env_id = f"rware-{shelf_time_limit}st-top{penalty_str}-{num_agents}ag-v{version_name}"
    gym.register(
        id=env_id,
        entry_point="robotic_warehouse.warehouse:Warehouse",
        kwargs={
            "column_height": 8,
            "shelf_rows": 1,
            "shelf_columns": 3,
            "n_agents": num_agents,
            "msg_bits": 0,
            "sensor_range": 1,
            "request_queue_size": num_agents,
            "max_inactivity_steps": None,
            "max_steps": 500,
            "reward_type": RewardType.INDIVIDUAL,
            "shelf_time_limit": shelf_time_limit,
            "shelf_timeout_penalty": timeout_penalty,
        }
    )



for num_agents in range(2, 11):
    env_id = f"rware-toy-agentzerogen-{num_agents}ag-v{toy_version_name}"
    gym.register(
        id=env_id,
        entry_point="robotic_warehouse.warehouse_toy:WarehouseToy",
        kwargs={
            "n_agents": num_agents,
        }
    )


kwargs = {
    "obstacles": np.array([
        [1, 0, 1, 1, 1],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [1, 0, 1, 1, 1]
    ]),
    "agents": [(1, 3), (1, 4), (2, 4)],
    "shelves": [(0, 1), (1, 1), (2, 1)],
    "req_shelf": (0, 1),
    "goals": [(3, 1)],
}
env_id = f"rware-toy-agentzerogen-wider-3ag-v{custom_version_name}"
gym.register(
    id=env_id,
    entry_point="robotic_warehouse.warehouse_custom:WarehouseCustom",
    kwargs=kwargs
)


####### for IJCAI25

kwargs = dict(
    obstacles = np.zeros((9, 7)),
    shelves = list(itertools.product([1, 2, 6, 7], [1, 2, 3, 4])),
    goals = [(4, 6)],
    add_rand_obst = True,
    rand_obst_pos_list = list(itertools.product([3, 4, 5], [1, 2, 3, 4])),
    rand_obst_num = 3,
    random_agent_pos = True,
    rand_num_agents = 2,
    random_request = True,
    request_queue_size = 4,
    reward_type = RewardType.TWO_STAGE,
    block_shelf_to_shelf = False,
    terminate_on_deliv = False,
)
env_id = f"rware-ijcai25-v{custom_version_name}"
gym.register(
    id=env_id,
    entry_point="robotic_warehouse.warehouse_custom:WarehouseCustom",
    kwargs=kwargs,
)

# _kwargs = copy.deepcopy(kwargs)
# _kwargs["shelves"] = list(itertools.product([2, 3, 5, 6], [1, 2, 3, 4]))
# env_id = f"rware-ijcai25-shelfshift-v{custom_version_name}"
# gym.register(
#     id=env_id,
#     entry_point="robotic_warehouse.warehouse_custom:WarehouseCustom",
#     kwargs=_kwargs,
# )

_kwargs = copy.deepcopy(kwargs)
_kwargs["shelves"] = list(itertools.product([0, 1, 7, 8], [1, 2, 3, 4]))
env_id = f"rware-ijcai25-shelfshift-hard-v{custom_version_name}"
gym.register(
    id=env_id,
    entry_point="robotic_warehouse.warehouse_custom:WarehouseCustom",
    kwargs=_kwargs,
)

_kwargs = copy.deepcopy(kwargs)
_kwargs["goals"] = [(2, 6), (4, 6), (6, 6)]
env_id = f"rware-ijcai25-goalshift-v{custom_version_name}"
gym.register(
    id=env_id,
    entry_point="robotic_warehouse.warehouse_custom:WarehouseCustom",
    kwargs=_kwargs,
)

###### AAAI26

kwargs = dict(
    obstacles = np.zeros((5, 2)),
    shelves = list(itertools.product([0], [0, 1])),
    goals = [(2, 1)],
    add_rand_obst = False,
    random_agent_pos = True,
    rand_num_agents = 2,
    random_request = True,
    request_queue_size = 1,
    reward_type = RewardType.TWO_STAGE,
    block_shelf_to_shelf = False,
    terminate_on_deliv = False,
)
env_id = f"rware-aaai26-tinynorth-v{custom_version_name}"
gym.register(
    id=env_id,
    entry_point="robotic_warehouse.warehouse_custom:WarehouseCustom",
    kwargs=kwargs,
)

kwargs = dict(
    obstacles = np.zeros((5, 2)),
    shelves = list(itertools.product([4], [0, 1])),
    goals = [(2, 1)],
    add_rand_obst = False,
    random_agent_pos = True,
    rand_num_agents = 2,
    random_request = True,
    request_queue_size = 1,
    reward_type = RewardType.TWO_STAGE,
    block_shelf_to_shelf = False,
    terminate_on_deliv = False,
)
env_id = f"rware-aaai26-tinysouth-v{custom_version_name}"
gym.register(
    id=env_id,
    entry_point="robotic_warehouse.warehouse_custom:WarehouseCustom",
    kwargs=kwargs,
)