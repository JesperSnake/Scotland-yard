import argparse
from collections import OrderedDict
from env import environment
from game_constants import board_graph
from pathlib import Path
from contextlib import nullcontext
import multiprocessing as mp
import numpy as np
import os
import torch
from helper import state_to_input_detective, state_to_input_mrx
from mrxbrain import MrXPolicy, MrXValue
from detectivebrain import DetectivePolicy, DetectiveValue
import torch.nn.functional as F
from torch.distributions import Categorical

rounds = 24
detective_amount = 4
rollout_debug = False
env = environment(detective_amount)
env.setup_game()

rng = np.random.default_rng()
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

mrx_out_size = 1866
mrx_in_size = 108

detective_out_size = 930
detective_in_size = 106

TRAIN_ITERATIONS = 500
BUFFER_TARGET_SIZE = 2048
GAMMA = 0.99
GAE_LAMBDA = 0.95
PPO_EPOCHS = 4
PPO_MINIBATCH_SIZE = 256
POLICY_LR = 3e-4
VALUE_LR = 1e-3
CHECKPOINT_EVERY = 25
ARTIFACT_DIR = Path("training_artifacts")
ROLLOUT_WORKERS = max(1, (os.cpu_count() or 1) - 1)
CURRENT_OPPONENT_PROB = 0.4
RECENT_SNAPSHOT_SAMPLE_PROB = 0.7
PAST_OPPONENT_SNAPSHOT_EVERY = 10
PAST_OPPONENT_MAX_SNAPSHOTS = 20
HISTORICAL_SNAPSHOT_CACHE_SIZE = 32

MrXPolicyNet = MrXPolicy(mrx_in_size, 256, mrx_out_size).to(DEVICE)
MrXValueNet = MrXValue(mrx_in_size, 256, 1).to(DEVICE)
PastMrXPolicyNet = MrXPolicy(mrx_in_size, 256, mrx_out_size).to(DEVICE)

DetectivePolicyNet = DetectivePolicy(detective_in_size, 256, detective_out_size).to(DEVICE)
DetectiveValueNet = DetectiveValue(detective_in_size, 256, 1).to(DEVICE)
PastDetectivePolicyNet = DetectivePolicy(detective_in_size, 256, detective_out_size).to(DEVICE)

TRAJECTORY_KEYS = (
    "state",
    "action",
    "log_prob",
    "value",
    "reward",
    "done",
    "action_mask",
    "double_action",
    "double_mask",
)

# ---------------------------------------------------------------------
# Build global action table
# ---------------------------------------------------------------------

action_table = []

for src in range(1, 200):

    # Taxi / bus / metro
    for transport in ("taxi", "bus", "metro"):
        for dst in board_graph[src].get(transport, []):

            # Native ticket
            action_table.append(
                (src, dst, transport, False)
            )

            # Black ticket
            action_table.append(
                (src, dst, transport, True)
            )

    # Water (black ticket only)
    for dst in board_graph[src].get("water", []):
        action_table.append(
            (src, dst, "water", True)
        )

N_ACTIONS = len(action_table)
print(f"Number of actions: {N_ACTIONS}")

# ---------------------------------------------------------------------
# Precompute lookup tables
# ---------------------------------------------------------------------

# action ids leaving each source node
actions_from = [[] for _ in range(200)]

# Store action metadata in arrays for fast indexing
action_dst = np.empty(N_ACTIONS, dtype=np.int16)
action_transport = np.empty(N_ACTIONS, dtype=np.int8)
action_use_black = np.empty(N_ACTIONS, dtype=bool)
action_required_ticket = np.empty(N_ACTIONS, dtype=np.int8)

TRANSPORT_TO_INT = {
    "taxi": 0,
    "bus": 1,
    "metro": 2,
    "water": 3,
}

# ticket array order:
# [taxi, bus, metro, black]
for action_id, (src, dst, transport, use_black) in enumerate(action_table):

    actions_from[src].append(action_id)

    action_dst[action_id] = dst
    action_transport[action_id] = TRANSPORT_TO_INT[transport]
    action_use_black[action_id] = use_black

    # Black-ticket action (includes all water actions)
    if use_black:
        action_required_ticket[action_id] = 3
    else:
        action_required_ticket[action_id] = TRANSPORT_TO_INT[transport]

# Convert to numpy arrays for speed
actions_from = [
    np.asarray(x, dtype=np.int32)
    for x in actions_from
]

TRANSPORT_NAMES = ("taxi", "bus", "metro", "water")

# ---------------------------------------------------------------------
# Build detective action table
# ---------------------------------------------------------------------

# Detective action = (src, dst, transport)
# No black tickets, no double tickets, no water.
detective_action_table = []

for src in range(1, 200):
    for transport in ("taxi", "bus", "metro"):
        for dst in board_graph[src].get(transport, []):
            detective_action_table.append(
                (src, dst, transport)
            )

N_DETECTIVE_ACTIONS = len(detective_action_table)
print(f"Number of detective actions: {N_DETECTIVE_ACTIONS}")

# ---------------------------------------------------------------------
# Precompute detective lookup tables
# ---------------------------------------------------------------------

detective_actions_from = [[] for _ in range(200)]

detective_action_dst = np.empty(N_DETECTIVE_ACTIONS, dtype=np.int16)
detective_action_transport = np.empty(N_DETECTIVE_ACTIONS, dtype=np.int8)
detective_action_required_ticket = np.empty(N_DETECTIVE_ACTIONS, dtype=np.int8)

for action_id, (src, dst, transport) in enumerate(detective_action_table):

    detective_actions_from[src].append(action_id)

    detective_action_dst[action_id] = dst
    detective_action_transport[action_id] = TRANSPORT_TO_INT[transport]
    detective_action_required_ticket[action_id] = TRANSPORT_TO_INT[transport]

detective_actions_from = [
    np.asarray(x, dtype=np.int32)
    for x in detective_actions_from
]

DETECTIVE_TRANSPORT_NAMES = ("taxi", "bus", "metro")

# ---------------------------------------------------------------------
# Hot-path function
# ---------------------------------------------------------------------

def build_action_mask(legal_actions, total_actions, device):
    action_mask = torch.zeros(total_actions, dtype=torch.bool, device=device)
    legal_actions_tensor = torch.as_tensor(
        legal_actions,
        dtype=torch.long,
        device=device,
    )
    action_mask[legal_actions_tensor] = True
    return action_mask


def apply_action_mask(logits, action_mask):
    mask_value = torch.finfo(logits.dtype).min
    return logits.masked_fill(~action_mask, mask_value)


def sample_from_policy(policy_output, legal_actions, total_actions):
    if legal_actions.size == 0:
        return None, None, None

    if torch.is_tensor(policy_output):
        policy_tensor = policy_output.reshape(-1)
        device = policy_tensor.device
    else:
        policy_tensor = torch.as_tensor(policy_output, dtype=torch.float32)
        device = policy_tensor.device

    action_mask = build_action_mask(
        legal_actions,
        total_actions,
        device,
    )
    masked_logits = apply_action_mask(policy_tensor, action_mask)
    dist = torch.distributions.Categorical(logits=masked_logits)
    sampled_action = dist.sample()
    action_id = int(sampled_action.item())

    return action_id, dist.log_prob(sampled_action), action_mask.detach().cpu()


def sample_double_decision(double_logits):
    dist = Categorical(logits=double_logits.reshape(-1))
    double_action = dist.sample()
    return int(double_action.item()), dist.log_prob(double_action)


def create_trajectory():
    return {key: [] for key in TRAJECTORY_KEYS}


def append_transition(trajectory, transition):
    if transition is None or transition["action"] is None:
        return

    for key in TRAJECTORY_KEYS:
        trajectory[key].append(transition[key])


def finalize_trajectory(trajectory, gamma, gae_lambda):
    if not trajectory["reward"]:
        trajectory["reward"] = []
        trajectory["advantage"] = []
        return

    rewards = np.asarray(trajectory["reward"], dtype=np.float32)
    values = np.asarray(trajectory["value"], dtype=np.float32)
    dones = np.asarray(trajectory["done"], dtype=np.float32)

    advantages = np.zeros_like(rewards, dtype=np.float32)
    gae = 0.0
    next_value = 0.0

    for t in range(rewards.shape[0] - 1, -1, -1):
        non_terminal = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_value * non_terminal - values[t]
        gae = delta + gamma * gae_lambda * non_terminal * gae
        advantages[t] = gae
        next_value = values[t]

    returns = advantages + values

    trajectory["reward"] = returns.tolist()
    trajectory["advantage"] = advantages.tolist()


def sample_action_mrx(policy_logits, current_position, ticket_dict):

    if current_position is None:
        return None, None, None, None, None, None

    # Convert dict to fixed-order array
    tickets = np.array([
        ticket_dict["taxi"],
        ticket_dict["bus"],
        ticket_dict["metro"],
        ticket_dict["black"],
    ], dtype=np.int16)

    candidates = actions_from[current_position]

    # Vectorized legal action check
    legal_mask = tickets[action_required_ticket[candidates]] > 0
    legal_actions = candidates[legal_mask]

    if legal_actions.size == 0:
        return None, None, None, None, None, None

    action_id, log_prob, action_mask = sample_from_policy(
        policy_logits,
        legal_actions,
        N_ACTIONS,
    )

    return (
        int(action_id),
        int(action_dst[action_id]),
        TRANSPORT_NAMES[action_transport[action_id]],
        bool(action_use_black[action_id]),
        log_prob,
        action_mask,
    )


def sample_action_detective(policy_logits, current_position, ticket_dict, occupied_positions):

    if current_position is None:
        return None, None, None, None, None

    # Convert dict to fixed-order array
    tickets = np.array([
        ticket_dict["taxi"],
        ticket_dict["bus"],
        ticket_dict["metro"],
    ], dtype=np.int16)

    candidates = detective_actions_from[current_position]

    # Has matching ticket
    ticket_mask = tickets[detective_action_required_ticket[candidates]] > 0

    # Cannot move onto another detective
    destinations = detective_action_dst[candidates]
    occupied_mask = ~np.isin(destinations, occupied_positions)

    legal_mask = ticket_mask & occupied_mask
    legal_actions = candidates[legal_mask]

    if legal_actions.size == 0:
        return None, None, None, None, None

    action_id, log_prob, action_mask = sample_from_policy(
        policy_logits,
        legal_actions,
        N_DETECTIVE_ACTIONS,
    )

    return (
        int(action_id),
        int(detective_action_dst[action_id]),
        DETECTIVE_TRANSPORT_NAMES[detective_action_transport[action_id]],
        log_prob,
        action_mask,
    )




# ---------------------------------------------------------------------
# Game loop
# ---------------------------------------------------------------------

def move_mr_x(
    round,
    used_double=False,
    policy_net=None,
    value_net=None,
    collect_experience=True,
):
    # Let Mr X play
    state = env.mrx_state(round)
    input_vector = state_to_input_mrx(state)
    network_input = input_vector.to(DEVICE)
    mrx_pos = state['mr_x_location']
    mrx_tickets = state['mr_x_tickets']
    policy_net = MrXPolicyNet if policy_net is None else policy_net

    # Network output
    with torch.no_grad():
        policy_logits, double_logits = policy_net(network_input)
        value = None
        if collect_experience:
            value_net = MrXValueNet if value_net is None else value_net
            value = value_net(network_input).squeeze(-1)

    double_available = (
        not used_double
        and mrx_tickets["double"] > 0
    )
    double_action = 1
    double_log_prob = None
    use_double = False

    if double_available:
        double_action, double_log_prob = sample_double_decision(double_logits)
        use_double = double_action == 0

    action_id, next_pos, transport, use_black, log_prob, action_mask = sample_action_mrx(
        policy_logits,
        mrx_pos,
        mrx_tickets,
    )

    total_log_prob = None
    if collect_experience and log_prob is not None:
        total_log_prob = log_prob
        if double_available and double_log_prob is not None:
            total_log_prob = total_log_prob + double_log_prob

    mrx_experience = None
    if collect_experience:
        mrx_experience = {'state': input_vector,
                          'action': action_id,
                          'log_prob': None if total_log_prob is None else float(total_log_prob.item()),
                          'reward': 0,
                          'value': float(value.item()),
                          'done': 0,
                          'action_mask': action_mask,
                          'double_action': int(double_action),
                          'double_mask': bool(double_available)}

    if next_pos is None:
        return action_id, next_pos, transport, use_black, mrx_pos, False, mrx_experience

    env.apply_mrx_move(
        action_id,
        next_pos,
        transport,
        use_black,
        round
    )

    if use_double:
        env.mrx.mrx_tickets["double"] -= 1
        return action_id, next_pos, transport, use_black, mrx_pos, True, mrx_experience
    else:
        return action_id, next_pos, transport, use_black, mrx_pos, False, mrx_experience


def move_detective(
    detective_id,
    round,
    policy_net=None,
    value_net=None,
    collect_experience=True,
):
    state = env.detective_state(
        detective_id=detective_id,
        round=round,
    )
    input_vector = state_to_input_detective(state)
    network_input = input_vector.to(DEVICE)
    detective_tickets = state["my_tickets"]
    detective_pos = state["my_position"]
    policy_net = DetectivePolicyNet if policy_net is None else policy_net

    occupied_positions = [
        state["detective_locations"][j]
        for j in range(detective_amount)
        if j != detective_id
    ]

    # Network output
    with torch.no_grad():
        policy_logits = policy_net(network_input)
        value = None
        if collect_experience:
            value_net = DetectiveValueNet if value_net is None else value_net
            value = value_net(network_input).squeeze(-1)

    action_id, next_pos, transport, log_prob, action_mask = sample_action_detective(
        policy_logits,
        detective_pos,
        detective_tickets,
        occupied_positions,
    )
    detective_experience = None
    if collect_experience:
        detective_experience = {'state': input_vector,
                          'action': action_id,
                          'log_prob': None if log_prob is None else float(log_prob.item()),
                          'reward': 0,
                          'value': float(value.item()),
                          'done': 0,
                          'action_mask': action_mask,
                          'double_action': 1,
                          'double_mask': False}

    if next_pos is None:
        return action_id, next_pos, transport, detective_pos, detective_experience

    env.apply_detective_move(
        detective_id,
        action_id,
        next_pos,
        transport,
    )

    return action_id, next_pos, transport, detective_pos, detective_experience

def gather_experience(
    mrx_policy_net=None,
    mrx_value_net=None,
    collect_mrx=True,
    detective_policy_net=None,
    detective_value_net=None,
    collect_detective=True,
):
    mrx_trajectory = create_trajectory()
    detective_trajectory = create_trajectory()
    env.setup_game()

    mrx_policy_net = MrXPolicyNet if mrx_policy_net is None else mrx_policy_net
    detective_policy_net = DetectivePolicyNet if detective_policy_net is None else detective_policy_net

    for round in range(rounds):

        action_id, next_pos_mrx, transport, use_black, mrx_pos, use_double, mrx_experience = move_mr_x(
            round,
            policy_net=mrx_policy_net,
            value_net=mrx_value_net,
            collect_experience=collect_mrx,
        )
        append_transition(mrx_trajectory, mrx_experience)
        # No legal moves for mrx so he lost
        if next_pos_mrx is None:
            if rollout_debug:
                print("Mr X lost, no legal moves available")
            return -1, mrx_trajectory, detective_trajectory

        # This runs if mrx used double card
        if use_double:
            action_id, next_pos_mrx, transport, use_black, mrx_pos, _, mrx_experience = move_mr_x(
                round,
                True,
                policy_net=mrx_policy_net,
                value_net=mrx_value_net,
                collect_experience=collect_mrx,
            )
            append_transition(mrx_trajectory, mrx_experience)
            if next_pos_mrx is None:
                if rollout_debug:
                    print("Mr X lost, no legal moves available during double move")
                return -1, mrx_trajectory, detective_trajectory

        if round == rounds - 1:
            if rollout_debug:
                print("Mr X reached round 24 and wins immediately")
            return 1, mrx_trajectory, detective_trajectory


        # Detectivessss vo
        for detective_id in range(detective_amount):
            action_id, next_pos, transport, detective_pos, detective_experience = move_detective(
                detective_id,
                round,
                policy_net=detective_policy_net,
                value_net=detective_value_net,
                collect_experience=collect_detective,
            )
            append_transition(detective_trajectory, detective_experience)
            if next_pos == next_pos_mrx:
                if rollout_debug:
                    print(f"Detective {detective_id} has found mr X!")
                return -1, mrx_trajectory, detective_trajectory
            
            if next_pos is None:
                if rollout_debug:
                    print(f"Detective {detective_id} has no legal moves on round {round}")
                return 1, mrx_trajectory, detective_trajectory
    return 1, mrx_trajectory, detective_trajectory


def mark_terminal_transition(trajectory, reward):
    if not trajectory["done"]:
        return

    trajectory["done"][-1] = 1
    trajectory["reward"][-1] = reward
BUFFER_KEYS = (
    "state",
    "action",
    "log_prob",
    "value",
    "reward",
    "done",
    "action_mask",
    "double_action",
    "double_mask",
    "advantage",
)


def create_empty_buffer():
    return {key: [] for key in BUFFER_KEYS}


def merge_buffer_into(buffer, other_buffer):
    for key in BUFFER_KEYS:
        buffer[key].extend(other_buffer[key])


def truncate_buffer(buffer, target_size):
    for key in BUFFER_KEYS:
        buffer[key] = buffer[key][:target_size]


def extend_buffer(buffer, trajectory):
    for key in BUFFER_KEYS:
        buffer[key].extend(trajectory[key])


def state_dict_to_cpu(model):
    return {
        key: value.detach().cpu()
        for key, value in model.state_dict().items()
    }


def load_mrx_policy_state_dict(model, state_dict):
    load_result = model.load_state_dict(state_dict, strict=False)
    if load_result.unexpected_keys:
        print(f"Mr X policy had unexpected keys: {load_result.unexpected_keys}")


def load_detective_policy_state_dict(model, state_dict):
    model.load_state_dict(state_dict)


def move_optimizer_to_device(optimizer, device):
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device)


def find_latest_checkpoint(checkpoint_dir=ARTIFACT_DIR / "checkpoints"):
    checkpoint_paths = sorted(checkpoint_dir.glob("self_play_iter_*.pt"))
    if not checkpoint_paths:
        return None
    return checkpoint_paths[-1]


def load_policy_snapshot_from_checkpoint_path(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    iteration = int(checkpoint.get("iteration", 0))
    return {
        "iteration": iteration,
        "mrx_policy_state_dict": checkpoint["mrx_policy_state_dict"],
        "detective_policy_state_dict": checkpoint["detective_policy_state_dict"],
    }


def list_historical_snapshot_paths(checkpoint_dir, current_iteration):
    eligible_checkpoint_paths = []

    for checkpoint_path in sorted(checkpoint_dir.glob("self_play_iter_*.pt")):
        try:
            iteration = int(checkpoint_path.stem.split("_")[-1])
        except ValueError:
            continue

        if iteration >= current_iteration:
            continue

        eligible_checkpoint_paths.append(checkpoint_path)

    return eligible_checkpoint_paths


def bootstrap_opponent_snapshot_bank(
    checkpoint_dir,
    current_iteration,
    include_current_snapshot=True,
    max_snapshots=PAST_OPPONENT_MAX_SNAPSHOTS,
):
    eligible_checkpoint_paths = list_historical_snapshot_paths(
        checkpoint_dir,
        current_iteration,
    )

    if max_snapshots:
        keep_count = max_snapshots - (1 if include_current_snapshot else 0)
        if keep_count > 0:
            eligible_checkpoint_paths = eligible_checkpoint_paths[-keep_count:]
        else:
            eligible_checkpoint_paths = []

    snapshot_bank = [
        load_policy_snapshot_from_checkpoint_path(checkpoint_path)
        for checkpoint_path in eligible_checkpoint_paths
    ]

    if include_current_snapshot or not snapshot_bank:
        snapshot_bank.append(create_policy_snapshot(iteration=current_iteration))

    if max_snapshots and len(snapshot_bank) > max_snapshots:
        snapshot_bank = snapshot_bank[-max_snapshots:]

    return snapshot_bank


def load_training_checkpoint(
    checkpoint_path,
    mrx_policy_optimizer,
    mrx_value_optimizer,
    detective_policy_optimizer,
    detective_value_optimizer,
    device,
):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    load_mrx_policy_state_dict(MrXPolicyNet, checkpoint["mrx_policy_state_dict"])
    MrXValueNet.load_state_dict(checkpoint["mrx_value_state_dict"])
    load_detective_policy_state_dict(
        DetectivePolicyNet,
        checkpoint["detective_policy_state_dict"],
    )
    DetectiveValueNet.load_state_dict(checkpoint["detective_value_state_dict"])

    if "mrx_policy_optimizer_state_dict" in checkpoint:
        mrx_policy_optimizer.load_state_dict(checkpoint["mrx_policy_optimizer_state_dict"])
        move_optimizer_to_device(mrx_policy_optimizer, device)

    if "mrx_value_optimizer_state_dict" in checkpoint:
        mrx_value_optimizer.load_state_dict(checkpoint["mrx_value_optimizer_state_dict"])
        move_optimizer_to_device(mrx_value_optimizer, device)

    if "detective_policy_optimizer_state_dict" in checkpoint:
        detective_policy_optimizer.load_state_dict(
            checkpoint["detective_policy_optimizer_state_dict"]
        )
        move_optimizer_to_device(detective_policy_optimizer, device)

    if "detective_value_optimizer_state_dict" in checkpoint:
        detective_value_optimizer.load_state_dict(
            checkpoint["detective_value_optimizer_state_dict"]
        )
        move_optimizer_to_device(detective_value_optimizer, device)

    return {
        "iteration": int(checkpoint.get("iteration", 0)),
        "history": checkpoint.get("history", []),
    }


def create_policy_snapshot(iteration):
    return {
        "iteration": iteration,
        "mrx_policy_state_dict": state_dict_to_cpu(MrXPolicyNet),
        "detective_policy_state_dict": state_dict_to_cpu(DetectivePolicyNet),
    }


def append_policy_snapshot(snapshot_bank, snapshot, max_snapshots=PAST_OPPONENT_MAX_SNAPSHOTS):
    snapshot_bank.append(snapshot)
    if len(snapshot_bank) > max_snapshots:
        del snapshot_bank[:len(snapshot_bank) - max_snapshots]


def get_cached_policy_snapshot(
    checkpoint_path,
    snapshot_cache,
    cache_size=HISTORICAL_SNAPSHOT_CACHE_SIZE,
):
    if snapshot_cache is None:
        return load_policy_snapshot_from_checkpoint_path(checkpoint_path)

    if checkpoint_path in snapshot_cache:
        snapshot = snapshot_cache.pop(checkpoint_path)
        snapshot_cache[checkpoint_path] = snapshot
        return snapshot

    snapshot = load_policy_snapshot_from_checkpoint_path(checkpoint_path)
    snapshot_cache[checkpoint_path] = snapshot

    if cache_size and len(snapshot_cache) > cache_size:
        snapshot_cache.popitem(last=False)

    return snapshot


def sample_recent_policy_snapshot(snapshot_bank):
    if not snapshot_bank:
        return None

    snapshot_idx = int(rng.integers(0, len(snapshot_bank)))
    return snapshot_bank[snapshot_idx]


def sample_policy_snapshot(
    recent_snapshot_bank,
    historical_snapshot_paths=None,
    historical_snapshot_cache=None,
    recent_sample_prob=RECENT_SNAPSHOT_SAMPLE_PROB,
):
    recent_snapshot_bank = recent_snapshot_bank or []
    historical_snapshot_paths = historical_snapshot_paths or []

    has_recent = len(recent_snapshot_bank) > 0
    has_historical = len(historical_snapshot_paths) > 0

    if not has_recent and not has_historical:
        return None

    if has_recent and (
        not has_historical or rng.random() < recent_sample_prob
    ):
        return sample_recent_policy_snapshot(recent_snapshot_bank)

    if has_historical:
        checkpoint_idx = int(rng.integers(0, len(historical_snapshot_paths)))
        checkpoint_path = historical_snapshot_paths[checkpoint_idx]
        return get_cached_policy_snapshot(
            checkpoint_path,
            historical_snapshot_cache,
        )

    return sample_recent_policy_snapshot(recent_snapshot_bank)


def sample_self_play_mode(
    has_old_mrx,
    has_old_detective,
    need_mrx=True,
    need_detective=True,
    mrx_fill_ratio=1.0,
    detective_fill_ratio=1.0,
    mrx_current_games=0,
    mrx_old_games=0,
    detective_current_games=0,
    detective_old_games=0,
):
    if need_mrx and need_detective:
        total_fill = mrx_fill_ratio + detective_fill_ratio
        if total_fill <= 0:
            train_mrx = True
        else:
            train_mrx = rng.random() < (mrx_fill_ratio / total_fill)
    elif need_mrx:
        train_mrx = True
    elif need_detective:
        train_mrx = False
    else:
        return "current_mrx_vs_current_detective"

    if train_mrx:
        if not has_old_detective:
            return "current_mrx_vs_current_detective"

        mrx_total_games = mrx_current_games + mrx_old_games
        target_current_games = CURRENT_OPPONENT_PROB * (mrx_total_games + 1)
        if mrx_current_games < target_current_games:
            return "current_mrx_vs_current_detective"
        return "current_mrx_vs_old_detective"
    else:
        if not has_old_mrx:
            return "current_detective_vs_current_mrx"

        detective_total_games = detective_current_games + detective_old_games
        target_current_games = CURRENT_OPPONENT_PROB * (detective_total_games + 1)
        if detective_current_games < target_current_games:
            return "current_detective_vs_current_mrx"
        return "current_detective_vs_old_mrx"


def split_target_size(total_size, num_parts):
    base = total_size // num_parts
    remainder = total_size % num_parts
    return [
        base + (1 if idx < remainder else 0)
        for idx in range(num_parts)
    ]


def init_rollout_worker():
    global DEVICE, env, rng
    global MrXPolicyNet, MrXValueNet, DetectivePolicyNet, DetectiveValueNet
    global PastMrXPolicyNet, PastDetectivePolicyNet

    DEVICE = torch.device("cpu")
    seed = int.from_bytes(os.urandom(4), "little")
    np.random.seed(seed)
    rng = np.random.default_rng(seed)
    env = environment(detective_amount)
    env.setup_game()

    MrXPolicyNet = MrXPolicy(mrx_in_size, 256, mrx_out_size).to(DEVICE)
    MrXValueNet = MrXValue(mrx_in_size, 256, 1).to(DEVICE)
    PastMrXPolicyNet = MrXPolicy(mrx_in_size, 256, mrx_out_size).to(DEVICE)
    DetectivePolicyNet = DetectivePolicy(detective_in_size, 256, detective_out_size).to(DEVICE)
    DetectiveValueNet = DetectiveValue(detective_in_size, 256, 1).to(DEVICE)
    PastDetectivePolicyNet = DetectivePolicy(detective_in_size, 256, detective_out_size).to(DEVICE)


def collect_self_play_worker(task):
    (
        mrx_target_size,
        detective_target_size,
        gamma,
        gae_lambda,
        worker_seed,
        mrx_policy_state_dict,
        mrx_value_state_dict,
        detective_policy_state_dict,
        detective_value_state_dict,
        old_mrx_policy_state_dict,
        old_detective_policy_state_dict,
    ) = task

    global rng

    np.random.seed(worker_seed)
    rng = np.random.default_rng(worker_seed)

    load_mrx_policy_state_dict(MrXPolicyNet, mrx_policy_state_dict)
    MrXValueNet.load_state_dict(mrx_value_state_dict)
    load_detective_policy_state_dict(
        DetectivePolicyNet,
        detective_policy_state_dict,
    )
    DetectiveValueNet.load_state_dict(detective_value_state_dict)

    MrXPolicyNet.eval()
    MrXValueNet.eval()
    DetectivePolicyNet.eval()
    DetectiveValueNet.eval()

    return create_self_play_buffers(
        mrx_target_size=mrx_target_size,
        detective_target_size=detective_target_size,
        gamma=gamma,
        gae_lambda=gae_lambda,
        old_mrx_policy_state_dict=old_mrx_policy_state_dict,
        old_detective_policy_state_dict=old_detective_policy_state_dict,
    )

def create_self_play_buffers(
    mrx_target_size=BUFFER_TARGET_SIZE,
    detective_target_size=BUFFER_TARGET_SIZE,
    gamma=GAMMA,
    gae_lambda=GAE_LAMBDA,
    old_mrx_policy_state_dict=None,
    old_detective_policy_state_dict=None,
):
    mrx_buffer = create_empty_buffer()
    detective_buffer = create_empty_buffer()
    has_old_mrx = old_mrx_policy_state_dict is not None
    has_old_detective = old_detective_policy_state_dict is not None

    if has_old_mrx:
        load_mrx_policy_state_dict(PastMrXPolicyNet, old_mrx_policy_state_dict)
        PastMrXPolicyNet.eval()

    if has_old_detective:
        load_detective_policy_state_dict(
            PastDetectivePolicyNet,
            old_detective_policy_state_dict,
        )
        PastDetectivePolicyNet.eval()

    episodes = 0
    mrx_wins = 0
    detective_wins = 0
    mrx_steps = []
    detective_steps = []
    mode_counts = {
        "current_mrx_vs_current_detective": 0,
        "current_mrx_vs_old_detective": 0,
        "current_detective_vs_current_mrx": 0,
        "current_detective_vs_old_mrx": 0,
    }

    while (
        len(mrx_buffer["action"]) < mrx_target_size
        or len(detective_buffer["action"]) < detective_target_size
    ):
        need_mrx = len(mrx_buffer["action"]) < mrx_target_size
        need_detective = len(detective_buffer["action"]) < detective_target_size
        mrx_fill_ratio = (mrx_target_size - len(mrx_buffer["action"])) / max(mrx_target_size, 1)
        detective_fill_ratio = (
            (detective_target_size - len(detective_buffer["action"]))
            / max(detective_target_size, 1)
        )
        mode = sample_self_play_mode(
            has_old_mrx,
            has_old_detective,
            need_mrx=need_mrx,
            need_detective=need_detective,
            mrx_fill_ratio=mrx_fill_ratio,
            detective_fill_ratio=detective_fill_ratio,
            mrx_current_games=mode_counts["current_mrx_vs_current_detective"],
            mrx_old_games=mode_counts["current_mrx_vs_old_detective"],
            detective_current_games=mode_counts["current_detective_vs_current_mrx"],
            detective_old_games=mode_counts["current_detective_vs_old_mrx"],
        )
        mode_counts[mode] += 1

        if mode == "current_mrx_vs_current_detective":
            outcome, mrx_trajectory, detective_trajectory = gather_experience(
                collect_mrx=True,
                collect_detective=False,
            )
        elif mode == "current_mrx_vs_old_detective":
            outcome, mrx_trajectory, detective_trajectory = gather_experience(
                detective_policy_net=PastDetectivePolicyNet,
                collect_mrx=True,
                collect_detective=False,
            )
        elif mode == "current_detective_vs_current_mrx":
            outcome, mrx_trajectory, detective_trajectory = gather_experience(
                collect_mrx=False,
                collect_detective=True,
            )
        else:
            outcome, mrx_trajectory, detective_trajectory = gather_experience(
                mrx_policy_net=PastMrXPolicyNet,
                collect_mrx=False,
                collect_detective=True,
            )
        episodes += 1

        # Mr X wins, detectives lose
        if outcome == 1:
            mrx_wins += 1
            mark_terminal_transition(mrx_trajectory, 1)
            mark_terminal_transition(detective_trajectory, -1)

        # Detectives win, Mr X loses
        elif outcome == -1:
            detective_wins += 1
            mark_terminal_transition(mrx_trajectory, -1)
            mark_terminal_transition(detective_trajectory, 1)

        finalize_trajectory(mrx_trajectory, gamma, gae_lambda)
        finalize_trajectory(detective_trajectory, gamma, gae_lambda)

        if mrx_trajectory["action"]:
            mrx_steps.append(len(mrx_trajectory["action"]))

        if detective_trajectory["action"]:
            detective_steps.append(len(detective_trajectory["action"]))

        extend_buffer(mrx_buffer, mrx_trajectory)
        extend_buffer(detective_buffer, detective_trajectory)

    truncate_buffer(mrx_buffer, mrx_target_size)
    truncate_buffer(detective_buffer, detective_target_size)

    stats = {
        "episodes": episodes,
        "mrx_wins": mrx_wins,
        "detective_wins": detective_wins,
        "mrx_win_rate": mrx_wins / max(episodes, 1),
        "detective_win_rate": detective_wins / max(episodes, 1),
        "total_mrx_steps": int(np.sum(mrx_steps)) if mrx_steps else 0,
        "total_detective_steps": int(np.sum(detective_steps)) if detective_steps else 0,
        "mrx_training_episodes": len(mrx_steps),
        "detective_training_episodes": len(detective_steps),
        "avg_mrx_steps": float(np.mean(mrx_steps)) if mrx_steps else 0.0,
        "avg_detective_steps": float(np.mean(detective_steps)) if detective_steps else 0.0,
        "mrx_samples": len(mrx_buffer["action"]),
        "detective_samples": len(detective_buffer["action"]),
        "current_mrx_vs_current_detective_games": mode_counts["current_mrx_vs_current_detective"],
        "current_mrx_vs_old_detective_games": mode_counts["current_mrx_vs_old_detective"],
        "current_detective_vs_current_mrx_games": mode_counts["current_detective_vs_current_mrx"],
        "current_detective_vs_old_mrx_games": mode_counts["current_detective_vs_old_mrx"],
        "mrx_current_opponent_games": mode_counts["current_mrx_vs_current_detective"],
        "mrx_old_opponent_games": mode_counts["current_mrx_vs_old_detective"],
        "detective_current_opponent_games": mode_counts["current_detective_vs_current_mrx"],
        "detective_old_opponent_games": mode_counts["current_detective_vs_old_mrx"],
        "mrx_current_opponent_rate": (
            mode_counts["current_mrx_vs_current_detective"]
            / max(
                mode_counts["current_mrx_vs_current_detective"]
                + mode_counts["current_mrx_vs_old_detective"],
                1,
            )
        ),
        "detective_current_opponent_rate": (
            mode_counts["current_detective_vs_current_mrx"]
            / max(
                mode_counts["current_detective_vs_current_mrx"]
                + mode_counts["current_detective_vs_old_mrx"],
                1,
            )
        ),
    }

    return mrx_buffer, detective_buffer, stats


def create_self_play_buffers_parallel(
    mrx_target_size=BUFFER_TARGET_SIZE,
    detective_target_size=BUFFER_TARGET_SIZE,
    gamma=GAMMA,
    gae_lambda=GAE_LAMBDA,
    num_workers=ROLLOUT_WORKERS,
    pool=None,
    recent_snapshot_bank=None,
    historical_snapshot_paths=None,
    historical_snapshot_cache=None,
):
    num_workers = max(1, num_workers)

    if num_workers == 1:
        old_mrx_snapshot = sample_policy_snapshot(
            recent_snapshot_bank,
            historical_snapshot_paths,
            historical_snapshot_cache,
        )
        old_detective_snapshot = sample_policy_snapshot(
            recent_snapshot_bank,
            historical_snapshot_paths,
            historical_snapshot_cache,
        )
        return create_self_play_buffers(
            mrx_target_size=mrx_target_size,
            detective_target_size=detective_target_size,
            gamma=gamma,
            gae_lambda=gae_lambda,
            old_mrx_policy_state_dict=None if old_mrx_snapshot is None else old_mrx_snapshot["mrx_policy_state_dict"],
            old_detective_policy_state_dict=None if old_detective_snapshot is None else old_detective_snapshot["detective_policy_state_dict"],
        )

    if pool is None:
        raise ValueError("A multiprocessing pool is required when num_workers > 1")

    mrx_targets = split_target_size(mrx_target_size, num_workers)
    detective_targets = split_target_size(detective_target_size, num_workers)

    mrx_policy_state_dict = state_dict_to_cpu(MrXPolicyNet)
    mrx_value_state_dict = state_dict_to_cpu(MrXValueNet)
    detective_policy_state_dict = state_dict_to_cpu(DetectivePolicyNet)
    detective_value_state_dict = state_dict_to_cpu(DetectiveValueNet)

    base_seed = int(rng.integers(0, 2**31 - 1))
    tasks = []

    for worker_idx in range(num_workers):
        old_mrx_snapshot = sample_policy_snapshot(
            recent_snapshot_bank,
            historical_snapshot_paths,
            historical_snapshot_cache,
        )
        old_detective_snapshot = sample_policy_snapshot(
            recent_snapshot_bank,
            historical_snapshot_paths,
            historical_snapshot_cache,
        )
        tasks.append(
            (
                mrx_targets[worker_idx],
                detective_targets[worker_idx],
                gamma,
                gae_lambda,
                base_seed + worker_idx,
                mrx_policy_state_dict,
                mrx_value_state_dict,
                detective_policy_state_dict,
                detective_value_state_dict,
                None if old_mrx_snapshot is None else old_mrx_snapshot["mrx_policy_state_dict"],
                None if old_detective_snapshot is None else old_detective_snapshot["detective_policy_state_dict"],
            )
        )

    results = pool.map(collect_self_play_worker, tasks)

    merged_mrx_buffer = create_empty_buffer()
    merged_detective_buffer = create_empty_buffer()

    episodes = 0
    mrx_wins = 0
    detective_wins = 0
    total_mrx_steps = 0
    total_detective_steps = 0
    mrx_training_episodes = 0
    detective_training_episodes = 0
    current_mrx_vs_current_detective_games = 0
    current_mrx_vs_old_detective_games = 0
    current_detective_vs_current_mrx_games = 0
    current_detective_vs_old_mrx_games = 0
    mrx_current_opponent_games = 0
    mrx_old_opponent_games = 0
    detective_current_opponent_games = 0
    detective_old_opponent_games = 0

    for worker_mrx_buffer, worker_detective_buffer, worker_stats in results:
        merge_buffer_into(merged_mrx_buffer, worker_mrx_buffer)
        merge_buffer_into(merged_detective_buffer, worker_detective_buffer)

        episodes += worker_stats["episodes"]
        mrx_wins += worker_stats["mrx_wins"]
        detective_wins += worker_stats["detective_wins"]
        total_mrx_steps += worker_stats["total_mrx_steps"]
        total_detective_steps += worker_stats["total_detective_steps"]
        mrx_training_episodes += worker_stats["mrx_training_episodes"]
        detective_training_episodes += worker_stats["detective_training_episodes"]
        current_mrx_vs_current_detective_games += worker_stats["current_mrx_vs_current_detective_games"]
        current_mrx_vs_old_detective_games += worker_stats["current_mrx_vs_old_detective_games"]
        current_detective_vs_current_mrx_games += worker_stats["current_detective_vs_current_mrx_games"]
        current_detective_vs_old_mrx_games += worker_stats["current_detective_vs_old_mrx_games"]
        mrx_current_opponent_games += worker_stats["mrx_current_opponent_games"]
        mrx_old_opponent_games += worker_stats["mrx_old_opponent_games"]
        detective_current_opponent_games += worker_stats["detective_current_opponent_games"]
        detective_old_opponent_games += worker_stats["detective_old_opponent_games"]

    truncate_buffer(merged_mrx_buffer, mrx_target_size)
    truncate_buffer(merged_detective_buffer, detective_target_size)

    stats = {
        "episodes": episodes,
        "mrx_wins": mrx_wins,
        "detective_wins": detective_wins,
        "mrx_win_rate": mrx_wins / max(episodes, 1),
        "detective_win_rate": detective_wins / max(episodes, 1),
        "total_mrx_steps": total_mrx_steps,
        "total_detective_steps": total_detective_steps,
        "mrx_training_episodes": mrx_training_episodes,
        "detective_training_episodes": detective_training_episodes,
        "avg_mrx_steps": total_mrx_steps / max(mrx_training_episodes, 1),
        "avg_detective_steps": total_detective_steps / max(detective_training_episodes, 1),
        "mrx_samples": len(merged_mrx_buffer["action"]),
        "detective_samples": len(merged_detective_buffer["action"]),
        "current_mrx_vs_current_detective_games": current_mrx_vs_current_detective_games,
        "current_mrx_vs_old_detective_games": current_mrx_vs_old_detective_games,
        "current_detective_vs_current_mrx_games": current_detective_vs_current_mrx_games,
        "current_detective_vs_old_mrx_games": current_detective_vs_old_mrx_games,
        "mrx_current_opponent_games": mrx_current_opponent_games,
        "mrx_old_opponent_games": mrx_old_opponent_games,
        "detective_current_opponent_games": detective_current_opponent_games,
        "detective_old_opponent_games": detective_old_opponent_games,
        "mrx_current_opponent_rate": (
            mrx_current_opponent_games
            / max(mrx_current_opponent_games + mrx_old_opponent_games, 1)
        ),
        "detective_current_opponent_rate": (
            detective_current_opponent_games
            / max(detective_current_opponent_games + detective_old_opponent_games, 1)
        ),
    }

    return merged_mrx_buffer, merged_detective_buffer, stats

def ppo_update(
    policy_net,
    value_net,
    optimizer_policy,
    optimizer_value,
    buffer,
    device,
    supports_double_head=False,
    epochs=4,
    minibatch_size=256,
    clip_eps=0.2,
    value_coef=0.5,
    entropy_coef=0.01,
    max_grad_norm=0.5,
):
    states = torch.stack(buffer["state"]).to(device)

    actions = torch.tensor(
        buffer["action"],
        dtype=torch.long,
        device=device,
    )

    old_log_probs = torch.tensor(
        buffer["log_prob"],
        dtype=torch.float32,
        device=device,
    )

    action_masks = torch.stack(buffer["action_mask"]).to(device=device, dtype=torch.bool)

    double_actions = torch.tensor(
        buffer["double_action"],
        dtype=torch.long,
        device=device,
    )
    double_masks = torch.tensor(
        buffer["double_mask"],
        dtype=torch.bool,
        device=device,
    )

    advantages = torch.tensor(
        buffer["advantage"],
        dtype=torch.float32,
        device=device,
    )
    advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)

    returns = torch.tensor(
        buffer["reward"],
        dtype=torch.float32,
        device=device,
    )

    n = len(actions)
    metrics = {
        "policy_loss": 0.0,
        "value_loss": 0.0,
        "entropy": 0.0,
        "approx_kl": 0.0,
        "batches": 0,
    }

    for _ in range(epochs):
        indices = torch.randperm(n, device=device)

        for start in range(0, n, minibatch_size):
            batch_idx = indices[start:start + minibatch_size]

            batch_states = states[batch_idx]
            batch_actions = actions[batch_idx]
            batch_old_log_probs = old_log_probs[batch_idx]
            batch_action_masks = action_masks[batch_idx]
            batch_double_actions = double_actions[batch_idx]
            batch_double_masks = double_masks[batch_idx]
            batch_advantages = advantages[batch_idx]
            batch_returns = returns[batch_idx]

            # -------------------------
            # Policy forward
            # -------------------------
            policy_output = policy_net(batch_states)
            if supports_double_head:
                logits, double_logits = policy_output
            else:
                logits = policy_output
                double_logits = None
            masked_logits = apply_action_mask(logits, batch_action_masks)

            dist = Categorical(logits=masked_logits)

            new_log_probs = dist.log_prob(batch_actions)
            sample_entropy = dist.entropy()

            if supports_double_head and batch_double_masks.any():
                active_double_logits = double_logits[batch_double_masks]
                active_double_actions = batch_double_actions[batch_double_masks]
                double_dist = Categorical(logits=active_double_logits)
                double_log_probs = double_dist.log_prob(active_double_actions)

                new_log_probs = new_log_probs.clone()
                new_log_probs[batch_double_masks] = (
                    new_log_probs[batch_double_masks] + double_log_probs
                )

                sample_entropy = sample_entropy.clone()
                sample_entropy[batch_double_masks] = (
                    sample_entropy[batch_double_masks] + double_dist.entropy()
                )

            entropy = sample_entropy.mean()
            approx_kl = (batch_old_log_probs - new_log_probs).mean().abs()

            ratio = torch.exp(new_log_probs - batch_old_log_probs)

            unclipped = ratio * batch_advantages

            clipped = torch.clamp(
                ratio,
                1.0 - clip_eps,
                1.0 + clip_eps,
            ) * batch_advantages

            policy_loss = -torch.min(unclipped, clipped).mean()

            # -------------------------
            # Value forward
            # -------------------------
            values = value_net(batch_states).squeeze(-1)

            value_loss = F.mse_loss(values, batch_returns)

            # -------------------------
            # Update policy network
            # -------------------------
            optimizer_policy.zero_grad()

            policy_total_loss = policy_loss - entropy_coef * entropy

            policy_total_loss.backward()

            torch.nn.utils.clip_grad_norm_(
                policy_net.parameters(),
                max_grad_norm,
            )

            optimizer_policy.step()

            # -------------------------
            # Update value network
            # -------------------------
            optimizer_value.zero_grad()

            value_total_loss = value_coef * value_loss

            value_total_loss.backward()

            torch.nn.utils.clip_grad_norm_(
                value_net.parameters(),
                max_grad_norm,
            )

            optimizer_value.step()

            metrics["policy_loss"] += float(policy_loss.item())
            metrics["value_loss"] += float(value_loss.item())
            metrics["entropy"] += float(entropy.item())
            metrics["approx_kl"] += float(approx_kl.item())
            metrics["batches"] += 1

    if metrics["batches"] == 0:
        return {
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "entropy": 0.0,
            "approx_kl": 0.0,
            "samples": n,
            "mean_return": 0.0,
        }

    batches = metrics.pop("batches")
    return {
        "policy_loss": metrics["policy_loss"] / batches,
        "value_loss": metrics["value_loss"] / batches,
        "entropy": metrics["entropy"] / batches,
        "approx_kl": metrics["approx_kl"] / batches,
        "samples": n,
        "mean_return": float(np.mean(buffer["reward"])) if buffer["reward"] else 0.0,
    }


def save_checkpoint(
    iteration,
    history,
    mrx_policy_optimizer,
    mrx_value_optimizer,
    detective_policy_optimizer,
    detective_value_optimizer,
    checkpoint_dir=ARTIFACT_DIR / "checkpoints",
):
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / f"self_play_iter_{iteration:04d}.pt"

    torch.save(
        {
            "iteration": iteration,
            "history": history,
            "mrx_policy_state_dict": MrXPolicyNet.state_dict(),
            "mrx_value_state_dict": MrXValueNet.state_dict(),
            "detective_policy_state_dict": DetectivePolicyNet.state_dict(),
            "detective_value_state_dict": DetectiveValueNet.state_dict(),
            "mrx_policy_optimizer_state_dict": mrx_policy_optimizer.state_dict(),
            "mrx_value_optimizer_state_dict": mrx_value_optimizer.state_dict(),
            "detective_policy_optimizer_state_dict": detective_policy_optimizer.state_dict(),
            "detective_value_optimizer_state_dict": detective_value_optimizer.state_dict(),
        },
        checkpoint_path,
    )

    return checkpoint_path


def plot_training_metrics(history, output_dir=ARTIFACT_DIR):
    if not history:
        return None

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not installed, skipping metric plots")
        return None

    output_dir.mkdir(parents=True, exist_ok=True)

    iterations = [entry["iteration"] for entry in history]
    mrx_value_loss = [entry["mrx"]["value_loss"] for entry in history]
    detective_value_loss = [entry["detective"]["value_loss"] for entry in history]
    mrx_entropy = [entry["mrx"]["entropy"] for entry in history]
    detective_entropy = [entry["detective"]["entropy"] for entry in history]
    avg_mrx_steps = [entry["self_play"]["avg_mrx_steps"] for entry in history]
    avg_detective_steps = [entry["self_play"]["avg_detective_steps"] for entry in history]

    fig, axes = plt.subplots(3, 1, figsize=(10, 11), sharex=True)

    axes[0].plot(iterations, mrx_value_loss, label="Mr X", linewidth=2)
    axes[0].plot(iterations, detective_value_loss, label="Detective", linewidth=2)
    axes[0].set_title("Value Loss")
    axes[0].set_ylabel("Loss")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(iterations, mrx_entropy, label="Mr X", linewidth=2)
    axes[1].plot(iterations, detective_entropy, label="Detective", linewidth=2)
    axes[1].set_title("Policy Entropy")
    axes[1].set_ylabel("Entropy")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    axes[2].plot(iterations, avg_mrx_steps, label="Mr X", linewidth=2)
    axes[2].plot(iterations, avg_detective_steps, label="Detective", linewidth=2)
    axes[2].set_title("Average Steps Per Episode")
    axes[2].set_xlabel("Iteration")
    axes[2].set_ylabel("Steps")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()

    fig.tight_layout()
    plot_path = output_dir / "training_metrics.png"
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return plot_path

def train(
    num_iterations=TRAIN_ITERATIONS,
    buffer_target_size=BUFFER_TARGET_SIZE,
    gamma=GAMMA,
    gae_lambda=GAE_LAMBDA,
    ppo_epochs=PPO_EPOCHS,
    minibatch_size=PPO_MINIBATCH_SIZE,
    policy_lr=POLICY_LR,
    value_lr=VALUE_LR,
    device=DEVICE,
    checkpoint_every=CHECKPOINT_EVERY,
    rollout_workers=ROLLOUT_WORKERS,
    from_scratch=False,
    checkpoint_dir=ARTIFACT_DIR / "checkpoints",
):
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = Path(checkpoint_dir)

    mrx_policy_optimizer = torch.optim.Adam(MrXPolicyNet.parameters(), lr=policy_lr)
    mrx_value_optimizer = torch.optim.Adam(MrXValueNet.parameters(), lr=value_lr)
    detective_policy_optimizer = torch.optim.Adam(DetectivePolicyNet.parameters(), lr=policy_lr)
    detective_value_optimizer = torch.optim.Adam(DetectiveValueNet.parameters(), lr=value_lr)

    history = []
    start_iteration = 0
    latest_checkpoint_path = None

    if not from_scratch:
        latest_checkpoint_path = find_latest_checkpoint(checkpoint_dir)

    if latest_checkpoint_path is not None:
        checkpoint_state = load_training_checkpoint(
            latest_checkpoint_path,
            mrx_policy_optimizer,
            mrx_value_optimizer,
            detective_policy_optimizer,
            detective_value_optimizer,
            device,
        )
        start_iteration = checkpoint_state["iteration"]
        history = checkpoint_state["history"]
        recent_snapshot_bank = bootstrap_opponent_snapshot_bank(
            checkpoint_dir=checkpoint_dir,
            current_iteration=start_iteration,
            include_current_snapshot=False,
        )
        historical_snapshot_paths = list_historical_snapshot_paths(
            checkpoint_dir,
            start_iteration,
        )
        print(
            f"Resuming from checkpoint {latest_checkpoint_path} "
            f"at iteration {start_iteration}"
        )
    else:
        recent_snapshot_bank = [create_policy_snapshot(iteration=0)]
        historical_snapshot_paths = []
        if from_scratch:
            print("Starting training from scratch")
        else:
            print("No saved checkpoint found, starting training from scratch")

    historical_snapshot_cache = OrderedDict()

    print(
        f"Training on {device} | additional_iterations={num_iterations} | "
        f"buffer_size={buffer_target_size} | gamma={gamma} | rollout_workers={rollout_workers}"
    )

    ctx = mp.get_context("spawn")
    use_parallel_rollouts = rollout_workers > 1

    with ctx.Pool(processes=rollout_workers, initializer=init_rollout_worker) if use_parallel_rollouts else nullcontext() as rollout_pool:
        for iteration in range(start_iteration + 1, start_iteration + num_iterations + 1):
            MrXPolicyNet.eval()
            MrXValueNet.eval()
            DetectivePolicyNet.eval()
            DetectiveValueNet.eval()

            if use_parallel_rollouts:
                mrx_buffer, detective_buffer, self_play_stats = create_self_play_buffers_parallel(
                    mrx_target_size=buffer_target_size,
                    detective_target_size=buffer_target_size,
                    gamma=gamma,
                    gae_lambda=gae_lambda,
                    num_workers=rollout_workers,
                    pool=rollout_pool,
                    recent_snapshot_bank=recent_snapshot_bank,
                    historical_snapshot_paths=historical_snapshot_paths,
                    historical_snapshot_cache=historical_snapshot_cache,
                )
            else:
                old_mrx_snapshot = sample_policy_snapshot(
                    recent_snapshot_bank,
                    historical_snapshot_paths,
                    historical_snapshot_cache,
                )
                old_detective_snapshot = sample_policy_snapshot(
                    recent_snapshot_bank,
                    historical_snapshot_paths,
                    historical_snapshot_cache,
                )
                mrx_buffer, detective_buffer, self_play_stats = create_self_play_buffers(
                    mrx_target_size=buffer_target_size,
                    detective_target_size=buffer_target_size,
                    gamma=gamma,
                    gae_lambda=gae_lambda,
                    old_mrx_policy_state_dict=None if old_mrx_snapshot is None else old_mrx_snapshot["mrx_policy_state_dict"],
                    old_detective_policy_state_dict=None if old_detective_snapshot is None else old_detective_snapshot["detective_policy_state_dict"],
                )

            MrXPolicyNet.train()
            MrXValueNet.train()
            DetectivePolicyNet.train()
            DetectiveValueNet.train()

            mrx_metrics = ppo_update(
                policy_net=MrXPolicyNet,
                value_net=MrXValueNet,
                optimizer_policy=mrx_policy_optimizer,
                optimizer_value=mrx_value_optimizer,
                buffer=mrx_buffer,
                device=device,
                supports_double_head=True,
                epochs=ppo_epochs,
                minibatch_size=minibatch_size,
            )

            detective_metrics = ppo_update(
                policy_net=DetectivePolicyNet,
                value_net=DetectiveValueNet,
                optimizer_policy=detective_policy_optimizer,
                optimizer_value=detective_value_optimizer,
                buffer=detective_buffer,
                device=device,
                epochs=ppo_epochs,
                minibatch_size=minibatch_size,
            )

            iteration_stats = {
                "iteration": iteration,
                "self_play": self_play_stats,
                "mrx": mrx_metrics,
                "detective": detective_metrics,
            }
            history.append(iteration_stats)

            print(
                f"[Iter {iteration:03d}] episodes={self_play_stats['episodes']} "
                f"mrx_win_rate={self_play_stats['mrx_win_rate']:.3f} "
                f"detective_win_rate={self_play_stats['detective_win_rate']:.3f} "
                f"avg_mrx_steps={self_play_stats['avg_mrx_steps']:.1f} "
                f"avg_detective_steps={self_play_stats['avg_detective_steps']:.1f}"
            )
            print(
                f"  Opponents: recent={len(recent_snapshot_bank)} "
                f"historical={len(historical_snapshot_paths)} "
                f"mrx_vs_current_detective={self_play_stats['current_mrx_vs_current_detective_games']} "
                f"mrx_vs_old_detective={self_play_stats['current_mrx_vs_old_detective_games']} "
                f"detective_vs_current_mrx={self_play_stats['current_detective_vs_current_mrx_games']} "
                f"detective_vs_old_mrx={self_play_stats['current_detective_vs_old_mrx_games']} "
                f"mrx_current_rate={self_play_stats['mrx_current_opponent_rate']:.3f} "
                f"detective_current_rate={self_play_stats['detective_current_opponent_rate']:.3f}"
            )
            print(
                f"  MrX: samples={mrx_metrics['samples']} "
                f"mean_return={mrx_metrics['mean_return']:.3f} "
                f"policy_loss={mrx_metrics['policy_loss']:.4f} "
                f"value_loss={mrx_metrics['value_loss']:.4f} "
                f"entropy={mrx_metrics['entropy']:.4f} "
                f"approx_kl={mrx_metrics['approx_kl']:.4f}"
            )
            print(
                f"  Detective: samples={detective_metrics['samples']} "
                f"mean_return={detective_metrics['mean_return']:.3f} "
                f"policy_loss={detective_metrics['policy_loss']:.4f} "
                f"value_loss={detective_metrics['value_loss']:.4f} "
                f"entropy={detective_metrics['entropy']:.4f} "
                f"approx_kl={detective_metrics['approx_kl']:.4f}"
            )

            if checkpoint_every and iteration % checkpoint_every == 0:
                checkpoint_path = save_checkpoint(
                    iteration=iteration,
                    history=history,
                    mrx_policy_optimizer=mrx_policy_optimizer,
                    mrx_value_optimizer=mrx_value_optimizer,
                    detective_policy_optimizer=detective_policy_optimizer,
                    detective_value_optimizer=detective_value_optimizer,
                    checkpoint_dir=checkpoint_dir,
                )
                print(f"  Saved checkpoint: {checkpoint_path}")
                historical_snapshot_paths.append(checkpoint_path)

            if iteration % PAST_OPPONENT_SNAPSHOT_EVERY == 0:
                append_policy_snapshot(
                    recent_snapshot_bank,
                    create_policy_snapshot(iteration=iteration),
                )

    final_checkpoint_path = save_checkpoint(
        iteration=start_iteration + num_iterations,
        history=history,
        mrx_policy_optimizer=mrx_policy_optimizer,
        mrx_value_optimizer=mrx_value_optimizer,
        detective_policy_optimizer=detective_policy_optimizer,
        detective_value_optimizer=detective_value_optimizer,
        checkpoint_dir=checkpoint_dir,
    )
    plot_path = plot_training_metrics(history)

    print(f"Final checkpoint: {final_checkpoint_path}")
    if plot_path is not None:
        print(f"Saved training plot: {plot_path}")

    return history


def parse_train_args():
    parser = argparse.ArgumentParser(
        description="Train Scotland Yard self-play PPO agents.",
    )
    parser.add_argument(
        "--from-scratch",
        action="store_true",
        help="Ignore saved checkpoints and start training from scratch.",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=TRAIN_ITERATIONS,
        help="Number of additional training iterations to run.",
    )
    parser.add_argument(
        "--buffer-size",
        type=int,
        default=BUFFER_TARGET_SIZE,
        help="Target number of samples per side before each PPO update.",
    )
    parser.add_argument(
        "--rollout-workers",
        type=int,
        default=ROLLOUT_WORKERS,
        help="Number of rollout workers to use for self-play.",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=CHECKPOINT_EVERY,
        help="Save a checkpoint every N iterations. Use 0 to disable periodic saves.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    mp.freeze_support()
    args = parse_train_args()
    train(
        num_iterations=args.iterations,
        buffer_target_size=args.buffer_size,
        rollout_workers=args.rollout_workers,
        checkpoint_every=args.checkpoint_every,
        from_scratch=args.from_scratch,
    )


