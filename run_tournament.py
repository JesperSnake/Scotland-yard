import argparse
from pathlib import Path

import numpy as np
import torch

from detectivebrain import DetectivePolicy
from env import environment
from game_constants import board_graph
from helper import (
    detective_input_dim,
    mrx_input_dim,
    state_to_input_detective,
    state_to_input_mrx,
)
from mrxbrain import MrXPolicy


ROUNDS = 24
DETECTIVE_AMOUNT = 4
MRX_INPUT_SIZE = mrx_input_dim(DETECTIVE_AMOUNT)
MRX_OUTPUT_SIZE = 1866
DETECTIVE_INPUT_SIZE = detective_input_dim(DETECTIVE_AMOUNT)
DETECTIVE_OUTPUT_SIZE = 930
HIDDEN_SIZE = 256
GAME_SEED_MAX = int(np.iinfo(np.uint32).max)

TRANSPORT_TO_INT = {
    "taxi": 0,
    "bus": 1,
    "metro": 2,
    "water": 3,
}
TRANSPORT_NAMES = ("taxi", "bus", "metro", "water")
DETECTIVE_TRANSPORT_NAMES = ("taxi", "bus", "metro")


def build_mrx_action_tables():
    action_table = []

    for src in range(1, 200):
        for transport in ("taxi", "bus", "metro"):
            for dst in board_graph[src].get(transport, []):
                action_table.append((src, dst, transport, False))
                action_table.append((src, dst, transport, True))

        for dst in board_graph[src].get("water", []):
            action_table.append((src, dst, "water", True))

    n_actions = len(action_table)
    actions_from = [[] for _ in range(200)]
    action_dst = np.empty(n_actions, dtype=np.int16)
    action_transport = np.empty(n_actions, dtype=np.int8)
    action_use_black = np.empty(n_actions, dtype=bool)
    action_required_ticket = np.empty(n_actions, dtype=np.int8)

    for action_id, (src, dst, transport, use_black) in enumerate(action_table):
        actions_from[src].append(action_id)
        action_dst[action_id] = dst
        action_transport[action_id] = TRANSPORT_TO_INT[transport]
        action_use_black[action_id] = use_black
        action_required_ticket[action_id] = 3 if use_black else TRANSPORT_TO_INT[transport]

    actions_from = [np.asarray(x, dtype=np.int32) for x in actions_from]

    return actions_from, action_dst, action_transport, action_use_black, action_required_ticket


def build_detective_action_tables():
    action_table = []

    for src in range(1, 200):
        for transport in ("taxi", "bus", "metro"):
            for dst in board_graph[src].get(transport, []):
                action_table.append((src, dst, transport))

    n_actions = len(action_table)
    actions_from = [[] for _ in range(200)]
    action_dst = np.empty(n_actions, dtype=np.int16)
    action_transport = np.empty(n_actions, dtype=np.int8)
    action_required_ticket = np.empty(n_actions, dtype=np.int8)

    for action_id, (src, dst, transport) in enumerate(action_table):
        actions_from[src].append(action_id)
        action_dst[action_id] = dst
        action_transport[action_id] = TRANSPORT_TO_INT[transport]
        action_required_ticket[action_id] = TRANSPORT_TO_INT[transport]

    actions_from = [np.asarray(x, dtype=np.int32) for x in actions_from]

    return actions_from, action_dst, action_transport, action_required_ticket


(
    MRX_ACTIONS_FROM,
    MRX_ACTION_DST,
    MRX_ACTION_TRANSPORT,
    MRX_ACTION_USE_BLACK,
    MRX_ACTION_REQUIRED_TICKET,
) = build_mrx_action_tables()

(
    DETECTIVE_ACTIONS_FROM,
    DETECTIVE_ACTION_DST,
    DETECTIVE_ACTION_TRANSPORT,
    DETECTIVE_ACTION_REQUIRED_TICKET,
) = build_detective_action_tables()


def build_game_rng(game_seed):
    game_seed = int(game_seed)
    np.random.seed(game_seed)
    return np.random.default_rng(game_seed)


def sample_game_seed(master_rng):
    return int(master_rng.integers(0, GAME_SEED_MAX + 1, dtype=np.uint64))


def load_policy_weights(path, model, state_dict_key, strict=True):
    checkpoint = torch.load(path, map_location="cpu")

    if isinstance(checkpoint, dict) and state_dict_key in checkpoint:
        state_dict = checkpoint[state_dict_key]
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    model_state = model.state_dict()
    compatible_state_dict = {}
    skipped_keys = []
    unexpected_keys = []

    for key, value in state_dict.items():
        if key not in model_state:
            unexpected_keys.append(key)
            continue

        if model_state[key].shape != value.shape:
            skipped_keys.append(
                (
                    key,
                    tuple(value.shape),
                    tuple(model_state[key].shape),
                )
            )
            continue

        compatible_state_dict[key] = value

    load_result = model.load_state_dict(compatible_state_dict, strict=False)
    missing_keys = [
        key
        for key in model_state.keys()
        if key not in compatible_state_dict
    ]

    if strict and (skipped_keys or unexpected_keys or missing_keys):
        raise ValueError(
            f"Checkpoint {path} is incompatible with the current model shape. "
            f"skipped_keys={skipped_keys}, unexpected_keys={unexpected_keys}, "
            f"missing_keys={missing_keys}"
        )

    if skipped_keys or unexpected_keys or missing_keys:
        print(f"Loaded {path} with partial parameter reuse.")
        if skipped_keys:
            skipped_summary = ", ".join(
                f"{key}: ckpt{checkpoint_shape} != model{model_shape}"
                for key, checkpoint_shape, model_shape in skipped_keys
            )
            print(f"  skipped incompatible tensors: {skipped_summary}")
        if unexpected_keys:
            print(f"  unexpected keys: {unexpected_keys}")
        if missing_keys or load_result.missing_keys:
            print(f"  left at init values: {missing_keys}")


def sample_from_policy(policy_logits, legal_actions, rng, greedy=True):
    if legal_actions.size == 0:
        return None

    legal_actions_tensor = torch.as_tensor(
        legal_actions,
        dtype=torch.long,
        device=policy_logits.device,
    )
    legal_logits = policy_logits.reshape(-1)[legal_actions_tensor]

    if greedy:
        max_logit = legal_logits.max()
        best_indices = torch.nonzero(legal_logits == max_logit, as_tuple=False).reshape(-1)
        selected_best_idx = int(best_indices[int(rng.integers(0, len(best_indices)))].item())
        return int(legal_actions[selected_best_idx])

    probs = torch.softmax(legal_logits, dim=0)
    probs_np = probs.detach().cpu().numpy()

    sampled_index = int(rng.choice(len(legal_actions), p=probs_np))
    return int(legal_actions[sampled_index])


def sample_double_decision(double_logits, rng, greedy=True):
    if greedy:
        max_logit = double_logits.reshape(-1).max()
        best_indices = torch.nonzero(
            double_logits.reshape(-1) == max_logit,
            as_tuple=False,
        ).reshape(-1)
        return int(best_indices[int(rng.integers(0, len(best_indices)))].item())

    probs = torch.softmax(double_logits.reshape(-1), dim=0)
    probs_np = probs.detach().cpu().numpy()
    return int(rng.choice(len(probs_np), p=probs_np))


def mrx_ticket_array(ticket_dict):
    return np.asarray(
        [
            ticket_dict["taxi"],
            ticket_dict["bus"],
            ticket_dict["metro"],
            ticket_dict["black"],
        ],
        dtype=np.int16,
    )


def detective_ticket_array(ticket_dict):
    return np.asarray(
        [
            ticket_dict["taxi"],
            ticket_dict["bus"],
            ticket_dict["metro"],
        ],
        dtype=np.int16,
    )


def detective_positions(env):
    return np.asarray(
        [int(detective.detective_pos) for detective in env.detectives],
        dtype=np.int16,
    )


def legal_mrx_action_ids_from_state(position, ticket_array, occupied_positions):
    candidates = MRX_ACTIONS_FROM[int(position)]
    ticket_mask = ticket_array[MRX_ACTION_REQUIRED_TICKET[candidates]] > 0
    occupied_mask = ~np.isin(MRX_ACTION_DST[candidates], occupied_positions)
    return candidates[ticket_mask & occupied_mask]


def has_mrx_followup_after_action(env, action_id):
    tickets_after = mrx_ticket_array(env.mrx.mrx_tickets).copy()
    tickets_after[MRX_ACTION_REQUIRED_TICKET[int(action_id)]] -= 1
    next_position = int(MRX_ACTION_DST[int(action_id)])
    return (
        legal_mrx_action_ids_from_state(
            next_position,
            tickets_after,
            detective_positions(env),
        ).size
        > 0
    )


def legal_mrx_action_ids(env, require_double_followup=False):
    legal_actions = legal_mrx_action_ids_from_state(
        env.mrx.mrx_pos,
        mrx_ticket_array(env.mrx.mrx_tickets),
        detective_positions(env),
    )
    if not require_double_followup:
        return legal_actions

    if env.move_counter >= ROUNDS - 1:
        return np.asarray([], dtype=np.int32)

    filtered = [
        int(action_id)
        for action_id in legal_actions
        if has_mrx_followup_after_action(env, int(action_id))
    ]
    return np.asarray(filtered, dtype=np.int32)


def legal_detective_action_ids(env, detective_id):
    current_detective = env.detectives[detective_id]
    candidates = DETECTIVE_ACTIONS_FROM[int(current_detective.detective_pos)]
    tickets = detective_ticket_array(current_detective.detective_tickets)
    ticket_mask = tickets[DETECTIVE_ACTION_REQUIRED_TICKET[candidates]] > 0
    occupied_positions = [
        int(env.detectives[other_id].detective_pos)
        for other_id in range(DETECTIVE_AMOUNT)
        if other_id != detective_id
    ]
    occupied_mask = ~np.isin(DETECTIVE_ACTION_DST[candidates], occupied_positions)
    return candidates[ticket_mask & occupied_mask]


def sample_action_mrx(policy_logits, current_position, ticket_dict, rng, greedy=True):
    if current_position is None:
        return None, None, None, None

    tickets = np.array(
        [
            ticket_dict["taxi"],
            ticket_dict["bus"],
            ticket_dict["metro"],
            ticket_dict["black"],
        ],
        dtype=np.int16,
    )

    candidates = MRX_ACTIONS_FROM[current_position]
    legal_mask = tickets[MRX_ACTION_REQUIRED_TICKET[candidates]] > 0
    legal_actions = candidates[legal_mask]

    if legal_actions.size == 0:
        return None, None, None, None

    action_id = sample_from_policy(policy_logits, legal_actions, rng, greedy=greedy)

    return (
        int(action_id),
        int(MRX_ACTION_DST[action_id]),
        TRANSPORT_NAMES[MRX_ACTION_TRANSPORT[action_id]],
        bool(MRX_ACTION_USE_BLACK[action_id]),
    )


def sample_action_detective(
    policy_logits,
    current_position,
    ticket_dict,
    occupied_positions,
    rng,
    greedy=True,
):
    if current_position is None:
        return None, None, None

    tickets = np.array(
        [
            ticket_dict["taxi"],
            ticket_dict["bus"],
            ticket_dict["metro"],
        ],
        dtype=np.int16,
    )

    candidates = DETECTIVE_ACTIONS_FROM[current_position]
    ticket_mask = tickets[DETECTIVE_ACTION_REQUIRED_TICKET[candidates]] > 0
    destinations = DETECTIVE_ACTION_DST[candidates]
    occupied_mask = ~np.isin(destinations, occupied_positions)
    legal_actions = candidates[ticket_mask & occupied_mask]

    if legal_actions.size == 0:
        return None, None, None

    action_id = sample_from_policy(policy_logits, legal_actions, rng, greedy=greedy)

    return (
        int(action_id),
        int(DETECTIVE_ACTION_DST[action_id]),
        DETECTIVE_TRANSPORT_NAMES[DETECTIVE_ACTION_TRANSPORT[action_id]],
    )


def move_mr_x(env, mrx_policy_net, device, round_number, rng, used_double=False, greedy=True):
    round_number = env.move_counter if round_number is None else int(round_number)
    state = env.mrx_state(round_number)
    input_vector = state_to_input_mrx(state).to(device)
    mrx_pos = state["mr_x_location"]
    mrx_tickets = state["mr_x_tickets"]

    with torch.no_grad():
        policy_logits, double_logits = mrx_policy_net(input_vector)

    legal_actions = legal_mrx_action_ids(env)
    if legal_actions.size == 0:
        return None, None, None, None, mrx_pos, False

    use_double = False
    selectable_actions = legal_actions
    if not used_double and mrx_tickets["double"] > 0:
        double_actions = legal_mrx_action_ids(env, require_double_followup=True)
        if double_actions.size > 0:
            use_double = sample_double_decision(double_logits, rng, greedy=greedy) == 0
            if use_double:
                selectable_actions = double_actions

    action_id = sample_from_policy(
        policy_logits,
        selectable_actions,
        rng,
        greedy=greedy,
    )
    next_pos = int(MRX_ACTION_DST[action_id])
    transport = TRANSPORT_NAMES[int(MRX_ACTION_TRANSPORT[action_id])]
    use_black = bool(MRX_ACTION_USE_BLACK[action_id])

    env.apply_mrx_move(
        action_id,
        next_pos,
        transport,
        use_black,
        round_number,
    )

    if use_double:
        env.mrx.mrx_tickets["double"] -= 1

    return action_id, next_pos, transport, use_black, mrx_pos, use_double


def move_detective(
    env,
    detective_policy_net,
    device,
    detective_id,
    round_number,
    rng,
    greedy=True,
):
    round_number = env.move_counter if round_number is None else int(round_number)
    state = env.detective_state(
        detective_id=detective_id,
        round=round_number,
    )
    input_vector = state_to_input_detective(state).to(device)
    detective_tickets = state["my_tickets"]
    detective_pos = state["my_position"]
    legal_actions = legal_detective_action_ids(env, detective_id)

    with torch.no_grad():
        policy_logits = detective_policy_net(input_vector)

    if legal_actions.size == 0:
        return None, None, None, detective_pos

    action_id = sample_from_policy(
        policy_logits,
        legal_actions,
        rng,
        greedy=greedy,
    )
    next_pos = int(DETECTIVE_ACTION_DST[action_id])
    transport = DETECTIVE_TRANSPORT_NAMES[int(DETECTIVE_ACTION_TRANSPORT[action_id])]

    env.apply_detective_move(
        detective_id,
        action_id,
        next_pos,
        transport,
    )

    return action_id, next_pos, transport, detective_pos


def play_game(env, mrx_policy_net, detective_policy_net, device, rng, greedy=True):
    env.setup_game()

    while env.move_counter < ROUNDS:
        (
            _action_id,
            next_pos_mrx,
            _transport,
            _use_black,
            _mrx_pos,
            use_double,
        ) = move_mr_x(
            env,
            mrx_policy_net,
            device,
            env.move_counter,
            rng,
            greedy=greedy,
        )

        if next_pos_mrx is None:
            return "detective"

        if use_double:
            (
                _action_id,
                next_pos_mrx,
                _transport,
                _use_black,
                _mrx_pos,
                _,
            ) = move_mr_x(
                env,
                mrx_policy_net,
                device,
                env.move_counter,
                rng,
                used_double=True,
                greedy=greedy,
            )

            if next_pos_mrx is None:
                return "detective"

        if env.move_counter >= ROUNDS:
            return "mrx"

        for detective_id in range(DETECTIVE_AMOUNT):
            _action_id, next_pos, _transport, _detective_pos = move_detective(
                env,
                detective_policy_net,
                device,
                detective_id,
                env.move_counter,
                rng,
                greedy=greedy,
            )

            if next_pos == next_pos_mrx:
                return "detective"

            if next_pos is None:
                continue

    return "mrx"


def run_tournament(mrx_path, detective_path, games, device, seed=None, greedy=True):
    master_rng = np.random.default_rng(seed)
    env = environment(DETECTIVE_AMOUNT)

    mrx_policy_net = MrXPolicy(MRX_INPUT_SIZE, HIDDEN_SIZE, MRX_OUTPUT_SIZE).to(device)
    detective_policy_net = DetectivePolicy(DETECTIVE_INPUT_SIZE, HIDDEN_SIZE, DETECTIVE_OUTPUT_SIZE).to(device)

    load_policy_weights(
        mrx_path,
        mrx_policy_net,
        "mrx_policy_state_dict",
        strict=False,
    )
    load_policy_weights(
        detective_path,
        detective_policy_net,
        "detective_policy_state_dict",
        strict=False,
    )

    mrx_policy_net.eval()
    detective_policy_net.eval()

    mrx_wins = 0
    detective_wins = 0
    example_mrx_seed = None
    example_detective_seed = None

    for game_idx in range(1, games + 1):
        game_seed = sample_game_seed(master_rng)
        winner = play_game(
            env,
            mrx_policy_net,
            detective_policy_net,
            device,
            build_game_rng(game_seed),
            greedy=greedy,
        )

        if winner == "mrx":
            mrx_wins += 1
            if example_mrx_seed is None:
                example_mrx_seed = game_seed
        else:
            detective_wins += 1
            if example_detective_seed is None:
                example_detective_seed = game_seed

        if game_idx % 1000 == 0 or game_idx == games:
            print(
                f"Completed {game_idx}/{games} games | "
                f"Mr X wins: {mrx_wins} | Detective wins: {detective_wins}"
            )

    mrx_win_rate = mrx_wins / games
    detective_win_rate = detective_wins / games

    print()
    print("Tournament Results")
    print(f"Mr X checkpoint: {mrx_path}")
    print(f"Detective checkpoint: {detective_path}")
    print(f"Policy mode: {'greedy' if greedy else 'sampled'}")
    print(f"Tournament seed: {seed if seed is not None else 'random'}")
    print(f"Games played: {games}")
    print(f"Mr X wins: {mrx_wins} ({mrx_win_rate:.4%})")
    print(f"Detective wins: {detective_wins} ({detective_win_rate:.4%})")
    print(
        "Example Mr X-winning seed: "
        + (str(example_mrx_seed) if example_mrx_seed is not None else "none found")
    )
    print(
        "Example detective-winning seed: "
        + (
            str(example_detective_seed)
            if example_detective_seed is not None
            else "none found"
        )
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run a tournament between a Mr X policy and a detective policy.",
    )
    parser.add_argument(
        "--mrx-path",
        required=True,
        help="Path to a Mr X checkpoint or Mr X policy state dict.",
    )
    parser.add_argument(
        "--detective-path",
        required=True,
        help="Path to a detective checkpoint or detective policy state dict.",
    )
    parser.add_argument(
        "--games",
        type=int,
        default=10000,
        help="Number of games to play. Default: 10000",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device to use, for example cpu or cuda.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help=(
            "Optional tournament seed used to generate one reproducible seed per game."
        ),
    )
    parser.add_argument(
        "--sampled",
        action="store_true",
        help="Use stochastic policy sampling instead of greedy action selection.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_tournament(
        mrx_path=Path(args.mrx_path),
        detective_path=Path(args.detective_path),
        games=args.games,
        device=torch.device(args.device),
        seed=args.seed,
        greedy=not args.sampled,
    )
