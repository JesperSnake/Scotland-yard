import argparse
from pathlib import Path

import numpy as np
import torch

from detectivebrain import DetectivePolicy
from env import environment
from game_constants import board_graph
from helper import state_to_input_detective, state_to_input_mrx
from mrxbrain import MrXPolicy


ROUNDS = 24
DETECTIVE_AMOUNT = 4
DOUBLE_PROBABILITY = 0.1
MRX_INPUT_SIZE = 108
MRX_OUTPUT_SIZE = 1866
DETECTIVE_INPUT_SIZE = 106
DETECTIVE_OUTPUT_SIZE = 930
HIDDEN_SIZE = 256

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


def load_policy_weights(path, model, state_dict_key):
    checkpoint = torch.load(path, map_location="cpu")

    if isinstance(checkpoint, dict) and state_dict_key in checkpoint:
        state_dict = checkpoint[state_dict_key]
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)


def sample_from_policy(policy_logits, legal_actions, rng):
    if legal_actions.size == 0:
        return None

    legal_actions_tensor = torch.as_tensor(
        legal_actions,
        dtype=torch.long,
        device=policy_logits.device,
    )
    legal_logits = policy_logits.reshape(-1)[legal_actions_tensor]
    probs = torch.softmax(legal_logits, dim=0)
    probs_np = probs.detach().cpu().numpy()

    sampled_index = int(rng.choice(len(legal_actions), p=probs_np))
    return int(legal_actions[sampled_index])


def sample_action_mrx(policy_logits, current_position, ticket_dict, rng):
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

    action_id = sample_from_policy(policy_logits, legal_actions, rng)

    return (
        int(action_id),
        int(MRX_ACTION_DST[action_id]),
        TRANSPORT_NAMES[MRX_ACTION_TRANSPORT[action_id]],
        bool(MRX_ACTION_USE_BLACK[action_id]),
    )


def sample_action_detective(policy_logits, current_position, ticket_dict, occupied_positions, rng):
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

    action_id = sample_from_policy(policy_logits, legal_actions, rng)

    return (
        int(action_id),
        int(DETECTIVE_ACTION_DST[action_id]),
        DETECTIVE_TRANSPORT_NAMES[DETECTIVE_ACTION_TRANSPORT[action_id]],
    )


def move_mr_x(env, mrx_policy_net, device, round_number, rng, used_double=False):
    state = env.mrx_state(round_number)
    input_vector = state_to_input_mrx(state).to(device)
    mrx_pos = state["mr_x_location"]
    mrx_tickets = state["mr_x_tickets"]

    with torch.no_grad():
        policy_logits = mrx_policy_net(input_vector)

    use_double = (
        not used_double
        and mrx_tickets["double"] > 0
        and rng.random() < DOUBLE_PROBABILITY
    )

    action_id, next_pos, transport, use_black = sample_action_mrx(
        policy_logits,
        mrx_pos,
        mrx_tickets,
        rng,
    )

    if next_pos is None:
        return action_id, next_pos, transport, use_black, mrx_pos, False

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


def move_detective(env, detective_policy_net, device, detective_id, round_number, rng):
    state = env.detective_state(
        detective_id=detective_id,
        round=round_number,
    )
    input_vector = state_to_input_detective(state).to(device)
    detective_tickets = state["my_tickets"]
    detective_pos = state["my_position"]
    occupied_positions = [
        state["detective_locations"][j]
        for j in range(DETECTIVE_AMOUNT)
        if j != detective_id
    ]

    with torch.no_grad():
        policy_logits = detective_policy_net(input_vector)

    action_id, next_pos, transport = sample_action_detective(
        policy_logits,
        detective_pos,
        detective_tickets,
        occupied_positions,
        rng,
    )

    if next_pos is None:
        return action_id, next_pos, transport, detective_pos

    env.apply_detective_move(
        detective_id,
        action_id,
        next_pos,
        transport,
    )

    return action_id, next_pos, transport, detective_pos


def play_game(env, mrx_policy_net, detective_policy_net, device, rng):
    env.setup_game()

    for round_number in range(ROUNDS):
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
            round_number,
            rng,
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
                round_number,
                rng,
                used_double=True,
            )

            if next_pos_mrx is None:
                return "detective"

        if round_number == ROUNDS - 1:
            return "mrx"

        for detective_id in range(DETECTIVE_AMOUNT):
            _action_id, next_pos, _transport, _detective_pos = move_detective(
                env,
                detective_policy_net,
                device,
                detective_id,
                round_number,
                rng,
            )

            if next_pos == next_pos_mrx:
                return "detective"

            if next_pos is None:
                return "mrx"

    return "mrx"


def run_tournament(mrx_path, detective_path, games, device, seed=None):
    if seed is not None:
        np.random.seed(seed)

    rng = np.random.default_rng(seed)
    env = environment(DETECTIVE_AMOUNT)

    mrx_policy_net = MrXPolicy(MRX_INPUT_SIZE, HIDDEN_SIZE, MRX_OUTPUT_SIZE).to(device)
    detective_policy_net = DetectivePolicy(DETECTIVE_INPUT_SIZE, HIDDEN_SIZE, DETECTIVE_OUTPUT_SIZE).to(device)

    load_policy_weights(mrx_path, mrx_policy_net, "mrx_policy_state_dict")
    load_policy_weights(detective_path, detective_policy_net, "detective_policy_state_dict")

    mrx_policy_net.eval()
    detective_policy_net.eval()

    mrx_wins = 0
    detective_wins = 0

    for game_idx in range(1, games + 1):
        winner = play_game(
            env,
            mrx_policy_net,
            detective_policy_net,
            device,
            rng,
        )

        if winner == "mrx":
            mrx_wins += 1
        else:
            detective_wins += 1

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
    print(f"Games played: {games}")
    print(f"Mr X wins: {mrx_wins} ({mrx_win_rate:.4%})")
    print(f"Detective wins: {detective_wins} ({detective_win_rate:.4%})")


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
        help="Optional random seed for reproducibility.",
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
    )
