import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from env import environment
from game_constants import detective_start_positions, mrx_start_positions, reveal_rounds
from helper import state_to_input_mrx
from run_tournament import (
    DETECTIVE_ACTION_DST,
    DETECTIVE_ACTION_REQUIRED_TICKET,
    DETECTIVE_ACTION_TRANSPORT,
    DETECTIVE_ACTIONS_FROM,
    DETECTIVE_AMOUNT,
    ROUNDS,
    MRX_ACTION_DST,
    MRX_ACTION_REQUIRED_TICKET,
    MRX_ACTION_TRANSPORT,
    MRX_ACTION_USE_BLACK,
    MRX_ACTIONS_FROM,
    TRANSPORT_NAMES,
    DETECTIVE_TRANSPORT_NAMES,
    build_game_rng,
    move_detective,
    sample_double_decision,
    sample_from_policy,
)
from visualize_game import (
    ReplayViewer,
    build_frame,
    build_layout,
    describe_frame,
    load_policies,
    parse_generation_arg,
    resolve_checkpoint_path,
)

class UserQuit(Exception):
    pass


def normalize_side(text):
    value = text.strip().lower().replace(" ", "").replace("-", "")
    if value in {"mrx", "mrx.", "mr.x", "x"}:
        return "mrx"
    if value in {"detective", "detectives", "d", "4detectives", "fourdetectives"}:
        return "detectives"
    return None


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Play a live Scotland Yard game against the trained AI while "
            "manually entering the opposing side's moves."
        ),
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path("training_artifacts/checkpoints"),
        help="Directory containing self_play_iter_XXXX.pt checkpoints.",
    )
    parser.add_argument(
        "--mrx-generation",
        type=parse_generation_arg,
        default="latest",
        help="Mr X generation number to load, or 'latest'.",
    )
    parser.add_argument(
        "--detective-generation",
        type=parse_generation_arg,
        default="latest",
        help="Detective generation number to load, or 'latest'.",
    )
    parser.add_argument(
        "--mrx-path",
        type=Path,
        default=None,
        help="Optional explicit Mr X checkpoint path. Overrides --mrx-generation.",
    )
    parser.add_argument(
        "--detective-path",
        type=Path,
        default=None,
        help=(
            "Optional explicit detective checkpoint path. "
            "Overrides --detective-generation."
        ),
    )
    parser.add_argument(
        "--ai-side",
        choices=("mrx", "detectives"),
        default=None,
        help="Skip the side prompt and force the AI-controlled side.",
    )
    parser.add_argument(
        "--human-side",
        dest="legacy_human_side",
        choices=("mrx", "detectives"),
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device to use, for example cpu or cuda.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed used for AI tie-breaking and stochastic sampling. Default: 0",
    )
    parser.add_argument(
        "--sampled",
        action="store_true",
        help="Use stochastic policy sampling instead of greedy action selection.",
    )
    parser.add_argument(
        "--layout-seed",
        type=int,
        default=42,
        help="Seed used for the graph layout. Default: 42",
    )
    parser.add_argument(
        "--interval-ms",
        type=int,
        default=900,
        help="Viewer autoplay interval in milliseconds. Default: 900",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Do not open the live matplotlib board window.",
    )
    parser.add_argument(
        "--show-all-node-labels",
        action="store_true",
        help="Draw labels for all 199 board nodes instead of only key nodes.",
    )
    return parser.parse_args()


def read_input(prompt_text):
    value = input(prompt_text).strip()
    if value.lower() in {"q", "quit", "exit"}:
        raise UserQuit()
    return value


def prompt_yes_no(prompt_text, default=False):
    suffix = "[Y/n]" if default else "[y/N]"
    while True:
        value = read_input(f"{prompt_text} {suffix} ").lower()
        if not value:
            return default
        if value in {"y", "yes"}:
            return True
        if value in {"n", "no"}:
            return False
        print("Please answer yes or no, or type 'quit'.")


def prompt_side(explicit_side=None):
    if explicit_side is not None:
        return explicit_side

    while True:
        value = read_input("Which side should the AI control, Mr X or the 4 detectives? ")
        side = normalize_side(value)
        if side is not None:
            return side
        print("Enter 'mrx' or 'detectives'.")


def prompt_start_position(label, allowed_positions, occupied_positions):
    allowed_sorted = sorted(int(position) for position in allowed_positions)
    allowed_text = ", ".join(str(position) for position in allowed_sorted)

    while True:
        value = read_input(f"{label} start node ({allowed_text}): ")
        try:
            position = int(value)
        except ValueError:
            print("Enter a node number.")
            continue

        if position not in allowed_positions:
            print(f"{position} is not a valid start node for {label}.")
            continue
        if position in occupied_positions:
            print(f"Node {position} is already occupied.")
            continue
        return position


def prompt_initial_positions():
    occupied_positions = set()

    mrx_position = prompt_start_position(
        "Mr X",
        set(mrx_start_positions),
        occupied_positions,
    )
    occupied_positions.add(mrx_position)

    detective_positions = []
    for detective_id in range(DETECTIVE_AMOUNT):
        detective_position = prompt_start_position(
            f"Detective {detective_id + 1}",
            set(detective_start_positions),
            occupied_positions,
        )
        detective_positions.append(detective_position)
        occupied_positions.add(detective_position)

    return mrx_position, detective_positions


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


def mrx_action_details(action_id):
    action_id = int(action_id)
    return (
        int(MRX_ACTION_DST[action_id]),
        TRANSPORT_NAMES[int(MRX_ACTION_TRANSPORT[action_id])],
        bool(MRX_ACTION_USE_BLACK[action_id]),
    )


def detective_action_details(action_id):
    action_id = int(action_id)
    return (
        int(DETECTIVE_ACTION_DST[action_id]),
        DETECTIVE_TRANSPORT_NAMES[int(DETECTIVE_ACTION_TRANSPORT[action_id])],
    )


def format_mrx_action(action_id):
    destination, transport, use_black = mrx_action_details(action_id)
    if transport == "water":
        return f"{destination}(black via water)"
    if use_black:
        return f"{destination}(black via {transport})"
    return f"{destination}({transport})"


def format_detective_action(action_id):
    destination, transport = detective_action_details(action_id)
    return f"{destination}({transport})"


def format_legal_actions(action_ids, actor):
    if len(action_ids) == 0:
        return "none"
    action_ids = [int(action_id) for action_id in action_ids]
    if actor == "mrx":
        action_ids = sorted(
            action_ids,
            key=lambda action_id: (
                int(MRX_ACTION_DST[action_id]),
                int(MRX_ACTION_TRANSPORT[action_id]),
                bool(MRX_ACTION_USE_BLACK[action_id]),
            ),
        )
        return ", ".join(format_mrx_action(action_id) for action_id in action_ids)
    action_ids = sorted(
        action_ids,
        key=lambda action_id: (
            int(DETECTIVE_ACTION_DST[action_id]),
            int(DETECTIVE_ACTION_TRANSPORT[action_id]),
        ),
    )
    return ", ".join(
        format_detective_action(action_id)
        for action_id in action_ids
    )


def parse_move_tokens(text):
    return text.lower().replace(",", " ").split()


def parse_human_mrx_action(text, legal_action_ids):
    tokens = parse_move_tokens(text)
    if not tokens:
        return None

    destination = None
    transport = None
    use_black = False
    for token in tokens:
        if token.isdigit():
            destination = int(token)
        elif token in TRANSPORT_NAMES:
            transport = token
        elif token == "black":
            use_black = True
        elif token == "help":
            return "help"

    if destination is None or transport is None:
        return None

    if transport == "water":
        use_black = True

    matches = []
    for action_id in legal_action_ids:
        legal_destination, legal_transport, legal_use_black = mrx_action_details(action_id)
        if (
            destination == legal_destination
            and transport == legal_transport
            and use_black == legal_use_black
        ):
            matches.append(int(action_id))

    if not matches and not use_black and transport != "water":
        regular_matches = []
        for action_id in legal_action_ids:
            legal_destination, legal_transport, legal_use_black = mrx_action_details(action_id)
            if (
                destination == legal_destination
                and transport == legal_transport
                and not legal_use_black
            ):
                regular_matches.append(int(action_id))
        if len(regular_matches) == 1:
            return regular_matches[0]

    if len(matches) == 1:
        return matches[0]
    return None


def parse_human_detective_action(text, legal_action_ids):
    tokens = parse_move_tokens(text)
    if not tokens:
        return None

    destination = None
    transport = None
    for token in tokens:
        if token.isdigit():
            destination = int(token)
        elif token in DETECTIVE_TRANSPORT_NAMES:
            transport = token
        elif token == "help":
            return "help"

    if destination is None or transport is None:
        return None

    matches = []
    for action_id in legal_action_ids:
        legal_destination, legal_transport = detective_action_details(action_id)
        if destination == legal_destination and transport == legal_transport:
            matches.append(int(action_id))

    if len(matches) == 1:
        return matches[0]
    return None


def apply_human_mrx_action(env, round_number, action_id):
    round_number = env.move_counter if round_number is None else int(round_number)
    previous_position = int(env.mrx.mrx_pos)
    next_position, transport, use_black = mrx_action_details(action_id)
    env.apply_mrx_move(
        action_id,
        next_position,
        transport,
        use_black,
        round_number,
    )
    return action_id, next_position, transport, use_black, previous_position


def prompt_human_detective_action(env, detective_id):
    legal_actions = legal_detective_action_ids(env, detective_id)
    current_position = int(env.detectives[detective_id].detective_pos)

    if legal_actions.size == 0:
        return None, None, None, current_position

    print(
        f"Detective {detective_id + 1} from {current_position}. "
        f"Legal actions: {format_legal_actions(legal_actions, 'detective')}"
    )

    while True:
        value = read_input(
            f"Enter detective {detective_id + 1} action as "
            "'destination transport': "
        )
        action_id = parse_human_detective_action(value, legal_actions)
        if action_id == "help":
            print(
                f"Legal actions: {format_legal_actions(legal_actions, 'detective')}"
            )
            continue
        if action_id is None:
            print("That move is not legal from the current state.")
            continue

        next_position, transport = detective_action_details(action_id)
        env.apply_detective_move(
            detective_id,
            action_id,
            next_position,
            transport,
        )
        return action_id, next_position, transport, current_position


def choose_ai_mrx_action(
    env,
    mrx_policy_net,
    device,
    round_number,
    rng,
    greedy=True,
    used_double=False,
):
    round_number = env.move_counter if round_number is None else int(round_number)
    state = env.mrx_state(round_number)
    input_vector = state_to_input_mrx(state).to(device)
    current_position = int(state["mr_x_location"])
    legal_actions = legal_mrx_action_ids(env)

    if legal_actions.size == 0:
        return None, None, None, None, current_position, False

    with torch.no_grad():
        policy_logits, double_logits = mrx_policy_net(input_vector)

    use_double = False
    selectable_actions = legal_actions
    if not used_double and env.mrx.mrx_tickets["double"] > 0:
        double_actions = legal_mrx_action_ids(env, require_double_followup=True)
        if double_actions.size > 0:
            use_double = sample_double_decision(
                double_logits,
                rng,
                greedy=greedy,
            ) == 0
            if use_double:
                selectable_actions = double_actions

    action_id = sample_from_policy(
        policy_logits,
        selectable_actions,
        rng,
        greedy=greedy,
    )
    next_position, transport, use_black = mrx_action_details(action_id)
    env.apply_mrx_move(
        action_id,
        next_position,
        transport,
        use_black,
        round_number,
    )

    if use_double:
        env.mrx.mrx_tickets["double"] -= 1

    return (
        int(action_id),
        next_position,
        transport,
        use_black,
        current_position,
        use_double,
    )


def public_mrx_announcement(round_number, destination, transport, use_black):
    reveal_turn = (round_number + 1) in reveal_rounds
    if reveal_turn:
        if use_black:
            return f"Public reveal: Mr X is at {destination} after a black ticket."
        return f"Public reveal: Mr X is at {destination} via {transport}."

    if use_black:
        return "Public info: show a black ticket."
    return f"Public info: show a {transport} ticket."


def print_ticket_summary(label, ticket_dict, order):
    summary = " ".join(f"{ticket}:{ticket_dict[ticket]}" for ticket in order)
    print(f"{label}: {summary}")


def print_state_snapshot(env, round_number):
    detectives = ", ".join(
        f"D{detective_id + 1}={int(detective.detective_pos)}"
        for detective_id, detective in enumerate(env.detectives)
    )
    hidden_trail = " -> ".join(env.mrx.ticket_history) if env.mrx.ticket_history else "none"

    print(f"Move {round_number + 1}/{ROUNDS}")
    print(f"Mr X actual position: {int(env.mrx.mrx_pos)}")
    print(f"Public reveal node: {env.last_reveal_location}")
    print(f"Mr X hidden trail: {hidden_trail}")
    print(f"Detectives: {detectives}")
    print_ticket_summary(
        "Mr X tickets",
        env.mrx.mrx_tickets,
        ("taxi", "bus", "metro", "black", "double"),
    )
    for detective_id, detective in enumerate(env.detectives, start=1):
        print_ticket_summary(
            f"Detective {detective_id} tickets",
            detective.detective_tickets,
            ("taxi", "bus", "metro"),
        )


def append_frame_and_refresh(
    frames,
    viewer,
    env,
    step_index,
    round_number,
    actor_key,
    actor_label,
    source=None,
    destination=None,
    transport=None,
    use_black=False,
    note="",
    winner=None,
    win_reason=None,
):
    frame = build_frame(
        env,
        step_index=step_index,
        round_number=round_number,
        actor_key=actor_key,
        actor_label=actor_label,
        source=source,
        destination=destination,
        transport=transport,
        use_black=use_black,
        note=note,
        winner=winner,
        win_reason=win_reason,
    )
    frames.append(frame)
    print(describe_frame(frame))
    if actor_key == "mrx" and destination is not None:
        print(public_mrx_announcement(round_number, destination, transport, use_black))

    if viewer is not None:
        viewer.current_index = len(frames) - 1
        viewer.draw_frame(viewer.current_index)
        plt.pause(0.001)

    return frame


def create_viewer_if_needed(args, title, frames):
    if args.no_show:
        return None

    graph, positions = build_layout(args.layout_seed)
    plt.ion()
    viewer = ReplayViewer(
        frames,
        graph,
        positions,
        title=title,
        interval_ms=args.interval_ms,
        show_all_node_labels=args.show_all_node_labels,
    )
    viewer.current_index = len(frames) - 1
    viewer.draw_frame(viewer.current_index)
    plt.pause(0.001)
    return viewer


def maybe_prompt_double(env):
    if env.mrx.mrx_tickets["double"] <= 0:
        return False

    double_actions = legal_mrx_action_ids(env, require_double_followup=True)
    if double_actions.size == 0:
        return False

    return prompt_yes_no("Did the real Mr X use a double ticket this turn?", default=False)


def run_interactive_game(
    args,
    env,
    ai_side,
    mrx_policy_net,
    detective_policy_net,
    device,
    rng,
):
    mrx_position, detective_positions_list = prompt_initial_positions()
    env.setup_custom_game(mrx_position, detective_positions_list)

    frames = [
        build_frame(
            env,
            step_index=0,
            round_number=0,
            actor_key="setup",
            actor_label="Setup",
            note="Initial custom positions",
        )
    ]
    title = (
        f"Scotland Yard Live Game | AI: {ai_side} | "
        f"Mr X: {int(env.mrx.mrx_pos)}"
    )
    viewer = create_viewer_if_needed(args, title, frames)

    manual_side = "detectives" if ai_side == "mrx" else "mrx"
    print()
    print(f"AI side: {ai_side}")
    print(f"You enter moves for: {manual_side}")
    print("Type 'quit' at any prompt to stop the session.")
    print()

    winner = None
    win_reason = None
    step_index = 0

    while env.move_counter < ROUNDS:
        round_number = env.move_counter
        print("=" * 72)
        print_state_snapshot(env, round_number)
        print()

        if ai_side == "mrx":
            print("AI Mr X is choosing a move...")
            (
                _action_id,
                next_pos_mrx,
                transport,
                use_black,
                previous_position,
                use_double,
            ) = choose_ai_mrx_action(
                env,
                mrx_policy_net,
                device,
                round_number,
                rng,
                greedy=not args.sampled,
            )

            if next_pos_mrx is None:
                winner = "detective"
                win_reason = f"Mr X had no legal move on move {round_number + 1}."
                step_index += 1
                append_frame_and_refresh(
                    frames,
                    viewer,
                    env,
                    step_index,
                    round_number,
                    actor_key="mrx",
                    actor_label="Mr X",
                    source=previous_position,
                    destination=None,
                    note="Mr X could not move",
                    winner=winner,
                    win_reason=win_reason,
                )
                break
        else:
            use_double = maybe_prompt_double(env)
            legal_actions = legal_mrx_action_ids(
                env,
                require_double_followup=use_double,
            )
            previous_position = int(env.mrx.mrx_pos)

            if legal_actions.size == 0:
                winner = "detective"
                win_reason = f"Mr X had no legal move on move {round_number + 1}."
                step_index += 1
                append_frame_and_refresh(
                    frames,
                    viewer,
                    env,
                    step_index,
                    round_number,
                    actor_key="mrx",
                    actor_label="Mr X",
                    source=previous_position,
                    destination=None,
                    note="Mr X could not move",
                    winner=winner,
                    win_reason=win_reason,
                )
                break

            print(
                f"Mr X move from {previous_position}. "
                f"Legal actions: {format_legal_actions(legal_actions, 'mrx')}"
            )
            while True:
                value = read_input(
                    "Enter Mr X action as 'destination transport [black]': "
                )
                action_id = parse_human_mrx_action(value, legal_actions)
                if action_id == "help":
                    print(
                        f"Legal actions: {format_legal_actions(legal_actions, 'mrx')}"
                    )
                    continue
                if action_id is None:
                    print("That move is not legal from the current state.")
                    continue
                break

            if use_double:
                env.mrx.mrx_tickets["double"] -= 1

            (
                _action_id,
                next_pos_mrx,
                transport,
                use_black,
                previous_position,
            ) = apply_human_mrx_action(env, round_number, action_id)

        step_index += 1
        append_frame_and_refresh(
            frames,
            viewer,
            env,
            step_index,
            round_number,
            actor_key="mrx",
            actor_label="Mr X",
            source=previous_position,
            destination=next_pos_mrx,
            transport=transport,
            use_black=use_black,
            note="Mr X double move 1/2" if use_double else "Mr X move",
        )

        if use_double:
            if ai_side == "mrx":
                print("AI Mr X is choosing the second leg of the double move...")
                (
                    _action_id,
                    next_pos_mrx,
                    transport,
                    use_black,
                    previous_position,
                    _,
                ) = choose_ai_mrx_action(
                    env,
                    mrx_policy_net,
                    device,
                    env.move_counter,
                    rng,
                    greedy=not args.sampled,
                    used_double=True,
                )
            else:
                legal_actions = legal_mrx_action_ids(env)
                previous_position = int(env.mrx.mrx_pos)

                print(
                    f"Mr X second leg from {previous_position}. "
                    f"Legal actions: {format_legal_actions(legal_actions, 'mrx')}"
                )
                while True:
                    value = read_input(
                        "Enter Mr X second action as "
                        "'destination transport [black]': "
                    )
                    action_id = parse_human_mrx_action(value, legal_actions)
                    if action_id == "help":
                        print(
                            f"Legal actions: {format_legal_actions(legal_actions, 'mrx')}"
                        )
                        continue
                    if action_id is None:
                        print("That move is not legal from the current state.")
                        continue
                    break

                (
                    _action_id,
                    next_pos_mrx,
                    transport,
                    use_black,
                    previous_position,
                ) = apply_human_mrx_action(env, env.move_counter, action_id)

            if next_pos_mrx is None:
                winner = "detective"
                win_reason = (
                    f"Mr X could not complete the double move on move "
                    f"{env.move_counter + 1}."
                )
                step_index += 1
                append_frame_and_refresh(
                    frames,
                    viewer,
                    env,
                    step_index,
                    env.move_counter,
                    actor_key="mrx",
                    actor_label="Mr X",
                    source=previous_position,
                    destination=None,
                    note="Mr X failed to complete the double move",
                    winner=winner,
                    win_reason=win_reason,
                )
                break

            step_index += 1
            append_frame_and_refresh(
                frames,
                viewer,
                env,
                step_index,
                env.move_counter - 1,
                actor_key="mrx",
                actor_label="Mr X",
                source=previous_position,
                destination=next_pos_mrx,
                transport=transport,
                use_black=use_black,
                note="Mr X double move 2/2",
            )

        if env.move_counter >= ROUNDS:
            winner = "mrx"
            win_reason = f"Mr X survived all {ROUNDS} moves."
            break

        for detective_id in range(DETECTIVE_AMOUNT):
            actor_key = f"detective_{detective_id + 1}"
            actor_label = f"Detective {detective_id + 1}"

            if ai_side == "detectives":
                print(f"AI {actor_label} is choosing a move...")
                (
                    _action_id,
                    next_position,
                    transport,
                    previous_position,
                ) = move_detective(
                    env,
                    detective_policy_net,
                    device,
                    detective_id,
                    env.move_counter,
                    rng,
                    greedy=not args.sampled,
                )
            else:
                (
                    _action_id,
                    next_position,
                    transport,
                    previous_position,
                ) = prompt_human_detective_action(env, detective_id)

            step_index += 1

            if next_position is None:
                append_frame_and_refresh(
                    frames,
                    viewer,
                    env,
                    step_index,
                    max(0, env.move_counter - 1),
                    actor_key=actor_key,
                    actor_label=actor_label,
                    source=previous_position,
                    destination=None,
                    note="Detective skipped turn",
                )
                continue

            capture = int(next_position) == int(env.mrx.mrx_pos)
            if capture:
                winner = "detective"
                win_reason = (
                    f"{actor_label} caught Mr X on node {next_position} "
                    f"on move {env.move_counter}."
                )

            append_frame_and_refresh(
                frames,
                viewer,
                env,
                step_index,
                max(0, env.move_counter - 1),
                actor_key=actor_key,
                actor_label=actor_label,
                source=previous_position,
                destination=next_position,
                transport=transport,
                note="Capture move" if capture else "Detective move",
                winner=winner if capture else None,
                win_reason=win_reason if capture else None,
            )

            if capture:
                break

        if winner is not None:
            break

    if winner is None:
        winner = "mrx"
        win_reason = f"Mr X survived all {ROUNDS} moves."

    step_index += 1
    frames.append(
        build_frame(
            env,
            step_index=step_index,
            round_number=min(ROUNDS - 1, frames[-1].round_number),
            actor_key="game_over",
            actor_label="Game Over",
            note="Live session finished",
            winner=winner,
            win_reason=win_reason,
        )
    )

    if viewer is not None:
        viewer.current_index = len(frames) - 1
        viewer.draw_frame(viewer.current_index)
        plt.pause(0.001)

    print()
    print(f"Winner: {winner}")
    print(f"Reason: {win_reason}")
    print(
        "Note: this live simulator now counts each leg of a Mr X double move "
        "as its own move for reveal timing and move progression."
    )

    if viewer is not None:
        print("Close the matplotlib window when you are done reviewing the board.")
        plt.ioff()
        viewer.show()


def main():
    args = parse_args()

    checkpoint_dir = Path(args.checkpoint_dir)
    mrx_path = resolve_checkpoint_path(
        "Mr X",
        args.mrx_path,
        args.mrx_generation,
        checkpoint_dir,
    )
    detective_path = resolve_checkpoint_path(
        "Detective",
        args.detective_path,
        args.detective_generation,
        checkpoint_dir,
    )

    print(f"Mr X checkpoint: {mrx_path}")
    print(f"Detective checkpoint: {detective_path}")
    print(f"Policy mode: {'sampled' if args.sampled else 'greedy'}")
    print(f"AI seed: {args.seed}")
    print()

    device = torch.device(args.device)
    env = environment(DETECTIVE_AMOUNT)
    mrx_policy_net, detective_policy_net = load_policies(
        mrx_path,
        detective_path,
        device,
    )

    try:
        explicit_ai_side = args.ai_side
        if explicit_ai_side is None and args.legacy_human_side is not None:
            explicit_ai_side = (
                "detectives" if args.legacy_human_side == "mrx" else "mrx"
            )
            print(
                "Note: `--human-side` is now treated as a legacy option. "
                "Using the opposite side for the AI."
            )

        ai_side = prompt_side(explicit_ai_side)
        rng = build_game_rng(args.seed)
        run_interactive_game(
            args,
            env,
            ai_side,
            mrx_policy_net,
            detective_policy_net,
            device,
            rng,
        )
    except UserQuit:
        print()
        print("Session ended before the game finished.")
    except KeyboardInterrupt:
        print()
        print("Session interrupted.")


if __name__ == "__main__":
    main()
