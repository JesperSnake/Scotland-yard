from env import environment
from game_constants import board_graph
import numpy as np
import timeit

rounds = 24
detective_amount = 4
double_probability = 0.1
env = environment(detective_amount)
env.setup_game()

rng = np.random.default_rng()

# ---------------------------------------------------------------------
# Build global action table
# ---------------------------------------------------------------------

# Action = (src, dst, transport, use_black)
# Water edges ALWAYS require a black ticket.
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

def sample_action_mrx(pdf, current_position, ticket_dict):

    if current_position is None:
        return None, None, None, None

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
        return None, None, None, None

    # Filter and normalize policy
    p = pdf[legal_actions]
    p = p / p.sum()

    # Sample action
    action_id = rng.choice(legal_actions, p=p)

    return (
        int(action_id),
        int(action_dst[action_id]),
        TRANSPORT_NAMES[action_transport[action_id]],
        bool(action_use_black[action_id]),
    )


def sample_action_detective(pdf, current_position, ticket_dict, occupied_positions):

    if current_position is None:
        return None, None, None

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
        return None, None, None

    # Filter and normalize policy
    p = pdf[legal_actions]
    p = p / p.sum()

    # Sample action
    action_id = rng.choice(legal_actions, p=p)

    return (
        int(action_id),
        int(detective_action_dst[action_id]),
        DETECTIVE_TRANSPORT_NAMES[detective_action_transport[action_id]],
    )


# ---------------------------------------------------------------------
# Game loop
# ---------------------------------------------------------------------

def move_mr_x(used_double=False):
    # Let Mr X play
    mrx_tickets, mrx_pos = env.mrx_state()

    # Network output
    pdf = rng.dirichlet(np.ones(N_ACTIONS))
    use_double = (
        not used_double
        and mrx_tickets["double"] > 0
        and rng.random() < double_probability
    )

    action_id, next_pos, transport, use_black = sample_action_mrx(
        pdf,
        mrx_pos,
        mrx_tickets,
    )

    if next_pos is None:
        return action_id, next_pos, transport, use_black, mrx_pos, False

    env.apply_mrx_move(
        action_id,
        next_pos,
        transport,
        use_black,
    )

    if use_double:
        return action_id, next_pos, transport, use_black, mrx_pos, True
    else:
        return action_id, next_pos, transport, use_black, mrx_pos, False


def move_detective(detective_id):
    detective_tickets, detective_pos = env.detective_state(
        detective_id=detective_id
    )

    occupied_positions = [
        env.detective_state(detective_id=j)[1]
        for j in range(detective_amount)
        if j != detective_id
    ]

    # Network output
    pdf = rng.dirichlet(np.ones(N_DETECTIVE_ACTIONS))

    action_id, next_pos, transport = sample_action_detective(
        pdf,
        detective_pos,
        detective_tickets,
        occupied_positions,
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

def gather_experience():
    env.setup_game()

    for i in range(rounds):

        action_id, next_pos, transport, use_black, mrx_pos, use_double = move_mr_x()
        # print(
        #     f"Round {i + 1}: "
        #     f"{mrx_pos} -> {next_pos} "
        #     f"via {'black-' if use_black else ''}{transport}"
        # )

        # No legal moves for mrx so he lost
        if next_pos is None:
            print("Mr X lost, no legal moves available")
            break

        # This runs if mrx used double card
        if use_double:
            action_id, next_pos, transport, use_black, mrx_pos, _ = move_mr_x(True)

            if next_pos is None:
                print("Mr X lost, no legal moves available during double move")
                break

            # print(
            #     f"Round {i + 1}: "
            #     f"{mrx_pos} -> {next_pos} "
            #     f"via {'black-' if use_black else ''}{transport}",
            #     "Used a double mover"
            # )

        # Detectivessss vo
        for detective_id in range(detective_amount):
            action_id, next_pos, transport, detective_pos = move_detective(detective_id)

            if next_pos is None:
                print(f"Detective {detective_id} has no legal moves")
                continue

            # print(
            #     f"Detective {detective_id}: "
            #     f"{detective_pos} -> {next_pos} "
            #     f"via {transport}"
            # )

num_runs = 200
total_time = timeit.timeit(gather_experience, number=num_runs)

print(f"Average time: {total_time / num_runs:.8f} seconds")