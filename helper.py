import numpy as np
import torch
from game_constants import reveal_rounds
import numpy as np
"""
Feature layout reference
========================

Mr. X encoder
- round context: 3
- Mr. X position: 1
- Mr. X ticket inventory (count/has/exhausted for 5 ticket types): 15
- detective locations: N
- detective ticket inventories (count/has/exhausted for 3 ticket types each): 9 * N
- Mr. X history sequence (6 slots * 6 one-hot values): 36
- Mr. X history histogram: 5
- most recent history ticket one-hot: 6
- last reveal position: 1
- moves since reveal: 1
- total: 68 + 10 * N

Detective encoder
- round context: 3
- detective id one-hot: N
- own position: 1
- own ticket inventory (count/has/exhausted for 3 ticket types): 9
- all detective locations: N
- detective ticket inventories (count/has/exhausted for 3 ticket types each): 9 * N
- Mr. X history sequence (6 slots * 6 one-hot values): 36
- Mr. X history histogram: 5
- most recent history ticket one-hot: 6
- Mr. X last revealed position: 1
- moves since last reveal: 1
- total: 62 + 11 * N

With 4 detectives:
- Mr. X input size: 108
- detective input size: 106
"""


MAX_ROUNDS = 24
NUM_NODES = 199

DETECTIVE_TICKET_ORDER = ("taxi", "bus", "metro")
MRX_TICKET_ORDER = ("taxi", "bus", "metro", "black", "double")
HISTORY_TICKET_ORDER = ("taxi", "bus", "metro", "black", "double", "pad")

MAX_DETECTIVE_TICKETS = {
    "taxi": 11,
    "bus": 8,
    "metro": 4,
}
MAX_MRX_TICKETS = {
    "taxi": 23,
    "bus": 8,
    "metro": 6,
    "black": 5,
    "double": 2,
}

MAX_HISTORY = 6
MAX_REVEAL_GAP = max(
    reveal_rounds[0],
    max(curr - prev for prev, curr in zip(reveal_rounds, reveal_rounds[1:])),
)


class FeatureBuilder:
    """Small helper for assembling flat float32 feature tensors."""

    def __init__(self):
        self._parts = []

    def scalar(self, value, scale=1.0):
        self._parts.append(np.array([value / scale], dtype=np.float32))

    def binary(self, value):
        self._parts.append(np.array([float(bool(value))], dtype=np.float32))

    def one_hot_index(self, index, size):
        vec = np.zeros(size, dtype=np.float32)
        if 0 <= index < size:
            vec[index] = 1.0
        self._parts.append(vec)

    def one_hot_label(self, label, labels):
        vec = np.zeros(len(labels), dtype=np.float32)
        if label in labels:
            vec[labels.index(label)] = 1.0
        self._parts.append(vec)

    def position(self, node):
        self.scalar(node, NUM_NODES)

    def positions(self, nodes):
        self._parts.append(np.asarray(nodes, dtype=np.float32) / NUM_NODES)

    def ticket_inventory(self, ticket_dict, max_tickets, ticket_order):
        counts = np.asarray(
            [ticket_dict.get(ticket, 0) for ticket in ticket_order],
            dtype=np.float32,
        )
        capacity = np.asarray(
            [max_tickets[ticket] for ticket in ticket_order],
            dtype=np.float32,
        )

        # Continuous signal for how full the inventory is.
        self._parts.append(counts / capacity)
        # Binary availability flags are often easier for a network to use.
        self._parts.append((counts > 0).astype(np.float32))
        # Explicitly mark exhausted ticket types.
        self._parts.append((counts == 0).astype(np.float32))

    def detective_ticket_table(self, tickets_per_detective):
        for tickets in tickets_per_detective:
            self.ticket_inventory(
                tickets,
                MAX_DETECTIVE_TICKETS,
                DETECTIVE_TICKET_ORDER,
            )

    def ticket_history(self, history, max_length=MAX_HISTORY):
        trimmed_history = history[-max_length:]
        padded_history = ["pad"] * (max_length - len(trimmed_history)) + trimmed_history

        for ticket in padded_history:
            self.one_hot_label(ticket, HISTORY_TICKET_ORDER)

        counts = np.zeros(len(HISTORY_TICKET_ORDER) - 1, dtype=np.float32)
        for ticket in trimmed_history:
            if ticket in HISTORY_TICKET_ORDER[:-1]:
                counts[HISTORY_TICKET_ORDER.index(ticket)] += 1.0

        self._parts.append(counts / max_length)
        self.one_hot_label(
            trimmed_history[-1] if trimmed_history else "pad",
            HISTORY_TICKET_ORDER,
        )

    def build(self):
        if not self._parts:
            return torch.empty(0, dtype=torch.float32)

        vector = np.concatenate(self._parts).astype(np.float32, copy=False)
        return torch.from_numpy(vector)


def _add_round_context(builder, round_number):
    current_move = round_number + 1
    next_reveal = next(
        (reveal_round for reveal_round in reveal_rounds if reveal_round >= current_move),
        reveal_rounds[-1],
    )

    builder.scalar(round_number, MAX_ROUNDS)
    builder.scalar(next_reveal - current_move, MAX_REVEAL_GAP)
    builder.binary(current_move in reveal_rounds)


def state_to_input_detective(state):
    """Encode detective state into a flat float32 torch tensor."""
    builder = FeatureBuilder()
    num_detectives = len(state["detective_locations"])

    _add_round_context(builder, state["round"])
    builder.one_hot_index(state["detective_id"], num_detectives)
    builder.position(state["my_position"])
    builder.ticket_inventory(
        state["my_tickets"],
        MAX_DETECTIVE_TICKETS,
        DETECTIVE_TICKET_ORDER,
    )
    builder.positions(state["detective_locations"])
    builder.detective_ticket_table(state["detective_tickets"])
    builder.ticket_history(state["mr_x_ticket_history"])
    builder.position(state["mr_x_last_revealed_position"])
    builder.scalar(state["moves_since_last_reveal"], MAX_REVEAL_GAP)

    return builder.build()


def state_to_input_mrx(state):
    """Encode Mr. X state into a flat float32 torch tensor."""
    builder = FeatureBuilder()

    _add_round_context(builder, state["round"])
    builder.position(state["mr_x_location"])
    builder.ticket_inventory(
        state["mr_x_tickets"],
        MAX_MRX_TICKETS,
        MRX_TICKET_ORDER,
    )
    builder.positions(state["detective_locations"])
    builder.detective_ticket_table(state["detective_tickets"])
    builder.ticket_history(state["mr_x_ticket_history"])
    builder.position(state["last_reveal_location"])
    builder.scalar(state["moves_since_reveal"], MAX_REVEAL_GAP)

    return builder.build()


def compute_discounted_rewards(rewards, dones, gamma):
    out = np.empty_like(rewards, dtype=np.float32)

    running = 0.0

    for t in range(rewards.shape[0] - 1, -1, -1):
        if dones[t]:
            running = 0.0

        running = rewards[t] + gamma * running
        out[t] = running

    return out
