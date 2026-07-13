import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.lines import Line2D
import networkx as nx
import numpy as np
import torch

from detectivebrain import DetectivePolicy
from env import environment
from game_constants import board_graph, reveal_rounds
from mrxbrain import MrXPolicy
from run_tournament import (
    DETECTIVE_AMOUNT,
    DETECTIVE_INPUT_SIZE,
    DETECTIVE_OUTPUT_SIZE,
    HIDDEN_SIZE,
    MRX_INPUT_SIZE,
    MRX_OUTPUT_SIZE,
    ROUNDS,
    build_game_rng,
    load_policy_weights,
    move_detective,
    move_mr_x,
)


TRANSPORT_EDGE_COLORS = {
    "taxi": "#b8b8b8",
    "bus": "#f4a261",
    "metro": "#3d5a80",
    "water": "#2a9d8f",
}

DETECTIVE_STYLES = (
    {"face": "#d1495b", "edge": "#7f1d2d", "label": "D1"},
    {"face": "#3a86ff", "edge": "#123a73", "label": "D2"},
    {"face": "#43aa8b", "edge": "#1b5e4b", "label": "D3"},
    {"face": "#ffb703", "edge": "#9b6800", "label": "D4"},
)


@dataclass(frozen=True)
class ReplayFrame:
    step_index: int
    round_number: int
    actor_key: str
    actor_label: str
    source: int | None
    destination: int | None
    transport: str | None
    use_black: bool
    note: str
    winner: str | None
    win_reason: str | None
    mrx_position: int
    detective_positions: tuple[int, ...]
    last_reveal_location: int | None
    ticket_history: tuple[str, ...]
    mrx_tickets: dict[str, int]
    detective_tickets: tuple[dict[str, int], ...]


def parse_generation_arg(value):
    if value is None:
        return None

    text = str(value).strip().lower()
    if text == "latest":
        return "latest"

    try:
        return int(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"Generation must be an integer or 'latest', got {value!r}.",
        ) from exc


def extract_checkpoint_generation(checkpoint_path):
    try:
        return int(Path(checkpoint_path).stem.split("_")[-1])
    except ValueError:
        return None


def list_available_generations(checkpoint_dir):
    generations = []
    for checkpoint_path in sorted(checkpoint_dir.glob("self_play_iter_*.pt")):
        generation = extract_checkpoint_generation(checkpoint_path)
        if generation is not None:
            generations.append(generation)
    return generations


def find_latest_checkpoint_path(checkpoint_dir):
    checkpoint_paths = []
    for checkpoint_path in checkpoint_dir.glob("self_play_iter_*.pt"):
        generation = extract_checkpoint_generation(checkpoint_path)
        if generation is not None:
            checkpoint_paths.append((generation, checkpoint_path))

    if not checkpoint_paths:
        return None
    checkpoint_paths.sort(key=lambda item: item[0])
    return checkpoint_paths[-1][1]


def resolve_checkpoint_path(role_name, explicit_path, generation, checkpoint_dir):
    if explicit_path is not None:
        checkpoint_path = Path(explicit_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"{role_name} checkpoint path does not exist: {checkpoint_path}",
            )
        return checkpoint_path

    if generation is None:
        generation = "latest"

    if generation == "latest":
        checkpoint_path = find_latest_checkpoint_path(checkpoint_dir)
        if checkpoint_path is None:
            raise FileNotFoundError(
                f"No checkpoints found in {checkpoint_dir}",
            )
        return checkpoint_path

    checkpoint_path = checkpoint_dir / f"self_play_iter_{int(generation):04d}.pt"
    if checkpoint_path.exists():
        return checkpoint_path

    available_generations = list_available_generations(checkpoint_dir)
    if available_generations:
        preview = ", ".join(str(gen) for gen in available_generations[-10:])
        raise FileNotFoundError(
            f"{role_name} generation {generation} was not found in {checkpoint_dir}. "
            f"Available recent generations: {preview}",
        )

    raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}")


def load_policies(mrx_path, detective_path, device):
    mrx_policy_net = MrXPolicy(MRX_INPUT_SIZE, HIDDEN_SIZE, MRX_OUTPUT_SIZE).to(device)
    detective_policy_net = DetectivePolicy(
        DETECTIVE_INPUT_SIZE,
        HIDDEN_SIZE,
        DETECTIVE_OUTPUT_SIZE,
    ).to(device)

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
    return mrx_policy_net, detective_policy_net


def copy_ticket_dict(ticket_dict):
    return {key: int(value) for key, value in ticket_dict.items()}


def build_frame(
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
    return ReplayFrame(
        step_index=int(step_index),
        round_number=int(round_number),
        actor_key=actor_key,
        actor_label=actor_label,
        source=None if source is None else int(source),
        destination=None if destination is None else int(destination),
        transport=transport,
        use_black=bool(use_black),
        note=note,
        winner=winner,
        win_reason=win_reason,
        mrx_position=int(env.mrx.mrx_pos),
        detective_positions=tuple(int(d.detective_pos) for d in env.detectives),
        last_reveal_location=(
            None if env.last_reveal_location is None else int(env.last_reveal_location)
        ),
        ticket_history=tuple(env.mrx.ticket_history),
        mrx_tickets=copy_ticket_dict(env.mrx.mrx_tickets),
        detective_tickets=tuple(
            copy_ticket_dict(d.detective_tickets)
            for d in env.detectives
        ),
    )


def describe_frame(frame):
    if frame.actor_key == "setup":
        detectives = ", ".join(
            f"D{idx + 1}={position}"
            for idx, position in enumerate(frame.detective_positions)
        )
        return f"Setup | Mr X={frame.mrx_position} | {detectives}"

    if frame.actor_key == "game_over":
        return f"Game Over | Winner: {frame.winner} | {frame.win_reason}"

    round_label = frame.round_number + 1
    if frame.destination is None:
        return (
            f"Move {round_label:02d} | {frame.actor_label} | "
            f"no legal move from {frame.source} | {frame.win_reason}"
        )

    move_text = (
        f"Move {round_label:02d} | {frame.actor_label} | "
        f"{frame.source} -> {frame.destination} via {frame.transport}"
    )
    if frame.use_black:
        move_text += " (black ticket)"
    if frame.winner is not None:
        move_text += f" | winner={frame.winner}"
    return move_text


def play_visualized_game(
    env,
    mrx_policy_net,
    detective_policy_net,
    device,
    rng,
    greedy=True,
):
    frames = []
    env.setup_game()
    step_index = 0

    frames.append(
        build_frame(
            env,
            step_index=step_index,
            round_number=0,
            actor_key="setup",
            actor_label="Setup",
            note="Initial spawn positions",
        )
    )

    winner = None
    win_reason = None

    while env.move_counter < ROUNDS:
        round_number = env.move_counter
        (
            _action_id,
            next_pos_mrx,
            transport,
            use_black,
            previous_mrx_pos,
            use_double,
        ) = move_mr_x(
            env,
            mrx_policy_net,
            device,
            round_number,
            rng,
            greedy=greedy,
        )

        step_index += 1

        if next_pos_mrx is None:
            winner = "detective"
            win_reason = f"Mr X had no legal move on move {round_number + 1}."
            frames.append(
                build_frame(
                    env,
                    step_index=step_index,
                    round_number=round_number,
                    actor_key="mrx",
                    actor_label="Mr X",
                    source=previous_mrx_pos,
                    destination=None,
                    note="Mr X could not move",
                    winner=winner,
                    win_reason=win_reason,
                )
            )
            break

        frames.append(
            build_frame(
                env,
                step_index=step_index,
                round_number=round_number,
                actor_key="mrx",
                actor_label="Mr X",
                source=previous_mrx_pos,
                destination=next_pos_mrx,
                transport=transport,
                use_black=use_black,
                note="Mr X double move 1/2" if use_double else "Mr X move",
            )
        )

        if use_double:
            (
                _action_id,
                next_pos_mrx,
                transport,
                use_black,
                previous_mrx_pos,
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

            step_index += 1

            if next_pos_mrx is None:
                winner = "detective"
                win_reason = (
                    f"Mr X could not complete the double move on move "
                    f"{env.move_counter + 1}."
                )
                frames.append(
                    build_frame(
                        env,
                        step_index=step_index,
                        round_number=env.move_counter,
                        actor_key="mrx",
                        actor_label="Mr X",
                        source=previous_mrx_pos,
                        destination=None,
                        note="Mr X failed to complete the double move",
                        winner=winner,
                        win_reason=win_reason,
                    )
                )
                break

            frames.append(
                build_frame(
                    env,
                    step_index=step_index,
                    round_number=env.move_counter - 1,
                    actor_key="mrx",
                    actor_label="Mr X",
                    source=previous_mrx_pos,
                    destination=next_pos_mrx,
                    transport=transport,
                    use_black=use_black,
                    note="Mr X double move 2/2",
                )
            )

        if env.move_counter >= ROUNDS:
            winner = "mrx"
            win_reason = f"Mr X survived all {ROUNDS} moves."
            break

        for detective_id in range(DETECTIVE_AMOUNT):
            (
                _action_id,
                next_pos_detective,
                transport,
                previous_detective_pos,
            ) = move_detective(
                env,
                detective_policy_net,
                device,
                detective_id,
                env.move_counter,
                rng,
                greedy=greedy,
            )

            step_index += 1

            actor_key = f"detective_{detective_id + 1}"
            actor_label = f"Detective {detective_id + 1}"

            if next_pos_detective is None:
                frames.append(
                    build_frame(
                        env,
                        step_index=step_index,
                        round_number=max(0, env.move_counter - 1),
                        actor_key=actor_key,
                        actor_label=actor_label,
                        source=previous_detective_pos,
                        destination=None,
                        note="Detective skipped turn",
                    )
                )
                continue

            capture = next_pos_detective == next_pos_mrx
            if capture:
                winner = "detective"
                win_reason = (
                    f"Detective {detective_id + 1} caught Mr X on node "
                    f"{next_pos_detective} on move {env.move_counter}."
                )

            frames.append(
                build_frame(
                    env,
                    step_index=step_index,
                    round_number=max(0, env.move_counter - 1),
                    actor_key=actor_key,
                    actor_label=actor_label,
                    source=previous_detective_pos,
                    destination=next_pos_detective,
                    transport=transport,
                    note="Capture move" if capture else "Detective move",
                    winner=winner if capture else None,
                    win_reason=win_reason if capture else None,
                )
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
            note="Replay finished",
            winner=winner,
            win_reason=win_reason,
        )
    )

    return winner, win_reason, frames


def build_transport_edge_sets():
    transport_edges = {
        "taxi": set(),
        "bus": set(),
        "metro": set(),
        "water": set(),
    }

    for src, edge_map in board_graph.items():
        for transport, destinations in edge_map.items():
            for dst in destinations:
                transport_edges[transport].add(tuple(sorted((int(src), int(dst)))))

    return transport_edges


def build_layout(layout_seed):
    graph = nx.Graph()
    graph.add_nodes_from(range(1, 200))

    for edge_set in build_transport_edge_sets().values():
        graph.add_edges_from(edge_set)

    initial_pos = nx.kamada_kawai_layout(graph)
    final_pos = nx.spring_layout(
        graph,
        pos=initial_pos,
        seed=layout_seed,
        iterations=250,
        k=0.18,
    )
    return graph, final_pos


def format_ticket_summary(ticket_dict, order):
    return " ".join(f"{ticket[0].upper()}:{ticket_dict.get(ticket, 0)}" for ticket in order)


class ReplayViewer:
    def __init__(
        self,
        frames,
        graph,
        positions,
        title,
        interval_ms=900,
        show_all_node_labels=False,
    ):
        self.frames = frames
        self.graph = graph
        self.positions = positions
        self.title = title
        self.interval_ms = int(interval_ms)
        self.show_all_node_labels = show_all_node_labels
        self.current_index = 0
        self.autoplay = False

        self.transport_edges = build_transport_edge_sets()
        self.figure, (self.board_ax, self.info_ax) = plt.subplots(
            1,
            2,
            figsize=(16, 10),
            gridspec_kw={"width_ratios": [3.6, 1.5]},
        )

        xs = np.array([xy[0] for xy in positions.values()], dtype=np.float32)
        ys = np.array([xy[1] for xy in positions.values()], dtype=np.float32)
        x_margin = (xs.max() - xs.min()) * 0.08
        y_margin = (ys.max() - ys.min()) * 0.08
        self.x_limits = (float(xs.min() - x_margin), float(xs.max() + x_margin))
        self.y_limits = (float(ys.min() - y_margin), float(ys.max() + y_margin))
        self.marker_offset_radius = max(
            self.x_limits[1] - self.x_limits[0],
            self.y_limits[1] - self.y_limits[0],
        ) * 0.012

        self.figure.canvas.mpl_connect("key_press_event", self.on_key_press)
        self.timer = self.figure.canvas.new_timer(interval=self.interval_ms)
        self.timer.add_callback(self.advance_frame)

    def draw_frame(self, frame_index):
        frame = self.frames[frame_index]

        self.board_ax.clear()
        self.info_ax.clear()
        self.board_ax.set_facecolor("#fbfbfb")
        self.info_ax.set_facecolor("#ffffff")

        self._draw_base_graph()
        self._draw_highlighted_move(frame)
        self._draw_agents(frame)
        self._annotate_key_nodes(frame)
        self._draw_info_panel(frame, frame_index)

        self.board_ax.set_xlim(*self.x_limits)
        self.board_ax.set_ylim(*self.y_limits)
        self.board_ax.set_aspect("equal")
        self.board_ax.axis("off")
        self.info_ax.axis("off")

        self.figure.suptitle(self.title, fontsize=16, fontweight="bold", y=0.98)
        self.figure.tight_layout(rect=(0, 0, 1, 0.965))
        self.figure.canvas.draw_idle()

    def _draw_base_graph(self):
        self.board_ax.set_title("Board Replay", fontsize=13, fontweight="bold", loc="left")

        for transport in ("taxi", "bus", "metro", "water"):
            nx.draw_networkx_edges(
                self.graph,
                self.positions,
                edgelist=sorted(self.transport_edges[transport]),
                ax=self.board_ax,
                edge_color=TRANSPORT_EDGE_COLORS[transport],
                width=0.6 if transport == "taxi" else 1.0,
                alpha=0.28 if transport == "taxi" else 0.35,
                style="dashed" if transport == "water" else "solid",
            )

        nx.draw_networkx_nodes(
            self.graph,
            self.positions,
            nodelist=sorted(self.graph.nodes),
            node_size=32,
            node_color="#f3f3f3",
            edgecolors="#c7c7c7",
            linewidths=0.4,
            ax=self.board_ax,
        )

        if self.show_all_node_labels:
            nx.draw_networkx_labels(
                self.graph,
                self.positions,
                labels={node: str(node) for node in self.graph.nodes},
                font_size=4.5,
                font_color="#7a7a7a",
                ax=self.board_ax,
            )

    def _draw_highlighted_move(self, frame):
        if frame.last_reveal_location is not None:
            reveal_x, reveal_y = self.positions[frame.last_reveal_location]
            self.board_ax.scatter(
                [reveal_x],
                [reveal_y],
                s=260,
                facecolors="none",
                edgecolors="#e9c46a",
                linewidths=2.4,
                zorder=4,
            )

        if frame.source is None or frame.destination is None:
            return

        x1, y1 = self.positions[frame.source]
        x2, y2 = self.positions[frame.destination]
        transport_color = TRANSPORT_EDGE_COLORS.get(frame.transport, "#264653")

        if frame.use_black:
            self.board_ax.plot(
                [x1, x2],
                [y1, y2],
                color="#111111",
                linewidth=5.0,
                linestyle=(0, (2, 2)),
                alpha=0.95,
                zorder=4,
            )

        self.board_ax.plot(
            [x1, x2],
            [y1, y2],
            color=transport_color,
            linewidth=3.2,
            alpha=0.95,
            zorder=5,
            solid_capstyle="round",
        )

        self.board_ax.scatter(
            [x1, x2],
            [y1, y2],
            s=180,
            facecolors="none",
            edgecolors=transport_color,
            linewidths=1.8,
            zorder=6,
        )

    def _draw_agents(self, frame):
        occupants_by_node = {}
        occupants_by_node.setdefault(frame.mrx_position, []).append(("mrx", "X"))

        for detective_id, position in enumerate(frame.detective_positions, start=1):
            occupants_by_node.setdefault(position, []).append(
                (f"detective_{detective_id}", f"D{detective_id}"),
            )

        for node, occupants in occupants_by_node.items():
            base_x, base_y = self.positions[node]
            offsets = self._offsets_for_count(len(occupants))

            for offset, (actor_key, label) in zip(offsets, occupants):
                x = base_x + offset[0]
                y = base_y + offset[1]

                if actor_key == "mrx":
                    self.board_ax.scatter(
                        [x],
                        [y],
                        s=340,
                        marker="*",
                        c=["#111111"],
                        edgecolors="#ef476f",
                        linewidths=1.5,
                        zorder=8,
                    )
                    self.board_ax.text(
                        x,
                        y,
                        label,
                        ha="center",
                        va="center",
                        fontsize=8,
                        fontweight="bold",
                        color="white",
                        zorder=9,
                    )
                    continue

                style = DETECTIVE_STYLES[int(actor_key.split("_")[-1]) - 1]
                self.board_ax.scatter(
                    [x],
                    [y],
                    s=210,
                    marker="o",
                    c=[style["face"]],
                    edgecolors=style["edge"],
                    linewidths=1.4,
                    zorder=8,
                )
                self.board_ax.text(
                    x,
                    y,
                    label,
                    ha="center",
                    va="center",
                    fontsize=7,
                    fontweight="bold",
                    color="#111111",
                    zorder=9,
                )

    def _annotate_key_nodes(self, frame):
        node_labels = {
            frame.mrx_position,
            *frame.detective_positions,
        }
        if frame.source is not None:
            node_labels.add(frame.source)
        if frame.destination is not None:
            node_labels.add(frame.destination)
        if frame.last_reveal_location is not None:
            node_labels.add(frame.last_reveal_location)

        for node in sorted(node_labels):
            x, y = self.positions[node]
            self.board_ax.text(
                x,
                y + self.marker_offset_radius * 1.8,
                str(node),
                ha="center",
                va="bottom",
                fontsize=7,
                color="#333333",
                bbox={
                    "boxstyle": "round,pad=0.15",
                    "facecolor": "white",
                    "edgecolor": "#d0d0d0",
                    "linewidth": 0.6,
                    "alpha": 0.92,
                },
                zorder=10,
            )

    def _draw_info_panel(self, frame, frame_index):
        lines = [
            f"Frame: {frame_index + 1}/{len(self.frames)}",
            f"Move: {frame.round_number + 1 if frame.actor_key != 'setup' else 0}",
            f"Actor: {frame.actor_label}",
            "",
            f"Event: {frame.note or 'Move'}",
        ]

        if frame.destination is not None and frame.transport is not None:
            route = f"{frame.source} -> {frame.destination} via {frame.transport}"
            if frame.use_black:
                route += " (black ticket)"
            lines.append(f"Route: {route}")
        elif frame.source is not None and frame.destination is None:
            lines.append(f"Route: no legal move from {frame.source}")

        lines.extend(
            [
                "",
                f"Public reveal node: {frame.last_reveal_location}",
                "Mr X hidden trail: "
                + (" -> ".join(frame.ticket_history) if frame.ticket_history else "none"),
                "",
                f"Mr X actual position: {frame.mrx_position}",
                "Detectives: "
                + ", ".join(
                    f"D{idx + 1}={position}"
                    for idx, position in enumerate(frame.detective_positions)
                ),
                "",
                "Mr X tickets: "
                + format_ticket_summary(
                    frame.mrx_tickets,
                    ("taxi", "bus", "metro", "black", "double"),
                ),
            ]
        )

        for detective_id, ticket_dict in enumerate(frame.detective_tickets, start=1):
            lines.append(
                f"D{detective_id} tickets: "
                + format_ticket_summary(ticket_dict, ("taxi", "bus", "metro"))
            )

        if frame.winner is not None:
            lines.extend(
                [
                    "",
                    f"Winner: {frame.winner}",
                    f"Reason: {frame.win_reason}",
                ]
            )

        lines.extend(
            [
                "",
                "Reveal rounds: " + ", ".join(str(value) for value in reveal_rounds),
                "Controls: Left/Right step, Home/End jump, Space autoplay",
            ]
        )

        self.info_ax.text(
            0.03,
            0.98,
            "\n".join(lines),
            va="top",
            ha="left",
            fontsize=10,
            family="monospace",
            color="#222222",
        )

        legend_handles = [
            Line2D([0], [0], color=TRANSPORT_EDGE_COLORS["taxi"], lw=2, label="Taxi"),
            Line2D([0], [0], color=TRANSPORT_EDGE_COLORS["bus"], lw=2, label="Bus"),
            Line2D([0], [0], color=TRANSPORT_EDGE_COLORS["metro"], lw=2, label="Metro"),
            Line2D(
                [0],
                [0],
                color=TRANSPORT_EDGE_COLORS["water"],
                lw=2,
                linestyle="dashed",
                label="Water",
            ),
            Line2D(
                [0],
                [0],
                marker="*",
                color="w",
                markerfacecolor="#111111",
                markeredgecolor="#ef476f",
                markersize=12,
                linestyle="None",
                label="Mr X",
            ),
        ]

        for style in DETECTIVE_STYLES:
            legend_handles.append(
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="w",
                    markerfacecolor=style["face"],
                    markeredgecolor=style["edge"],
                    markersize=9,
                    linestyle="None",
                    label=style["label"],
                )
            )

        self.info_ax.legend(
            handles=legend_handles,
            loc="lower left",
            bbox_to_anchor=(0.02, 0.02),
            frameon=False,
            fontsize=9,
            ncol=2,
        )

    def _offsets_for_count(self, count):
        if count <= 1:
            return [(0.0, 0.0)]

        offsets = []
        for idx in range(count):
            angle = (2.0 * np.pi * idx) / count
            offsets.append(
                (
                    float(np.cos(angle) * self.marker_offset_radius),
                    float(np.sin(angle) * self.marker_offset_radius),
                )
            )
        return offsets

    def advance_frame(self):
        if self.current_index >= len(self.frames) - 1:
            self.timer.stop()
            self.autoplay = False
            return

        self.current_index += 1
        self.draw_frame(self.current_index)

    def on_key_press(self, event):
        if event.key == "right":
            self.timer.stop()
            self.autoplay = False
            self.current_index = min(self.current_index + 1, len(self.frames) - 1)
            self.draw_frame(self.current_index)
        elif event.key == "left":
            self.timer.stop()
            self.autoplay = False
            self.current_index = max(self.current_index - 1, 0)
            self.draw_frame(self.current_index)
        elif event.key == "home":
            self.timer.stop()
            self.autoplay = False
            self.current_index = 0
            self.draw_frame(self.current_index)
        elif event.key == "end":
            self.timer.stop()
            self.autoplay = False
            self.current_index = len(self.frames) - 1
            self.draw_frame(self.current_index)
        elif event.key == " ":
            if self.autoplay:
                self.timer.stop()
                self.autoplay = False
            else:
                self.timer.start()
                self.autoplay = True

    def _animation_step(self, frame_index):
        self.current_index = int(frame_index)
        self.draw_frame(self.current_index)
        return []

    def save_gif(self, output_path, fps):
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        animated = animation.FuncAnimation(
            self.figure,
            self._animation_step,
            frames=len(self.frames),
            interval=self.interval_ms,
            repeat=False,
            blit=False,
        )
        animated.save(output_path, writer=animation.PillowWriter(fps=fps), dpi=140)
        self.draw_frame(self.current_index)

    def show(self):
        self.draw_frame(self.current_index)
        plt.show()


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Replay and visualize one Scotland Yard game using selected "
            "Mr X and detective checkpoints."
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
        help="Optional explicit detective checkpoint path. Overrides --detective-generation.",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device to use, for example cpu or cuda.",
    )
    parser.add_argument(
        "--seed",
        "--game-seed",
        dest="seed",
        type=int,
        default=0,
        help=(
            "Per-game replay seed. Use a winning seed printed by run_tournament.py. "
            "Default: 0"
        ),
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
        help="Autoplay interval between frames in milliseconds. Default: 900",
    )
    parser.add_argument(
        "--gif-path",
        type=Path,
        default=None,
        help="Optional path to save the replay as a GIF.",
    )
    parser.add_argument(
        "--gif-fps",
        type=int,
        default=2,
        help="Frames per second for the saved GIF. Default: 2",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Do not open the interactive matplotlib window.",
    )
    parser.add_argument(
        "--show-all-node-labels",
        action="store_true",
        help="Draw labels for all 199 board nodes instead of just important nodes.",
    )
    return parser.parse_args()


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
    print(f"Replay seed: {args.seed}")
    print()

    rng = build_game_rng(args.seed)
    device = torch.device(args.device)
    env = environment(DETECTIVE_AMOUNT)
    mrx_policy_net, detective_policy_net = load_policies(
        mrx_path,
        detective_path,
        device,
    )

    winner, win_reason, frames = play_visualized_game(
        env,
        mrx_policy_net,
        detective_policy_net,
        device,
        rng,
        greedy=not args.sampled,
    )

    for frame in frames:
        if frame.actor_key == "game_over":
            continue
        print(describe_frame(frame))

    print()
    print(f"Winner: {winner}")
    print(f"Reason: {win_reason}")

    graph, positions = build_layout(args.layout_seed)
    title = (
        f"Scotland Yard Replay | Mr X: {mrx_path.stem} | "
        f"Detective: {detective_path.stem} | Seed: {args.seed}"
    )
    viewer = ReplayViewer(
        frames,
        graph,
        positions,
        title=title,
        interval_ms=args.interval_ms,
        show_all_node_labels=args.show_all_node_labels,
    )

    if args.gif_path is not None:
        viewer.save_gif(args.gif_path, fps=args.gif_fps)
        print(f"Saved replay GIF: {args.gif_path}")

    if not args.no_show:
        viewer.show()


if __name__ == "__main__":
    main()
