from env import environment
from game_constants import board_graph
from pathlib import Path
import numpy as np
import torch
from helper import state_to_input_detective, state_to_input_mrx
from mrxbrain import MrXPolicy, MrXValue
from detectivebrain import DetectivePolicy, DetectiveValue
import torch.nn.functional as F
from torch.distributions import Categorical

rounds = 24
detective_amount = 4
double_probability = 0.1
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

MrXPolicyNet = MrXPolicy(mrx_in_size, 256, mrx_out_size).to(DEVICE)
MrXValueNet = MrXValue(mrx_in_size, 256, 1).to(DEVICE)

DetectivePolicyNet = DetectivePolicy(detective_in_size, 256, detective_out_size).to(DEVICE)
DetectiveValueNet = DetectiveValue(detective_in_size, 256, 1).to(DEVICE)

TRAJECTORY_KEYS = ("state", "action", "log_prob", "value", "reward", "done", "action_mask")

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


def create_trajectory():
    return {key: [] for key in TRAJECTORY_KEYS}


def append_transition(trajectory, transition):
    if transition["action"] is None:
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

def move_mr_x(round, used_double=False):
    # Let Mr X play
    state = env.mrx_state(round)
    input_vector = state_to_input_mrx(state)
    network_input = input_vector.to(DEVICE)
    mrx_pos = state['mr_x_location']
    mrx_tickets = state['mr_x_tickets']

    # Network output
    with torch.no_grad():
        policy_logits = MrXPolicyNet(network_input)
        value = MrXValueNet(network_input).squeeze(-1)

    use_double = (
        not used_double
        and mrx_tickets["double"] > 0
        and rng.random() < double_probability
    )

    action_id, next_pos, transport, use_black, log_prob, action_mask = sample_action_mrx(
        policy_logits,
        mrx_pos,
        mrx_tickets,
    )

    mrx_experience = {'state': input_vector,
                      'action': action_id,
                      'log_prob': None if log_prob is None else float(log_prob.item()),
                      'reward': 0,
                      'value': float(value.item()),
                      'done': 0,
                      'action_mask': action_mask}

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


def move_detective(detective_id, round):
    state = env.detective_state(
        detective_id=detective_id,
        round=round,
    )
    input_vector = state_to_input_detective(state)
    network_input = input_vector.to(DEVICE)
    detective_tickets = state["my_tickets"]
    detective_pos = state["my_position"]

    occupied_positions = [
        state["detective_locations"][j]
        for j in range(detective_amount)
        if j != detective_id
    ]

    # Network output
    with torch.no_grad():
        policy_logits = DetectivePolicyNet(network_input)
        value = DetectiveValueNet(network_input).squeeze(-1)

    action_id, next_pos, transport, log_prob, action_mask = sample_action_detective(
        policy_logits,
        detective_pos,
        detective_tickets,
        occupied_positions,
    )
    detective_experience = {'state': input_vector,
                      'action': action_id,
                      'log_prob': None if log_prob is None else float(log_prob.item()),
                      'reward': 0,
                      'value': float(value.item()),
                      'done': 0,
                      'action_mask': action_mask}

    if next_pos is None:
        return action_id, next_pos, transport, detective_pos, detective_experience

    env.apply_detective_move(
        detective_id,
        action_id,
        next_pos,
        transport,
    )

    return action_id, next_pos, transport, detective_pos, detective_experience

def gather_experience():
    mrx_trajectory = create_trajectory()
    detective_trajectory = create_trajectory()
    env.setup_game()

    for round in range(rounds):

        action_id, next_pos_mrx, transport, use_black, mrx_pos, use_double, mrx_experience = move_mr_x(round)
        append_transition(mrx_trajectory, mrx_experience)
        # No legal moves for mrx so he lost
        if next_pos_mrx is None:
            if rollout_debug:
                print("Mr X lost, no legal moves available")
            return -1, mrx_trajectory, detective_trajectory

        # This runs if mrx used double card
        if use_double:
            action_id, next_pos_mrx, transport, use_black, mrx_pos, _, mrx_experience = move_mr_x(round, True)
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
            action_id, next_pos, transport, detective_pos, detective_experience = move_detective(detective_id, round)
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
    "advantage",
)


def create_empty_buffer():
    return {key: [] for key in BUFFER_KEYS}


def truncate_buffer(buffer, target_size):
    for key in BUFFER_KEYS:
        buffer[key] = buffer[key][:target_size]


def extend_buffer(buffer, trajectory):
    for key in BUFFER_KEYS:
        buffer[key].extend(trajectory[key])

def create_self_play_buffers(
    mrx_target_size=BUFFER_TARGET_SIZE,
    detective_target_size=BUFFER_TARGET_SIZE,
    gamma=GAMMA,
    gae_lambda=GAE_LAMBDA,
):
    mrx_buffer = create_empty_buffer()
    detective_buffer = create_empty_buffer()

    episodes = 0
    mrx_wins = 0
    detective_wins = 0
    mrx_steps = []
    detective_steps = []

    while (
        len(mrx_buffer["action"]) < mrx_target_size
        or len(detective_buffer["action"]) < detective_target_size
    ):
        outcome, mrx_trajectory, detective_trajectory = gather_experience()
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

        mrx_steps.append(len(mrx_trajectory["action"]))
        detective_steps.append(len(detective_trajectory["action"]))

        extend_buffer(mrx_buffer, mrx_trajectory)
        extend_buffer(detective_buffer, detective_trajectory)

    truncate_buffer(mrx_buffer, mrx_target_size)
    truncate_buffer(detective_buffer, detective_target_size)

    stats = {
        "episodes": episodes,
        "mrx_win_rate": mrx_wins / max(episodes, 1),
        "detective_win_rate": detective_wins / max(episodes, 1),
        "avg_mrx_steps": float(np.mean(mrx_steps)) if mrx_steps else 0.0,
        "avg_detective_steps": float(np.mean(detective_steps)) if detective_steps else 0.0,
        "mrx_samples": len(mrx_buffer["action"]),
        "detective_samples": len(detective_buffer["action"]),
    }

    return mrx_buffer, detective_buffer, stats

def ppo_update(
    policy_net,
    value_net,
    optimizer_policy,
    optimizer_value,
    buffer,
    device,
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
            batch_advantages = advantages[batch_idx]
            batch_returns = returns[batch_idx]

            # -------------------------
            # Policy forward
            # -------------------------
            logits = policy_net(batch_states)
            masked_logits = apply_action_mask(logits, batch_action_masks)

            dist = Categorical(logits=masked_logits)

            new_log_probs = dist.log_prob(batch_actions)
            entropy = dist.entropy().mean()
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
):
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    mrx_policy_optimizer = torch.optim.Adam(MrXPolicyNet.parameters(), lr=policy_lr)
    mrx_value_optimizer = torch.optim.Adam(MrXValueNet.parameters(), lr=value_lr)
    detective_policy_optimizer = torch.optim.Adam(DetectivePolicyNet.parameters(), lr=policy_lr)
    detective_value_optimizer = torch.optim.Adam(DetectiveValueNet.parameters(), lr=value_lr)

    history = []

    print(
        f"Training on {device} | iterations={num_iterations} | "
        f"buffer_size={buffer_target_size} | gamma={gamma}"
    )

    for iteration in range(1, num_iterations + 1):
        MrXPolicyNet.eval()
        MrXValueNet.eval()
        DetectivePolicyNet.eval()
        DetectiveValueNet.eval()

        mrx_buffer, detective_buffer, self_play_stats = create_self_play_buffers(
            mrx_target_size=buffer_target_size,
            detective_target_size=buffer_target_size,
            gamma=gamma,
            gae_lambda=gae_lambda,
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
            )
            print(f"  Saved checkpoint: {checkpoint_path}")

    final_checkpoint_path = save_checkpoint(
        iteration=num_iterations,
        history=history,
        mrx_policy_optimizer=mrx_policy_optimizer,
        mrx_value_optimizer=mrx_value_optimizer,
        detective_policy_optimizer=detective_policy_optimizer,
        detective_value_optimizer=detective_value_optimizer,
    )
    plot_path = plot_training_metrics(history)

    print(f"Final checkpoint: {final_checkpoint_path}")
    if plot_path is not None:
        print(f"Saved training plot: {plot_path}")

    return history


if __name__ == "__main__":
    train()


