# TODO: Sesión de debugeo mirando acción por acción que todo esté bien
from clearml import Task, TaskTypes
import numpy as np
import tensorflow as tf
from tqdm import trange
from dataclasses import dataclass, field
from src.agents.alphaholdem.agent import AlphaHoldemAgent
from src.agents.alphaholdem.model import build_alphaholdem
from src.agents.alphaholdem.stats import SelfPlayStatistics, calc_self_play_stats
from src.utils.environment import Trace, env
from src.utils.visualizations import visualize_trace


def train():
    params = {
        "seed": 1234,
        "iterations": 1000,
        "hands_per_iter": 128,
        "n_players": 2,
        "batch_size": 20,
        "replay_buffer_size": 1000,
        "agents_pool_size": 10,
        "update_steps": 128,
        "ppo_γ": 0.999,
        "ppo_λ": 0.95,
        "ppo_ε": 0.2,
        "critic_factor": 0.5,
        "entropy_factor": 0.001,
        "lr": 3e-6,  # 0.0003,
        "grad_clip": 50,
    }

    tf.random.set_seed(params["seed"])

    task = Task.init(
        project_name="ReinforcementPoker/AlphaHoldem",
        task_type=TaskTypes.training,
        task_name="Test",
    )
    task.connect(params)

    replay_buffer: list[Trace] = []
    model = build_alphaholdem(n_actions=5)
    models_pool = [tf.keras.models.clone_model(model) for _ in range(params["agents_pool_size"])]
    models_ratings = [1000.0] * params["agents_pool_size"]

    cum_batch_iterations = 0
    for i in range(params["iterations"]):
        new_traces, self_play_metrics = k_best_selfplay(
            models_pool,
            models_ratings,
            params["hands_per_iter"],
            params["n_players"],
            params["seed"],
        )
        models_ratings = self_play_metrics.elo_ratings
        replay_buffer = update_replay_buffer(
            replay_buffer, new_traces, params["replay_buffer_size"]
        )

        for m, v in self_play_metrics.__dict__.items():
            if isinstance(v, list):
                task.get_logger().report_histogram(
                    title=m, series="self-play", values=np.array(v), iteration=i
                )
            else:
                task.get_logger().report_scalar(title=m, series="self-play", value=v, iteration=i)

        model = tf.keras.models.clone_model(models_pool[-1])
        cum_batch_iterations = ppo_training(
            it=i,
            cum_batch_iterations=cum_batch_iterations,
            traces=replay_buffer,
            model=model,
            n_players=params["n_players"],
            n_actions=5,
            batch_size=params["batch_size"],
            update_steps=params["update_steps"],
            ppo_γ=params["ppo_γ"],
            ppo_λ=params["ppo_λ"],
            ppo_ε=params["ppo_ε"],
            critic_factor=params["critic_factor"],
            entropy_factor=params["entropy_factor"],
            lr=params["lr"],
            grad_clip=params["grad_clip"],
            seed=params["seed"],
            task=task,
        )

        # update pool
        last_rating = models_ratings[-1]
        worst_index = np.argmin(models_ratings)
        models_pool.pop(worst_index)
        models_ratings.pop(worst_index)
        models_pool.append(model)
        models_ratings.append(last_rating)

        model.save("./.checkpts/model.h5")


# TODO: Sacarlo en una clase más sofisticada
def update_replay_buffer(
    replay_buffer: list[Trace], new_traces: list[Trace], size: int
) -> list[Trace]:
    len_diff = max(0, len(replay_buffer) + len(new_traces) - size)
    return replay_buffer[len_diff:] + new_traces


def k_best_selfplay(
    models_pool: list[tf.keras.Model],
    models_scores: list[float],
    hands: int,
    n_players: int,
    seed: int,
) -> tuple[list[Trace], SelfPlayStatistics]:
    def play(match_indices: list[int]) -> tuple[Trace, np.ndarray]:
        environment = env(num_players=n_players)
        environment.reset(seed=seed)
        agents = {
            agent_id: AlphaHoldemAgent(i, models_pool[match_indices[i]])
            for i, agent_id in enumerate(environment.agents)
        }

        rewards = np.zeros(len(agents))
        for agent_id in environment.agent_iter():
            agent = agents[agent_id]
            observation, reward, termination, truncation, info = environment.last()

            if observation is None or termination or truncation:
                action = None
                rewards[agent.player_id] = reward
            else:
                action = agent.step_batch([observation["state"]])[0]

            environment.step(action)

        sorted_players = match_indices[np.argsort(-rewards)]
        places = np.full(len(models_pool), np.nan)
        places[sorted_players] = np.arange(1, n_players + 1)

        return environment.trace, places

    rng = np.random.default_rng(seed=seed)
    matches_indices = np.array(
        [rng.choice(np.arange(len(models_pool)), n_players, replace=False) for _ in range(hands)]
    )
    res = [play(match) for match in matches_indices]
    traces = [r[0] for r in res]
    places = np.array([r[1] for r in res])

    stats = calc_self_play_stats(traces, matches_indices, places, models_scores)

    return traces, stats


def ppo_training(
    it: int,
    cum_batch_iterations: int,
    traces: list[Trace],
    model: tf.keras.Model,
    n_players: int,
    n_actions: int,
    batch_size: int,
    update_steps: int,
    ppo_γ: float,
    ppo_λ: float,
    ppo_ε: float,
    critic_factor: float,
    entropy_factor: float,
    lr: float,
    grad_clip: float,
    seed: int,
    task: Task,
):
    rng = np.random.default_rng(seed=seed)
    action_indices = [
        (trace_index, action_index)
        for trace_index, trace in enumerate(traces)
        for action_index, _ in enumerate(trace)
        if action_index != len(trace) - 1
    ]
    rng.shuffle(action_indices)

    batches_iterator = trange(0, len(action_indices), batch_size)
    for batch_i in batches_iterator:
        # print(f"batch {batch_i}")
        batch_action_indices = action_indices[batch_i : batch_i + batch_size]
        # Batch of partial trajectories
        traces_batch = [
            traces[action_index[0]][action_index[1] :] for action_index in batch_action_indices
        ]
        N = len(traces_batch)
        # print("N:", N)

        # print("traces_batch:")
        # for ti, t in enumerate(traces_batch):
        #    print(f"\n\n--- {ti} ---\n")
        #    print(visualize_trace(t))

        # actions_encoding: ΣT_i x (actions_encoding_dim)
        actions_encodings = np.array(
            [
                AlphaHoldemAgent.encode_actions(state, state.current_player, n_players, n_actions)
                for trace in traces_batch
                for state in trace
            ]
        )
        # print("actions_encodings:", actions_encodings)
        # cards_encoding: ΣT_i x (cards_encoding_dim)
        cards_encodings = np.array(
            [
                AlphaHoldemAgent.encode_cards(state, state.current_player)
                for trace in traces_batch
                for state in trace
            ]
        )
        # print("cards_encodings:", cards_encodings)

        trace_lens = [len(trace) for trace in traces_batch]
        # initial_states_indices: N
        initial_states_indices = np.concatenate([[0], np.cumsum(trace_lens)[:-1]])
        # print("mask_indices:", initial_states_indices)

        # traces_mask: N x ΣT_i
        traces_mask = np.zeros((N, np.sum(trace_lens)), dtype=np.int32)
        for i in range(1, len(initial_states_indices)):
            traces_mask[i - 1, initial_states_indices[i - 1 : i]] = 1
        traces_mask = tf.convert_to_tensor(traces_mask, dtype=tf.int32)
        # print("traces_mask:", traces_mask)

        # initial_performed_actions: N
        # Actions performed at the start of each trace
        initial_performed_actions = [
            traces[trace_index][-1].action_record[state_index].action.value
            for trace_index, state_index in batch_action_indices
        ]
        initial_performed_actions = tf.convert_to_tensor(initial_performed_actions, dtype=tf.int64)
        # print("initial_performed_actions:", initial_performed_actions)

        ppo_policy_indices = tf.stack([initial_states_indices, initial_performed_actions], axis=-1)
        # print("ppo_policy_indices:", ppo_policy_indices)

        # Current model policy
        # π0: ΣT_i x num_actions
        # values: ΣT_i
        π_0, values_0 = model(
            {
                "actions": actions_encodings,
                "cards": cards_encodings,
            }
        )
        # print("π_0", π_0.shape)
        # print("values_0", values_0.shape)
        # advantages: N
        advantages = generalized_advantage_estimation(values_0, traces_mask, ppo_γ, ppo_λ)
        # print("advantages:", advantages.shape)

        # cumulative_rewards: ΣT_i
        cumulative_rewards = np.array(
            [
                trace[-1].players_state[state.current_player].reward
                for trace in traces_batch
                for state in trace
            ]
        )
        # print("cumulative_rewards:", cumulative_rewards)
        cumulative_rewards_discounts = ppo_γ ** np.array(
            [
                len([s for s in trace[i:] if s.current_player == trace[i].current_player]) - 1
                for trace in traces_batch
                for i in range(len(trace))
            ]
        )
        # print("cumulative_rewards_discounts:", cumulative_rewards_discounts)
        cumulative_rewards = cumulative_rewards * cumulative_rewards_discounts
        cumulative_rewards = tf.convert_to_tensor(cumulative_rewards, dtype=tf.float32)
        # print("cumulative_rewards:", cumulative_rewards)

        optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
        for update_i in range(update_steps):
            with tf.GradientTape() as tape:
                # π: ΣT_i x num_actions
                # values: ΣT_i
                π, values = model({"actions": actions_encodings, "cards": cards_encodings})
                # print("π", π.shape)
                # print("values", values.shape)
                train_actor_loss = ppo_loss(
                    advantages,
                    tf.gather_nd(π, ppo_policy_indices),
                    tf.gather_nd(π_0, ppo_policy_indices),
                    ε=ppo_ε,
                )
                train_critic_loss = critic_factor * critic_loss(cumulative_rewards, values)
                train_entropy_loss = entropy_factor * entropy_loss(π)
                train_loss = train_actor_loss + train_critic_loss + train_entropy_loss

            grads = tape.gradient(train_loss, model.trainable_weights)
            grads, _ = tf.clip_by_global_norm(grads, grad_clip)
            optimizer.apply_gradients(zip(grads, model.trainable_weights))

            task.get_logger().report_scalar(
                title="gradient_norm",
                series="gradients",
                value=tf.linalg.global_norm(grads),
                iteration=cum_batch_iterations * update_steps + update_i,
            )

            # TODO: Visualizar evolución en la distribución de las policy

        task.get_logger().report_scalar(
            title="loss", series="train", value=train_critic_loss, iteration=cum_batch_iterations
        )
        task.get_logger().report_scalar(
            title="actor_loss",
            series="train",
            value=train_actor_loss,
            iteration=cum_batch_iterations,
        )
        task.get_logger().report_scalar(
            title="critic_loss",
            series="train",
            value=train_critic_loss,
            iteration=cum_batch_iterations,
        )
        task.get_logger().report_scalar(
            title="entropy_loss",
            series="train",
            value=train_entropy_loss,
            iteration=cum_batch_iterations,
        )

        batches_iterator.set_description(f"It {it+1}")
        batches_iterator.set_postfix(
            loss=train_loss.numpy(),
            actor_loss=train_actor_loss.numpy(),
            critic_loss=train_critic_loss.numpy(),
            entropy_loss=train_entropy_loss.numpy(),
        )

        cum_batch_iterations += 1

    return cum_batch_iterations


def generalized_advantage_estimation(
    values: tf.Tensor, mask: tf.Tensor, γ: float, λ: float = 0.95
) -> tf.Tensor:
    mask = tf.cast(mask, tf.float32)
    # V: N x ΣT_i
    V = mask * tf.expand_dims(values, axis=0)
    # Δ: N x (ΣT_i - 1)
    Δ = γ * V[:, 1:] - V[:, :-1]
    # Δ: N x ΣT_i
    zeros = tf.zeros((tf.shape(Δ)[0], 1))
    Δ = tf.concat([Δ, zeros], axis=1)

    # discounts: N x ΣT_i
    indices = tf.math.cumsum(mask, axis=-1) - 1
    discounts = tf.math.pow(γ * λ, indices) * mask

    # advantages: N
    advantages = tf.reduce_sum(discounts * Δ, axis=-1)

    return advantages


def clipped_ppo_loss(
    advantages: tf.Tensor, π: tf.Tensor, π_0: tf.Tensor, ε: float = 0.2
) -> tf.Tensor:
    assert advantages.shape == π.shape
    assert advantages.shape == π_0.shape

    r = π / π_0
    raw_loss = r * advantages
    clipped_loss = tf.clip(r, 1 - ε, 1 + ε) * advantages

    return tf.mean(tf.reduce_sum(tf.minimum(raw_loss, clipped_loss), axis=-1))


# Source: https://github.com/ChintanTrivedi/rl-bot-football/blob/master/train.py#L63
def ppo_loss(advantages: tf.Tensor, π: tf.Tensor, π_0: tf.Tensor, ε: float = 0.2) -> tf.Tensor:
    ratio = tf.math.exp(tf.math.log(π + 1e-12) - tf.math.log(π_0 + 1e-12))
    p1 = ratio * advantages
    p2 = tf.clip_by_value(ratio, clip_value_min=1 - ε, clip_value_max=1 + ε) * advantages
    return -tf.reduce_mean(tf.minimum(p1, p2))


def critic_loss(rewards: tf.Tensor, values: tf.Tensor) -> tf.Tensor:
    return tf.reduce_mean((rewards - values) ** 2)


def entropy_loss(π: tf.Tensor) -> tf.Tensor:
    return tf.reduce_mean(π * tf.math.log(π + 1e-12))


if __name__ == "__main__":
    train()
