# TODO: Sesión de debugeo mirando acción por acción que todo esté bien
from clearml import Task, TaskTypes
import jax
import jax.numpy as jnp
from jaxtyping import Key
import flax
import optax
import pokers as pkrs
from tqdm import tqdm
from src.agents.alphaholdem.agent import AlphaHoldemAgent
from src.agents.alphaholdem.model import AlphaHoldem, ActionsObservation, CardsObservation
from src.agents.alphaholdem.replay_buffer import ReplayBuffer, BufferRecord
from src.agents.alphaholdem.stats import SelfPlayStatistics, calc_self_play_stats


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

    task = Task.init(
        project_name="ReinforcementPoker/AlphaHoldem",
        task_type=TaskTypes.training,
        task_name="Test",
    )
    task.connect(params)

    key_init, key_buffer, key_selfplay, key_ppo = jax.random.split(
        jax.random.PRNGKey(params["seed"]), 4
    )
    model = AlphaHoldem()
    model_params = model.init(
        key_init, jnp.zeros((24, 4, params["n_players"] + 2)), jnp.zeros((4, 13, 6))
    )
    models_pool = [model_params.copy(add_or_replace={}) for _ in range(params["agents_pool_size"])]  # type: ignore
    models_ratings = [1000.0] * params["agents_pool_size"]
    replay_buffer = ReplayBuffer(size=params["replay_buffer_size"], key=key_buffer)

    cum_batch_iterations = 0
    for i in range(params["iterations"]):
        self_play_metrics, key_selfplay = k_best_selfplay(
            model=model,
            models_pool=models_pool,
            models_scores=models_ratings,
            replay_buffer=replay_buffer,
            n_hands=params["hands_per_iter"],
            n_players=params["n_players"],
            key=key_selfplay,
        )
        models_ratings = self_play_metrics.elo_ratings

        for m, v in self_play_metrics.__dict__.items():
            if isinstance(v, list):
                task.get_logger().report_histogram(
                    title=m, series="self-play", values=v, iteration=i
                )
            else:
                task.get_logger().report_scalar(title=m, series="self-play", value=v, iteration=i)

        model_params = models_pool[-1].copy(add_or_replace={})
        cum_batch_iterations = ppo_training(
            it=i,
            cum_batch_iterations=cum_batch_iterations,
            replay_buffer=replay_buffer,
            model=model,
            model_params=model_params,
            batch_size=params["batch_size"],
            update_steps=params["update_steps"],
            ppo_γ=params["ppo_γ"],
            ppo_λ=params["ppo_λ"],
            ppo_ε=params["ppo_ε"],
            critic_factor=params["critic_factor"],
            entropy_factor=params["entropy_factor"],
            lr=params["lr"],
            grad_clip=params["grad_clip"],
            key=key_ppo,
            task=task,
        )

        # update pool
        last_rating = models_ratings[-1]
        worst_index = jnp.argmin(jnp.array(models_ratings))
        models_pool.pop(worst_index)
        models_ratings.pop(worst_index)
        models_pool.append(model_params)
        models_ratings.append(last_rating)

        model.save("./.checkpts/model.h5")


def k_best_selfplay(
    model: AlphaHoldem,
    models_pool: list[flax.core.FrozenDict],
    models_scores: list[float],
    replay_buffer: ReplayBuffer,
    n_hands: int,
    n_players: int,
    key: Key,
) -> tuple[SelfPlayStatistics, Key]:
    def play(match_indices: list[int], key: Key) -> list[pkrs.State]:
        game_seed = int(jax.random.randint(key, shape=(1,), minval=0, maxval=10000))
        initial_state = pkrs.State.from_seed(
            n_players=n_players, button=0, sb=0.5, bb=1.0, stake=float("inf"), seed=game_seed
        )
        observations = []
        trace = [initial_state]
        print(pkrs.visualize_trace(trace))
        while not trace[-1].final_state:
            actions_observation = AlphaHoldemAgent.encode_actions(trace)
            cards_observation = AlphaHoldemAgent.encode_cards(trace[-1])
            policy, _ = model.apply(
                models_pool[match_indices[trace[-1].current_player]],
                actions_observation,
                cards_observation,
                train=False,
            )
            key = jax.random.split(key, 1)
            action = AlphaHoldemAgent.choose_action(policy, trace[-1], key)
            trace.append(trace[-1].apply_action(action))
            observations.append((actions_observation, cards_observation, action))
            print(pkrs.visualize_state(trace[-1]))

        if trace[-1] != pkrs.StateStatus.Ok:
            last_action = trace[-1].from_action
            assert last_action is not None
            trace[-1].players_state[last_action.player].reward = -1e12

        rewards = jnp.array([ps.reward for ps in trace[-1].players_state])
        replay_buffer.update(observations, rewards)
        # sorted_players = match_indices[jnp.argsort(-rewards)]
        # places = jnp.full(len(models_pool), jnp.nan)
        # places[sorted_players] = jnp.arange(1, n_players + 1)

        return trace

    keys = jax.random.split(key, n_hands)
    matches_indices = jnp.array(
        [
            jax.random.choice(k, jnp.arange(len(models_pool)), (n_players,), replace=False)
            for k in keys
        ]
    )
    traces = [play(match, key=k) for match, k in tqdm(zip(matches_indices, keys))]

    stats = calc_self_play_stats(traces, matches_indices, models_scores)

    return stats, jax.random.split(key, 1)


def ppo_training(
    replay_buffer: ReplayBuffer,
    model: AlphaHoldem,
    model_params: flax.core.FrozenDict,
    batch_size: int,
    update_steps: int,
    ppo_γ: float,
    ppo_λ: float,
    ppo_ε: float,
    critic_factor: float,
    entropy_factor: float,
    lr: float,
    grad_clip: float,
    it: int,
    cum_batch_iterations: int,
    key: Key,
    task: Task,
) -> int:
    initial_model_params = model_params.copy(add_or_replace={})
    batched_initial_model = jax.vmap(
        lambda obs: model.apply(initial_model_params, obs[0], obs[1])[0]
    )

    optimizer = optax.adam(learning_rate=lr)
    opt_state = optimizer.init(model_params)
    batch_iter = tqdm(replay_buffer.batch_iter(batch_size))
    for batch in batch_iter:
        batched_model = jax.vmap(lambda obs: model.apply(model_params, obs[0], obs[1])[0])
        batch_π_0 = [batched_initial_model(jnp.array(r.observations))[..., 0] for r in batch]
        print("batch_π_0:", batch_π_0)
        exit()
        # La pérdida se calcula según la política seguida en todos los estados de cada traza

        # Esto va en el loss
        # batch_values = [batched_model(jnp.array(r.observations))[..., 1] for r in batch]
        # batch_advantages = [generalized_advantage_estimation(values) for values in batch_values]

        # No queda otra que flattear y usar una máscara
        # Fit model
        # loss_grad_fn = jax.value_and_grad(build_loss_fn(π_0, advantages))
        # loss_val, grads = loss_grad_fn(model_params, batch.observations)
        # updates, opt_state = optimizer.update(grads, opt_state)
        # model_params = optax.apply_updates(model_params, updates)

    return cum_batch_iterations


def build_loss_fn(model: AlphaHoldem, θ_0: flax.core.FrozenDict, θ_k: flax.core.FrozenDict):
    # TODO: para que esto funcione BufferRecord tiene que ser un struct de arrays
    def loss_fn(buffer_record: BufferRecord):
        π_0, _ = model.apply(θ_0, observation[0], observation[1])
        π_k, value = model.apply(θ_k, observation[0], observation[1])
        train_actor_loss = ppo_loss(
            advantage,
            π,
            π_0,
            ε=ppo_ε,
        )
        train_critic_loss = critic_factor * critic_loss(cumulative_reward, value)
        train_entropy_loss = entropy_factor * entropy_loss(π)
        train_loss = train_actor_loss + train_critic_loss + train_entropy_loss

    return loss_fn


# def __ppo_training(
#     it: int,
#     cum_batch_iterations: int,
#     traces: list[list[pkrs.State]],
#     model: tf.keras.Model,
#     n_players: int,
#     n_actions: int,
#     batch_size: int,
#     update_steps: int,
#     ppo_γ: float,
#     ppo_λ: float,
#     ppo_ε: float,
#     critic_factor: float,
#     entropy_factor: float,
#     lr: float,
#     grad_clip: float,
#     seed: int,
#     task: Task,
# ):
#     rng = np.random.default_rng(seed=seed)
#     action_indices = [
#         (trace_index, action_index)
#         for trace_index, trace in enumerate(traces)
#         for action_index, _ in enumerate(trace)
#         if action_index != len(trace) - 1
#     ]
#     rng.shuffle(action_indices)

#     batches_iterator = trange(0, len(action_indices), batch_size)
#     for batch_i in batches_iterator:
#         # print(f"batch {batch_i}")
#         batch_action_indices = action_indices[batch_i : batch_i + batch_size]
#         # Batch of partial trajectories
#         traces_batch = [
#             traces[action_index[0]][action_index[1] :] for action_index in batch_action_indices
#         ]
#         N = len(traces_batch)
#         # print("N:", N)

#         # print("traces_batch:")
#         # for ti, t in enumerate(traces_batch):
#         #    print(f"\n\n--- {ti} ---\n")
#         #    print(visualize_trace(t))

#         # actions_encoding: ΣT_i x (actions_encoding_dim)
#         actions_encodings = np.array(
#             [
#                 AlphaHoldemAgent.encode_actions(
#                     traces[action_index[0]][: action_index[1] + i],
#                     traces[action_index[0]][action_index[1] + i - 1].current_player,
#                     n_players,
#                     n_actions,
#                 )
#                 for action_index in batch_action_indices
#                 for i in range(1, len(traces[action_index[0]][action_index[1] :]))
#             ]
#         )
#         # print("actions_encodings:", actions_encodings)
#         # cards_encoding: ΣT_i x (cards_encoding_dim)
#         cards_encodings = np.array(
#             [
#                 AlphaHoldemAgent.encode_cards(state, state.current_player)
#                 for trace in traces_batch
#                 for state in trace
#             ]
#         )
#         # print("cards_encodings:", cards_encodings)

#         trace_lens = [len(trace) for trace in traces_batch]
#         # initial_states_indices: N
#         initial_states_indices = np.concatenate([[0], np.cumsum(trace_lens)[:-1]])
#         # print("mask_indices:", initial_states_indices)

#         # traces_mask: N x ΣT_i
#         traces_mask = np.zeros((N, np.sum(trace_lens)), dtype=np.int32)
#         for i in range(1, len(initial_states_indices)):
#             traces_mask[i - 1, initial_states_indices[i - 1 : i]] = 1
#         traces_mask = tf.convert_to_tensor(traces_mask, dtype=tf.int32)
#         # print("traces_mask:", traces_mask)

#         # initial_performed_actions: N
#         # Actions performed at the start of each trace
#         initial_performed_actions = [
#             int(traces[trace_index][state_index + 1].from_action.action.action)  # type: ignore
#             for trace_index, state_index in batch_action_indices
#             if traces[trace_index][state_index + 1].from_action is not None
#         ]
#         initial_performed_actions = tf.convert_to_tensor(initial_performed_actions, dtype=tf.int64)
#         # print("initial_performed_actions:", initial_performed_actions)

#         ppo_policy_indices = tf.stack([initial_states_indices, initial_performed_actions], axis=-1)
#         # print("ppo_policy_indices:", ppo_policy_indices)

#         # Current model policy
#         # π0: ΣT_i x num_actions
#         # values: ΣT_i
#         π_0, values_0 = model(
#             {
#                 "actions": actions_encodings,
#                 "cards": cards_encodings,
#             }
#         )
#         # print("π_0", π_0.shape)
#         # print("values_0", values_0.shape)
#         # advantages: N
#         advantages = generalized_advantage_estimation(values_0, traces_mask, ppo_γ, ppo_λ)
#         # print("advantages:", advantages.shape)

#         # cumulative_rewards: ΣT_i
#         cumulative_rewards = np.array(
#             [
#                 trace[-1].players_state[state.current_player].reward
#                 for trace in traces_batch
#                 for state in trace
#             ]
#         )
#         # print("cumulative_rewards:", cumulative_rewards)
#         cumulative_rewards_discounts = ppo_γ ** np.array(
#             [
#                 len([s for s in trace[i:] if s.current_player == trace[i].current_player]) - 1
#                 for trace in traces_batch
#                 for i in range(len(trace))
#             ]
#         )
#         # print("cumulative_rewards_discounts:", cumulative_rewards_discounts)
#         cumulative_rewards = cumulative_rewards * cumulative_rewards_discounts
#         cumulative_rewards = tf.convert_to_tensor(cumulative_rewards, dtype=tf.float32)
#         # print("cumulative_rewards:", cumulative_rewards)

#         optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
#         for update_i in range(update_steps):
#             with tf.GradientTape() as tape:
#                 # π: ΣT_i x num_actions
#                 # values: ΣT_i
#                 π, values = model({"actions": actions_encodings, "cards": cards_encodings})
#                 # print("π", π.shape)
#                 # print("values", values.shape)
#                 train_actor_loss = ppo_loss(
#                     advantages,
#                     tf.gather_nd(π, ppo_policy_indices),
#                     tf.gather_nd(π_0, ppo_policy_indices),
#                     ε=ppo_ε,
#                 )
#                 train_critic_loss = critic_factor * critic_loss(cumulative_rewards, values)
#                 train_entropy_loss = entropy_factor * entropy_loss(π)
#                 train_loss = train_actor_loss + train_critic_loss + train_entropy_loss

#             grads = tape.gradient(train_loss, model.trainable_weights)
#             grads, _ = tf.clip_by_global_norm(grads, grad_clip)
#             optimizer.apply_gradients(zip(grads, model.trainable_weights))

#             task.get_logger().report_scalar(
#                 title="gradient_norm",
#                 series="gradients",
#                 value=tf.linalg.global_norm(grads),
#                 iteration=cum_batch_iterations * update_steps + update_i,
#             )

#             # TODO: Visualizar evolución en la distribución de las policy

#         task.get_logger().report_scalar(
#             title="loss", series="train", value=train_critic_loss, iteration=cum_batch_iterations
#         )
#         task.get_logger().report_scalar(
#             title="actor_loss",
#             series="train",
#             value=train_actor_loss,
#             iteration=cum_batch_iterations,
#         )
#         task.get_logger().report_scalar(
#             title="critic_loss",
#             series="train",
#             value=train_critic_loss,
#             iteration=cum_batch_iterations,
#         )
#         task.get_logger().report_scalar(
#             title="entropy_loss",
#             series="train",
#             value=train_entropy_loss,
#             iteration=cum_batch_iterations,
#         )

#         batches_iterator.set_description(f"It {it+1}")
#         batches_iterator.set_postfix(
#             loss=train_loss.numpy(),
#             actor_loss=train_actor_loss.numpy(),
#             critic_loss=train_critic_loss.numpy(),
#             entropy_loss=train_entropy_loss.numpy(),
#         )

#         cum_batch_iterations += 1

#     return cum_batch_iterations


def generalized_advantage_estimation(
    values: jax.Array, mask: jax.Array, γ: float, λ: float = 0.95
) -> jax.Array:
    # mask = tf.cast(mask, tf.float32)
    # # V: N x ΣT_i
    # V = mask * tf.expand_dims(values, axis=0)
    # # Δ: N x (ΣT_i - 1)
    # Δ = γ * V[:, 1:] - V[:, :-1]
    # # Δ: N x ΣT_i
    # zeros = tf.zeros((tf.shape(Δ)[0], 1))
    # Δ = tf.concat([Δ, zeros], axis=1)

    # # discounts: N x ΣT_i
    # indices = tf.math.cumsum(mask, axis=-1) - 1
    # discounts = tf.math.pow(γ * λ, indices) * mask

    # # advantages: N
    # advantages = tf.reduce_sum(discounts * Δ, axis=-1)

    # return advantages
    return jnp.zeros(10)


def clipped_ppo_loss(
    advantages: jax.Array, π: jax.Array, π_0: jax.Array, ε: float = 0.2
) -> jax.Array:
    assert advantages.shape == π.shape
    assert advantages.shape == π_0.shape

    r = π / π_0
    raw_loss = r * advantages
    clipped_loss = jnp.clip(r, 1 - ε, 1 + ε) * advantages

    return jnp.mean(jnp.sum(jnp.minimum(raw_loss, clipped_loss), axis=-1))


# Source: https://github.com/ChintanTrivedi/rl-bot-football/blob/master/train.py#L63
def ppo_loss(advantages: jax.Array, π: jax.Array, π_0: jax.Array, ε: float = 0.2) -> jax.Array:
    ratio = jnp.exp(jnp.log(π + 1e-12) - jnp.log(π_0 + 1e-12))
    p1 = ratio * advantages
    p2 = jnp.clip(ratio, clip_value_min=1 - ε, clip_value_max=1 + ε) * advantages
    return -jnp.mean(jnp.minimum(p1, p2))


def critic_loss(rewards: jax.Array, values: jax.Array) -> jax.Array:
    return jnp.mean((rewards - values) ** 2)


def entropy_loss(π: jax.Array) -> jax.Array:
    return jnp.mean(π * jnp.log(π + 1e-12))


if __name__ == "__main__":
    train()
