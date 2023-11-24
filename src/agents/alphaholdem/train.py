# Lo que está pasando es que el value repercute y se amplifica en
# el advantage, que a su vez hace que el actor loss baje más de lo
# que el critic loss sube, así que le compensa hacer que el value sea
# lo más alto posible. ¿Está bien el actor loss? Habrá que hacer que
# no sea tan sensible al advantage. ¿Está bien el critic loss?
# Además la entropía da 0 cuando la distribución no es realmente determinista
# ¿por qué?
from clearml import Task, TaskTypes
import jax
import jax.numpy as jnp
from jaxtyping import Array, Scalar, Float
import flax
import optax
import pokers as pkrs
from tqdm import tqdm
from src.agents.alphaholdem.agent import AlphaHoldemAgent
from src.agents.alphaholdem.model import AlphaHoldem, ActionsObservation, CardsObservation
from src.agents.alphaholdem.replay_buffer import ReplayBuffer, BufferRecord
from src.agents.alphaholdem.self_play import k_best_selfplay
from src.utils.hand_ranker import get_hand_score
from src.utils.poker_agent import RandomAgent, OnlyCallsAgent

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_debug_nans", True)


def train():
    params = {
        "seed": 123456,  # 123456, <- nan temprano
        "iterations": 10000,
        "hands_per_iter": 80,  # 128,
        "n_players": 2,
        "batch_size": 20,
        "agents_pool_size": 10,
        "ppo_γ": 0.999,
        "ppo_λ": 0.95,
        "ppo_ε": 0.2,
        "critic_factor": 0.5,
        "entropy_factor": 0.01,
        "lr": 3e-3,
        "grad_clip": 10.0,
    }

    task = Task.init(
        project_name="ReinforcementPoker/AlphaHoldem",
        task_type=TaskTypes.training,
        task_name="Play against only calls agent",
    )
    task.connect(params)
    #task.execute_remotely(queue_name="kaggle-bruno")

    key_init, key_buffer, key_selfplay, key_ppo = jax.random.split(
        jax.random.PRNGKey(params["seed"]), 4
    )
    model = AlphaHoldem()
    model_params = model.init(
        key_init, jnp.zeros((24, params["n_players"] + 2, 4)), jnp.zeros((4, 13, 6))
    )
    agents_pool = [OnlyCallsAgent(key_selfplay)]
    models_ratings = [1000.0] * params["agents_pool_size"]
    replay_buffer = ReplayBuffer(key=key_buffer)

    # optimizer = optax.adam(learning_rate=params["lr"])
    optimizer = optax.chain(
        optax.clip_by_global_norm(params["grad_clip"]), optax.adamw(learning_rate=params["lr"])
    )
    opt_state = optimizer.init(model_params)

    cum_batch_iterations = 0
    for i in range(params["iterations"]):
        replay_buffer.reset()
        replay_buffer, self_play_metrics, key_selfplay = k_best_selfplay(
            model=model,
            model_params=model_params,  # type: ignore
            agents_pool=agents_pool,  # type: ignore
            agents_scores=models_ratings,
            replay_buffer=replay_buffer,
            n_hands=params["hands_per_iter"],
            n_players=params["n_players"],
            key=key_selfplay,
        )
        models_ratings = self_play_metrics.elo_ratings

        if i % 25 == 0:
            for m, v in self_play_metrics.__dict__.items():
                if isinstance(v, tuple):
                    task.get_logger().report_histogram(
                        title=f"self-play | {m}", series=m, values=v[0], xlabels=v[1], iteration=i
                    )
                elif isinstance(v, list):
                    task.get_logger().report_histogram(
                        title=f"self-play | {m}", series=m, values=v, iteration=i
                    )
                else:
                    task.get_logger().report_scalar(
                        title=f"self-play | {m}", series=m, value=v, iteration=i
                    )

        model_params, opt_state, cum_batch_iterations = ppo_training(
            i=i,
            cum_batch_iterations=cum_batch_iterations,
            replay_buffer=replay_buffer,
            model=model,
            model_params=model_params,  # type: ignore
            batch_size=params["batch_size"],
            ppo_γ=params["ppo_γ"],
            ppo_λ=params["ppo_λ"],
            ppo_ε=params["ppo_ε"],
            critic_factor=params["critic_factor"],
            entropy_factor=params["entropy_factor"],
            optimizer=optimizer,
            opt_state=opt_state,
            task=task,
        )

        # update pool
        # last_rating = models_ratings[-1]
        # worst_index = (
        #     int(jnp.argmin(jnp.array(models_ratings[1:]))) + 1
        # )  # Fix 0-th agent for benchmarking
        # models_pool.pop(worst_index)
        # models_ratings.pop(worst_index)
        # models_pool.append(model_params)
        # models_ratings.append(last_rating)

        # TODO: model.save("./.checkpts/model.h5")


# def k_best_selfplay(
#     model: AlphaHoldem,
#     models_pool: list[flax.core.FrozenDict],
#     models_scores: list[float],
#     replay_buffer: ReplayBuffer,
#     n_hands: int,
#     n_players: int,
#     key: jax.random.KeyArray,
# ) -> tuple[SelfPlayStatistics, jax.random.KeyArray]:
#     # TODO: Quedarse unicamente con las observaciones del agente principal -> El reward de otros agentes no sirve como estimación
#     # TODO: Meter un agente extra aleatorio para medir progresión y aumentar la exploración
#     def play(match_indices: list[int], key: jax.random.KeyArray) -> list[pkrs.State]:
#         # TEMPORAL
#         overfit_seeds = jnp.arange(10, dtype=jnp.int32)
#         game_seed = int(jax.random.choice(key, overfit_seeds))
#         # game_seed = int(jax.random.randint(key, shape=(1,), minval=0, maxval=10000))
#         initial_state = pkrs.State.from_seed(
#             n_players=n_players, button=0, sb=0.5, bb=1.0, stake=float("inf"), seed=game_seed
#         )
#         actions_observations_record = []
#         cards_observations_record = []
#         actions_taken_record = []
#         hand_scores_record = []
#         trace = [initial_state]
#         self_play_it = 0
#         while not trace[-1].final_state and self_play_it < 24:
#             self_play_it += 1
#             actions_observation = AlphaHoldemAgent.encode_actions(trace)
#             cards_observation = AlphaHoldemAgent.encode_cards(trace[-1])

#             policy, _, _ = model.apply(
#                 models_pool[match_indices[trace[-1].current_player]],
#                 actions_observation,
#                 cards_observation,
#                 train=False,
#             )

#             _, key = jax.random.split(key)
#             action_index = AlphaHoldemAgent.choose_action(policy, key, trace[-1].legal_actions)
#             action = AlphaHoldemAgent.index_to_action(action_index, trace[-1])

#             if trace[-1].current_player == n_players - 1:
#                 assert not jnp.any(jnp.isnan(actions_observation))
#                 assert not jnp.any(jnp.isnan(cards_observation))
#                 actions_observations_record.append(actions_observation)
#                 cards_observations_record.append(cards_observation)
#                 actions_taken_record.append(action_index)
#                 hand_scores_record.append(calc_hand_score(trace[-1]))

#             trace.append(trace[-1].apply_action(action))

#         rewards = [ps.reward for ps in trace[-1].players_state]
#         assert trace[-1].status == pkrs.StateStatus.Ok
#         print("hand_len:", len(trace))
#         if trace[-1].status != pkrs.StateStatus.Ok:
#             last_action = trace[-1].from_action
#             assert last_action is not None
#             rewards[last_action.player] = -100

#         # print(pkrs.visualize_trace(trace))
#         # print(rewards)

#         # print(actions_observations_record[0][-1][0])
#         # exit()

#         if (
#             len(trace) < 25
#             and actions_observations_record
#             and cards_observations_record
#             and actions_taken_record
#             and hand_scores_record
#         ):
#             replay_buffer.add_agent_play(
#                 actions_observations=actions_observations_record,
#                 cards_observations=cards_observations_record,
#                 actions_taken=actions_taken_record,
#                 reward=rewards[-1],
#                 hand_scores=hand_scores_record,
#             )
#         # sorted_players = match_indices[jnp.argsort(-rewards)]
#         # places = jnp.full(len(models_pool), jnp.nan)
#         # places[sorted_players] = jnp.arange(1, n_players + 1)

#         return trace

#     keys = jax.random.split(key, n_hands)
#     matches_indices = jnp.array(
#         [
#             jnp.concatenate(
#                 [
#                     jax.random.choice(
#                         k, jnp.arange(len(models_pool) - 1), (n_players - 1,), replace=False
#                     ),
#                     jnp.array([len(models_pool) - 1]),
#                 ]
#             )
#             for k in keys
#         ]
#     )
#     traces = [play(match, key=k) for match, k in tqdm(zip(matches_indices, keys), desc="Self play")]

#     stats = calc_self_play_stats(traces, matches_indices, models_scores)

#     return stats, jax.random.split(key, 1)[1]


# def calc_hand_score(state: pkrs.State) -> float:
#     public_cards = state.public_cards
#     player_cards = state.players_state[state.current_player].hand
#     public_cards_str = [
#         f"{str(c.suit).split('.')[-1][0]}{str(c.rank).split('.')[-1][-1]}" for c in public_cards
#     ]
#     player_cards_str = [
#         f"{str(c.suit).split('.')[-1][0]}{str(c.rank).split('.')[-1][-1]}" for c in player_cards
#     ]
#     # print(player_cards_str)
#     # exit()
#     score = get_hand_score(player_cards_str, public_cards_str)
#     return score


def ppo_training(
    replay_buffer: ReplayBuffer,
    model: AlphaHoldem,
    model_params: flax.core.FrozenDict,
    batch_size: int,
    ppo_γ: float,
    ppo_λ: float,
    ppo_ε: float,
    critic_factor: float,
    entropy_factor: float,
    optimizer,
    opt_state,
    i: int,
    cum_batch_iterations: int,
    task: Task,
):
    initial_params = flax.core.copy(model_params, add_or_replace={})
    batch_metrics: dict[str, list[float]] = dict()

    batch_iter = tqdm(
        replay_buffer.batch_iter(batch_size),
        desc="Training",
        total=replay_buffer.n_batches(batch_size),
    )
    for batch in batch_iter:
        loss_fn = build_ppo_loss(
            model=model,
            θ_0=initial_params,  # type: ignore
            γ=ppo_γ,
            λ=ppo_λ,
            ppo_ε=ppo_ε,
            critic_factor=critic_factor,
            entropy_factor=entropy_factor,
        )

        loss_grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
        try:
            (loss_val, metrics), grads = loss_grad_fn(model_params, batch)
        except FloatingPointError:
            jax.debug.print("Invalid backpropagation, skipping")
            continue
        updates, opt_state = optimizer.update(grads, opt_state, params=model_params)
        model_params = optax.apply_updates(model_params, updates)  # type: ignore

        batch_iter.set_postfix(metrics)
        for m, v in metrics.items():
            task.get_logger().report_scalar(f"train | {m}", m, v, cum_batch_iterations)
            batch_metrics[m] = batch_metrics.get(m, []) + [v]

        norm = optax.global_norm(updates)
        task.get_logger().report_scalar("grads", "grads_norm", float(norm), cum_batch_iterations)
        cum_batch_iterations += 1

    median_batch_metrics = {f"{m}": jnp.median(jnp.array(v)) for m, v in batch_metrics.items()}
    print(median_batch_metrics)

    return model_params, opt_state, cum_batch_iterations


def build_ppo_loss(
    model: AlphaHoldem,
    θ_0: flax.core.FrozenDict,
    γ: float,
    λ: float,
    ppo_ε: float,
    critic_factor: float,
    entropy_factor: float,
):
    def loss_fn(θ_k: flax.core.FrozenDict, buffer_record_batch: list[BufferRecord]):
        batch_advantages = []
        batch_values = []
        batch_ranks = []
        batch_rewards = []
        batch_actions_taken = []
        batch_π_0 = []
        batch_π_k = []
        for buffer_record in buffer_record_batch:
            π_0: Float[Array, "hand_len n_actions"]
            π_0, _, _ = jax.vmap(lambda a_obs, c_obs: model.apply(θ_0, a_obs, c_obs, train=False))(  # type: ignore
                buffer_record.actions_observations, buffer_record.cards_observations
            )
            batch_π_0.append(π_0)

            π_k: Float[Array, "hand_len n_actions"]
            values: Float[Array, "hand_len"]
            # TODO: BatchNorm params
            π_k, values, hand_scores = jax.vmap(  # type: ignore
                lambda a_obs, c_obs: model.apply(θ_k, a_obs, c_obs, train=False)
            )(buffer_record.actions_observations, buffer_record.cards_observations)
            batch_π_k.append(π_k)
            batch_values.append(values)
            batch_ranks.append(hand_scores)

            advantages = generalized_advantage_estimation(
                jnp.concatenate([values, jnp.array([buffer_record.reward])]), γ, λ
            )
            batch_advantages.append(advantages)

            discounted_rewards = buffer_record.reward * (γ ** jnp.arange(len(values) - 1, -1, -1))
            batch_rewards.append(discounted_rewards)
            jax.debug.print("rew={}, val={}", discounted_rewards, values)
            jax.debug.print(
                "real_hand_scores={}, pred_hand_scores={}", buffer_record.hand_scores, hand_scores
            )

            batch_actions_taken.append(buffer_record.actions_taken)

        batch_advantages = jnp.concatenate(batch_advantages)
        batch_values = jnp.concatenate(batch_values)
        batch_ranks = jnp.concatenate(batch_ranks)
        batch_rewards = jnp.concatenate(batch_rewards)
        batch_actions_taken = jnp.concatenate(batch_actions_taken)
        batch_π_0 = jnp.concatenate(batch_π_0)
        batch_π_k = jnp.concatenate(batch_π_k)

        actor_loss_value, kl_div = actor_loss(
            batch_advantages,
            batch_π_0[:, batch_actions_taken],
            batch_π_k[:, batch_actions_taken],
            ε=ppo_ε,
        )
        critic_loss_value = critic_factor * critic_loss(batch_rewards, batch_values)
        potential_loss_value = critic_factor * critic_loss(jnp.abs(batch_rewards), batch_values)
        hand_score_factor = 2.5
        hand_score_loss_value = hand_score_factor * jnp.mean(
            (
                batch_ranks
                - jnp.concatenate(
                    [buffer_record.hand_scores for buffer_record in buffer_record_batch]
                )
            )
            ** 2
        )
        entropy_loss_value = -entropy_factor * entropy(batch_π_k)
        loss_value = (
            actor_loss_value + critic_loss_value + entropy_loss_value + hand_score_loss_value
        )
        # loss_value = hand_score_loss_value + entropy_loss_value

        return loss_value, {
            "loss": loss_value,
            "actor_loss": actor_loss_value,
            "hand_score_loss": hand_score_loss_value,
            "critic_loss": critic_loss_value,
            "potential_loss": potential_loss_value,
            "entropy_loss": entropy_loss_value,
            "advantage_mean": batch_advantages.mean(),
            "kl_divergence": kl_div,
        }

    return loss_fn


def generalized_advantage_estimation(
    values: Float[Array, "hand_len"], γ: float, λ: float
) -> Float[Array, "hand_len-1"]:
    # Δ = [δ1, δ2, ..., δL]
    Δ: Float[Array, "hand_len-1"]
    Δ = γ * values[1:] - values[:-1]

    # discounts = [1, γλ, (γλ)^2, ..., (γλ)^(L-1)]
    discounts: Float[Array, "hand_len-1"]
    discounts = jnp.power(γ * λ, jnp.arange(len(Δ)))
    # shift discounts to calc the proper advantage at each step
    indices = jnp.arange(len(discounts))
    shifted_inidces = shifted_inidces = indices - indices[:, jnp.newaxis]

    # D = |1, γλ, (γλ)^2, ..., (γλ)^(L-1)|
    #     |0, 1 ,     γλ, ..., (γλ)^(L-2)|
    #     |0, 0 ,      1, ..., (γλ)^(L-3)|
    #     |..............................|
    #     |0, 0 ,     0,  ...,          1|
    D: Float[Array, "hand_len-1 hand_len-1"]
    D = discounts[shifted_inidces].at[shifted_inidces < 0].set(0)

    return jnp.sum(D * Δ, axis=1)


# Source: https://github.com/ChintanTrivedi/rl-bot-football/blob/master/train.py#L63
def actor_loss(
    advantages: Float[Array, "hand_len"],
    π_0: Float[Array, "hand_len n_actions"],
    π: Float[Array, "hand_len n_actions"],
    ε: float = 0.2,
) -> tuple[Float[Scalar, ""], Float[Scalar, ""]]:
    logratio = jnp.log(π + 1e-12) - jnp.log(π_0 + 1e-12)
    ratio = jnp.exp(logratio)

    norm_advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-12)
    # norm_advantages = advantages
    p1 = -norm_advantages * ratio
    p2 = -norm_advantages * jnp.clip(ratio, 1 - ε, 1 + ε)
    loss_value = jnp.maximum(p1, p2).mean()

    # KL divergence for metric purposes
    approx_kl = ((ratio - 1) - logratio).mean()

    return loss_value, approx_kl


# TODO: Comprobar si hay que cambiarla (e.g returs = values + advantages)
def critic_loss(rewards: jax.Array, values: jax.Array) -> jax.Array:
    return jnp.mean(jnp.abs(rewards - values) / jnp.abs(rewards))
    # return jnp.mean((rewards - values) ** 2)


def entropy(π: jax.Array) -> jax.Array:
    return -jnp.mean(π * jnp.log(π + 1e-12))


if __name__ == "__main__":
    train()
