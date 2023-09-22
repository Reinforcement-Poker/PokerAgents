import jax
import jax.numpy as jnp
from jaxtyping import Float, Array, Int
import flax
import pokers as pkrs
from tqdm import tqdm
from dataclasses import dataclass
from typing import cast
from src.utils.poker_agent import PokerAgent
from src.agents.alphaholdem.model import AlphaHoldem
from src.agents.alphaholdem.agent import AlphaHoldemAgent
from src.agents.alphaholdem.replay_buffer import ReplayBuffer, BufferRecord
from src.utils.hand_ranker import get_hand_score


@dataclass
class SelfPlayStatistics:
    rewards: tuple[list[float], list[str]]
    avg_reward: float
    milli_big_blinds_per_hand: float
    elo_ratings: list[float]
    avg_hand_length: float
    illegal_actions_proportion: float
    actions_distribution: tuple[list[int], list[str]]
    win_rate: float
    no_played_rate: float


def k_best_selfplay(
    model: AlphaHoldem,
    model_params: flax.core.FrozenDict,
    agents_pool: list[PokerAgent],
    agents_scores: list[float],
    replay_buffer: ReplayBuffer,
    n_hands: int,
    n_players: int,
    key: jax.random.KeyArray,
) -> tuple[ReplayBuffer, SelfPlayStatistics, jax.random.KeyArray]:
    assert n_players >= 2
    assert n_players <= len(agents_pool) + 1

    print(f"Starting self play: {n_hands} hands with {n_players} players")

    # Initialize n_hands random matchups and their associated initial states
    matchups: list[list[int]] = []
    traces: list[list[pkrs.State]] = []
    for _ in range(n_hands):
        matchup = jax.random.choice(
            key, jnp.arange(1, len(agents_pool) + 1), (n_players - 1,), replace=False
        )
        matchup = jax.random.permutation(key, jnp.concatenate([jnp.array([0]), matchup]))
        matchups.append(list(matchup))
        traces.append(
            [
                pkrs.State.from_seed(
                    n_players=n_players,
                    button=0,
                    sb=0.5,
                    bb=1.0,
                    stake=float("inf"),
                    seed=int(jax.random.randint(key, (1,), 0, 10000)),
                )
            ]
        )
        _, key = jax.random.split(key)

    # Structure to record play observations
    buffer_records: list[dict] = [
        dict(
            actions_observations=[],
            cards_observations=[],
            actions_taken=[],
            reward=0.0,
            hand_scores=[],
        )
        for _ in range(n_hands)
    ]

    bar = tqdm()
    while not all([trace[-1].final_state for trace in traces]):
        # Current players in matchup terms
        players = [matchup[trace[-1].current_player] for trace, matchup in zip(traces, matchups)]

        # Group same player turns to batch the inference
        unique_players = jnp.unique(jnp.array(players))

        # Action selected for each hand in this turn
        actions: list[pkrs.Action | None] = [None for _ in range(n_hands)]
        for p in unique_players:
            # Hand in which each player is playing
            p_indices = jnp.where(jnp.array(players) == p)[0]
            # Complete trace of each player hand
            player_traces = [traces[i] for i in p_indices]
            if p == 0:  # Player 0 is always the main model
                actions_observations = jnp.array(
                    [AlphaHoldemAgent.encode_actions(trace) for trace in player_traces]
                )
                cards_observations = jnp.array(
                    [AlphaHoldemAgent.encode_cards(trace[-1]) for trace in player_traces]
                )

                policies, _, _ = jax.vmap(  # type: ignore
                    lambda actions_obs, cards_obs: model.apply(
                        model_params,
                        actions_obs,
                        cards_obs,
                        train=False,
                    )
                )(actions_observations, cards_observations)

                action_indices = []
                for i, policy, trace in zip(
                    p_indices,
                    policies,
                    player_traces,
                ):
                    _, key = jax.random.split(key)
                    action_index = AlphaHoldemAgent.choose_action(
                        policy, key, trace[-1].legal_actions
                    )
                    action_indices.append(action_index)
                    action = AlphaHoldemAgent.index_to_action(action_index, trace[-1])
                    # Save action to apply it later to its corresponding state in a batch fashion
                    actions[i] = action

                # Save plays in buffer
                for i, trace, action_obs, card_obs, action_index in zip(
                    p_indices,
                    player_traces,
                    actions_observations,
                    cards_observations,
                    action_indices,
                ):
                    if trace[-1].final_state:
                        continue
                    buffer_records[i]["actions_observations"].append(action_obs)
                    buffer_records[i]["cards_observations"].append(card_obs)
                    buffer_records[i]["actions_taken"].append(action_index)
                    buffer_records[i]["hand_scores"].append(calc_hand_score(trace[-1]))

            else:  # Rival agents
                agent = agents_pool[p - 1]
                chosen_actions = agent.batch_step(traces=player_traces)
                assert len(chosen_actions) == len(p_indices)
                # Save actions to apply them later to their corresponding state in a batch fashion
                for i, action in zip(p_indices, chosen_actions):
                    actions[i] = action

        assert all(actions)
        new_states = pkrs.parallel_apply_action(
            [trace[-1] for trace in traces], actions  # type: ignore
        )
        for i, state in enumerate(new_states):
            traces[i].append(state)
            if state.final_state:
                main_player = jnp.where(jnp.array(matchups[i]) == 0)[0][0]
                buffer_records[i]["reward"] = state.players_state[main_player].reward

        bar.update(1)
    bar.close()

    for record in buffer_records:
        if record["actions_observations"]:
            replay_buffer.add_agent_play(
                BufferRecord(
                    actions_observations=jnp.array(record["actions_observations"]),
                    cards_observations=jnp.array(record["cards_observations"]),
                    actions_taken=jnp.array(record["actions_taken"]),
                    reward=jnp.array(record["reward"]),
                    hand_scores=jnp.array(record["hand_scores"]),
                )
            )

    stats = calc_self_play_stats(
        traces=traces, matches_indices=jnp.array(matchups), models_scores=agents_scores
    )

    return replay_buffer, stats, key


def calc_hand_score(state: pkrs.State) -> float:
    public_cards = state.public_cards
    player_cards = state.players_state[state.current_player].hand
    public_cards_str = [
        f"{str(c.suit).split('.')[-1][0]}{str(c.rank).split('.')[-1][-1]}" for c in public_cards
    ]
    player_cards_str = [
        f"{str(c.suit).split('.')[-1][0]}{str(c.rank).split('.')[-1][-1]}" for c in player_cards
    ]
    # print(player_cards_str)
    # exit()
    score = get_hand_score(player_cards_str, public_cards_str)
    return score


def calc_self_play_stats(
    traces: list[list[pkrs.State]],
    matches_indices: Int[Array, "n_players n_hands"],
    models_scores: list[float],
) -> SelfPlayStatistics:
    assert len(traces) == len(matches_indices)

    # TODO: Adicionalmente, solo los rewards del main player
    rewards = [ps.reward for t in traces for ps in t[-1].players_state]
    rewards_distribution, rewards_bins = jnp.histogram(jnp.array(rewards), bins=20)
    rewards_labels = [
        f"{rewards_bins[i]:.2f} - {rewards_bins[i+1]:.2f}" for i in range(len(rewards_bins) - 1)
    ]

    # TODO: La distribución de acciones solo debería ser del main player
    actions = [int(a.action.action) for t in traces for s in t if (a := s.from_action) is not None]
    actions_distribution, _ = jnp.histogram(jnp.array(actions), bins=jnp.arange(5) - 0.5)

    places = jnp.zeros((matches_indices.shape[1], len(models_scores)), dtype=jnp.int32)
    for i, (t, match_indices) in enumerate(zip(traces, matches_indices)):
        sorted_players = match_indices[
            jnp.argsort(jnp.array([-ps.reward for ps in t[-1].players_state]))
        ]
        places = places.at[i, sorted_players].set(jnp.arange(1, len(sorted_players) + 1))
    elo_ratings = elo_rating(places, jnp.array(models_scores))

    return SelfPlayStatistics(
        rewards=(rewards_distribution.tolist(), rewards_labels),
        avg_reward=sum(rewards) / len(rewards),
        milli_big_blinds_per_hand=0.0,
        elo_ratings=elo_ratings.tolist(),
        avg_hand_length=sum([len(t) for t in traces]) / len(traces),
        illegal_actions_proportion=0.0,
        actions_distribution=(actions_distribution.tolist(), ["Fold", "Check", "Call", "Raise"]),
        win_rate=0.0,
        no_played_rate=0.0,
    )


# Source: https://towardsdatascience.com/developing-a-generalized-elo-rating-system-for-multiplayer-games-b9b495e87802
def elo_rating(
    places: Int[Array, "n_hands pool_size"],
    past_ratings: Float[Array, "pool_size"],
    K: float = 40,
    D: float = 400,
    α: float = 1.2,
) -> Float[Array, "pool_size"]:
    """
    Args:
        places: n_games x pool_size
                a_ij = place of player j in game i, nan when player j is not in game i
        past_ratings: pool_size

    Returns:
        new_ratings: pool_size
    """

    assert places.shape[1] == past_ratings.size

    # players_mask: n_games x pool_size
    players_mask = ~jnp.isnan(places)
    n_players_all_games = jnp.sum(players_mask, axis=-1)
    n_players = n_players_all_games[0]
    assert jnp.all(
        n_players_all_games == n_players
    ), f"The number of players needs to be the same in all games {n_players_all_games}"

    # scores: n_games x pool_size
    scores = (α ** (n_players - places) - 1) / jnp.expand_dims(
        jnp.sum(α ** (n_players - jnp.arange(1, n_players + 1)) - 1), axis=0
    )

    ratings = jnp.copy(past_ratings)
    for s, m in zip(scores, players_mask):
        # ratings_diff: pool_size x pool_size
        ratings_diff = ratings[m][jnp.newaxis, :] - ratings[m][:, jnp.newaxis]
        # sigmoids: pool_size x pool_size
        sigmoids = 1 / (1 + 10 ** (ratings_diff / D))
        # expectations: pool_size
        expectations = (
            2 / (n_players * (n_players - 1)) * jnp.sum(sigmoids * (1 - jnp.eye(n_players)), axis=1)
        )
        ratings = ratings.at[m].set(K * (n_players - 1) * (s[m] - expectations))

    return ratings
