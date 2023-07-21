import jax.numpy as jnp
from jaxtyping import Float, Array, Int
import pokers as pkrs
from dataclasses import dataclass


@dataclass
class SelfPlayStatistics:
    rewards: list[float]
    avg_reward: float
    milli_big_blinds_per_hand: float
    elo_ratings: list[float]
    avg_hand_length: float
    illegal_actions_proportion: float
    actions_distribution: list[float]
    win_rate: float
    no_played_rate: float


def calc_self_play_stats(
    traces: list[list[pkrs.State]],
    matches_indices: Int[Array, "n_players n_hands"],
    models_scores: list[float],
) -> SelfPlayStatistics:
    assert len(traces) == len(matches_indices)
    return SelfPlayStatistics(
        rewards=[0.0],
        avg_reward=0.0,
        milli_big_blinds_per_hand=0.0,
        elo_ratings=[0.0],
        avg_hand_length=0.0,
        illegal_actions_proportion=0.0,
        actions_distribution=[0.0],
        win_rate=0.0,
        no_played_rate=0.0,
    )

    pool_size = len(models_scores)
    # The ids of the main player (last model of the pool) in each game
    main_player_ids = np.argmax(matches_indices == pool_size - 1, axis=1)
    main_player_ids[~np.any(matches_indices == pool_size - 1, axis=1)] = -1

    new_ratings = elo_rating(places, np.array(models_scores))

    rewards = np.zeros((len(traces), pool_size))
    for i, (match, t) in enumerate(zip(matches_indices, traces)):
        for p_id, ps in enumerate(t[-1].players_state):
            rewards[i, match[p_id]] = ps.reward
    avg_rewards = list(np.nanmean(rewards, axis=0))

    actions_distribution = np.full(5, 0.0)
    n_illegal_actions = 0
    n_no_played = 0
    n_wins = 0
    for i, (p_id, t) in enumerate(zip(main_player_ids, traces)):
        first_player_action = True
        for a in t[-1].action_record:
            if a.player == p_id:
                actions_distribution[a.action.value] += 1
                if first_player_action and a.action == Action.FOLD:
                    n_no_played += 1
                first_player_action = False

        if np.all(rewards[i, :-1] == 0) and rewards[i, -1] < 0:
            n_illegal_actions += 1

        if np.all(rewards[i] <= rewards[i][-1]):
            n_wins += 1

    n_actions = np.sum(actions_distribution)
    actions_distribution /= n_actions
    illegal_actions_proportion = n_illegal_actions / n_actions
    no_played_rate = n_no_played / len(main_player_ids[main_player_ids != -1])
    win_rate = n_wins / len(main_player_ids[main_player_ids != -1])

    return SelfPlayStatistics(
        rewards=avg_rewards,
        avg_reward=avg_rewards[-1],
        milli_big_blinds_per_hand=float(1000 * np.nanmean(rewards[..., -1]) / 2),
        elo_ratings=list(new_ratings),
        avg_hand_length=float(np.mean([len(t) for t in traces])),
        illegal_actions_proportion=illegal_actions_proportion,
        actions_distribution=list(actions_distribution),
        win_rate=win_rate,
        no_played_rate=no_played_rate,
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
        ratings[m] += K * (n_players - 1) * (s[m] - expectations)

    return ratings
