import pokers as pkrs
import jax.numpy as jnp
from src.agents.alphaholdem.agent import AlphaHoldemAgent


def test_encode_actions():
    state = pkrs.State.from_seed(
        n_players=2, button=0, sb=0.5, bb=1.0, stake=float("inf"), seed=1234
    )
    trace = [state]
    trace.append(trace[-1].apply_action(pkrs.Action(pkrs.ActionEnum.Raise, 1.25)))
    print(pkrs.visualize_trace(trace))
    actions_obs = AlphaHoldemAgent.encode_actions(trace)
    print("current player:", trace[-1].current_player)
    print(actions_obs)

    assert actions_obs[0].tolist() == [
        [0, 0, 0, 0],
        [0, 0, 0, 0.5],
        [0, 0, 0, 0.5],
        [1.0, 0, 1.0, 1.0],
    ]

    trace.append(trace[-1].apply_action(pkrs.Action(pkrs.ActionEnum.Fold)))
    pkrs.visualize_trace(trace)
    actions_obs = AlphaHoldemAgent.encode_actions(trace)
    print("current player:", trace[-1].current_player)
    print(actions_obs)

    assert actions_obs[0].tolist() == [[0, 0, 0, 0.5], [1, 0, 0, 0], [1, 0, 0, 0.5], [0, 0, 0, 0]]
    assert jnp.sum(actions_obs[1:]) == 0


# def test_encode_cards():
#     state = pkrs.State.from_seed(
#         n_players=2, button=0, sb=0.5, bb=1.0, stake=float("inf"), seed=1234
#     )
#     cards_obs = AlphaHoldemAgent.encode_cards(state)
#     for i in range(6):
#         print(cards_obs[..., i])

#     assert False
