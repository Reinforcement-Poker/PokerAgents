import jax
import jax.numpy as jnp
from jaxtyping import Key
import pokers as pkrs
from src.agents.alphaholdem.model import AlphaHoldem, CardsObservation, ActionsObservation, Policy


class AlphaHoldemAgent(object):
    """An agent to wrap the AlphaHoldem model"""

    def __init__(self, player_id: int, model: AlphaHoldem):
        self.player_id = player_id
        self.model = model

    def step(self, trace: list[pkrs.State]) -> pkrs.Action:
        pass

    def step_batch(self, traces: list[list[pkrs.State]]) -> list[pkrs.Action]:
        pass

    @staticmethod
    def index_to_action(action_index: int, state: pkrs.State) -> pkrs.Action:
        r = 0
        match action_index:
            case 0:
                action_enum = pkrs.ActionEnum.Call
            case 1:
                action_enum = pkrs.ActionEnum.Check
            case 2:
                action_enum = pkrs.ActionEnum.Fold
            case 3:
                action_enum = pkrs.ActionEnum.Raise
                r = 1 / 2
            case 4:
                action_enum = pkrs.ActionEnum.Raise
                r = 2 / 3
            case 5:
                action_enum = pkrs.ActionEnum.Raise
                r = 1.0
            case 6:
                action_enum = pkrs.ActionEnum.Raise
                r = 3 / 2
            case _:
                raise Exception("Incorrect action")

        if action_enum == pkrs.ActionEnum.Raise:
            return pkrs.Action(
                action_enum,
                ((state.players_state[state.current_player].bet_chips - state.min_bet) + state.pot)
                * r,
            )
        else:
            return pkrs.Action(action_enum)

    @staticmethod
    def encode_actions(trace: list[pkrs.State]) -> ActionsObservation:
        assert len(trace) > 0
        num_players = len(trace[0].players_state)
        num_actions = 4  # len(pkrs.ActionEnum)
        current_player = trace[-1].current_player

        actions_encoding = jnp.zeros((24, num_actions, num_players + 2), dtype=jnp.float32)

        state_by_stage: dict[int, list[tuple[pkrs.State, pkrs.ActionRecord]]] = {
            stage: []
            for stage in (
                int(pkrs.Stage.Preflop),
                int(pkrs.Stage.Flop),
                int(pkrs.Stage.Turn),
                int(pkrs.Stage.River),
            )
        }
        for i in range(0, len(trace) - 1):
            s = trace[i]
            a = trace[i + 1].from_action
            assert a is not None
            state_by_stage[int(s.stage)].append((s, a))  # type: ignore

        for stage, stage_states in state_by_stage.items():
            cycle = 0
            for state, record in stage_states:
                if record.player == current_player:
                    # Legal actions
                    legal_actions_indices = [int(a) for a in record.legal_actions]  # type: ignore
                    actions_encoding = actions_encoding.at[
                        cycle + 6 * stage, legal_actions_indices, -1
                    ].set(1.0)
                    cycle += 1

                # Players actions
                player_position = (record.player - current_player) % num_players
                action_index = int(record.action.action)  # type: ignore
                if record.action.action == pkrs.ActionEnum.Raise:
                    actions_encoding = actions_encoding.at[
                        cycle + 6 * stage, action_index, player_position
                    ].set(record.action.amount / (state.pot + state.min_bet))
                else:
                    actions_encoding = actions_encoding.at[
                        cycle + 6 * stage, action_index, player_position
                    ].set(1.0)

        # All players actions
        actions_encoding = actions_encoding.at[..., -2].set(
            jnp.sum(actions_encoding[..., :-2], axis=-1)
        )

        return actions_encoding

    @staticmethod
    def encode_cards(state: pkrs.State) -> CardsObservation:
        cards = jnp.zeros((4, 13, 6), dtype=jnp.float32)

        player_state = state.players_state[state.current_player]
        # hole cards
        hand_index_1 = AlphaHoldemAgent.card_to_index(player_state.hand[0])
        hand_index_2 = AlphaHoldemAgent.card_to_index(player_state.hand[1])
        cards = cards.at[hand_index_1[0], hand_index_1[1], 0].set(1.0)
        cards = cards.at[hand_index_2[0], hand_index_2[1], 0].set(1.0)

        # flop cards
        if len(state.public_cards) >= 3:
            flop_indices = jnp.array(
                [AlphaHoldemAgent.card_to_index(card) for card in state.public_cards[:3]]
            )
            cards = cards.at[flop_indices[:, 0], flop_indices[:, 1], 1].set(1.0)

        # turn card
        if len(state.public_cards) >= 4:
            turn_index = AlphaHoldemAgent.card_to_index(state.public_cards[3])
            cards = cards.at[turn_index[0], turn_index[1], 2].set(1.0)

        # river card
        if len(state.public_cards) >= 5:
            river_index = AlphaHoldemAgent.card_to_index(state.public_cards[4])
            cards = cards.at[river_index[0], river_index[1], 3].set(1.0)

        # all public cards
        cards = cards.at[..., 4].set(jnp.sum(cards[..., 1:4], axis=-1))

        # all hole and public cards
        cards = cards.at[..., 5].set(cards[..., 0] + cards[..., 4])

        return cards

    @staticmethod
    def card_to_index(card: pkrs.Card) -> tuple[int, int]:
        return int(card.suit), int(card.rank)  # type: ignore

    @staticmethod
    def choose_action(policy: Policy, state: pkrs.State, key: Key) -> pkrs.Action:
        action_index = jax.random.choice(key, jnp.arange(len(policy)), p=policy)
        return AlphaHoldemAgent.index_to_action(int(action_index), state)
