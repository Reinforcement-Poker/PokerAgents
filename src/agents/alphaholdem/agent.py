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

    @staticmethod
    def index_to_action(action_index: int, state: pkrs.State) -> pkrs.Action:
        r = 0
        match action_index:
            case 0:
                action_enum = pkrs.ActionEnum.Fold
            case 1:
                action_enum = pkrs.ActionEnum.Check
            case 2:
                action_enum = pkrs.ActionEnum.Call
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

        actions_encoding = jnp.zeros((24, num_players + 2, num_actions), dtype=jnp.float32)

        cycle = 0
        for i in range(1, len(trace)):
            if trace[i].current_player == current_player:
                # Legal actions
                legal_actions_indices = [int(a) for a in trace[i].legal_actions]  # type: ignore
                actions_encoding = actions_encoding.at[cycle, -2, legal_actions_indices].set(1.0)
                actions_encoding = actions_encoding.at[cycle, -1].set(int(trace[i].stage))
                cycle += 1

            action_record = trace[i].from_action
            assert action_record is not None
            player_position = (action_record.player - current_player) % num_players
            action_index = int(action_record.action.action)  # type: ignore
            if action_record.action.action == pkrs.ActionEnum.Raise:
                actions_encoding = actions_encoding.at[cycle, player_position, action_index].set(
                    action_record.action.amount  # / (trace[i - 1].pot + trace[i - 1].min_bet)
                )
            else:
                actions_encoding = actions_encoding.at[cycle, player_position, action_index].set(
                    1.0
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
    def choose_action(policy: Policy, key: Key, legal_actions: list[pkrs.ActionEnum]) -> int:
        # Elimino temporalmente la posibilidad de acciones ilegales y de fold para forzar resultados más fáciles de predecir
        legal_actions_mask = jnp.zeros(7, dtype=bool)
        for a in legal_actions:
            if a == pkrs.ActionEnum.Raise:
                legal_actions_mask = legal_actions_mask.at[3:7].set(True)
            else:
                legal_actions_mask = legal_actions_mask.at[int(a)].set(True)

        # policy = policy.at[0].set(0.05)  # Bloquear fold

        # Balance raise probability since 4 policy fields lead to make a raise
        policy = policy.at[3:7].divide(4)
        policy = policy / (policy.sum() + 1e-12)
        policy = policy.at[~legal_actions_mask].set(0.0)
        action_index = jax.random.choice(key, jnp.arange(len(policy)), p=policy)
        return int(action_index)
