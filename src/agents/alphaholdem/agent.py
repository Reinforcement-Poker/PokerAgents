import tensorflow as tf
import numpy as np
from typing import Tuple
from utils.environment import Action, Stage, State, Card, ActionRecord


class AlphaHoldemAgent(object):
    """An agent to wrap the AlphaHoldem model"""

    def __init__(self, player_id: int, model: tf.keras.Model):
        self.player_id = player_id
        self.model = model
        self.use_raw = False
        self.num_actions = len(list(Action))
        self.num_players = 2

        assert (
            model.output_shape[0][1] == self.num_actions
        ), f"Model output ({model.output_shape[0][1]}) is not equal to the number of actions {self.num_actions}"

    def step_batch(self, states_batch: list[State]) -> list[int]:
        actions_encoding = np.array([self.encode_actions(state) for state in states_batch])
        cards_encoding = np.array([self.encode_cards(state) for state in states_batch])

        policy, pred_rewards = self.model({"actions": actions_encoding, "cards": cards_encoding})
        return list(np.argmax(policy.numpy(), axis=-1))

    def encode_actions(self, state: State) -> np.ndarray:
        assert (
            len(state.players_state) == self.num_players
        ), f"AlphaHoldemAgent only supports {self.num_players} players"

        actions_encoding = np.zeros((24, self.num_actions, self.num_players + 2), dtype=np.float32)

        action_record_by_stage: dict[Stage, list[ActionRecord]] = {
            stage: [] for stage in list(Stage)
        }
        for record in state.action_record:
            action_record_by_stage[record.stage].append(record)

        for stage, stage_records in action_record_by_stage.items():
            cycle = 0
            for record in stage_records:
                if record.player == self.player_id:
                    # Legal actions
                    legal_actions_indices = [a.value for a in record.legal_actions]
                    actions_encoding[cycle + 6 * stage.value, legal_actions_indices, -1] = 1.0
                    cycle += 1

                # Players actions
                player_position = (record.player - self.player_id) % self.num_players
                action_index = record.action.value
                actions_encoding[cycle + 6 * stage.value, action_index, player_position] = 1.0

        # All players actions
        actions_encoding[..., -2] = np.sum(actions_encoding[..., :-2], axis=-1)

        return actions_encoding

    def encode_cards(self, state: State) -> np.ndarray:
        cards = np.zeros((4, 13, 6), dtype=np.float32)

        player_state = state.players_state[self.player_id]
        # hole cards
        hand_index_1 = AlphaHoldemAgent.card_to_index(player_state.hand[0])
        hand_index_2 = AlphaHoldemAgent.card_to_index(player_state.hand[1])
        cards[hand_index_1[0], hand_index_1[1], 0] = 1.0
        cards[hand_index_2[0], hand_index_2[1], 0] = 1.0

        # flop cards
        if len(state.public_cards) >= 3:
            flop_indices = np.array(
                [AlphaHoldemAgent.card_to_index(card) for card in state.public_cards[:3]]
            )
            cards[flop_indices[:, 0], flop_indices[:, 1], 1] = 1.0

        # turn card
        if len(state.public_cards) >= 4:
            turn_index = AlphaHoldemAgent.card_to_index(state.public_cards[3])
            cards[turn_index[0], turn_index[1], 2] = 1.0

        # river card
        if len(state.public_cards) >= 5:
            river_index = AlphaHoldemAgent.card_to_index(state.public_cards[4])
            cards[river_index[0], river_index[1], 3] = 1.0

        # all public cards
        cards[..., 4] = np.sum(cards[..., 1:4], axis=-1)

        # all hole and public cards
        cards[..., 5] = cards[..., 0] + cards[..., 4]

        return cards

    @staticmethod
    def card_to_index(card: Card) -> Tuple[int, int]:
        suit = card[0]
        number = card[1]

        if suit == "D":  # Diamonds
            i = 0
        elif suit == "C":  # Clubs
            i = 1
        elif suit == "H":  # Hearts
            i = 2
        elif suit == "S":  # Spades
            i = 3
        else:
            raise Exception(f"Invalid card suit {suit}")

        if number.isnumeric():
            j = int(number) - 1
        elif number == "A":
            j = 0
        elif number == "T":
            j = 9
        elif number == "J":
            j = 10
        elif number == "Q":
            j = 11
        elif number == "K":
            j = 12
        else:
            raise Exception(f"Invalid card number {number}")

        return i, j

    def eval_step_batch(self, states_batch: list[State]) -> tuple[list[int], list[np.ndarray]]:
        """Predict the action given the current state for evaluation.
            Since the random agents are not trained. This function is equivalent to step function

        Args:
            state (dict): An dictionary that represents the current state

        Returns:
            action (int): The action predicted by the agent
            probs (list): The list of action probabilities
        """

        return self.step_batch(states_batch), [np.zeros(self.num_actions) for _ in states_batch]
