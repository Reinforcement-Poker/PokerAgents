from typing import OrderedDict, TypedDict

from rlcard.games.nolimitholdem.game import Action, Stage


class RawObservations(TypedDict):
    hand: list[str]
    public_cards: list[str]
    all_chips: list[int]
    legal_actions: list[Action]
    stakes: list[int]
    current_player: int
    pot: int
    stage: Stage


class RlcardState(TypedDict):
    legal_actions: OrderedDict
    raw_obs: RawObservations
    raw_legal_actions: list[Action]
    action_record: list[int, Action]
