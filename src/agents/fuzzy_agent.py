from typing import Callable

import numpy as np
from skfuzzy.control import (
    Antecedent,
    Consequent,
    ControlSystem,
    ControlSystemSimulation,
    Rule,
)

from utils.hand_ranker import get_hand_score
from utils.environment import RlcardState


class FuzzyModel:
    def __init__(
        self,
        hand_ranker: Callable[[list[str], list[str]], float] = get_hand_score,
    ) -> None:
        self.use_raw = False
        self.hand_ranker = hand_ranker

        # Fuzzy variables
        hand_rank = Antecedent(np.arange(0, 1, 0.001), "hand_rank")
        pot = Antecedent(np.arange(0, 25, 1), "pot")
        cost = Antecedent(np.arange(0, 1, 0.001), "cost")
        action = Consequent(np.arange(0, 101, 1), "action")

        hand_rank.automf(5)
        pot.automf(3, names=["low", "medium", "high"])
        cost.automf(3, names=["low", "medium", "high"])
        action.automf(5, names=["fold", "call", "raise_low", "raise_high", "all_in"])

        all_pot = pot["low"] | pot["medium"] | pot["high"]
        all_cost = cost["low"] | cost["medium"] | cost["high"]
        no_high_pot = pot["medium"] | pot["low"]

        # Rules
        r1 = Rule(hand_rank["poor"] & all_pot & all_cost, action["fold"])
        r2 = Rule(hand_rank["mediocre"] & all_pot & all_cost, action["fold"])
        r3 = Rule(hand_rank["average"] & pot["high"] & cost["high"], action["fold"])
        r4 = Rule(hand_rank["average"] & pot["high"] & cost["medium"], action["fold"])
        r5 = Rule(hand_rank["average"] & pot["high"] & cost["low"], action["call"])
        r6 = Rule(hand_rank["average"] & no_high_pot & all_cost, action["call"])
        r7 = Rule(hand_rank["decent"] & pot["high"] & all_cost, action["fold"])
        r8 = Rule(hand_rank["decent"] & no_high_pot & all_cost, action["raise_high"])
        r9 = Rule(hand_rank["good"] & pot["high"] & all_cost, action["all_in"])
        rA = Rule(hand_rank["good"] & no_high_pot & all_cost, action["raise_low"])

        rule_list = [r1, r2, r3, r4, r5, r6, r7, r8, r9, rA]
        action_ctrl = ControlSystem(rule_list)
        self.action_sim = ControlSystemSimulation(action_ctrl)

    def step(self, state: RlcardState) -> int:
        obs = state["raw_obs"]
        board = obs["public_cards"]
        hand = obs["hand"]
        pot = obs["pot"]
        my_chips = obs["my_chips"]
        max_chips = max(obs["all_chips"])
        actions = obs["legal_actions"]

        n_actions = len(actions)
        cost = (max_chips - my_chips) / pot

        hand_rank = self.hand_ranker(hand, board)
        score = self.make_prediction(hand_rank, pot, cost)
        action_index = int(n_actions * score / 100)

        return actions[action_index].value

    def eval_step(self, state: RlcardState) -> tuple[int, float]:
        return self.step(state), 0

    def make_prediction(self, hand_rank: float, pot: float, cost: float) -> float:
        self.action_sim.input["hand_rank"] = hand_rank
        self.action_sim.input["pot"] = pot
        self.action_sim.input["cost"] = cost
        self.action_sim.compute()

        return self.action_sim.output["action"]
