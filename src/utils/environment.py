from dataclasses import dataclass
from typing import Any, OrderedDict, TypedDict

import numpy as np
import pettingzoo.classic.rlcard_envs.texas_holdem_no_limit as thnl
from pettingzoo.utils import wrappers
from pettingzoo.utils.env import AgentID
from rlcard.games.nolimitholdem.game import Action, Stage

Card = str


@dataclass
class PlayerState:
    player: int
    hand: tuple[Card, Card]
    chips: int
    stake: int
    legal_actions: list[Action]
    reward: float


@dataclass
class ActionRecord:
    player: int
    stage: Stage
    action: Action
    legal_actions: list[Action]


class RawObservations(TypedDict):
    hand: list[Card]
    public_cards: list[Card]
    all_chips: list[int]
    my_chips: int
    legal_actions: list[Action]
    stakes: list[int]
    current_player: int
    pot: int
    stage: Stage


class RlcardState(TypedDict):
    legal_actions: OrderedDict
    raw_obs: RawObservations
    raw_legal_actions: list[Action]
    action_record: list[tuple[int, Action]]


@dataclass
class State:
    current_player: int
    players_state: dict[int, PlayerState]
    public_cards: list[Card]
    stage: Stage
    button: int
    action_record: list[ActionRecord]
    rlcard_state: RlcardState
    pettingzoo_observation: Any


Trace = list[State]


class Observation(TypedDict):
    state: State
    action_mask: np.ndarray


class raw_env(thnl.raw_env):
    def __init__(
        self, num_players: int = 2, render_mode: str | None = None, seed: int | None = None
    ):
        super().__init__(num_players, render_mode)
        self.reset(seed=seed)

    def reset(self, seed=None, options=None):
        super().reset(seed, options)
        self.action_record: list[ActionRecord] = []
        current_player = self._name_to_int(self.agent_selection)
        self.button: int = (current_player - 3) % self.num_players
        self.trace: Trace = [self.observe(self.agent_selection)["state"]]

    def observe(self, agent) -> Observation:
        # Returns the state for every agent, the parameter agent is only needed for super().observe() invocation
        pettingzoo_observation = super().observe(agent)

        players_state = dict()
        for a_ind in sorted([self._name_to_int(a) for a in self.agents]):
            a_obs = self.env.get_state(a_ind)
            players_state[a_ind] = PlayerState(
                player=a_ind,
                hand=a_obs["raw_obs"]["hand"],
                chips=a_obs["raw_obs"]["my_chips"],
                stake=a_obs["raw_obs"]["stakes"][a_ind],
                legal_actions=a_obs["raw_obs"]["legal_actions"],
                reward=self._cumulative_rewards[self._int_to_name(a_ind)],
            )

        current_player = self._name_to_int(self.agent_selection)
        obs = self.env.get_state(current_player)

        state = State(
            current_player=current_player,
            players_state=players_state,
            public_cards=obs["raw_obs"]["public_cards"],
            stage=obs["raw_obs"]["stage"],
            button=self.button,
            action_record=self.action_record.copy(),
            rlcard_state=obs,
            pettingzoo_observation=pettingzoo_observation["observation"],
        )

        return {"state": state, "action_mask": pettingzoo_observation["action_mask"]}

    def step(self, action: int | None):
        player = self._name_to_int(self.agent_selection)
        state = self.observe(self.agent_selection)["state"]
        stage = state.stage
        legal_actions = state.players_state[player].legal_actions
        super().step(action)
        if action is not None:
            self.action_record.append(
                ActionRecord(
                    player=player, stage=stage, action=Action(action), legal_actions=legal_actions
                )
            )
            observation = self.observe(self.agent_selection)
            self.trace.append(observation["state"])

    def last(
        self, _: bool = True
    ) -> tuple[Observation, float, bool, bool, dict[AgentID, dict[str, Any]]]:
        agent = self.agent_selection
        assert agent
        observation = self.observe(agent)
        return (
            observation,
            self._cumulative_rewards[agent],
            self.terminations[agent],
            self.truncations[agent],
            self.infos[agent],
        )


class TerminateIllegalWrapper(wrappers.TerminateIllegalWrapper):
    def __init__(self, env: raw_env, illegal_reward: float):
        super().__init__(env, illegal_reward)
        self.env: raw_env

    def __getattr__(self, item):
        return getattr(self.env, item)


class AssertOutOfBoundsWrapper(wrappers.AssertOutOfBoundsWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.env: raw_env

    def __getattr__(self, item):
        return getattr(self.env, item)


class OrderEnforcingWrapper(wrappers.OrderEnforcingWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.env: raw_env

    def __getattr__(self, item):
        return getattr(self.env, item)


def env(**kwargs):
    env = raw_env(**kwargs)
    env = TerminateIllegalWrapper(env, illegal_reward=-1)
    env = AssertOutOfBoundsWrapper(env)
    env = OrderEnforcingWrapper(env)
    return env
