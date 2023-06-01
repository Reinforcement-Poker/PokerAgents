from pettingzoo import AECEnv
from tqdm import trange
import time
from typing import Any
from utils import environment
from rlcard.agents import RandomAgent
from agents import FuzzyModel
from agents.alphaholdem.agent import AlphaHoldemAgent
from agents.alphaholdem.model import build_alphaholdem


def eval_agents(
    env: AECEnv, agents: dict[environment.AgentID, Any], n: int = 1000, sleep: int = 0
) -> dict[environment.AgentID, int]:
    rewards = dict()

    for i in trange(n):
        env.reset(seed=i)

        for agent_id in env.agent_iter():
            agent = agents[agent_id]
            observation, reward, termination, truncation, info = env.last()

            if observation is None or termination or truncation:
                action = None
                rewards[agent_id] = rewards.get(agent_id, 0) + reward
            else:
                if hasattr(agent, "step_batch") and callable(agent.step_batch):
                    action = agent.step_batch([observation["state"]])[0]
                else:
                    action = agent.step(observation["state"].rlcard_state)

            env.step(action)
            time.sleep(sleep)

    return rewards


if __name__ == "__main__":
    num_players = 2
    env = environment.env(num_players=num_players)
    env.reset(seed=1234)
    agents = dict()
    agents[env.agents[0]] = FuzzyModel()
    model = build_alphaholdem(len(list(environment.Action)))
    agents[env.agents[1]] = AlphaHoldemAgent(player_id=1, model=model)

    rewards = eval_agents(env, agents, n=1000)
    env.close()
    print(rewards)

    # With visualizaton
    # env = environment.env(num_players=num_players, render_mode="human", seed=1234)
    # eval_agents(env, agents, n=1, sleep=2)
    # time.sleep(10)
