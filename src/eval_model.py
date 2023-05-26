import rlcard
from rlcard.agents import RandomAgent

from models import FuzzyModel


def eval_model(env, n_games: int = 1000) -> int:
    profit = 0
    for _ in range(n_games):
        _, (*_, agent) = env.run()
        profit += agent

    return profit


if __name__ == "__main__":
    env = rlcard.make("no-limit-holdem", config={"seed": 0, "game_num_players": 6})
    agents = [RandomAgent(env.num_actions) for _ in range(env.num_players - 1)]
    agents.append(FuzzyModel())
    env.set_agents(agents)

    print(f"Profit in 1000 hands: {eval_model(env)}")