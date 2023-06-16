# %%
from src.utils.environment import Trace


def visualize_trace(trace: Trace) -> str:
    players = trace[0].players_state.keys()
    vis = ""
    vis += "            " + "        ".join(
        [f"{p}·" if trace[0].button == p else f"{p} " for p in players]
    )
    vis += "\n"
    hands = [f"|{s.hand[0]} {s.hand[1]}|" for s in trace[0].players_state.values()]
    vis += "         " + "   ".join(hands)
    vis += "\n"

    for state in trace:
        if len(state.action_record) > 0:
            last_action = state.action_record[-1]
            action_offset = 13 + 10 * last_action.player
            vis += "↓".rjust(action_offset, " ")
            vis += f" <{last_action.action.value}:{last_action.action.name}>"
            vis += "\n"

        vis += f"{state.stage._name_}:".ljust(9, " ")
        chips = [
            f"{str(ps.chips).rjust(3, ' ')}/{str(ps.stake).ljust(3, ' ')}"
            for ps in state.players_state.values()
        ]
        vis += "   ".join(chips)
        if len(state.public_cards) > 0:
            vis += f"   |{' '.join(state.public_cards)}|"
        vis += "\n"

    rewards = [str(ps.reward).rjust(3, " ") for ps in trace[-1].players_state.values()]
    vis += f"rewards:".ljust(10, " ")
    vis += "       ".join(rewards)

    return vis


# %%
