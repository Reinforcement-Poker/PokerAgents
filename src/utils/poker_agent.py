import pokers as pkrs
import jax


class PokerAgent:
    def step(self, trace: list[pkrs.State]) -> pkrs.Action:
        ...

    def batch_step(self, traces: list[list[pkrs.State]]) -> list[pkrs.Action]:
        ...


class RandomAgent(PokerAgent):
    def __init__(self, key: jax.random.KeyArray):
        self.key = key

    action_enums = [
        pkrs.ActionEnum.Call,
        pkrs.ActionEnum.Raise,
        pkrs.ActionEnum.Fold,
        pkrs.ActionEnum.Check,
    ]

    def step(self, trace: list[pkrs.State]) -> pkrs.Action:
        _, self.key = jax.random.split(self.key)
        action = jax.random.randint(
            self.key,
            (1,),
            minval=0,
            maxval=4,
        )

        amount = float(jax.random.uniform(self.key, minval=1, maxval=100))

        return pkrs.Action(self.action_enums[int(action)], amount)

    def batch_step(self, traces: list[list[pkrs.State]]) -> list[pkrs.Action]:
        _, self.key = jax.random.split(self.key)
        actions = []
        for trace in traces:
            if trace[-1].final_state:
                actions.append(pkrs.ActionEnum.Fold)
            else:
                a_i = int(
                    jax.random.randint(
                        self.key,
                        (1,),
                        minval=0,
                        maxval=len(trace[-1].legal_actions),
                    )
                )
                actions.append(trace[-1].legal_actions[a_i])
                _, self.key = jax.random.split(self.key)

        amounts = jax.random.uniform(self.key, (len(traces),), minval=1, maxval=100)

        return [pkrs.Action(action, amount) for action, amount in zip(actions, amounts)]


class OnlyCallsAgent(PokerAgent):
    def __init__(self, key: jax.random.KeyArray):
        self.key = key

    action_enums = [
        pkrs.ActionEnum.Call,
        pkrs.ActionEnum.Raise,
        pkrs.ActionEnum.Fold,
        pkrs.ActionEnum.Check,
    ]

    def step(self, trace: list[pkrs.State]) -> pkrs.Action:
        if pkrs.ActionEnum.Call in trace[-1].legal_actions:
            return pkrs.Action(pkrs.ActionEnum.Call)
        else:
            return pkrs.Action(pkrs.ActionEnum.Check)

    def batch_step(self, traces: list[list[pkrs.State]]) -> list[pkrs.Action]:
        return [self.step(t) for t in traces]
