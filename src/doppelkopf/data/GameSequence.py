import numpy as np
import copy
from doppelkopf.game.GameState import GameState
from doppelkopf.utils.Console import Console

class GameSequence():
    KEY_HISTORY = "history"
    KEY_ACTIONS = "actions"
    KEY_REWARDS = "rewards"
    KEY_STATEINDICES = "stateIndices"

    def __init__(self, history=None, stateIndices=None, actions=None, rewards=None, nextStateIndices=None):
        self.history = history if history is not None else np.array([])
        self.stateIndices = stateIndices if stateIndices is not None else []
        self.actions = actions if actions is not None else []
        self.rewards = rewards if rewards is not None else []
        self.nextStateIndices = nextStateIndices if nextStateIndices is not None else []
        self.currentIterIndex = 0

    def __iter__(self):
        self.currentIterIndex = 0
        return self

    def __next__(self):
        self.currentIterIndex += 1
        if self.currentIterIndex < self.Length():
            return self.TimestepAt(self.currentIterIndex)
        raise StopIteration

    def DeepDopy(self):
        return copy.deepcopy(self)
        
    def History(self) -> np.ndarray:
        return self.history

    def StateIndices(self):
        return self.stateIndices

    def Actions(self):
        return self.actions

    def Rewards(self):
        return self.rewards

    def NextStateIndices(self):
        return self.nextStateIndices

    def Zip(self) -> zip: # Get a zip of all four relevant lists
        return zip(self.stateIndices, self.actions, self.rewards, self.nextStateIndices)

    def TimestepAt(self, index) -> tuple:
        state = self.history[index]
        action = None
        reward = None
        nextStateIndex = None
        try:
            i = self.stateIndices.index(index)
            action = self.actions[i]
            reward = self.rewards[i]
            nextStateIndex = self.nextStateIndices[i]
        except ValueError:
            pass
        return state, action, reward, nextStateIndex

    def AddStateToHistory(self, state: np.ndarray):
        self.history = np.concatenate((self.history, state.reshape((1, -1)))) if len(self.history) > 0 else state.reshape((1, -1))

    def AddActionRewardForState(self, stateIndex: int, action: int, reward: float, nextStateIndex: int):
        self.stateIndices.append(stateIndex)
        self.actions.append(action)
        self.rewards.append(reward)
        self.nextStateIndices.append(nextStateIndex)

    def AddTimestep(self, state: np.ndarray, action: int, reward: float):
        self.AddStateToHistory(state)
        if action is not None and reward is not None:
            self.AddActionRewardForState(self.Length() - 1, action, reward, self.Length())

    def Length(self) -> int: # Should be equivalent to the number of timesteps within this sequence
        return self.history.shape[0] # First dimension of history

    def NextStateOf(self, index) -> GameState:
        return self.history[index + 1] if index < self.Length() else None

    def ForStorage(self):
        return {
            GameSequence.KEY_HISTORY: self.history,
            GameSequence.KEY_STATEINDICES: self.stateIndices,
            GameSequence.KEY_ACTIONS: self.actions,
            GameSequence.KEY_REWARDS: self.rewards
            }

    @staticmethod
    def FromStorage(values_dict: dict):
        return GameSequence(
            history=values_dict[GameSequence.KEY_HISTORY],
            stateIndices=values_dict[GameSequence.KEY_STATEINDICES],
            actions=values_dict[GameSequence.KEY_ACTIONS],
            rewards=values_dict[GameSequence.KEY_REWARDS]
        )

    def CheckIntegrity(self):
        isIntegrous = len(self.stateIndices) == len(self.actions) and len(self.stateIndices) == len(self.rewards) and len(self.stateIndices) == len(self.nextStateIndices)
        if isIntegrous:
            Console.WriteInfo(len(self.stateIndices))
        return isIntegrous

    def __str__(self):
        return "History: %s, stateIndices:%s, actions: %s, rewards: %s, nextStateIndices: %s" % (str(self.history), str(self.stateIndices), str(self.actions), str(self.rewards), str(self.nextStateIndices))