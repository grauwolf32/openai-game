from gym.core import Env

class GameEnv(Env):
    def _step(self, action): raise NotImplementedError
    def _reset(self): raise NotImplementedError
    def _render(self, mode='human', close=False): return
    def _seed(self, seed=None): return []