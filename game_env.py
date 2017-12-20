from gym import error, spaces
from gym import utils
from gym.utils import seeding

class GameEnv(Env):
    def __init__(self):
        world_shape = (1152, 784)
        controller_1 = StrategyController()
        controller_2 = AIControlledStrategy()

        p1_pos = Vec(randint(0, world_shape[0]), randint(0, world_shape[1]))
        p2_pos = Vec(randint(0, world_shape[0]), randint(0, world_shape[1]))

        player_1 = ControlledUnit(id=1, position=p1_pos,direction=Vec(1.0,0.0), radious=20.0, max_ds=6.0, max_dw=2, friction_k=0.013, controller=controller_1)
        player_2 = ControlledUnit(id=2, position=p2_pos,direction=Vec(1.0,0.0), radious=20.0, max_ds=6.0, max_dw=2, friction_k=0.013, controller=controller_2)

        player_1.controller.bindStrategy(TargerFollowStrategy())
        player_2.controller.bindStrategy(TargerFollowStrategy())

        target_1 = Target(id=3, position=Vec(0.0,0.0), direction=Vec(1.0,0.0), radious=20)
        target_2 = Target(id=4, position=Vec(0.0,0.0), direction=Vec(1.0,0.0), radious=20)

        world = World(players=[player_1, player_2], targets=[target_1, target_2], shape=world_shape)
        self.world = world
        self.actor = player_2

    def _step(self, action): 
        alpha, beta = action
        self.actor.controller.strategy.setControl(alpha, beta)
        self.world.updateState(dt=1.0/10.0)
        
        state = self.world.getState()
        actor_score = state["player_{}".format(self.actor.id)]
        
        opponent_score = 0.0
        for player in self.world.players:
            if player.id == self.actor.id:
                continue
            opponent_score += state["player_{}".format(player.id)]

        n_opponents = len(self.world.players) - 1.0
        if n_opponents > 0.0:
            opponent_score /= n_opponents

        reward = actor_score - 0.5*opponent_score

        
    def _reset(self): 
        self.world.resetState()

    def _render(self, mode='human', close=False): return
    def _seed(self, seed=None): return []