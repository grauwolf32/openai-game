from game_env import *
import pygame as pg

def main():
    env = GameEnv(visualization=True)
    while True:
        alpha = 0.2
        beta = 0.001
        observation, reward, done, info = env.step(action=(alpha, beta))
        print observation
        if done:
            env.close()
            pg.quit()
            sys.exit()

        env.render()


if __name__=="__main__":
    main()