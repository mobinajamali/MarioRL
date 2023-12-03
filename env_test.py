import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from nes_py.wrappers import JoypadSpace

ENV_NAME = 'SuperMarioBros-v0'

env = gym_super_mario_bros.make(ENV_NAME, render_mode='human', apply_api_compatibility=True)
env = JoypadSpace(env, SIMPLE_MOVEMENT)

done = False
env.reset()
while not done:
    action = env.action_space.sample()
    _, _, done, _, _ = env.step(action)
    env.render












