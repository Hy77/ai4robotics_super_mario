import gym
from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT

class MarioEnv:
    def __init__(self, gym_env):
        self.env = gym_super_mario_bros.make(gym_env, apply_api_compatibility=True, render_mode="human")
        self.env = JoypadSpace(self.env, SIMPLE_MOVEMENT)

    @property
    def action_space(self):
        # return action space
        return self.env.action_space

    def reset(self):
        # reset game env
        state = self.env.reset()
        return state

    def step(self, action):
        # execute next step then return the feedback
        next_state, reward, terminated, truncated, info = self.env.step(action)
        return next_state, reward, terminated, truncated, info

    def close(self):
        # turn off game env
        self.env.close()
