# mario_env.py
import gym
import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import RIGHT_ONLY
from gym.wrappers import FrameStack, GrayScaleObservation, TransformObservation


class SkipFrame(gym.Wrapper):
    def __init__(self, env, skip):
        super().__init__(env)
        self._skip = skip

    def step(self, action):
        total_reward = 0.0
        done = False
        for i in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            total_reward += reward
            if done:
                break
        return obs, total_reward, done, info


def make_env(skip_frames):
    env = gym_super_mario_bros.make('SuperMarioBros-v0')
    env = SkipFrame(env, skip=skip_frames)
    env = JoypadSpace(env, RIGHT_ONLY)
    # env = JoypadSpace(env, [["right"], ["right", "A", "B"]])
    env = GrayScaleObservation(env, keep_dim=True)
    env = TransformObservation(env, f=lambda x: x / 255.)
    env = FrameStack(env, num_stack=4)
    return env
