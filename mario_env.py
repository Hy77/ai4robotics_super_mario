import gym
import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
import cv2
import numpy as np
from gym.wrappers import FrameStack, GrayScaleObservation, TransformObservation
from gym.spaces import Box


def make_env(skip_frames):
    env = gym_super_mario_bros.make('SuperMarioBros-1-1-v0')
    env = SkipFrame(env, skip=skip_frames)
    env = JoypadSpace(env, [['right'], ['right', 'A']])  # RIGHT AND JUMP ONLY
    env = GrayScaleObservation(env, keep_dim=False)
    env = ResizeObv(env, shape=84)
    env = TransformObservation(env, f=lambda x: x / 255.)
    env = FrameStack(env, num_stack=4)
    return env


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


class ResizeObv(gym.ObservationWrapper):
    def __init__(self, env, shape):
        super().__init__(env)
        self.shape = (shape, shape) if isinstance(shape, int) else tuple(shape)

        channels = self.observation_space.shape[2] if len(self.observation_space.shape) > 2 else 1
        obs_shape = self.shape + (channels,)
        self.observation_space = Box(low=0, high=255, shape=obs_shape, dtype=np.uint8)

    def observation(self, observation):
        resized_observation = cv2.resize(observation, self.shape, interpolation=cv2.INTER_AREA)
        return resized_observation


