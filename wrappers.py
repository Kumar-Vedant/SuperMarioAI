# import numpy as np
from gym import Wrapper
from gym.wrappers import GrayScaleObservation, ResizeObservation, FrameStack

class SkipFrame(Wrapper):
    def __init__(self, env, skip):
        super().__init__(env)
        self.skip = skip

    def step(self, action):
        total_reward = 0.0
        done = False

        # take the same action for the next n(skip) steps
        for _ in range(self.skip):
            # take the step
            next_state, reward, done, trunc, info = self.env.step(action)
            total_reward += reward
            if done:
                break;
        return next_state, reward, done, trunc, info

# apply 4 wrappers - SkipFrame, ResizeObservation, GrayScaleObservation and FrameStack
def apply_wrappers(env):
    # skip 4 frames (apply the same action)
    env = SkipFrame(env, skip=4)
    # resize frame from 240x256 to 84x84
    env = ResizeObservation(env, shape=84)
    # convert to grayscale
    env = GrayScaleObservation(env)
    # stack 4 frames to give a sense of motion
    env = FrameStack(env, num_stack=4, lz4_compress=True)

    return env