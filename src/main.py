from environment.tetris_env import TetrisEnv
from interactions.tetris_actions import TetrisActions
from interactions.tetris_vision import Vision
from agent.agent import Agent

grid_shape = (10, 10)
env = TetrisEnv(grid_shape)

action_class = TetrisActions(env)
actions = [action_class.move_down, action_class.move_left, action_class.move_right, action_class.rotate_cw]

observe = Vision(TetrisEnv, view_radius=1)

tetris_agent = Agent(actions, observe)