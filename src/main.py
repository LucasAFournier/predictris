from .environment import TetrisEnv
from .interactions import TetrisActions, TetrisVision
from .agent import Agent
from .rendering import TetrisRenderer
from .graph import ExplorationTree

import numpy as np
from typing import List, Callable, Optional
import time
import random


AUTHORIZED_ACTIONS = [
    # "move_tetromino_down",
    "move_tetromino_left",
    "move_tetromino_right",
    "rotate_tetromino_cw",
    # "rotate_tetromino_ccw",
    "move_view_down",
    "move_view_left",
    "move_view_right",
    "move_view_up",
]


def perform_action_sequence(
    actions: List[Callable[[], None]],
    vision: TetrisVision,
    show: Optional[bool] = False,
) -> List[np.ndarray]:
    """
    Perform a sequence of actions.
    """
    observations = []

    for action in actions:
        action()
        observation = np.reshape(vision.observe(), vision.view_shape)
        observations.append(observation)

        if show: print(observation)
    
    return observations


def render_action_sequence(
    env: TetrisEnv,
    vision: TetrisVision,
    action_sequence: List[Callable],
    interval: Optional[float] = 1.0
):
    """
    Render a sequence of actions in the Tetris environment with a live view and a fixed time interval between actions.
    """
    renderer = TetrisRenderer(env, vision) # Initialize the renderer
    renderer.render() # Render the initial state
    time.sleep(interval) # Wait for the specified interval

    try:
        for action in action_sequence:
            action() # Perform the action
            renderer.render() # Render the new state
            time.sleep(interval) # Wait for the specified interval

    except KeyboardInterrupt:
        print("Action sequence rendering interrupted by user.") # Handle user interruption (e.g., pressing Ctrl+C)

    finally:
        renderer.close() # Ensure the renderer is closed


def create_exploration_tree(n_steps, grid_shape, view_radius, batch_size):
    # Initialize ExplorationTree
    tree = ExplorationTree(n_steps=n_steps, 
                           grid_shape=grid_shape, 
                           view_radius=view_radius,
                           batch_size=batch_size)

    # Visualize the tree
    tree.visualize()


# Example usage:
if __name__ == "__main__":
    grid_shape = (10, 20)  # Example grid shape
    # env = TetrisEnv(grid_shape=grid_shape)
    
    view_radius = 1
    # vision = TetrisVision(env, view_radius=view_radius)
    
    # # Create action handlers
    # actions = TetrisActions(env, vision)
    
    # Define a sequence of actions
    # action_sequence = [
    #     actions.move_tetromino_left,
    #     actions.move_tetromino_right,
    #     actions.move_tetromino_down,
    #     actions.rotate_tetromino_cw,
    #     actions.rotate_tetromino_ccw,
    #     actions.move_view_right,
    #     actions.move_view_left,
    #     actions.move_view_up,
    #     actions.move_view_down,
    # ]

    # n_actions = 20
    # action_sequence = [actions.__getattribute__(random.choice(AUTHORIZED_ACTIONS)) for i in range(n_actions)]


    # # Execute the sequence
    # observations = perform_action_sequence(action_sequence, vision, show=True)
        
    # # Execute and render the sequence
    # render_action_sequence(env, vision, action_sequence, interval=0.5)

    # Define parameters for the exploration
    n_steps = 2  # Depth of exploration tree
    batch_size = 10  # Batch size for exploration

    create_exploration_tree(n_steps, grid_shape, view_radius, batch_size)