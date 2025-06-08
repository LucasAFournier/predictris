import argparse

import pygame
from predictris.tetris.env import TetrisEnvironment
from predictris.vis.game_renderer import TetrisRenderer


def parse_args():
    parser = argparse.ArgumentParser(description='Run Tetris simulation with manual control')
    parser.add_argument('--tetromino', type=str,
                       help='Tetromino string identifier')
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Initialize environment with tetromino
    env = TetrisEnvironment(tetromino_state={'name': 'J', 'position': (0, 0), 'orientation': 0})
    
    # Create agent agent
    agent = env.build_agent()
    
    # Initialize renderer
    renderer = TetrisRenderer(env, agent)
    
    # Main game loop
    running = True
    while running:
        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key in renderer.key_action_map:
                    action_func = renderer.key_action_map[event.key]
                    action_func(agent)
        
        # Render frame
        renderer.render()
    
    renderer.close()


if __name__ == '__main__':
    main()