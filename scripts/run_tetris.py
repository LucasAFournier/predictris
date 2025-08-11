import argparse

import pygame
from predictris.tetris import TetrisEnvironment
from predictris.vis import TetrisRenderer


def parse_args():
    parser = argparse.ArgumentParser(description='Run Tetris simulation with manual control')
    parser.add_argument('--tetromino', type=str,
                       help='Tetromino string identifier')
    return parser.parse_args()


def main():
    args = parse_args()
    
    env = TetrisEnvironment(state=(args.tetromino, 0, 0, 0))    
    agent = env.build_agent(depth=1)
    
    renderer = TetrisRenderer(env, agent)
    
    # Main game loop
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key in renderer.key_action_map:
                    action_func = renderer.key_action_map[event.key]
                    action_func()
        
        renderer.render()
    
    renderer.close()


if __name__ == '__main__':
    main()