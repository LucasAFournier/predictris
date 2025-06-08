import argparse
from pathlib import Path
import pygame
import time

from predictris.tetris.env import TetrisEnvironment
from predictris.agent import StandardAgent
from predictris.vis.game_renderer import TetrisRenderer
from predictris.tetris.encoders import perceptions_from_nameset
from predictris.tetris.constants import TETROMINO_NAMES
from predictris.utils import create_random_environment


def parse_args():
    parser = argparse.ArgumentParser(description='Run Tetris simulation with smart agent')
    parser.add_argument('--origin', type=str, required=True, help='Directory containing trained trees')
    parser.add_argument('--delay', type=float, default=0.5, help='Delay between actions (seconds)')
    parser.add_argument('--steps', type=int, default=100, help='Steps before environment reset')
    parser.add_argument('--context', type=int, default=1, help='Context size for smart learning')
    parser.add_argument('--verbose', action='store_true', help='Verbose mode')
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Load trained agent
    perception = next(iter(perceptions_from_nameset('vision_only')))
    input_dir = Path('results') / args.origin
    agent = StandardAgent.load(
        dir_path=input_dir,
        perception=perception,
        verbose=args.verbose
    )
    
    # Initialize environment and agent
    while True:
        env = create_random_environment()
        agent.body = env.get_body()
        if agent.init_smart_learn(args.context) != 'aborted':
            break

    # Initialize renderer
    renderer = TetrisRenderer(env=env, body=agent.body)
    
    # Main game loop
    running = True
    last_action = time.time()
    step_count = 0
    
    while running:
        # Handle quit events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        # Let agent choose action if enough time has passed
        now = time.time()
        if now - last_action >= args.delay:
            result = agent.smart_learn()
            renderer.render()

            if args.verbose:
                print(f"Step {step_count}/{args.steps} - Result: {result}")

            if result == 'aborted' or step_count >= args.steps:
                while True:
                    env = create_random_environment()
                    agent.body = env.get_body()
                    if agent.init_smart_learn(args.context) != 'aborted':
                        break
                renderer.env = env
                renderer.body = agent.body
                step_count = 0
                if args.verbose:
                    print("\nNew environment created")
            else:
                step_count += 1
                    
            last_action = now

    renderer.close()


if __name__ == '__main__':
    main()
