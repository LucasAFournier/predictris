import argparse
from predictris.tetris import TetrisEnvironment
from predictris.agent import StandardAgent


def main():
    parser = argparse.ArgumentParser(description='Evaluate Tetris agent')
    parser.add_argument('--model-path', required=True)
    parser.add_argument('--episodes', type=int, default=100)
    args = parser.parse_args()
    
    # Load agent and run evaluation
    env = TetrisEnvironment()
    agent = StandardAgent.load(args.model_path)
    
    metrics = []
    for episode in range(args.episodes):
        episode_metrics = evaluate_episode(agent, env)
        metrics.append(episode_metrics)
        
    print_evaluation_results(metrics)


if __name__ == '__main__':
    main()