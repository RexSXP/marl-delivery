import argparse
import json
import os
from main import run_experiment

def main():
    parser = argparse.ArgumentParser(description='Run MARL delivery experiments')
    parser.add_argument('--config', type=str, default='configs/test_config.json',
                      help='Path to configuration file')
    parser.add_argument('--map', type=str, default=None,
                      help='Override map file from config')
    parser.add_argument('--agent', type=str, default=None,
                      help='Override agent type from config')
    args = parser.parse_args()

    # Load configuration
    with open(args.config, 'r') as f:
        config = json.load(f)

    # Override config with command line arguments
    if args.map:
        config['environment']['map_file'] = args.map
    if args.agent:
        config['agent']['type'] = args.agent

    # Run experiment
    run_experiment(config)

if __name__ == '__main__':
    main() 