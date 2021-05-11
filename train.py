# Usage: 
# python train.py -y LunarLander-v2 -m DQN -s True -t 50000 -w 4 -g 0.95 0.93 -l 1e-4 1e-6
# python train.py -y LunarLander-v2 -m DQN -s False -c configs/test.json

import argparse
import json
import matplotlib.pyplot as plt
import pandas as pd
import ray
import seaborn as sns

from ray import tune
from ray.rllib import agents


sns.set()
parser = argparse.ArgumentParser()

parser.add_argument('-y', '--gym-id', type=str, help="Id of the Gym environment")
parser.add_argument('-m', '--model-id', type=str, help="Id of the model (DQN, VPG, etc.)")
parser.add_argument('-s', '--grid-search', type=str, default="True", help="Whether to run in grid search \
                                                    mode (finding the best parameters) or test it normally")
parser.add_argument('-t', '--timesteps', type=int, default=50000, help="How many timesteps to run for")
parser.add_argument('-w', '--num-workers', type=int, default=3, help="Number of workers to spin up")
parser.add_argument('-g', '--gamma', type=float, nargs="+", default=[0.999, 0.8], help="Values of gamma to test")
parser.add_argument('-l', '--lr', type=float, nargs="+", default=[1e-2, 1e-3, 1e-4], help="Values of learning rates to test")
parser.add_argument('-c', '--config', type=str, default=None, help="Path to config file")

args = parser.parse_args()
if args.grid_search == "False" or args.grid_search == "false": args.grid_search = False
else: args.grid_search = True


# TODO: Generate training graphs / reward plots either in tune or run_model

if args.grid_search:
    results = tune.run(
        args.model_id, 
        stop={
            'timesteps_total': args.timesteps
        },
        config={
        "env": args.gym_id,
        "num_workers": args.num_workers,
        "gamma" : tune.grid_search(args.gamma),
        "lr": tune.grid_search(args.lr),
        }
    )

    # TODO: Parse results, generate json file containing best parameters, and store it in configs/

elif args.config is not None:

    ray.init()
    config_file = open(args.config)
    config = json.load(config_file)

    # TODO: Implement different agents / case on args.model_id
    trainer2 = agents.dqn.DQNTrainer(env=args.gym_id, config=config)
    results2 = trainer2.train()

