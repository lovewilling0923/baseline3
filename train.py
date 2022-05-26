import argparse
import re
import sys

parser = argparse.ArgumentParser()
parser.add_argument("-T", "--timeout", type=int, default=60 * 5)  # The time length of one game (sec)
parser.add_argument("-R", "--time-scale", type=int, default=10)
parser.add_argument("-M", "--map-id", type=int, default=1)
parser.add_argument("-S", "--random-seed", type=int, default=0)
parser.add_argument("--start-location", type=float, nargs=3, default=[0, 0, 0])
parser.add_argument("--target-location", type=float, nargs=3, default=[0, 0, 0])
parser.add_argument("--base-worker-port", type=int, default=50000)
parser.add_argument("--engine-dir", type=str, default="../wildscav-linux-backend-v1.0-benchmark")
parser.add_argument("--map-dir", type=str, default="../101_104")
parser.add_argument("--num-workers", type=int, default=10)
parser.add_argument("--eval-interval", type=int, default=None)
parser.add_argument("--record", action="store_true")
parser.add_argument("--replay-suffix", type=str, default="")
parser.add_argument("--checkpoint-dir", type=str, default="checkpoints-")
parser.add_argument("--detailed-log", action="store_true", help="whether to print detailed logs")
parser.add_argument("--run", type=str, default="ppo", help="The RLlib-registered algorithm to use.")
parser.add_argument("--stop-iters", type=int, default=300)
parser.add_argument("--stop-timesteps", type=int, default=1e8)
parser.add_argument("--stop-reward", type=float, default=98)
parser.add_argument("--use-depth", action="store_true")
parser.add_argument("--stop-episodes", type=float, default=200000)
parser.add_argument("--dmp-width", type=int, default=42)
parser.add_argument("--dmp-height", type=int, default=42)
parser.add_argument("--dmp-far", type=int, default=200)
parser.add_argument("--train-batch-size", type=int, default=1000)
parser.add_argument("--reload", type=bool, default=False)
parser.add_argument("--reload-dir", type=str, default="")
if __name__ == "__main__":
    import os
    import ray
    from ray.tune.logger import pretty_print
    from ray.rllib.agents.ppo import PPOTrainer
    from ray.rllib.agents.a3c import A3CTrainer
    # from ray.rllib.agents.sac import SACTrainer
    from ray.rllib.agents.impala import ImpalaTrainer
    # from ray.rllib.agents.ddpg import DDPGTrainer
    # from ray.rllib.agents.dqn.apex import ApexTrainer
    from ray.rllib.agents.ppo.appo import APPOTrainer
    from envs.env_train import NavigationEnv

    args = parser.parse_args()
    eval_cfg = vars(args).copy()
    eval_cfg["in_evaluation"] = True
    ray.init()
    alg = args.run
    if alg == 'ppo':
        trainer = PPOTrainer(
            config={
                "env": NavigationEnv,
                "env_config": vars(args),
                "framework": "torch",
                "num_workers": args.num_workers,
                "evaluation_interval": args.eval_interval,
                "model": {
                    # Auto-wrap the custom(!) model with an LSTM.
                    "use_lstm": True,
                    # To further customize the LSTM auto-wrapper.
                    "lstm_cell_size": 64, },
                "train_batch_size": args.train_batch_size,  # default of ray is 4000
                "evaluation_config": {"env_config": eval_cfg},
                "evaluation_num_workers": 10,
            }
        )
    elif alg == 'appo':
        trainer = APPOTrainer(
            config={
                "env": NavigationEnv,
                "env_config": vars(args),
                "framework": "torch",
                "num_workers": args.num_workers,
                "evaluation_interval": args.eval_interval,
                "num_gpus": 0,
                "model": {
                    # Auto-wrap the custom(!) model with an LSTM.
                    "use_lstm": True,
                    # To further customize the LSTM auto-wrapper.
                    "lstm_cell_size": 64, },
                "evaluation_config": {"env_config": eval_cfg},
                "evaluation_num_workers": 10,
            }
        )
    elif alg == 'a3c':
        trainer = A3CTrainer(
            config={
                "env": NavigationEnv,
                "env_config": vars(args),
                "framework": "torch",
                "num_workers": args.num_workers,
                "evaluation_interval": args.eval_interval,
                "num_gpus": 0,
                "model": {
                    # Auto-wrap the custom(!) model with an LSTM.
                    "use_lstm": True,
                    # To further customize the LSTM auto-wrapper.
                    "lstm_cell_size": 64, },
                "evaluation_config": {"env_config": eval_cfg},
                "evaluation_num_workers": 10,
            }
        )
    elif alg == 'impala':
        trainer = ImpalaTrainer(
            config={
                "env": NavigationEnv,
                "env_config": vars(args),
                "framework": "torch",
                "num_workers": args.num_workers,
                "evaluation_interval": args.eval_interval,
                "num_gpus": 0,
                "model": {
                    # Auto-wrap the custom(!) model with an LSTM.
                    "use_lstm": True,
                    # To further customize the LSTM auto-wrapper.
                    "lstm_cell_size": 64, },
                "evaluation_config": {"env_config": eval_cfg},
                "evaluation_num_workers": 10,
            }
        )
    else:
        raise ValueError('No such algorithm')
    step = 0

    while True:
        step += 1
        result = trainer.train()
        reward = result["episode_reward_mean"]
        e = result["episodes_total"]
        print(f"current_training_steps:{step},episodes_total:{e},current_alg:{alg},current_reward:{reward}")

        if step != 0 and step % 200 == 0:
            os.makedirs(args.checkpoint_dir + f"{alg}" + str(args.map_id), exist_ok=True)
            trainer.save(args.checkpoint_dir + f"{alg}" + str(args.map_id))
            print("trainer save a checkpoint")
        if result["episodes_total"] >= args.stop_episodes:
            os.makedirs(args.checkpoint_dir + f"{alg}" + str(args.map_id), exist_ok=True)
            trainer.save(args.checkpoint_dir + f"{alg}" + str(args.map_id))
            trainer.stop()
            break

    print("the training has done!!")
    ray.shutdown()
    sys.exit()


