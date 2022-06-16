import random
from typing import Dict

import gym
import numpy as np
from gym import spaces
from ray.rllib.env import EnvContext
from inspirai_fps.utils import get_distance, get_position,get_picth_yaw
from inspirai_fps.gamecore import Game
from inspirai_fps.gamecore import ActionVariable as A

BASE_WORKER_PORT = 50000


class NavigationEnv(gym.Env):
    def __init__(self, config: EnvContext):
        self.config = config
        self.render_scale = config.get("render_scale", 1)

        env_seed = config.get("random_seed", 0) + config.worker_index

        # only 240 * 320 can be aotu transform into conv model
        dmp_width = config["dmp_width"]
        dmp_height = config["dmp_height"]
        dmp_far = config["dmp_far"]

        obs_space_1 = spaces.Box(low=-100, high=100, shape=(3,), dtype=np.float32)
        obs_space_3 = spaces.Box(low=0, high=360, shape=(2,), dtype=np.float32)
        obs_space_2 = spaces.Box(
            low=0, high=dmp_far, shape=(dmp_height, dmp_width), dtype=np.float32
        )
        self.observation_space = spaces.Tuple([obs_space_1, obs_space_2, obs_space_3])
        # self.observation_space = obs_space_1
        self.action_dict = {
            "move": [
                [(A.WALK_DIR, 0), (A.WALK_SPEED, 0)],
                [(A.WALK_DIR, 0), (A.WALK_SPEED, 8)],
                [(A.WALK_DIR, 90), (A.WALK_SPEED, 8)],
                [(A.WALK_DIR, 180), (A.WALK_SPEED, 8)],
                [(A.WALK_DIR, 270), (A.WALK_SPEED, 8)],
                # [(A.JUMP, True)],
            ],
            "turn_lr": [
                [(A.TURN_LR_DELTA, -1)],
                [(A.TURN_LR_DELTA, 0)],
                [(A.TURN_LR_DELTA, 1)],
                [(A.TURN_LR_DELTA, -0.4)],
                [(A.TURN_LR_DELTA, 0)],
                [(A.TURN_LR_DELTA, 0.4)],
                [(A.TURN_LR_DELTA, -0.2)],
                [(A.TURN_LR_DELTA, 0)],
                [(A.TURN_LR_DELTA, 0.2)],
            ],
        }

        self.action_space = spaces.Dict({
            k: spaces.Discrete(len(v)) for k, v in self.action_dict.items()
        })

        self.replay_suffix = config.get("replay_suffix", "")
        self.print_log = config.get("detailed_log", False)
        # self.seed(env_seed)
        self.server_port = (
                config.get("base_worker_port", BASE_WORKER_PORT) + config.worker_index
        )
        if self.config.get("in_evaluation", False):
            self.server_port += 100
        print(f">>> New instance {self} on port: {self.server_port}")
        # print(
        #     f"Worker Index: {config.worker_index}, VecEnv Index: {config.vector_index}"
        # )

        self.game = Game(
            map_dir=config["map_dir"],
            engine_dir=config["engine_dir"],
            server_port=self.server_port,
        )
        self.seed(env_seed)
        self.game.set_random_seed(env_seed)
        self.game.set_map_id(config["map_id"])
        self.game.set_episode_timeout(config["timeout"])
        # self.game.set_random_seed(env_seed)
        self.start_location = config.get("start_location", [0, 0, 0])
        if self.config.get("record", False):
            self.game.turn_on_record()
        self.game.turn_on_depth_map()
        self.game.set_game_replay_suffix(self.replay_suffix)
        self.game.set_game_mode(Game.MODE_NAVIGATION)
        self.game.set_depth_map_size(dmp_width, dmp_height, far=dmp_far)
        self.target_location = config.get("target_location", [0, 0, 0])
        self.start_loc = config.get("start_location", [0, 0, 0])
        self.game.set_target_location(self.target_location)

        # 101 - [x(-100,100) z(-100,100)]
        # 102 - [x(-25,175) z(-50,150)]
        # 103 - [x(-100,0) z(-50,50)]
        # 104 - [x(-125,-25) z(-110,-30)]
        self.map_select = {
            101: [[-100, 100], [-100, 100]],
            102: [[-25, 175], [-50, 150]],
            103: [[-100, 0], [-50, 50]],
            104: [[-125, -25], [-110, -30]],
        }
        limit = self.map_select.get(config["map_id"], [[-500, 500], [-500, 500]])
        locations = self.game.get_valid_locations()

        def in_map(loc):
            return limit[0][0] <= loc[0] <= limit[0][1] and limit[1][0] <= loc[2] <= limit[1][1]

        self.outdoor_loc = list(filter(in_map, locations["outdoor"]))

        self.loc_20 = []
        self.loc_50 = []
        self.loc_80 = []
        for loc in self.outdoor_loc:
            if get_distance(loc, self.target_location) <= 20:
                self.loc_20.append(loc)
            elif get_distance(loc, self.target_location) <= 40:
                self.loc_50.append(loc)
            elif get_distance(loc, self.target_location) <= 60:
                self.loc_80.append(loc)

        self.limit = 500

        self.game.init()
        self.episodes = 0
        self.episode_reward = 0

    def _get_obs(self):
        cur_pos = np.asarray(get_position(self.state))
        tar_pos = np.asarray(self.target_location)
        pitch,yaw = get_picth_yaw(tar_pos[0]-cur_pos[0],tar_pos[1]-cur_pos[1],tar_pos[2]-cur_pos[2])

        return [
            tar_pos - cur_pos,
            np.asarray([self.state.yaw,yaw]),
            self.state.depth_map.copy(),
        ]

    def step(self, action):
        # action = self._action_process(action_idxs)
        # self.game.make_action({0: action})
        action_list = self._action_process(action)
        self.game.make_action_by_list({0: action_list})
        state = self.game.get_state()
        done = self.game.is_episode_finished()

        self.running_steps += 1
        cur_pos = get_position(state)
        tar_pos = self.target_location
        # reward = -get_distance(cur_pos, tar_pos)
        reward = -0.1
        reward += get_distance(get_position(self.state), tar_pos) - get_distance(cur_pos, tar_pos)
        pitch,yaw = get_picth_yaw(tar_pos[0]-cur_pos[0],tar_pos[1]-cur_pos[1],tar_pos[2]-cur_pos[2])
        reward -= abs(state.yaw-yaw)%180
        self.state = state
        if get_distance(cur_pos, tar_pos) <= 1:
            reward += 1000
            done = True
        if get_distance(cur_pos, tar_pos) >= self.limit * 1.5:
            done = True
            reward = -100

        if done:
            if self.print_log:
                Start = np.round(np.asarray(self.start_loc), 2).tolist()
                Target = np.round(np.asarray(self.target_location), 2).tolist()
                End = np.round(np.asarray(get_position(self.state)), 2).tolist()
                Step = self.running_steps
                Reward = reward
                print(f"{Start=}\t{Target=}\t{End=}\t{Step=}\t{Reward=}")
            self.episode_reward += reward

        return self._get_obs(), reward, done, {}

    def reset(self):

        self.start_loc = random.choice(self.outdoor_loc)

        

        self.game.set_start_location(self.start_loc)
        self.target_location = random.choice(self.outdoor_loc)
        self.game.set_target_location(self.target_location)
        self.limit = get_distance(self.target_location, self.start_loc)
        

        self.game.new_episode()
        self.state = self.game.get_state()

        self.running_steps = 0
        self.episodes += 1
        return self._get_obs()

    def close(self):
        self.game.close()
        return super().close()

    def _action_process(self, action: Dict[str, int]):
        action_list = []
        for action_name, action_idx in action.items():
            action_list.extend(self.action_dict[action_name][action_idx])
        return action_list


import argparse
import re
import sys

parser = argparse.ArgumentParser()
parser.add_argument("-T", "--timeout", type=int, default=60 * 5)  # The time length of one game (sec)
parser.add_argument("-R", "--time-scale", type=int, default=10)
parser.add_argument("-M", "--map-id", type=int, default=101)
parser.add_argument("-S", "--random-seed", type=int, default=0)
parser.add_argument("--start-location", type=float, nargs=3, default=[0, 0, 0])
parser.add_argument("--target-location", type=float, nargs=3, default=[0, 0, 0])
parser.add_argument("--base-worker-port", type=int, default=50000)
parser.add_argument("--engine-dir", type=str, default="/root/game-engine")
parser.add_argument("--map-dir", type=str, default="/root/map-data")
parser.add_argument("--num-workers", type=int, default=80)
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
parser.add_argument("--dmp-width", type=int, default=10)
parser.add_argument("--dmp-height", type=int, default=2)
parser.add_argument("--dmp-far", type=int, default=200)
parser.add_argument("--train-batch-size", type=int, default=4000)
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
                "train_batch_size": args.train_batch_size,  # default of ray is 4000
                "num_gpus": 0,
                "ignore_worker_failures":True,
            }
        )
    elif alg == 'a3c':
        trainer = A3CTrainer(
            config={
                "env": NavigationEnv,
                "env_config": vars(args),
                "framework": "torch",
                "num_workers": args.num_workers,
                "num_gpus": 0,
                "train_batch_size": args.train_batch_size,  # default of ray is 4000
                "ignore_worker_failures":True,

            }
        )
    elif alg == 'appo':
        trainer = APPOTrainer(
            config={
                "env": NavigationEnv,
                "env_config": vars(args),
                "framework": "torch",
                "num_workers": args.num_workers,
                "num_gpus": 0,
                "train_batch_size": args.train_batch_size,  # default of ray is 4000
                "ignore_worker_failures":True,

            }
        )
    elif alg == 'impala':
        trainer = ImpalaTrainer(
            config={
                "env": NavigationEnv,
                "env_config": vars(args),
                "framework": "torch",
                "num_workers": args.num_workers,
                "num_gpus": 0,
                "train_batch_size": args.train_batch_size,  # default of ray is 4000
                "ignore_worker_failures":True,
            }
        )
    else:
        raise ValueError('No such algorithm')
    step = 0
    if args.reload:
        trainer.restore(args.reload_dir)

    while True:
        step += 1
        result = trainer.train()
        reward = result["episode_reward_mean"]
        e = result["episodes_total"]
        len1 = result["episode_len_mean"]
        s = result["agent_timesteps_total"]
        print(f"current_alg:{alg},current_training_steps:{s},episodes_total:{e},current_reward:{reward},current_len:{len1}")

        if step != 0 and step % 200 == 0:
            os.makedirs(args.checkpoint_dir + f"{alg}" + str(args.map_id), exist_ok=True)
            trainer.save(args.checkpoint_dir + f"{alg}" + str(args.map_id))
            print("trainer save a checkpoint")
        if result["agent_timesteps_total"] >= args.stop_timesteps:
            os.makedirs(args.checkpoint_dir + f"{alg}" + str(args.map_id), exist_ok=True)
            trainer.save(args.checkpoint_dir + f"{alg}" + str(args.map_id))
            trainer.stop()
            break

    print("the training has done!!")
    ray.shutdown()
    sys.exit()
