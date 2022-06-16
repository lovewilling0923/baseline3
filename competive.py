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


class BattleEnv(gym.Env):
    def __init__(self, config: EnvContext):
        self.config = config
        self.render_scale = config.get("render_scale", 1)

        env_seed = config.get("random_seed", 0) + config.worker_index

        # only 240 * 320 can be aotu transform into conv model
        dmp_width = config["dmp_width"]
        dmp_height = config["dmp_height"]
        dmp_far = config["dmp_far"]

        obs_space_1 = spaces.Box(low=-np.Inf, high=np.Inf, shape=(3,), dtype=np.float32)
        obs_space_2 = spaces.Box(low=-np.Inf, high=np.Inf, shape=(4,), dtype=np.float32)
        obs_space_3 = spaces.Box(low=-np.Inf, high=np.Inf, shape=(6,), dtype=np.float32)
        obs_space_4 = spaces.Box(
            low=0, high=dmp_far, shape=(dmp_height, dmp_width), dtype=np.float32
        )
        self.observation_space = spaces.Tuple([obs_space_1, obs_space_2,obs_space_3,obs_space_4])
        # self.observation_space = obs_space_1
        self.action_dict = {
            "move": [
                [(A.WALK_DIR, 0), (A.WALK_SPEED, 0)],
                [(A.WALK_DIR, 0), (A.WALK_SPEED, 8)],
                [(A.WALK_DIR, 90), (A.WALK_SPEED, 8)],
                [(A.WALK_DIR, 180), (A.WALK_SPEED, 8)],
                [(A.WALK_DIR, 270), (A.WALK_SPEED, 8)],
            ],
            "turn_lr_or_up": [
                [(A.TURN_LR_DELTA, 0)],
                [(A.TURN_LR_DELTA, -1)],
                [(A.TURN_LR_DELTA, 1)],
                [(A.TURN_LR_DELTA, -2)],
                [(A.TURN_LR_DELTA, 2)],
                [(A.TURN_LR_DELTA, -0.5)],
                [(A.TURN_LR_DELTA, 0.5)],
                [(A.TURN_LR_DELTA, -0.2)],
                [(A.TURN_LR_DELTA, 0.2)],
                [(A.LOOK_UD_DELTA, -0.2)],
                [(A.LOOK_UD_DELTA, 0)],
                [(A.LOOK_UD_DELTA, 0.2)],
            ],
            "attack_or_reload":[
                [(A.ATTACK,True)],
                [(A.ATTACK,False)],
            ]

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
        self.start_location = config.get("start_location", [-54, 2.55, 2])
        self.game.set_start_location(self.start_location)
        if self.config.get("record", False):
            self.game.turn_on_record()
        self.game.turn_on_depth_map()
        self.game.set_game_replay_suffix(self.replay_suffix)
        self.game.set_game_mode(Game.MODE_SUP_GATHER)
        self.game.set_depth_map_size(dmp_width, dmp_height, far=dmp_far)

        self.agent_id = set()
        self.agent_id.add(0)
        #中心点为[-50,0,0]
        for agent_id in range(1, config["num_agents"]):
            self.game.add_agent(num_clip_ammo=10000)
            self.agent_id.add(agent_id)
            x = np.random.randint(-60, -40)
            y = 5
            z = np.random.randint(-10, 10)
            self.game.set_start_location([x, y, z], agent_id)




        self.game.init()
        self.episodes = 0
        self.episode_reward = 0

    def _get_obs(self,state):
        self.np_enemy_states = []
                
        for enemy in state.enemy_states.values():
             if enemy.health > 0:
                self.np_enemy_states.append([
                    enemy.position_x,
                    enemy.position_y,
                    enemy.position_z,
                    enemy.health/100.,
                    enemy.id,
                    1 if enemy.is_invincible else 0,
                ])
        
        # enemy_distance = [get_distance([enemy[0],enemy[1],enemy[2]], get_position(state)) for enemy in self.np_enemy_states]

        self.np_enemy_states.sort(key= lambda x:get_distance([x[0],x[1],x[2]], get_position(state)))

        if self.np_enemy_states:
            enemy_states = np.asarray(self.np_enemy_states[0])
            self.enemy_id = enemy_states[4]
        else:
            enemy_states = np.asarray([0 for i in range(6)])
            self.enemy_id = 0
        

        cur_pos = np.asarray(get_position(state))
        self_future = []

        enemy_pos = enemy_states[0:3]
        op_pitch,op_yaw = get_picth_yaw(enemy_pos[0]-cur_pos[0],enemy_pos[1]-cur_pos[1],enemy_pos[2]-cur_pos[2])
        self_future.append(state.yaw)
        self_future.append(state.pitch)
        self_future.append(op_pitch)
        self_future.append(op_yaw)
        self_future = np.asarray(self_future)

        return [
            cur_pos,
            self_future,
            enemy_states,
            state.depth_map.copy(),
        ]

    def step(self, action):
        # action = self._action_process(action_idxs)
        # self.game.make_action({0: action})

        action_0 = self._action_process(action)
        action_list = {0:action_0}
        for i in range(1,self.config["num_agents"]):
            action_list[i]=self._action_process({
            "move": 0,
            "turn_lr_or_up": 0,
            "attack_or_reload":1
            })
        self.game.make_action_by_list(action_list)
        state = self.game.get_state()
        done = self.game.is_episode_finished()
        reward = self._compute_reward(state)


        self.running_steps += 1

        
        self.state = state
        self.state_all = self.game.get_state_all()
        self.episode_reward += reward

        return self._get_obs(state), reward, done, {}
    def _compute_reward(self, state):
        """reward process method

        Parameters
        ----------
        state: AgentState object got from backend env
        action: action list got from agent
        """
        state_all = self.game.get_state_all()
        reward =0

        if state_all[self.enemy_id].is_waiting_respawn and not self.state_all[self.enemy_id].is_waiting_respawn:
            reward+=100
        if state.hit_enemy and state.hit_enemy_id ==self.enemy_id:
            reward+=1
        return  reward

    def reset(self):
        self.game.new_episode()
        self.state = self.game.get_state()
        self.state_all = self.game.get_state_all()


        self.running_steps = 0
        self.episodes += 1
        self.enemy_id = 0
        return self._get_obs(self.state)

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
parser.add_argument("-T", "--timeout", type=int, default=60 * 3)  # The time length of one game (sec)
parser.add_argument("-R", "--time-scale", type=int, default=10)
parser.add_argument("-M", "--map-id", type=int, default=103)
parser.add_argument("-S", "--random-seed", type=int, default=0)
parser.add_argument("--start-location", type=float, nargs=3, default=[-54, 2.55, 2])
parser.add_argument("--base-worker-port", type=int, default=50000)
parser.add_argument("--engine-dir", type=str, default="/root/game-engine")
parser.add_argument("--map-dir", type=str, default="/root/map-data")
parser.add_argument("--num-workers", type=int, default=180)
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
parser.add_argument("--train-batch-size", type=int, default=4000)
parser.add_argument("--reload", type=bool, default=False)
parser.add_argument("--reload-dir", type=str, default="")
parser.add_argument("--num-agents", type=int, default=10)
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
                "env": BattleEnv,
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
                "env": BattleEnv,
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
                "env": BattleEnv,
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
                "env": BattleEnv,
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
