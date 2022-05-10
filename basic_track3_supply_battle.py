from re import S
import time
import random
import argparse
from turtle import st

from rich.progress import track
from rich.console import Console

console = Console()
import numpy as np
from inspirai_fps import Game, ActionVariable
from inspirai_fps.utils import get_position


parser = argparse.ArgumentParser()
parser.add_argument("--port", type=int, default=50051)
parser.add_argument("--timeout", type=int, default=10)
parser.add_argument("--map-id", type=int, default=1)
parser.add_argument("--random-seed", type=int, default=0)
parser.add_argument("--num-episodes", type=int, default=1)
parser.add_argument("--engine-dir", type=str, default="../unity3d")
parser.add_argument("--map-dir", type=str, default="../data")
parser.add_argument("--num-agents", type=int, default=10)
parser.add_argument("--use-depth-map", action="store_true")
parser.add_argument("--record", action="store_true")
parser.add_argument("--replay-suffix", type=str, default="")
parser.add_argument("--start-location", type=float, nargs=3, default=[0, 0, 0])
parser.add_argument("--walk-speed", type=float, default=1)
args = parser.parse_args()

def get_picth_yaw(x, y, z):
            pitch = np.arctan2(y, (x**2 + z**2)**0.5) / np.pi * 180
            yaw = np.arctan2(x, z) / np.pi * 180
            return pitch, yaw

def act(state,last_enemy):
            state = state_all[0]
            x,y,z = state.position_x,state.position_y,state.position_z
            l_x,l_y,l_z = last_enemy.position_x,last_enemy.position_y,last_enemy.position_z
            x_,y_,z_ = x-l_x,y-l_y,z-l_z
            pitch_,yaw = get_picth_yaw(x_,y_,z_)
            cur_pitch,cur_yaw = get_picth_yaw(x,y,z)
            pitch=pitch - cur_pitch
            yaw-=cur_yaw

            return [pitch_,pitch,yaw]




# Define a random policy
def stay_policy(state):

    return [
        0,  # walk_dir
        0,  # walk_speed
        0,  # turn left right
        0,  # look up down
        random.random()>0.5,
        True,  # attack
        True,  # reload
        True,  # collect
    ]

def random_policy(state):
    return [
        random.randint(0, 270),  # walk_dir
        random.randint(1, 10),  # walk_speed
        random.choice([-1, 0, 1]),  # turn_lr_delta
        random.choice([-1, 0, 1]),  # turn_ud_delta
        random.random() > 0.5,  # jump
        False,
        False,
        True,
    ]
#Define a rule fight policy
def rule_policy(state,last_enemy):
    if len(state.enemy_states)==0:
        last_enemy = []
        return last_enemy ,random_policy(state)
    else:
        if last_enemy in state.enemy_states:
            walk_dir,pitch,yaw = act(state,last_enemy)
        else:
            last_enemy = state.enemy_states[0]
            walk_dir,pitch,yaw = act(state,last_enemy)
    return last_enemy,[
        walk_dir,
        args.walk_speed,
        yaw,
        pitch,
        random.random()<0.5,
        True,
        False if state.weapon_ammo>0 else True,
        True,]


# valid actions
used_actions = [
    ActionVariable.WALK_DIR,
    ActionVariable.WALK_SPEED,
    ActionVariable.TURN_LR_DELTA,
    ActionVariable.LOOK_UD_DELTA,
    ActionVariable.JUMP,
    ActionVariable.ATTACK,
    ActionVariable.RELOAD,
    ActionVariable.PICKUP,
]

# instantiate Game
game = Game(map_dir=args.map_dir, engine_dir=args.engine_dir)
game.set_game_mode(Game.MODE_SUP_BATTLE)
game.set_supply_heatmap_center([args.start_location[0], args.start_location[2]])
game.set_supply_heatmap_radius(50)
game.set_supply_indoor_richness(80)
game.set_supply_outdoor_richness(20)
game.set_supply_indoor_quantity_range(10, 50)
game.set_supply_outdoor_quantity_range(1, 5)
game.set_supply_spacing(5)
game.set_episode_timeout(args.timeout)
game.set_start_location(args.start_location)  # set start location of the first agent
game.set_available_actions(used_actions)
game.set_map_id(args.map_id)

if args.use_depth_map:
    game.turn_on_depth_map()

if args.record:
    game.turn_on_record()

for agent_id in range(1, args.num_agents):
    game.add_agent()
    game.random_start_location(agent_id)

game.init()

for ep in track(range(args.num_episodes), description="Running Episodes ..."):
    game.set_game_replay_suffix(f"{args.replay_suffix}_episode_{ep}")
    game.new_episode()
    last_enemy = []
    while not game.is_episode_finished():
        ts = game.get_time_step()

        t = time.perf_counter()
        state_all = game.get_state_all()
        action_all = {
            agent_id: stay_policy(state_all[agent_id], ts) for agent_id in state_all
        }
        last_enemy, action_all[0] = rule_policy(state_all[0],last_enemy)
        game.make_action(action_all)
        dt = time.perf_counter() - t



        agent_id = 0
        state = state_all[agent_id]
        step_info = {
            "Episode": ep,
            "TimeStep": ts,
            "AgentID": agent_id,
            "Location": get_position(state),
            "pitch":state.pitch,
            "yaw":state.yaw,
            "walk_dir":[state.move_dir_x,state.move_dir_y,state.move_dir_z],
            "Action": {
                name: val for name, val in zip(used_actions, action_all[agent_id])
            },
            "#EnemyInfo": state.enemy_states,
            "last_enemy":last_enemy,
        }
        if args.use_depth_map:
            step_info["DepthMap"] = state.depth_map.shape
        console.print(step_info, style="bold magenta")

    print("episode ended ...")

game.close()
