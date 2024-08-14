############################### Import libraries ###############################


import os
import glob
import time
from datetime import datetime

import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
from torch.distributions import Categorical

import numpy as np
from env.make_env import TradingEnv
import env.config as default_config

import gym
from net import RolloutBuffer, ActorCritic, PPO
# import roboschool
# import pybullet_envs
from tqdm import tqdm
from env.utils import time_delete
import subprocess
import json
import pandas as pd
from datetime import datetime

def convert_time_format(time_str):
    # 去掉前面的 'z ' 并去掉冒号
    time_str = time_str[2:].replace(':', '') + '000'
    # 去掉开头的 '0'
    return time_str.lstrip('0')
def formate_date(date):
    '''
    input: date(example: '2023/8/29')
    output: date(example: '20230829')
    '''
    date_obj = datetime.strptime(date, '%Y/%m/%d')
    formatted_date_str = date_obj.strftime('%Y%m%d')
    return formatted_date_str
def adjust_row(row):
    sample = row[1]
    direction = sample['side']
    date = formate_date(sample['date'])
    sym = sample['sym']
    side = sample['side'].lower()
    start_time = convert_time_format(sample['start-time'])
    end_time = convert_time_format(sample['end-time'])
    volume = sample['volume']
    return direction, date, sym, side, start_time, end_time, volume
    
def set_config(default_config, date, sym, side, start_time, end_time, volume):
    config = default_config
    config.StrategyParam.trading_day = date.replace('-', '')
    config.StrategyParam.instrument = sym
    config.TradeParam.volume = int(0.05 * volume)
    # split_num = int(min(config.TradeParam.volume//100, time_delete(config.TradeParam.end_time, config.TradeParam.start_time)//(3*config.StrategyParam.time_window)))
    # config.TradeParam.split_num = split_num
    config.TradeParam.start_time = start_time
    config.TradeParam.end_time = end_time
    return config
def adjust_sample(t_result, tot_vwap):
    state = t_result[0]['features']
    action, action_logprob, state_val = t_result[1]['action'], t_result[1]['action_logprob'], t_result[1]['state_val']
    reward_data = t_result[2]
    reward = reward_data[0] - reward_data[1]*tot_vwap
    return (state, action, action_logprob, state_val, reward)   

################################## set device ##################################

print("============================================================================================")


# set device to cpu or cuda
device = torch.device('cpu')

if(torch.cuda.is_available()): 
    device = torch.device('cuda:0') 
    torch.cuda.empty_cache()
    print("Device set to : " + str(torch.cuda.get_device_name(device)))
else:
    print("Device set to : cpu")
    
print("============================================================================================")


################################### Training ###################################


####### initialize environment hyperparameters ######

env_name = "sim_trade"
has_continuous_action_space = False

max_ep_len = 50                    # max timesteps in one episode
max_training_timesteps = int(1e5)   # break training loop if timeteps > max_training_timesteps

print_freq = max_ep_len * 4     # print avg reward in the interval (in num timesteps)
log_freq = max_ep_len * 2       # log avg reward in the interval (in num timesteps)
save_model_freq = int(2e4)      # save model frequency (in num timesteps)

action_std = None


#####################################################


## Note : print/log frequencies should be > than max_ep_len


################ PPO hyperparameters ################


update_timestep = max_ep_len * 4      # update policy every n timesteps
K_epochs = 40               # update policy for K epochs
eps_clip = 0.2              # clip parameter for PPO
gamma = 0.99                # discount factor

lr_actor = 0.0003       # learning rate for actor network
lr_critic = 0.001       # learning rate for critic network

random_seed = 0         # set random seed if required (0 = no random seed)

#####################################################



print("training environment name : " + env_name)

# env = TradingEnv()

# state space dimension
# state_dim = env.observation_space.shape[0]
state_dim = 7
# action space dimension

# action_dim = env.action_space.n
action_dim = 11
###################### logging ######################

#### log files for multiple runs are NOT overwritten

log_dir = "PPO_logs"
if not os.path.exists(log_dir):
      os.makedirs(log_dir)

log_dir = log_dir + '/' + env_name + '/'
if not os.path.exists(log_dir):
      os.makedirs(log_dir)


#### get number of log files in log directory
run_num = 0
current_num_files = next(os.walk(log_dir))[2]
run_num = len(current_num_files)


#### create new log file for each run 
log_f_name = log_dir + '/PPO_' + env_name + "_log_" + str(run_num) + ".csv"

print("current logging run number for " + env_name + " : ", run_num)
print("logging at : " + log_f_name)

#####################################################


################### checkpointing ###################

run_num_pretrained = 0      #### change this to prevent overwriting weights in same env_name folder

directory = "PPO_preTrained"
if not os.path.exists(directory):
      os.makedirs(directory)

directory = directory + '/' + env_name + '/'
if not os.path.exists(directory):
      os.makedirs(directory)


checkpoint_path = directory + "PPO_{}_{}_{}.pth".format(env_name, random_seed, run_num_pretrained)
training_checkpoint_path = directory + "training_PPO_{}_{}_{}.pth".format(env_name, random_seed, run_num_pretrained)
print("save checkpoint path : " + checkpoint_path)

#####################################################


############# print all hyperparameters #############

print("--------------------------------------------------------------------------------------------")

print("max training timesteps : ", max_training_timesteps)
print("max timesteps per episode : ", max_ep_len)

print("model saving frequency : " + str(save_model_freq) + " timesteps")
print("log frequency : " + str(log_freq) + " timesteps")
print("printing average reward over episodes in last : " + str(print_freq) + " timesteps")

print("--------------------------------------------------------------------------------------------")

print("state space dimension : ", state_dim)
print("action space dimension : ", action_dim)

print("--------------------------------------------------------------------------------------------")

print("Initializing a discrete action space policy")

print("--------------------------------------------------------------------------------------------")

print("PPO update frequency : " + str(update_timestep) + " timesteps") 
print("PPO K epochs : ", K_epochs)
print("PPO epsilon clip : ", eps_clip)
print("discount factor (gamma) : ", gamma)

print("--------------------------------------------------------------------------------------------")

print("optimizer learning rate actor : ", lr_actor)
print("optimizer learning rate critic : ", lr_critic)

if random_seed:
    print("--------------------------------------------------------------------------------------------")
    print("setting random seed to ", random_seed)
    torch.manual_seed(random_seed)
    env.seed(random_seed)
    np.random.seed(random_seed)

#####################################################

print("============================================================================================")

################# training procedure ################

# initialize a PPO agent
ppo_agent = PPO(state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, device, action_std)
# save initial model
ppo_agent.save(training_checkpoint_path)

# track total training time
start_time = datetime.now().replace(microsecond=0)
print("Started training at (GMT) : ", start_time)

print("============================================================================================")
# logging file
log_f = open(log_f_name,"w+")
log_f.write('episode,timestep,reward\n')

# printing and logging variables
print_running_reward = 0
print_running_episodes = 0

log_running_reward = 0
log_running_episodes = 0

time_step = 0
i_episode = 0   
# train set 
train_set = pd.read_csv('./plans/train_sell.csv')
for row in tqdm(train_set.iterrows(), desc='training plans\n'):
    direction, date, sym, side, start_time, end_time, volume = adjust_row(row)
    config = set_config(default_config, date, sym, side, start_time, end_time, volume)
    command = ['python', './env/make_env.py',  
         '--inst', str(config.StrategyParam.instrument), 
         '--td', str(config.StrategyParam.trading_day), 
        '--volume', str(config.TradeParam.volume),
         '--start_time', str(config.TradeParam.start_time),
         '--end_time', str(config.TradeParam.end_time),
        '--policy', os.path.abspath(training_checkpoint_path),
        '--direction', 'sell' 
        ]
    # print("command:", '\n', command)
    result = subprocess.run(command, 
        capture_output=True, text=True
    )
    output_lines = result.stdout.splitlines()
    json_output_lines = []
    capture_json = False
    for line in output_lines:
        if "JSON_OUTPUT_START" in line:
            capture_json = True
            continue
        if "JSON_OUTPUT_END" in line:
            capture_json = False
            continue
        if capture_json:
            json_output_lines.append(line)
    
    json_output = "\n".join(json_output_lines)
    output_data = json.loads(json_output)
    train_data = output_data["train_data"]
    tot_vwap = output_data["tot_vwap"]
    
    current_ep_reward = 0
    for t in range(len(train_data)):
        t_result = train_data[t]
        state, action, action_logprob, state_val, reward = adjust_sample(t_result, tot_vwap)
        done = (t == (len(train_data)-1))
        state, action, action_logprob, state_val, reward, done = torch.tensor(state), torch.tensor(action), torch.tensor(action_logprob), torch.tensor(state_val), torch.tensor(reward), torch.tensor(done)
        ppo_agent.buffer.states.append(state)
        ppo_agent.buffer.actions.append(action)
        ppo_agent.buffer.logprobs.append(action_logprob)
        ppo_agent.buffer.state_values.append(state_val)
        ppo_agent.buffer.rewards.append(reward)
        ppo_agent.buffer.is_terminals.append(done)
        
        time_step +=1
        current_ep_reward += reward.item()

        # update PPO agent
        if time_step % update_timestep == 0:
            ppo_agent.update()
            ppo_agent.save(training_checkpoint_path)
            print('-------------update and save PPO agent-------------')
        # log in logging file
        if time_step % log_freq == 0:

            # log average reward till last episode
            log_avg_reward = log_running_reward / log_running_episodes
            log_avg_reward = round(log_avg_reward, 4)

            log_f.write('{},{},{}\n'.format(i_episode, time_step, log_avg_reward))
            log_f.flush()

            log_running_reward = 0
            log_running_episodes = 0
        # printing average reward
        if time_step % print_freq == 0:

            # print average reward till last episode
            print_avg_reward = print_running_reward / print_running_episodes
            print_avg_reward = round(print_avg_reward, 2)

            print("Episode : {} \t\t Timestep : {} \t\t Average Reward : {}".format(i_episode, time_step, print_avg_reward))

            print_running_reward = 0
            print_running_episodes = 0
            
        # save model weights
        if time_step % save_model_freq == 0:
            print("--------------------------------------------------------------------------------------------")
            print("saving model at : " + checkpoint_path)
            ppo_agent.save(checkpoint_path)
            print("model saved")
            print("Elapsed Time  : ", datetime.now().replace(microsecond=0) - start_time)
            print("--------------------------------------------------------------------------------------------")
            
        # break; if the episode is over
        if done:
            break

    print_running_reward += current_ep_reward
    print_running_episodes += 1

    log_running_reward += current_ep_reward
    log_running_episodes += 1

    i_episode += 1

# print total training time
print("============================================================================================")
end_time = datetime.now().replace(microsecond=0)
print("Started training at (GMT) : ", start_time)
print("Finished training at (GMT) : ", end_time)
print("Total training time  : ", end_time - start_time)
print("============================================================================================")
