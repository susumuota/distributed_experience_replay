# copied parameters from https://github.com/ray-project/rl-experiments/blob/master/atari-apex/atari-apex.yaml
# changed num_workers to 6

from ray import tune
from ray.rllib.agents.dqn import ApexTrainer
from ray.rllib.agents import dqn

config = dqn.apex.APEX_DEFAULT_CONFIG.copy()

config["env"] = "PongNoFrameskip-v4"

config["double_q"] = False
config["dueling"] = False
config["num_atoms"] = 1
config["noisy"] = False
config["n_step"] = 3
config["lr"] = .0001
config["adam_epsilon"] = .00015
config["hiddens"] = [512]
config["buffer_size"] = 1000000
config["schedule_max_timesteps"] = 2000000
config["exploration_final_eps"] = 0.01
config["exploration_fraction"] = .1
config["prioritized_replay_alpha"] = 0.5
config["beta_annealing_fraction"] = 1.0
config["final_prioritized_replay_beta"] = 1.0
config["num_gpus"] = 0

config["num_workers"] = 6
config["num_envs_per_worker"] = 8
config["sample_batch_size"] = 20
config["train_batch_size"] = 512
config["target_network_update_freq"] = 50000
config["timesteps_per_iteration"] = 25000

tune.run(ApexTrainer, config=config, queue_trials=True)
