import gymnasium as gym
from gymnasium.utils.env_checker import check_env

from typing import Callable, List, Any, Optional, Sequence, Type
import os

from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.utils import set_random_seed, safe_mean
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.logger import configure
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import HParam
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.monitor import Monitor
import multiprocessing as mp
import torch as th
import numpy as np

from gym_chrono.envs.wheeled.off_road_art import off_road_art
from gym_chrono.train.custom_networks.artCustomCNN import CustomCombinedExtractor
from gym_chrono.envs.utils.SafeSubproc_vec_env import SafeSubprocVecEnv

class TensorboardCallback(BaseCallback):
    """
    Custom callback for plotting additional values in tensorboard.
    """

    def __init__(self, verbose=0):
        super(TensorboardCallback, self).__init__(verbose)
        self.old_episode_num = 0
        self.old_timeout_count = 0
        self.old_fallen_count = 0
        self.old_success_count = 0
        self.old_crash_count = 0
        self.last_ep_rew_mean = 0.0

    def _on_training_start(self) -> None:
        """
        This method is called before the first rollout starts.
        """
        hparam_dict = {
            "algorithm": self.model.__class__.__name__,
            "learning rate": self.model.learning_rate,
            "gamma": self.model.gamma,
        }
        metric_dict = {
            "rollout/ep_rew_mean": 0,
            "train/value_loss": 0.0,
        }
        self.logger.record(
            "hparams",
            HParam(hparam_dict, metric_dict),
            exclude=("stdout", "log", "json", "csv"),
        )

    def _on_rollout_start(self) -> None:
        """
        A rollout is the collection of environment interaction
        using the current policy.
        This event is triggered before collecting new samples.
        """
        return True
    
    def _on_step(self) -> bool:  
        return True

    def _on_rollout_end(self) -> None:
        """
        This event is triggered before updating the policy.
        Aggregate data from all environments
        """
        total_success_count = sum(self.training_env.get_attr("m_success_count"))
        total_fallen_count = sum(self.training_env.get_attr("m_fallen_count"))
        total_timeout_count = sum(self.training_env.get_attr("m_timeout_count"))
        total_episode_num = sum(self.training_env.get_attr("m_episode_num"))
        total_crash_count = sum(self.training_env.get_attr("m_crash_count"))

        # Log the rates
        self.logger.record("rollout/total_success", total_success_count)
        self.logger.record("rollout/total_fallen", total_fallen_count)
        self.logger.record("rollout/total_timeout", total_timeout_count)
        self.logger.record("rollout/total_episode_num", total_episode_num)
        self.logger.record("rollout/total_crashes", total_crash_count)
        
        self.old_episode_num = total_episode_num
        self.old_timeout_count = total_timeout_count
        self.old_fallen_count = total_fallen_count
        self.old_success_count = total_success_count
        self.old_crash_count = total_crash_count
        
        if len(self.model.ep_info_buffer) > 0 and len(self.model.ep_info_buffer[0]) > 0:
            self.last_ep_rew_mean = safe_mean([ep_info["r"] for ep_info in self.model.ep_info_buffer])

        return True

    def _on_training_end(self) -> None:
        print("Training ended")
        print("Total episodes ran: ", self.old_episode_num)
        print("Total success count: ", self.old_success_count)
        print("Total fallen count: ", self.old_fallen_count)
        print("Total timeout count: ", self.old_timeout_count)
        print("Total crash count: ", self.old_crash_count)
        return True

def make_env(stage: int, rank: int = 0, seed: int = 0) -> Callable:
    """
    Utility function for multiprocessed env.

    :param stage: (int) the terrain stage
    :param rank: (int) index of the subprocess
    :param seed: (int) the inital seed for RNG
    :return: (Callable)
    """
    def _init() -> gym.Env:
        try:
            env = off_road_art()
            env.update_terrain_stage(stage)
            env.set_nice_vehicle_mesh()
            env.reset(seed=seed + rank)
            return env
        except Exception as e:
            print(f"Failed to initialize environment in subprocess {rank} with seed {seed}: {str(e)}")
            raise e

    set_random_seed(seed)
    return _init


if __name__ == '__main__':
    device = th.device("cuda" if th.cuda.is_available() else "cpu")
    
    # Maximum num is 32
    num_procs = 20
    total_stages = 5
    stage_success_thresholds = [1, 1, 0.8, 0.6, 0.5]
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      
    log_path = "./logs/vws_logs_Hopper2/"
    os.makedirs(log_path, exist_ok=True)
    
    policy_kwargs = dict(
        features_extractor_class=CustomCombinedExtractor,
        features_extractor_kwargs={'features_dim': 32},
        activation_fn=th.nn.ReLU,
        net_arch=dict(activation_fn=th.nn.ReLU, pi=[64, 64], vf=[64, 64])
        # net_arch=dict(activation_fn=th.nn.ReLU, pi=[64, 32, 16], vf=[64, 32, 16])
    )
    
    # Update policy every num_episodes = num_procs * num_group
    # n_steps = max_time * control_freq * num_group
    n_steps = 20 * 10 * 2
    success_rate_eval = 0
    
    num_updates = 20
    total_timesteps = num_updates * n_steps * num_procs
    
    checkpoint_dir = './logs/vws_ppo_checkpoints_Hopper2/'
    os.makedirs(checkpoint_dir, exist_ok=True)

    best_checkpoint_file = None
    best_checkpoint_filename = None

    start_stage = 0
    checkpoint_file = None

    # # If training is interrupted, set the checkpoint file
    # checkpoint_file = os.path.join(checkpoint_dir, f"ppo_checkpoint_stage0_iter2")

    for stage in range(start_stage, total_stages):
        print(f"Training stage {stage+1}/{total_stages}")
        
        # Vectorized environment
        env = make_vec_env(env_id=make_env(stage), n_envs=num_procs, vec_env_cls=SubprocVecEnv)

        # Initialize the model
        if stage == start_stage and checkpoint_file:
            model = PPO.load(checkpoint_file, env, device=device)
            start_iteration = 1
        else:
            if stage == start_stage:
                model = PPO('MlpPolicy', env, learning_rate=5e-4, n_steps=n_steps, batch_size=n_steps // 2,
                        policy_kwargs=policy_kwargs, verbose=1, tensorboard_log=log_path, device=device)
                print(model.policy)
                start_iteration = 1
            else:
                # Load the best checkpoint from last stage
                model = PPO.load(best_checkpoint_file, env, device=device)
                start_iteration = 1
        
        best_success_rate = 0.0
        best_checkpoint_file = None
        best_checkpoint_filename = None
        best_ep_rew_mean = -np.inf
        
        # Single environment for evaluation
        env_single = off_road_art()
        env_single.update_terrain_stage(stage)

        for i in range(start_iteration, 51):
            # Create a unique log directory for each iteration
            iter_log_path = os.path.join(log_path, f"stage{stage}_iter{i}")
            os.makedirs(iter_log_path, exist_ok=True)
            # set up logger
            new_logger = configure(iter_log_path, ["stdout", "csv", "tensorboard"])
            model.set_logger(new_logger)

            # Instantiate the callback
            callback = TensorboardCallback()
            
            model.learn(total_timesteps, progress_bar=True, callback=callback)
            checkpoint_file = os.path.join(checkpoint_dir, f"ppo_checkpoint_stage{stage}_iter{i}")
            model.save(checkpoint_file)
            th.save(model.policy.state_dict(), os.path.join(checkpoint_dir, f"ppo_checkpoint_stage{stage}_iter{i}.pt"))
            del model
            model = PPO.load(checkpoint_file, env, device=device)

            # last reward in episode
            ep_rew_mean = callback.last_ep_rew_mean
            print(f"Iteration {i} in stage {stage+1} has ep_rew_mean: {ep_rew_mean}")
            
            # Save the best model based on `rollout/ep_rew_mean`
            if ep_rew_mean > best_ep_rew_mean:
                best_ep_rew_mean = ep_rew_mean
                best_checkpoint_file = os.path.join(checkpoint_dir, f"ppo_checkpoint_stage{stage}_iter{i}")
                best_checkpoint_filename = f"ppo_checkpoint_stage{stage}_iter{i}"

            # Evaluate the loaded model
            success_count = 0
            for _ in range(0, 10):
                obs, _ = env_single.reset()
                done = False
                while not done:
                    action, _states = model.predict(obs, deterministic=True)
                    obs, rewards, terminated, truncated, info = env_single.step(action)
                    done = terminated or truncated

                success_count += env_single.m_success_count_eval

            success_rate_eval = success_count / 10
            print(f"Iteration {i} in stage {stage+1} has success_rate_eval: {success_rate_eval}")
            
            if (success_rate_eval >= stage_success_thresholds[stage]):
                # Write to a file that the model with checkpoint i is good
                with open(f"./{log_path}good_models.txt", "a") as f:
                    f.write(f"ppo_checkpoint_stage{stage}_iter{i}, Success Rate: {success_rate_eval:.4f}\n")
                print(f"Stage {stage+1} completed with ep_rew_mean: {ep_rew_mean}")
                break

        # Save the best checkpoint file after each stage
        if best_checkpoint_filename:
            with open(f"./{log_path}best_models.txt", "a") as f:
                f.write(f"Best checkpoint for stage {stage}: {best_checkpoint_filename}\n")
        