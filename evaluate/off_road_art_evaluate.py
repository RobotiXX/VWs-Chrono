import gymnasium as gym
from stable_baselines3 import PPO
from sb3_plus import MultiOutputPPO, MultiOutputEnv
from gym_chrono.envs.wheeled.off_road_art import off_road_art
from gymnasium.utils.env_checker import check_env

import numpy as np
import os

render = True

def evaluate_model(env, model, num_trials=25, max_steps=1000, render=False):
    success_count = 0
    traversal_times = []
    episode_roll_means = []
    episode_pitch_means = []

    for trial in range(num_trials):
        print(f"Trial {trial + 1}")
        obs, _ = env.reset(seed=trial)
        if render:
            env.render('follow')
        
        step_count = 0
        roll_angles = []
        pitch_angles = []
        
        while step_count < max_steps:
            action, _states = model.predict(obs, deterministic=True)
            print(f"Step {step_count + 1}")
            print("Action: ", action)
            obs, reward, terminated, truncated, info = env.step(action)
            print("obs=", obs, "reward=", reward, "done=", (terminated or truncated))
            if render:
                env.render('follow')
            
            step_count += 1
            
            # Collect roll and pitch angles at each step
            euler_angles = env.m_vehicle.GetVehicle().GetRot().Q_to_Euler123()
            roll_angles.append(np.degrees(abs(euler_angles.x)))
            pitch_angles.append(np.degrees(abs(euler_angles.y)))
            
            if terminated or truncated:
                if terminated and step_count < 150:  # Successful trial
                    success_count += 1
                    traversal_times.append(env.m_system.GetChTime())
                break
        
        # Calculate mean roll and pitch angles for this episode
        episode_roll_means.append(np.mean(roll_angles))
        episode_pitch_means.append(np.mean(pitch_angles))

    mean_traversal_time = np.mean(traversal_times) if traversal_times else 0
    avg_roll_angle = np.mean(episode_roll_means)
    avg_pitch_angle = np.mean(episode_pitch_means)

    return success_count, mean_traversal_time, avg_roll_angle, avg_pitch_angle

if __name__ == '__main__':
    env = off_road_art()
    env.update_terrain_stage(stage=4)

    checkpoint_dir = '../train/logs/vws_ppo_checkpoints_HopperNorm'
    loaded_model = PPO.load(os.path.join(checkpoint_dir, f"ppo_checkpoint_stage4_iter46"), env)

    #Render and test model
    sim_time = 100
    timeStep = 0.1

    totalSteps = int(sim_time / timeStep)

    env.set_nice_vehicle_mesh()
    obs, _ = env.reset()
    if render:
        env.render('follow')
    for step in range(totalSteps):
        action, _states = loaded_model.predict(obs, deterministic=True)
        print(f"Step {step + 1}")
        print("Action: ", action)
        obs, reward, teriminated, truncated, info = env.step(action)
        print("obs=", obs, "reward=", reward, "done=", (teriminated or truncated))
        if render:
            env.render('follow')
        if (teriminated or truncated):
            break

    # num_trials = 25
    # success_count, mean_traversal_time, avg_roll_angle, avg_pitch_angle = evaluate_model(env, loaded_model, num_trials=num_trials, render=False)
    # success_rate = success_count / 25

    # print('--------------------------------------------------------------')
    # print(f"Number of successful trials (out of {num_trials}): {success_count}")
    # print(f'Success rate: {success_rate * 100:.2f}%')
    # print(f"Mean traversal time of successful trials (seconds): {mean_traversal_time}")
    # print(f"Average roll angle (degrees): {avg_roll_angle}")
    # print(f"Average pitch angle (degrees): {avg_pitch_angle}")
    # print('--------------------------------------------------------------')
