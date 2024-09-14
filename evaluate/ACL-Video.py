import gymnasium as gym
from stable_baselines3 import PPO
from sb3_plus import MultiOutputPPO, MultiOutputEnv
from gym_chrono.envs.wheeled.off_road_artACL import off_road_art
from gymnasium.utils.env_checker import check_env
import torch as th

import numpy as np
import os
import re
import subprocess

render = True

def evaluate_checkpoint(env, model, checkpoint_name, video_output_dir, max_steps=1000, render=False):
    env.set_nice_vehicle_mesh()
    
    # Set up the directory for this checkpoint's frames
    env.video_frames_dir = os.path.join('./ACL-frames', checkpoint_name)
    env.render_frame = 0  # Reset the frame counter
    
    if not os.path.exists(env.video_frames_dir):
        os.makedirs(env.video_frames_dir)
    
    obs, _ = env.reset(seed=0)
    env.render('follow', headless=True)
    
    for step in range(max_steps):
        action, _states = model.predict(obs, deterministic=True)
        print(f"Step {step + 1}")
        print("Action: ", action)
        obs, reward, terminated, truncated, info = env.step(action)
        print("obs=", obs, "reward=", reward, "done=", (terminated or truncated))
        
        if render:
            env.render('follow', headless=True)
        env.capture_frame()
        
        if terminated or truncated:
            break
    
    # After simulation, create video from saved frames
    video_filename = os.path.join(video_output_dir, f"{checkpoint_name}.mp4")
    create_video_from_frames(env.video_frames_dir, video_filename)
    
    
def create_video_from_frames(frame_dir, video_filename, fps=10):
    command = [
        'ffmpeg', 
        '-framerate', str(fps), 
        '-i', os.path.join(frame_dir, 'img_%04d.jpg'),
        '-c:v', 'libx264', 
        '-pix_fmt', 'yuv420p', 
        video_filename
    ]
    subprocess.run(command)
    
if __name__ == '__main__':
    env = off_road_art()
    checkpoint_dir = '../train/logs/vws_ppo_checkpoints_PLRNorm/'
    video_output_base_dir = './PLR-videos'

    # Ensure the video output base directory exists
    if not os.path.exists(video_output_base_dir):
        os.makedirs(video_output_base_dir)
    
    start_iter = 138
    final_iter = 139
    for iter_num in range(start_iter, final_iter):
        for checkpoint_file in sorted(os.listdir(checkpoint_dir)):
            # Extract the iteration and level from the filename
            match = re.match(rf'ppo_checkpoint_iter{iter_num}_level(\d+)\.zip', checkpoint_file)
            if match:
                level_num = int(match.group(1))
                checkpoint_name = f"iter{iter_num}_level{level_num}"

                # Update the terrain stage
                env.update_terrain_stage(level_index=level_num)

                # Load the model
                loaded_model = PPO.load(os.path.join(checkpoint_dir, checkpoint_file), env)

                # Evaluate the checkpoint and save the video
                evaluate_checkpoint(env, loaded_model, checkpoint_name, video_output_base_dir, max_steps=1000, render=render)
                print(f"Completed evaluation for {checkpoint_name}")
                
