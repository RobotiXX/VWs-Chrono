import gymnasium as gym
from gym_chrono.envs.wheeled.off_road_artACL import off_road_art
from gymnasium.utils.env_checker import check_env

render = True
if __name__ == '__main__':
    env = off_road_art()    
    env.set_nice_vehicle_mesh()

    env.update_terrain_stage(level_index=90)
    obs, _ = env.reset()
    if render:
        env.render('follow')

    print(env.observation_space.shape)
    print(env.action_space)
    print(env.action_space.sample())

    n_steps = 1000
    
    for step in range(n_steps):
        print(f"Step {step + 1}")
        '''
        Steering: -1 is right, 1 is left
        '''
        obs, reward, terminated, truncated, info = env.step([0, 1])
        print(obs, reward)
        print("Terminated=", terminated, "Truncated=", truncated)
        done = terminated or truncated
        if render:
            env.render('follow')
        if done:
            print("reward=", reward)
            break

