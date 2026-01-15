from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

from envs.single_intersection_env import SimpleTrafficEnv


def main():
    # Vectorized environment (required by SB3)
    env = make_vec_env(SimpleTrafficEnv, n_envs=4)

    model = PPO(
        policy="MlpPolicy",
        env=env,
        verbose=1,
        learning_rate=3e-4,
        gamma=0.99,
        n_steps=256,
        batch_size=64
    )

    model.learn(total_timesteps=50_000)

    model.save("ppo_traffic")


if __name__ == "__main__":
    main()
