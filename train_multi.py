from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

from envs.multi_intersection_env import MultiIntersectionEnv


def main():
    # Create vectorized environment
    env = make_vec_env(
        MultiIntersectionEnv,
        n_envs=4  # parallel rollouts for stability
    )

    # PPO model with shared policy
    model = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=3e-4,
        gamma=0.99,
        n_steps=512,
        batch_size=128,
        verbose=1
    )

    # Train the model
    model.learn(total_timesteps=200_000)

    # Save trained model
    model.save("ppo_multi_traffic")

    print("✅ Multi-intersection PPO training complete.")


if __name__ == "__main__":
    main()
