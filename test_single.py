from stable_baselines3 import PPO
from envs.single_intersection_env import SimpleTrafficEnv

model = PPO.load("ppo_traffic")
env = SimpleTrafficEnv()

state, _ = env.reset()

for step in range(10):
    action, _ = model.predict(state, deterministic=True)
    state, reward, _, _, _ = env.step(action)

    print(f"Step {step}, Action: {action}")
    print("Queues:", state[:4])
    print("Waits :", state[4:])
    print("Reward:", reward)
    print("-" * 30)
