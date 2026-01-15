from envs.multi_intersection_env import MultiIntersectionEnv
import numpy as np

env = MultiIntersectionEnv()
state, _ = env.reset()

for step in range(5):
    actions = np.random.randint(0, 2, size=4)
    state, reward, _, _, _ = env.step(actions)

    print(f"Step {step}, Actions: {actions}, Reward: {reward}")
    env.render()
