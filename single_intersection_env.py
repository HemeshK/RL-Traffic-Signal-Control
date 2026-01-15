import gymnasium as gym
from gymnasium import spaces
import numpy as np


class SimpleTrafficEnv(gym.Env):
    """
    Single-intersection traffic signal control environment
    with waiting time, switch penalty, and minimum green time.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self):
        super().__init__()

        # Actions:
        # 0 -> North-South green
        # 1 -> East-West green
        self.action_space = spaces.Discrete(2)

        # State:
        # [q_N, q_S, q_E, q_W, w_N, w_S, w_E, w_W]
        self.observation_space = spaces.Box(
            low=0,
            high=1000,
            shape=(8,),
            dtype=np.float32
        )

        # Traffic state
        self.queues = np.zeros(4, dtype=np.float32)
        self.wait_times = np.zeros(4, dtype=np.float32)

        # Signal control state
        self.last_action = None
        self.green_timer = 0
        self.min_green_time = 3
        self.switch_penalty = 2.0

        # Exposed signal phase for visualization
        self.current_phase = 0

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.queues = np.zeros(4, dtype=np.float32)
        self.wait_times = np.zeros(4, dtype=np.float32)

        self.last_action = None
        self.green_timer = 0
        self.current_phase = 0

        state = np.concatenate([self.queues, self.wait_times])
        return state, {}

    def step(self, action):
        """
        action: 0 (NS green) or 1 (EW green)
        """

        # 1️⃣ Vehicle arrivals (stochastic traffic)
        arrivals = np.random.poisson(lam=2, size=4)
        self.queues += arrivals

        # 2️⃣ Waiting time accumulation
        self.wait_times += self.queues

        # 3️⃣ Enforce minimum green time (decide effective_action)
        if self.last_action is None:
            effective_action = action
            self.green_timer = 1
        else:
            if action != self.last_action and self.green_timer < self.min_green_time:
                effective_action = self.last_action
                self.green_timer += 1
            else:
                effective_action = action
                if action != self.last_action:
                    self.green_timer = 1
                else:
                    self.green_timer += 1

        # 🔑 Expose executed phase for renderer
        self.current_phase = effective_action

        # 4️⃣ Vehicles pass through green lanes
        if effective_action == 0:
            # North & South green
            cleared_n = min(3, self.queues[0])
            cleared_s = min(3, self.queues[1])

            self.queues[0] -= cleared_n
            self.queues[1] -= cleared_s

            self.wait_times[0] = 0
            self.wait_times[1] = 0

        else:
            # East & West green
            cleared_e = min(3, self.queues[2])
            cleared_w = min(3, self.queues[3])

            self.queues[2] -= cleared_e
            self.queues[3] -= cleared_w

            self.wait_times[2] = 0
            self.wait_times[3] = 0

        # 5️⃣ Reward (congestion + fairness)
        reward = -(
            np.sum(self.queues)
            + 0.1 * np.sum(self.wait_times)
        )

        # 6️⃣ Switching penalty (only if a real switch happened)
        if self.last_action is not None and effective_action != self.last_action:
            reward -= self.switch_penalty

        # 7️⃣ Update signal state
        self.last_action = effective_action

        terminated = False
        truncated = False

        state = np.concatenate([self.queues, self.wait_times])
        return state, reward, terminated, truncated, {}

    def render(self):
        print(f"Phase: {self.current_phase}")
        print(f"Queues [N S E W]: {self.queues}")
        print(f"Waits  [N S E W]: {self.wait_times}")
