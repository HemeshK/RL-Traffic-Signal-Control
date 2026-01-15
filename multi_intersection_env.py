import gymnasium as gym
from gymnasium import spaces
import numpy as np


class MultiIntersectionEnv(gym.Env):
    """
    2x2 grid multi-intersection traffic signal control environment
    with direction-aware downstream pressure (Option B).
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self):
        super().__init__()

        self.num_intersections = 4

        # Each intersection chooses NS or EW
        self.action_space = spaces.MultiDiscrete([2] * self.num_intersections)

        # State per intersection:
        # [qN, qS, qE, qW, wN, wS, wE, wW, downstream_pressure]
        self.state_dim_per_intersection = 9

        self.observation_space = spaces.Box(
            low=0,
            high=5000,
            shape=(self.num_intersections * self.state_dim_per_intersection,),
            dtype=np.float32
        )

        # Traffic state
        self.queues = np.zeros((self.num_intersections, 4), dtype=np.float32)
        self.wait_times = np.zeros((self.num_intersections, 4), dtype=np.float32)

        # Signal control
        self.last_actions = [None] * self.num_intersections
        self.green_timers = [0] * self.num_intersections
        self.min_green_time = 3
        self.switch_penalty = 2.0

        # Grid neighbors (N, S, E, W)
        self.neighbors = {
            0: {"N": -1, "S": 2, "E": 1, "W": -1},
            1: {"N": -1, "S": 3, "E": -1, "W": 0},
            2: {"N": 0, "S": -1, "E": 3, "W": -1},
            3: {"N": 1, "S": -1, "E": -1, "W": 2},
        }

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.queues[:] = 0
        self.wait_times[:] = 0
        self.last_actions = [None] * self.num_intersections
        self.green_timers = [0] * self.num_intersections

        return self._get_state(), {}

    def step(self, actions):

        total_reward = 0.0

        # 1️⃣ External arrivals
        arrivals = np.random.poisson(lam=1.2, size=(self.num_intersections, 4))
        self.queues += arrivals

        # 2️⃣ Waiting time accumulation
        self.wait_times += self.queues

        # 3️⃣ Minimum green time logic
        effective_actions = []

        for i in range(self.num_intersections):
            action = actions[i]
            last = self.last_actions[i]

            if last is None:
                effective = action
                self.green_timers[i] = 1
            else:
                if action != last and self.green_timers[i] < self.min_green_time:
                    effective = last
                    self.green_timers[i] += 1
                else:
                    effective = action
                    if action != last:
                        self.green_timers[i] = 1
                        total_reward -= self.switch_penalty
                    else:
                        self.green_timers[i] += 1

            effective_actions.append(effective)
            self.last_actions[i] = effective

        # 4️⃣ Vehicle movement
        for i in range(self.num_intersections):
            if effective_actions[i] == 0:  # NS green
                self._move_vehicle(i, 0, "N")
                self._move_vehicle(i, 1, "S")
            else:  # EW green
                self._move_vehicle(i, 2, "E")
                self._move_vehicle(i, 3, "W")

        # 5️⃣ Global reward
        total_reward -= (
            np.sum(self.queues)
            + 0.1 * np.sum(self.wait_times)
        )

        return self._get_state(), total_reward, False, False, {}

    def _move_vehicle(self, idx, lane, direction):
        cleared = min(3, self.queues[idx, lane])
        self.queues[idx, lane] -= cleared
        self.wait_times[idx, lane] = 0

        neighbor = self.neighbors[idx][direction]

        if neighbor != -1:
            if direction == "N":
                self.queues[neighbor, 1] += cleared
            elif direction == "S":
                self.queues[neighbor, 0] += cleared
            elif direction == "E":
                self.queues[neighbor, 3] += cleared
            elif direction == "W":
                self.queues[neighbor, 2] += cleared

    def _compute_downstream_pressure(self, idx):
        pressure = 0.0

        for direction, neighbor in self.neighbors[idx].items():
            if neighbor == -1:
                continue

            if direction == "N":
                pressure += self.queues[neighbor, 1]
            elif direction == "S":
                pressure += self.queues[neighbor, 0]
            elif direction == "E":
                pressure += self.queues[neighbor, 3]
            elif direction == "W":
                pressure += self.queues[neighbor, 2]

        return pressure

    def _get_state(self):
        states = []

        for i in range(self.num_intersections):
            downstream_pressure = self._compute_downstream_pressure(i)
            local_state = np.concatenate(
                [self.queues[i], self.wait_times[i], [downstream_pressure]]
            )
            states.append(local_state)

        return np.concatenate(states)

    def render(self):
        for i in range(self.num_intersections):
            print(f"Intersection {i}")
            print("Queues:", self.queues[i])
            print("Waits :", self.wait_times[i])
            print("Downstream pressure:", self._compute_downstream_pressure(i))
            print("-" * 25)
