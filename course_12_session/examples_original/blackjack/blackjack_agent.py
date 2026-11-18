from collections import defaultdict
import gymnasium as gym
import numpy as np


class BlackjackAgent:
    def __init__(
        self,
        env: gym.Env,
        learning_rate: float,
        initial_epsilon: float,
        epsilon_decay: float,
        final_epsilon: float,
        discount_factor: float = 0.95,
        strategy: str = "TD",
    ):
        """Initialize a Q-Learning agent.

        Args:
            env: The training environment
            learning_rate: How quickly to update Q-values (0-1)
            initial_epsilon: Starting exploration rate (usually 1.0)
            epsilon_decay: How much to reduce epsilon each episode
            final_epsilon: Minimum exploration rate (usually 0.1)
            discount_factor: How much to value future rewards (0-1)
        """
        self.env = env

        # Q-table: maps (state, action) to expected reward
        # defaultdict automatically creates entries with zeros for new states
        self.q_values = defaultdict(lambda: np.zeros(env.action_space.n))

        self.lr = learning_rate
        self.discount_factor = discount_factor  # How much we care about future rewards
        self.strategy = strategy.upper()
        
        # Exploration parameters
        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon

        # Isolated RNG stream from environment instance
        self.np_random = env.unwrapped.np_random

        # Track learning progress
        self.training_error = []

    def get_action(self, obs: tuple[int, int, bool]) -> int:
        """Choose an action using epsilon-greedy strategy.

        Returns:
            action: 0 (stand) or 1 (hit)
        """
        # With probability epsilon: explore (random action)
        #replace global np.random with self.np_random to ensure reproducibility
        if self.np_random.random() < self.epsilon: 
            return int(self.env.action_space.sample())
        # With probability (1-epsilon): exploit (best known action)
        return int(np.argmax(self.q_values[obs]))

    def update_td(
        self,
        obs: tuple[int, int, bool],
        action: int,
        reward: float,
        terminated: bool,
        next_obs: tuple[int, int, bool],
    ):
        """Update Q-value based on experience.

        This is the heart of Q-learning: learn from (state, action, reward, next_state)
        https://huggingface.co/learn/deep-rl-course/unit2/q-learning
        """
        # What's the best we could do from the next state?
        # (Zero if episode terminated - no future rewards possible)
        future_q_value = (not terminated) * np.max(self.q_values[next_obs])

        # What should the Q-value be? (Bellman equation)
        target = reward + self.discount_factor * future_q_value

        # How wrong was our current estimate?
        # TD approach: updates its action-value function at each step instead of at the end of the episode.
        temporal_difference = target - self.q_values[obs][action]

        # Update our estimate in the direction of the error
        # Learning rate controls how big steps we take
        self.q_values[obs][action] = (
            self.q_values[obs][action] + self.lr * temporal_difference
        )

        # Track learning progress (useful for debugging)
        self.training_error.append(temporal_difference)

        print(
            f"Updated Q-value for state {obs} and action {action}: "
            f"{self.q_values[obs][action]:.4f} (TD error: {temporal_difference:.4f})"
        )

    def update_mc(self, trajectory: list[tuple]):
        """First-Visit Monte Carlo Control update."""
        G = 0.0
        visited = set()

        for obs, action, reward in reversed(trajectory):
            G = self.discount_factor * G + reward
            state_action = (obs, action)

            if state_action not in visited:
                visited.add(state_action)
                mc_error = G - self.q_values[obs][action]
                self.q_values[obs][action] += self.lr * mc_error
                self.training_error.append(mc_error)

    def decay_epsilon(self):
        """Reduce exploration rate after each episode."""
        self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay)