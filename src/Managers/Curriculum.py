# curriculum_manager.py
import torch

class CurriculumManager:
    """
    Time-based linear curriculum for CoopLiftEnv.
    Each parameter grows from 0.0 to 1.0 over a specified number of steps.
    """
    def __init__(self, env):
        self.env = env
        
        # Current progress (0.0 to 1.0) for each category
        self.factors = {
            "mass":   0.0,
            "goal":   0.0,
            "reward": 0.0
        }
        
        # Total control steps to reach 1.0 (1,000,000 = ~4.6 hours at 60Hz)
        self.steps_to_full = {
            "mass":   2_000_000,  # Slowest: take time to introduce heavy weights
            "goal":   1_000_000,  # Medium: push height/offset after basic lift is learned
            "reward": 500_000,    # Fastest: enable penalties/smoothness early
        }
        
        # Internal step counter
        self.total_steps = 0

    def update(self):
        """Called once per env step to progress all factors."""
        self.total_steps += 1
        
        for key in self.factors:
            # Linear growth: progress = current_steps / max_steps
            progress = self.total_steps / self.steps_to_full[key]
            self.factors[key] = min(1.0, progress)

    def get_factor(self, key: str) -> float:
        return self.factors.get(key, 0.0)
