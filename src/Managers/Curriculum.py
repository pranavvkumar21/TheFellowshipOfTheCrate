# curriculum_manager.py
import torch


class CurriculumManager:
    """
    Staggered curriculum — rewards activate in order of learning priority:

      Phase 1 (0 → 64k):   proximity only — drones learn not to collide
      Phase 2 (32k → 160k): goal_dist ramps — learn to move crate toward goal
      Phase 3 (128k → 256k): formation_deviation ramps — refine coordination

    Overlap between phases is intentional: smooth gradient transitions.
    
    goal_dist_potential is NOT curriculum-gated here — instead, reduce its
    weight significantly in reward_manager.py (see note below).
    """

    def __init__(self, env):
        self.env = env
        self.total_steps: int = 0

        self.factors: dict[str, float] = {
            "proximity":             0.0,
            "goal_dist":             0.0,
            "formation_deviation":   0.0,
        }

        # Staggered ramp schedule: (start_step, end_step)
        self._schedule: dict[str, tuple[int, int]] = {
            "proximity":           (0,       64_000),   # Phase 1
            "goal_dist":           (32_000,  160_000),  # Phase 2 — starts mid Phase 1
            "formation_deviation": (128_000, 256_000),  # Phase 3 — only after lifting learned
        }

        print("[CurriculumManager] staggered curriculum:")
        for k, (s, e) in self._schedule.items():
            print(f"  {k}: ramps {s} → {e} steps")

    def update(self):
        """Call once per env step (not per physics sub-step)."""
        self.total_steps += 1
        for key, (start, end) in self._schedule.items():
            if self.total_steps < start:
                self.factors[key] = 0.0
            elif self.total_steps >= end:
                self.factors[key] = 1.0
            else:
                self.factors[key] = (self.total_steps - start) / (end - start)

    def get_factor(self, key: str) -> float:
        return self.factors.get(key, 1.0)

    def get_log_dict(self) -> dict[str, float]:
        return {
            f"curriculum/{k}": v for k, v in self.factors.items()
        } | {"curriculum/step": float(self.total_steps)}