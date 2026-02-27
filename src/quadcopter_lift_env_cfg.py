from __future__ import annotations
from dataclasses import field

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.envs import DirectMARLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass
from isaaclab_assets import CRAZYFLIE_CFG

NUM_DRONES: int = 4
_ACTION_DIM: int = 4   # [thrust, mx, my, mz]
_OBS_DIM: int = 26     # placeholder
CRATE_SIZE: tuple = (0.4, 0.4, 0.2)   # (x, y, z) in metres

# Drone spawn corner offsets per env (x, y, z)
# DRONE_INIT_POSITIONS = [
#     ( 0.2,  0.2, 1.7),
#     (-0.2,  0.2, 1.7),
#     (-0.2, -0.2, 1.7),
#     ( 0.2, -0.2, 1.7),
# ]
# # Rope attachment points on crate top face (crate-local frame)
# ROPE_ATTACH_OFFSETS = [
#     ( 0.18,  0.18, 0.10),
#     (-0.18,  0.18, 0.10),
#     (-0.18, -0.18, 0.10),
#     ( 0.18, -0.18, 0.10),
# ]
def _make_drone_cfg(pos: tuple, id: int) -> ArticulationCfg:
    ox, oy, oz = pos
    return CRAZYFLIE_CFG.replace(
        prim_path=f"/World/envs/env_.*/drone_{id}",
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(ox, oy, oz),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )


@configclass
class CoopLiftEnvCfg(DirectMARLEnvCfg):

    decimation: int         = 10
    episode_length_s: float = 10.0

    possible_agents: list    = [f"drone_{i}" for i in range(NUM_DRONES)]
    action_spaces: dict      = {f"drone_{i}": _ACTION_DIM for i in range(NUM_DRONES)}
    observation_spaces: dict = {f"drone_{i}": _OBS_DIM    for i in range(NUM_DRONES)}
    state_space: int         = -1

    sim: SimulationCfg = SimulationCfg(
        dt=1 / 600,
        render_interval=decimation,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )

    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=4,
        env_spacing=4.0,
        replicate_physics=True,
    )

    crate: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/crate",
        spawn=sim_utils.CuboidCfg(
            size=CRATE_SIZE,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=False),
            mass_props=sim_utils.MassPropertiesCfg(mass=2.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.6, 0.4, 0.1)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.1),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )

    # Physics
    thrust_to_weight: float = 17.5
    moment_scale: float     = 0.01

    # Rewards
    rew_scale_alive:        float =  1.0
    rew_scale_terminated:   float = -2.0
    rew_scale_crate_height: float =  5.0
    rew_scale_goal_dist:    float = 10.0
    rew_scale_lin_vel:      float = -0.05
    rew_scale_ang_vel:      float = -0.01

    # Termination
    max_drone_height: float = 5.0
    min_drone_height: float = 0.05
    goal_pos: tuple         = (0.0, 0.0, 1.5)

    crate_size = CRATE_SIZE
    crate_mass_range: tuple = (0.8, 1.0)

    thrust_delta_scale: float = 0.05
    torque_delta_scale: float = 0.05

    drone_collision_radius: float = 0.15
    drone_crate_radius:     float = 0.25
    max_crate_tilt:         float = 0.524

    # ── Set rope_length here. Everything else is derived in __post_init__ ──
    rope_length: float = 0.5
    rope_length_tolerance: float = 0.05  # ±5cm around rope_length
    reset_grace_steps: int = 120  # ~2s at 60Hz control (no termination after reset)

    # Placeholders — populated by __post_init__, do NOT set manually
    rope_max_distance:    float = 0.0
    DRONE_INIT_POSITIONS: list  = field(default_factory=list)
    ROPE_ATTACH_OFFSETS:  list  = field(default_factory=list)

    drone_0: ArticulationCfg = None  # type: ignore
    drone_1: ArticulationCfg = None  # type: ignore
    drone_2: ArticulationCfg = None  # type: ignore
    drone_3: ArticulationCfg = None  # type: ignore

    def __post_init__(self):
        # ── Fixed crate geometry ──────────────────────────────────────────
        crate_centre_z = self.crate.init_state.pos[2]   # 0.1
        half_z  = self.crate.spawn.size[2] / 2          # 0.1
        half_xy = self.crate.spawn.size[0] / 2          # 0.2

        # ── Account for rope attachment offsets when positioning drones ────
        # Crate attachment is at +half_z from crate center (top face)
        # Drone attachment is at -0.02 from drone center (2cm below)
        # So drone must be higher by 0.02m to maintain rope_length distance
        drone_attachment_offset = 0.02
        drone_z = crate_centre_z + half_z + self.rope_length + drone_attachment_offset

        # ── Derived geometry ──────────────────────────────────────────────
        self.DRONE_INIT_POSITIONS = [
            ( half_xy,  half_xy, drone_z),
            (-half_xy,  half_xy, drone_z),
            (-half_xy, -half_xy, drone_z),
            ( half_xy, -half_xy, drone_z),
        ]
        self.ROPE_ATTACH_OFFSETS = [
            ( half_xy,  half_xy, half_z),
            (-half_xy,  half_xy, half_z),
            (-half_xy, -half_xy, half_z),
            ( half_xy, -half_xy, half_z),
        ]
        # PhysX only supports max distance on D6 joints, not min
        # So rope_max_distance is the hard limit the rope can't exceed
        self.rope_max_distance = self.rope_length

        # ── Rebuild ArticulationCfgs with correct spawn heights ───────────
        self.drone_0 = _make_drone_cfg(self.DRONE_INIT_POSITIONS[0], 0)
        self.drone_1 = _make_drone_cfg(self.DRONE_INIT_POSITIONS[1], 1)
        self.drone_2 = _make_drone_cfg(self.DRONE_INIT_POSITIONS[2], 2)
        self.drone_3 = _make_drone_cfg(self.DRONE_INIT_POSITIONS[3], 3)
