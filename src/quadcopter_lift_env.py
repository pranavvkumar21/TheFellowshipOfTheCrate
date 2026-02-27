from __future__ import annotations

from collections.abc import Sequence

import torch
from pxr import UsdPhysics, UsdGeom, Gf, Sdf
import omni.usd
from pxr import PhysxSchema
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import DirectMARLEnv
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane

from Managers import ObservationManager, ActionManager, TerminationManager, RewardManager, CommandManager
from quadcopter_lift_env_cfg import (
    CoopLiftEnvCfg, NUM_DRONES,
    
)



def _create_rope_d6(
    stage,
    joint_path: str,
    body0_path: str,
    body1_path: str,
    local_pos0: tuple,
    local_pos1: tuple,
    max_dist: float,
    rope_damping: float = 10.0,
    rope_stiffness: float = 1.0,
) -> None:
    
    # --- Base D6 joint (UsdPhysics.Joint = D6 in PhysX) ---
    joint      = UsdPhysics.Joint.Define(stage, joint_path)
    joint_prim = joint.GetPrim()

    # --- Wire bodies ---
    joint.CreateBody0Rel().SetTargets([Sdf.Path(body0_path)])
    joint.CreateBody1Rel().SetTargets([Sdf.Path(body1_path)])

    # --- Local attachment frames ---
    joint.CreateLocalPos0Attr().Set(Gf.Vec3f(*local_pos0))
    joint.CreateLocalPos1Attr().Set(Gf.Vec3f(*local_pos1))
    joint.CreateLocalRot0Attr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    joint.CreateLocalRot1Attr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))

    # --- Enable joint, no collision through anchor ---
    joint.CreateJointEnabledAttr().Set(True)
    joint.CreateCollisionEnabledAttr().Set(False)


    dist_limit = UsdPhysics.LimitAPI.Apply(joint_prim, UsdPhysics.Tokens.distance)
    dist_limit.CreateLowAttr().Set(0.0)  # Must be 0, not max_dist - PhysX requirement
    dist_limit.CreateHighAttr().Set(max_dist)

    cone_limit = 45.0   # degrees
    for axis in ["rotX", "rotY", "rotZ"]:
        limit = UsdPhysics.LimitAPI.Apply(joint_prim, axis)
        limit.CreateLowAttr().Set(-cone_limit)
        limit.CreateHighAttr().Set( cone_limit)

    # --- Add damping drives on rotY and rotZ to stop oscillation ---
    # This is the pattern from the Isaac Sim rope demo [web:69]
    for axis in ["rotY", "rotZ"]:
        drive = UsdPhysics.DriveAPI.Apply(joint_prim, axis)
        drive.CreateTypeAttr().Set("force")
        drive.CreateDampingAttr().Set(rope_damping)
        drive.CreateStiffnessAttr().Set(rope_stiffness)

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

class CoopLiftEnv(DirectMARLEnv):
    cfg: CoopLiftEnvCfg

    # -----------------------------------------------------------------------
    def __init__(self, cfg: CoopLiftEnvCfg, render_mode=None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        

        # Sanity check obs_dim matches config
        # assert self._obs_manager.obs_dim == self.cfg.observation_spaces["drone_0"], (
        #         f"obs_dim mismatch: manager={self._obs_manager.obs_dim} "
        #         f"cfg={self.cfg.observation_spaces['drone_0']}"
        # )
        

        # Body id for force application — same "body" link on every Crazyflie
        # Each Articulation has one instance per env, so find_bodies returns
        # a list with one index.
        self._body_ids = {
            f"drone_{i}": self._drones[f"drone_{i}"].find_bodies("body")[0]
            for i in range(NUM_DRONES)
        }
        # Stores current crate dimensions per env for rope offset computation
        self._crate_size = torch.zeros(self.num_envs, 3, device=self.device)
        self._crate_size[:] = torch.tensor(self.cfg.crate_size, device=self.device)
        
        # Tracks current crate mass per env (for obs/reward use)
        self._crate_mass = torch.ones(self.num_envs, device=self.device) * 2.0

        # Drone hover weight
        drone_mass         = self._drones["drone_0"].root_physx_view.get_masses()[0].sum()
        gravity_mag        = torch.tensor(self.sim.cfg.gravity, device=self.device).norm()
        self._drone_weight = (drone_mass * gravity_mag).item()

        # Buffers
        self._actions = torch.zeros(
            self.num_envs, NUM_DRONES, self.cfg.action_spaces["drone_0"],
            device=self.device,
        )
        self._thrust = {
            f"drone_{i}": torch.zeros(self.num_envs, 1, 3, device=self.device)
            for i in range(NUM_DRONES)
        }
        self._moment = {
            f"drone_{i}": torch.zeros(self.num_envs, 1, 3, device=self.device)
            for i in range(NUM_DRONES)
        }


        self._obs_manager = ObservationManager(self)
        self._action_manager = ActionManager(self)
        self._termination_manager = TerminationManager(self)
        self._reward_manager = RewardManager(self)
        self._command_manager = CommandManager(self)

        # Goal
        self._goal_pos_w = torch.tensor(
            self.cfg.goal_pos, device=self.device
        ).unsqueeze(0).expand(self.num_envs, -1).clone()

        # Episode accumulators
        self._episode_sums = {
            k: torch.zeros(self.num_envs, device=self.device)
            for k in ["alive", "crate_height", "goal_dist", "lin_vel", "ang_vel"]
        }

    # -----------------------------------------------------------------------
    def _setup_scene(self):
        # --- One Articulation per drone, registered separately ---
        self._drones: dict[str, Articulation] = {}
        for i in range(NUM_DRONES):
            name = f"drone_{i}"
            drone = Articulation(getattr(self.cfg, name))
            self._drones[name] = drone
            self.scene.articulations[name] = drone

        # --- Crate ---
        self._crate = RigidObject(self.cfg.crate)
        self.scene.rigid_objects["crate"] = self._crate

        # --- Ground ---
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())

        # --- Clone BEFORE joints ---
        self.scene.clone_environments(copy_from_source=False)

        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[])

        # --- Lights ---
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

        # --- Rope joints will be created on first reset when bodies exist ---
        self._joints_created = False
    # -----------------------------------------------------------------------
    def _print_prim_tree(self, root_path: str, max_depth: int = 4):
        """Debug helper — call once to see actual prim tree after cloning."""
        stage = omni.usd.get_context().get_stage()
        from pxr import Usd
        root = stage.GetPrimAtPath(root_path)
        if not root.IsValid():
            print(f"[DEBUG] Prim not found: {root_path}")
            return
        def _walk(prim, depth=0):
            if depth > max_depth:
                return
            apis = list(prim.GetAppliedSchemas())

            print(f"{'  ' * depth}{prim.GetPath()}  [{prim.GetTypeName()}]  apis={apis}")
            for child in prim.GetChildren():
                _walk(child, depth + 1)
        _walk(root)
    def _get_spawn_geometry(self, env_idx: int):
        """
        Returns per-drone (drone_spawn_local, crate_attach_local) tuples
        computed from the actual randomised crate size for this env.
        Drone spawns directly above the crate corner at exactly rope_length height.
        """
        sx = self._crate_size[env_idx, 0].item() / 2   # half-extents
        sy = self._crate_size[env_idx, 1].item() / 2
        sz = self._crate_size[env_idx, 2].item() / 2   # top face offset from crate centre

        # corners in crate-local frame (crate centre = origin)
        corners = [
            ( sx,  sy),
            (-sx,  sy),
            (-sx, -sy),
            ( sx, -sy),
        ]

        rope_len = self.cfg.rope_length
        crate_centre_z = self.cfg.crate.init_state.pos[2]   # e.g. 0.1

        results = []
        for (cx, cy) in corners:
            # Crate attach: top face corner (crate-local z = +sz)
            crate_attach = (cx, cy, sz)

            # Drone spawn: directly above crate attach, rope_len higher
            # World z = crate_centre_z + sz + rope_len
            drone_spawn_z = crate_centre_z + sz + rope_len
            drone_spawn = (cx, cy, drone_spawn_z)   # same x,y as corner

            results.append((drone_spawn, crate_attach))

        return results


    def _find_rigid_body_path(self, root_path: str) -> str | None:
        """
        Walk the prim subtree rooted at root_path and return the path of the
        first prim that has UsdPhysics.RigidBodyAPI applied.
        Returns None if not found.
        """
        stage = omni.usd.get_context().get_stage()
        from pxr import Usd
        root = stage.GetPrimAtPath(root_path)
        if not root.IsValid():
            return None
        for prim in Usd.PrimRange(root):
            if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                return str(prim.GetPath())
        return None


    def _create_rope_joints(self):
        """
        Discover actual rigid body prim paths at runtime, then wire D6 joints.
        Joints live under /World/Joints/ — outside all articulation trees.
        """
        stage = omni.usd.get_context().get_stage()

        # --- Debug: print env_0 tree once so you can verify paths ---
        print("\n[DEBUG] Prim tree for env_0/drone_0:")
        self._print_prim_tree(f"/World/envs/env_0/drone_0", max_depth=5)
        print("\n[DEBUG] Prim tree for env_0/crate:")
        self._print_prim_tree(f"/World/envs/env_0/crate", max_depth=3)

        # Ensure parent Xform exists
        if not stage.GetPrimAtPath("/World/Joints"):
            UsdGeom.Xform.Define(stage, "/World/Joints")

        for env_idx in range(self.num_envs):
            env_joints_path = f"/World/Joints/env_{env_idx}"
            if not stage.GetPrimAtPath(env_joints_path):
                UsdGeom.Xform.Define(stage, env_joints_path)

            # --- Discover crate rigid body path ---
            crate_rb_path = self._find_rigid_body_path(
                f"/World/envs/env_{env_idx}/crate"
            )
            if crate_rb_path is None:
                # Fallback: crate itself is the rigid body root
                crate_rb_path = f"/World/envs/env_{env_idx}/crate"
                print(f"[WARN] No RigidBodyAPI found under crate for env_{env_idx}, "
                    f"using: {crate_rb_path}")

            for i in range(NUM_DRONES):
                # --- Discover drone rigid body path ---
                drone_rb_path = self._find_rigid_body_path(
                    f"/World/envs/env_{env_idx}/drone_{i}"
                )
                if drone_rb_path is None:
                    print(f"[WARN] No RigidBodyAPI found under drone_{i} for env_{env_idx}, "
                        f"skipping rope joint.")
                    continue

                joint_path = f"/World/Joints/env_{env_idx}/rope_drone_{i}"

                _create_rope_d6(
                    stage        = stage,
                    joint_path   = joint_path,
                    body0_path   = drone_rb_path,
                    body1_path   = crate_rb_path,
                    local_pos0   = (0.0, 0.0, -0.02),
                    local_pos1   = self.cfg.ROPE_ATTACH_OFFSETS[i],
                    max_dist     = self.cfg.rope_max_distance,
                )

            if env_idx == 0:
                print(f"\n[DEBUG] env_0 rope joints wired:")
                print(f"  crate body  → {crate_rb_path}")
                for i in range(NUM_DRONES):
                    rb = self._find_rigid_body_path(
                        f"/World/envs/env_0/drone_{i}"
                    )
                    print(f"  drone_{i} body → {rb}")
        
    # _pre_physics_step:
    def _pre_physics_step(self, actions: dict[str, torch.Tensor]) -> None:
        self._action_manager.step(actions)

    # _apply_action:
    def _apply_action(self) -> None:
        forces, torques = self._action_manager.get_forces_and_torques()
        for name in self.cfg.possible_agents:
            self._drones[name].set_external_force_and_torque(
                forces   = forces[name],
                torques  = torques[name],
                body_ids = self._body_ids[name],
            )

    # -----------------------------------------------------------------------
    def _get_observations(self) -> dict[str, torch.Tensor]:
        return self._obs_manager.compute()

    # -----------------------------------------------------------------------
    def _get_rewards(self) -> dict[str, torch.Tensor]:
        terminated = self._termination_manager.compute()[0]["drone_0"]
        timed_out  = self._termination_manager.compute()[1]["drone_0"]
        return self._reward_manager.compute(terminated, timed_out) 
    # -----------------------------------------------------------------------
    def _get_dones(self) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        # Termination manager already checks all conditions including height
        return self._termination_manager.compute() 

    # -----------------------------------------------------------------------
    def _find_prim_by_type(self, root_prim, type_name: str):
        """Walk subtree and return first prim matching type_name."""
        from pxr import Usd
        for prim in Usd.PrimRange(root_prim):
            if prim.GetTypeName() == type_name:
                return prim
        return None

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self._drones["drone_0"]._ALL_INDICES
        super()._reset_idx(env_ids)

        #reset action manager state for these envs
        self._action_manager.reset(env_ids)
        self._reward_manager.reset(env_ids)

        # Create joints ONCE on first reset (after super() initializes physics bodies)
        if not self._joints_created:
            self._create_rope_joints()
            self._joints_created = True

        n = len(env_ids)

        # ------------------------------------------------------------------
        # Randomise crate mass only — geometry cannot be changed at runtime
        # without invalidating PhysX's tensor view. Mass is safe because it
        # is written via the physics tensor API, not by modifying USD prims.
        # ------------------------------------------------------------------
        mass_min, mass_max = self.cfg.crate_mass_range
        new_masses = torch.empty(n, device="cpu").uniform_(mass_min, mass_max)

        # get_masses / set_masses operate on CPU tensors
        masses = self._crate.root_physx_view.get_masses()   # shape (num_envs, 1)
        masses[env_ids, 0] = new_masses
        self._crate.root_physx_view.set_masses(masses, torch.arange(self.num_envs, device="cpu"))

        # Store for obs use (broadcast to match num_envs dim)
        self._crate_mass[env_ids] = new_masses.to(self.device)

        # ------------------------------------------------------------------
        # Reset crate pose — fixed geometry, explicit position
        # ------------------------------------------------------------------
        crate_state = self._crate.data.default_root_state[env_ids].clone()
        crate_state[:, 0:3] = 0.0  # Reset to origin in local frame
        crate_state[:, 2] = self.cfg.crate.init_state.pos[2]  # Fixed z = 0.1
        crate_state[:, :3] += self.scene.env_origins[env_ids]
        self._crate.write_root_state_to_sim(crate_state, env_ids)
        
        if 0 in env_ids:
            print(f"\n[DEBUG RESET] Crate position AFTER reset: {crate_state[0, :3]}")
            print(f"  Crate Z should be: {self.cfg.crate.init_state.pos[2]} + env_origin_z")

        # ------------------------------------------------------------------
        # Reset each drone to its spawn corner — fully vectorised
        # ------------------------------------------------------------------

        # Crate half-extents per env: shape (num_envs, 3) → (|env_ids|, 3)
        half = self._crate_size[env_ids] / 2   # (n, 3)

        # Crate centre Z per env (same for all, from config)
        crate_z = self.cfg.crate.init_state.pos[2]   # scalar

        # Corner signs per drone: shape (4, 2) → broadcast with (n, 2)
        corner_signs = torch.tensor([
            [ 1.,  1.],
            [-1.,  1.],
            [-1., -1.],
            [ 1., -1.],
        ], device=self.device)  # (NUM_DRONES, 2)

        for i in range(NUM_DRONES):
            name = f"drone_{i}"
            drone = self._drones[name]

            sx = corner_signs[i, 0] * half[:, 0]   # (n,)
            sy = corner_signs[i, 1] * half[:, 1]   # (n,)
            sz = half[:, 2]                         # (n,)

            # Account for drone attachment offset (rope attaches 2cm below drone center)
            drone_attachment_offset = 0.02
            spawn_z = crate_z + sz + self.cfg.rope_length + drone_attachment_offset  # (n,)

            state = drone.data.default_root_state[env_ids].clone()  # (n, 13)

            # Set position in local frame, then add env origins
            state[:, 0] = sx
            state[:, 1] = sy
            state[:, 2] = spawn_z
            state[:, :3] += self.scene.env_origins[env_ids]
            
            # Clear velocities
            state[:, 7:] = 0.0

            drone.write_root_pose_to_sim(state[:, :7], env_ids=env_ids)
            drone.write_root_velocity_to_sim(state[:, 7:], env_ids=env_ids)
            
            if 0 in env_ids and i == 0:
                idx = (env_ids == 0).nonzero(as_tuple=True)[0][0]
                print(f"[DEBUG RESET] Drone_0 position AFTER reset: {state[idx, :3]}")
                print(f"  Expected Z: {self.cfg.crate.init_state.pos[2]} + {self.cfg.crate_size[2]/2} + {self.cfg.rope_length} + 0.02 = {spawn_z[idx]}")
                crate_attach_z = self.cfg.crate.init_state.pos[2] + self.cfg.crate_size[2]/2
                drone_attach_z = state[idx, 2] - 0.02
                rope_dist = drone_attach_z - crate_attach_z
                print(f"  Rope attachment distance: {rope_dist:.3f}m (should be {self.cfg.rope_length}m)\n")

        # ------------------------------------------------------------------
        # Clear buffers
        # ------------------------------------------------------------------
        self._actions[env_ids] = 0.0
        for name in self.cfg.possible_agents:
            self._thrust[name][env_ids] = 0.0
            self._moment[name][env_ids] = 0.0
        for k in self._episode_sums:
            self._episode_sums[k][env_ids] = 0.0

    def _get_states(self):
        # state_space = -1 in cfg, so this is never called
        # but required by DirectMARLEnv abstract interface
        return None

