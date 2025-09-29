import torch

from dataclasses import MISSING

from isaaclab.utils import configclass
from isaaclab.assets import AssetBaseCfg

from ..template.bi_arm_env_cfg import (
    BiArmTaskSceneCfg,
    BiArmTaskEnvCfg,
    BiArmObservationsCfg,
    BiArmTerminationsCfg,
)
from ...utils.general_assets import parse_usd_and_create_subassets
from ...assets.scenes.toyroom import LIGHTWHEEL_TOYROOM_CFG, LIGHTWHEEL_TOYROOM_USD_PATH


@configclass
class CustomBiArmSceneCfg(BiArmTaskSceneCfg):

    scene: AssetBaseCfg = LIGHTWHEEL_TOYROOM_CFG.replace(prim_path="{ENV_REGEX_NS}/Scene")


@configclass
class CustomBiArmEnvCfg(BiArmTaskEnvCfg):
    """Bimanual environment with configurable arm gap and camera positions.

    Runtime-tunable attributes (can be set by caller before instantiation):
    - arm_gap: float distance along X between right and left arm bases
    - center_pos: tuple[float,float,float] world position for midpoint between arm bases
    - top_cam_pos: optional tuple position for top camera in world space
    - top_cam_quat_ros: optional quaternion (wxyz) for top camera in ROS convention
    """

    scene: CustomBiArmSceneCfg = CustomBiArmSceneCfg(env_spacing=8.0)

    observations: BiArmObservationsCfg = BiArmObservationsCfg()

    terminations: BiArmTerminationsCfg = BiArmTerminationsCfg()

    # User-tunable fields
    arm_gap: float | None = None
    center_pos: tuple[float, float, float] | None = None
    top_cam_pos: tuple[float, float, float] | None = None
    top_cam_quat_ros: tuple[float, float, float, float] | None = None

    def __post_init__(self) -> None:
        super().__post_init__()

        # default viewer params
        self.viewer.eye = (-1.5, -2.0, 1.5)
        self.viewer.lookat = (-0.2, -0.3, 0.5)

        # default arm bases if not overridden
        if self.center_pos is None:
            self.center_pos = (-0.375, -0.2, 0.43)
        if self.arm_gap is None:
            self.arm_gap = 0.45

        cx, cy, cz = self.center_pos
        gap = self.arm_gap

        # place left/right bases along X with specified gap
        self.scene.left_arm.init_state.pos = (cx - gap / 2.0, cy, cz)
        self.scene.right_arm.init_state.pos = (cx + gap / 2.0, cy, cz)

        # optional top camera world placement
        if self.top_cam_pos is not None:
            # When setting world poses later in env build, use events util if needed.
            # Here we override the sensor offset so the spawned camera starts near desired world pose.
            # Keep orientation if not provided.
            if self.top_cam_quat_ros is None:
                self.top_cam_quat_ros = (0.1650476, -0.9862856, 0.0, 0.0)
            self.scene.top.offset.pos = self.top_cam_pos
            self.scene.top.offset.rot = self.top_cam_quat_ros

        # load scene USD
        parse_usd_and_create_subassets(LIGHTWHEEL_TOYROOM_USD_PATH, self)



