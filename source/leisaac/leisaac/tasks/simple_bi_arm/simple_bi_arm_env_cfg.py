import torch

from isaaclab.assets import AssetBaseCfg
from isaaclab.utils import configclass

from leisaac.assets.scenes.simple import TABLE_WITH_CUBE_CFG, TABLE_WITH_CUBE_USD_PATH
from leisaac.utils.general_assets import parse_usd_and_create_subassets

from ..template import BiArmTaskSceneCfg, BiArmTaskEnvCfg, BiArmObservationsCfg, BiArmTerminationsCfg


@configclass
class SimpleBiArmSceneCfg(BiArmTaskSceneCfg):
    """Scene configuration for the simple bi-arm task using table with cube."""

    scene: AssetBaseCfg = TABLE_WITH_CUBE_CFG.replace(prim_path="{ENV_REGEX_NS}/Scene")


@configclass
class SimpleBiArmEnvCfg(BiArmTaskEnvCfg):
    """Configuration for the simple bi-arm environment."""

    scene: SimpleBiArmSceneCfg = SimpleBiArmSceneCfg(env_spacing=8.0)

    observations: BiArmObservationsCfg = BiArmObservationsCfg()

    terminations: BiArmTerminationsCfg = BiArmTerminationsCfg()

    def __post_init__(self) -> None:
        super().__post_init__()

        # Adjust robot positions for bi-arm setup
        self.scene.left_arm.init_state.pos = (0.2, -0.64, 0.01)
        self.scene.right_arm.init_state.pos = (0.5, -0.64, 0.01)

        parse_usd_and_create_subassets(TABLE_WITH_CUBE_USD_PATH, self)
