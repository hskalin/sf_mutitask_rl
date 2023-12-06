from .pointMass3D import PointMass3D
from .pointMass2D import PointMass2D
from .pointMass2D_rand import PointMass2DRand
from .pointer2D import Pointer
from .pointer2D_rand import PointerRand
from .ant import Ant
from .humanoid import Humanoid
from .cartpole import Cartpole
from .quadcopter import Quadcopter
from .arm import Arm
from .blimp import Blimp


# Mappings from strings to environments
env_map = {
    "PointMass3D": PointMass3D,
    "PointMass2D": PointMass2D,
    "PointMass2DRand": PointMass2DRand,
    "Pointer2D": Pointer,
    "Pointer2DRand": PointerRand,
    "Ant": Ant,
    "Humanoid": Humanoid,
    "Cartpole": Cartpole,
    "Quadcopter": Quadcopter,
    "Arm": Arm,
    "Blimp": Blimp,
}
