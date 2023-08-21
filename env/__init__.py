from .pointMass3D import PointMass3D
from .pointMass2D import PointMass2D
from .pointer2D import Pointer
from .ant import Ant
from .humanoid import Humanoid


# Mappings from strings to environments
env_map = {
    "PointMass3D": PointMass3D,
    "PointMass2D": PointMass2D,
    "Pointer2D": Pointer,
    "Ant": Ant,
    "Humanoid": Humanoid,
}
