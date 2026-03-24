# Mechanics (always available)
# Data (always available)
from hyper_surrogate.data.dataset import MaterialDataset, Normalizer, create_datasets
from hyper_surrogate.data.deformation import DeformationGenerator
from hyper_surrogate.mechanics.kinematics import Kinematics
from hyper_surrogate.mechanics.materials import (
    Demiray,
    Fung,
    GasserOgdenHolzapfel,
    Guccione,
    HolzapfelOgden,
    HolzapfelOgdenBiaxial,
    Material,
    MooneyRivlin,
    NeoHooke,
    Ogden,
    Yeoh,
)
from hyper_surrogate.mechanics.symbolic import SymbolicHandler
from hyper_surrogate.reporting.reporter import Reporter

# ML (requires torch)
try:
    from hyper_surrogate.models.cann import CANN  # noqa: F401
    from hyper_surrogate.models.icnn import ICNN  # noqa: F401
    from hyper_surrogate.models.mlp import MLP  # noqa: F401
    from hyper_surrogate.models.polyconvex import PolyconvexICNN  # noqa: F401
    from hyper_surrogate.training.losses import (  # noqa: F401
        EnergyStressLoss,
        SparseLoss,
        StressLoss,
        StressTangentLoss,
    )
    from hyper_surrogate.training.trainer import Trainer, TrainingResult  # noqa: F401
except ImportError:
    pass

# Export (requires torch for weights extraction)
try:
    from hyper_surrogate.export.fortran.emitter import FortranEmitter  # noqa: F401
    from hyper_surrogate.export.fortran.hybrid import HybridUMATEmitter  # noqa: F401
    from hyper_surrogate.export.weights import ExportedModel, extract_weights  # noqa: F401
except ImportError:
    pass

__all__ = [
    "DeformationGenerator",
    "Demiray",
    "Fung",
    "GasserOgdenHolzapfel",
    "Guccione",
    "HolzapfelOgden",
    "HolzapfelOgdenBiaxial",
    "Kinematics",
    "Material",
    "MaterialDataset",
    "MooneyRivlin",
    "NeoHooke",
    "Normalizer",
    "Ogden",
    "Reporter",
    "SymbolicHandler",
    "Yeoh",
    "create_datasets",
]
