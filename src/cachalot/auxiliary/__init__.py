"""
Auxiliary inference: a small model beside the target model, off by default.

The first backend is Apple's on-device system model through the Foundation Models
Python SDK (`apple-fm-sdk`), which runs on the Neural Engine in a system service.
Nothing in the target model's load, prefill, decode or sampling path imports this
package; see docs/APPLE-FM-AUXILIARY.md.
"""

from cachalot.auxiliary.config import AuxiliaryConfig, load_aux_config
from cachalot.auxiliary.engine import AuxiliaryEngine, build_provider
from cachalot.auxiliary.errors import (
    AuxBusy,
    AuxCancelled,
    AuxContextOverflow,
    AuxDisabled,
    AuxError,
    AuxInvalidOutput,
    AuxProviderError,
    AuxRefused,
    AuxTimeout,
    AuxUnavailable,
)
from cachalot.auxiliary.provider import (
    AuxiliaryProvider,
    Capabilities,
    GenerationParams,
    GenerationResult,
)

__all__ = [
    "AuxBusy", "AuxCancelled", "AuxContextOverflow", "AuxDisabled", "AuxError", "AuxInvalidOutput",
    "AuxProviderError", "AuxRefused", "AuxTimeout", "AuxUnavailable", "AuxiliaryConfig", "AuxiliaryEngine",
    "AuxiliaryProvider", "Capabilities", "GenerationParams", "GenerationResult", "build_provider",
    "load_aux_config",
]
