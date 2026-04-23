"""
Neuron device detection and management.

AWS Neuron exposes its accelerator cores through a custom device backend
integrated into PyTorch via the ``torch-neuronx`` package.  This module
wraps the detection logic so the rest of the library can gracefully fall
back to CPU when running on non-Neuron hardware (CI, developer laptops, etc.).

Hardware tiers
--------------
Inf1 (Inferentia 1)
  • First-generation Inferentia chip; 4 NeuronCores-v1 per chip
  • Supports float16 inference only (no training)
  • Accessed via the older ``torch.neuron`` (PyTorch 1.x) SDK

Inf2 (Inferentia 2)
  • Uses NeuronCores-v2, supports bfloat16 + float32
  • Accessed via ``torch-neuronx`` (PyTorch 2.x) — same SDK as Trn1

Trn1 (Trainium 1)
  • Training-class chip; 2 NeuronCores-v2 per chip, 32 GiB HBM each
  • Supports float32, bfloat16, tf32 + automatic mixed precision
  • Backward-pass (autograd) compiled together with the forward graph

All Trn1/Inf2 hardware uses the same ``torch-neuronx`` package and the
device string ``'xla'`` via PyTorch/XLA (the Neuron runtime implements an
XLA backend).
"""

from __future__ import annotations

import os
from enum import Enum, auto
from typing import Optional

import torch


class NeuronHardware(Enum):
    """Detected Neuron hardware generation."""
    INF1    = auto()   # Inferentia 1 (torch.neuron  / NeuronCore-v1)
    INF2    = auto()   # Inferentia 2 (torch-neuronx / NeuronCore-v2)
    TRN1    = auto()   # Trainium 1   (torch-neuronx / NeuronCore-v2)
    UNKNOWN = auto()   # Neuron SDK present but chip generation unknown
    NONE    = auto()   # No Neuron hardware detected


def _detect_hardware() -> NeuronHardware:
    """Attempt to detect the Neuron hardware generation from the environment."""
    # ``NEURON_RT_NUM_CORES`` is set by the Neuron runtime on Trainium/Inferentia
    # instances.  The instance type is available via the EC2 metadata service or
    # the ``AWS_DEFAULT_REGION`` + instance-type combination.
    instance_type = os.environ.get("INSTANCE_TYPE", "").lower()
    if "trn1" in instance_type:
        return NeuronHardware.TRN1
    if "inf2" in instance_type:
        return NeuronHardware.INF2
    if "inf1" in instance_type:
        return NeuronHardware.INF1

    # Fall back to probing the torch-neuronx package
    try:
        import torch_neuronx  # noqa: F401
        # If torch-neuronx is present we're almost certainly on Trn1/Inf2
        return NeuronHardware.UNKNOWN
    except ImportError:
        pass

    try:
        import torch.neuron  # noqa: F401 — older Inf1 SDK
        return NeuronHardware.INF1
    except ImportError:
        pass

    return NeuronHardware.NONE


class NeuronDevice:
    """Thin wrapper around Neuron device state.

    Attributes:
        hardware: Detected :class:`NeuronHardware` generation.
        device:   PyTorch device string (``'xla'`` on Trn1/Inf2,
                  ``'cpu'`` when no Neuron hardware is available).
        num_cores: Number of NeuronCores visible to this process.
    """

    def __init__(self) -> None:
        self.hardware = _detect_hardware()
        self._torch_xla_available = False

        if self.hardware in (
            NeuronHardware.TRN1,
            NeuronHardware.INF2,
            NeuronHardware.UNKNOWN,
        ):
            try:
                import torch_xla.core.xla_model as xm  # noqa: F401
                self._torch_xla_available = True
            except ImportError:
                pass

        self.num_cores = int(os.environ.get("NEURON_RT_NUM_CORES", "1"))

    @property
    def device(self) -> torch.device:
        if self._torch_xla_available:
            import torch_xla.core.xla_model as xm
            return xm.xla_device()
        # Graceful fallback to CPU
        return torch.device("cpu")

    @property
    def is_neuron(self) -> bool:
        return self.hardware is not NeuronHardware.NONE

    def synchronize(self) -> None:
        """Block until all pending Neuron computations have completed."""
        if self._torch_xla_available:
            import torch_xla.core.xla_model as xm
            xm.mark_step()

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"NeuronDevice(hardware={self.hardware.name}, "
            f"device={self.device}, num_cores={self.num_cores})"
        )


# Module-level singleton — created lazily on first access
_DEVICE: Optional[NeuronDevice] = None


def get_device() -> NeuronDevice:
    """Return the process-level :class:`NeuronDevice` singleton."""
    global _DEVICE
    if _DEVICE is None:
        _DEVICE = NeuronDevice()
    return _DEVICE


def is_neuron_available() -> bool:
    """Return True if Neuron hardware or SDK is detected."""
    return get_device().is_neuron
