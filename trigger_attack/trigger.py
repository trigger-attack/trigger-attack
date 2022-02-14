from dataclasses import dataclass
from typing import Any
import torch


@dataclass
class Trigger:
    input_ids: torch.Tensor
    location: str
    source_labels: Any
