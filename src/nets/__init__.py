# /home/ilias/weedsgalore/src/nets/__init__.py
from .deeplabv3plus.modeling import deeplabv3plus_resnet50
from .deeplabv3plus_do.modeling import deeplabv3plus_resnet50_do
from .segdino.modeling import segdino_vitb16, segdino_vits16

__all__ = [
    'deeplabv3plus_resnet50',
    'deeplabv3plus_resnet50_do',
    'segdino_vitb16',
    'segdino_vits16'
]