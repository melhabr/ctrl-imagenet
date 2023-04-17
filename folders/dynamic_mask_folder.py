import torchvision.datasets as datasets
import os.path as p
from typing import Optional, Callable, Any, Tuple

from PIL import Image

def pil_loader_with_id(filepath: str) -> Image.Image:
    with open(filepath, "rb") as f:
        img = Image.open(f)
        return (p.splitext(p.basename(filepath))[0], img.convert("RGB"))

class DynamicMaskFolder(datasets.ImageFolder):

    def __init__(self, root: str, mask: dict[str, dict[str, bool]], transform: Optional[Callable] = None):

        self.mask = mask
        super().__init__(
            root,
            loader=pil_loader_with_id,
            transform=transform
        )

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        path, target = self.samples[index]
        id, sample = self.loader(path)

        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        if id in self.mask:
            target = -1

        return sample, target
