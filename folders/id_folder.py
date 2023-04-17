import torchvision.datasets as datasets
import os.path as p
from typing import Optional, Callable, Any, Tuple

from PIL import Image

def pil_loader_with_id(filepath: str) -> Image.Image:
    with open(filepath, "rb") as f:
        img = Image.open(f)
        return (p.splitext(p.basename(filepath))[0], img.convert("RGB"))

class IDFolder(datasets.ImageFolder):

    def __init__(self, root: str, 
                 transform: Optional[Callable] = None,
                 is_valid_file: Optional[Callable[[str], bool]] = None,):

        super().__init__(
            root,
            loader=pil_loader_with_id,
            transform=transform,
            is_valid_file=is_valid_file
        )

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = (sample[0], self.transform(sample[1]))
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target
