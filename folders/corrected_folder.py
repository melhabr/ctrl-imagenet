import torchvision.datasets as datasets
import os.path as p
from typing import Optional, Callable, Any, Tuple

class CorrectedFolder(datasets.ImageFolder):

    def __init__(self, 
                 root: str,
                 remove: set = [], 
                 replace: dict = {}, 
                 transform: Optional[Callable] = None):

        is_valid_file = None
        if remove:
            def is_valid_file(filepath):
                filename = p.splitext(p.basename(filepath))[0]
                return filename not in remove
        
        self.replace = replace
        super().__init__(
            root,
            transform=transform,
            is_valid_file=is_valid_file
        )

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        path, target = self.samples[index]
        filename = p.splitext(p.basename(path))[0]
        sample = self.loader(path)

        if self.transform is not None:
            sample = self.transform(sample)

        if filename in self.replace:
            target = self.replace[filename]

        return sample, target
