import torch
from typing import Optional, Generator, List


class PipelineABC:
    """
    We delete the save_pretrained function.
    and save training function will add in the next repo.
    """
    def __init__(self):
        pass

    @classmethod
    def from_pretrained(cls):
        return cls()
    
    def process(self, *args, **kwds):
        pass
    
    def __call__(self, *args, **kwds):
        pass

    def stream(self, *args, **kwds)-> Generator[torch.Tensor, List[str], None]:
        pass
