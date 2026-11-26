# DATA LOADER (SensoDat LOADER)

## 1st IDEA
- Index files by ID {parallel}
  - Load files {parallel}
    - Select [MAYBE store indexes] and Extract Features [MAYBE store Tensors]
      - Split (train/validation/test) and Execute Training
    - Convert to JSON Lines and Store
      - Select and Extract Features [MAYBE store Tensors]
        - Split (train/validation/test) and Execute Training
- Load JSONL {streaming}
  - Select and Extract Features [MAYBE store Tensors]
    - Split (train/validation/test) and Execute Training
- Load Tensors {streaming}
  - Split (train/validation/test) and Execute Training

## 2nd IDEA

- Index files by ID {parallel}
  - Load files {parallel} | (XODR U JSON) ⇒ dict, \
    [MAYBE Convert to JSONL and Store | dict ⇒ JSONL]
    - Select [MAYBE store indexes | dict ⇒ JSON], \
      Extract Features | dict ⇒ Tensor, \
      [MAYBE store Tensors | Tensor ⇒ PT file]
      - Split (train/validation/test), Execute Training
- Load JSONL {streaming} JSONL ⇒ dict \
  Index lines by offset
  - Select [MAYBE store indexes | list ⇒ JSON], \
    Extract Features | dict ⇒ Tensor,\
    [MAYBE store Tensors | Tensor ⇒ PT file]
    - Split (train/validation/test), Execute Training
- Load Tensors {streaming} | PT File ⇒ Tensor
  - Split (train/validation/test), Execute Training

### Possible Code

```python
from torch.utils.data import IterableDataset
from types import NoneType
from typing import Callable

class Loader:
    def __init__(self):
        ...
    def __len__(self):
        ...
    def __iter__(self):
        ...

class JSONLoader:
    def __init__(self, jsonl_path: str = '', loader: Loader|NoneType = None):
        pass

class TensorLoader(IterableDataset):
    def __init__(self, filepath: str = '', loader: Loader|NoneType = None):
        pass
    def __iter__(self):
        pass
    
class SensoDatLoader(Loader):
    def __init__(self, path: str, jsonl_file:str):
        super(Loader, self).__init__(path)
    def retrieve_indexes(self):
        pass
    def export_jsonl(self, jsonl_file:str):
        pass
    def select(self, criterium:Callable[..., bool], store_indexes=False):
        pass
    def compute_features(self, road, trajectory):
        pass
    def export_tensors(self, pytorch_file:str):
        pass

```