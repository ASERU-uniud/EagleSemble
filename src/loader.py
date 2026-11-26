from torch.utils.data import IterableDataset, DataLoader, get_worker_info
from concurrent.futures import ThreadPoolExecutor

from pathlib import Path
from collections import defaultdict
import json
import os
import re
import xml.etree.ElementTree as ET

class MyIterableDataset(IterableDataset):
    def __init__(self, path:str):
        super(MyIterableDataset, self).__init__()
        assert Path(path).exists(), "Path does not exist"
        self.path = path

    def __iter__(self):
        worker_info = get_worker_info()
        xodr_files = [str(xodr_file) for xodr_file in Path(self.path).rglob('*.xodr')]
        json_files = [str(sim) for sim in sorted(Path(self.path).rglob('*.json'))]
        result = defaultdict(dict)
        if worker_info is not None:
            worker_id = worker_info.id
            n_workers = worker_info.num_workers
        else:
            n_workers = 1
            worker_id = 0
        for i in range(worker_id, len(xodr_files), n_workers):
            test_id = extract_xodr_id(xodr_files[i])
            if test_id:
                result[test_id]["xodr"] = xodr_files[i]

        for i in range(worker_id, len(json_files), n_workers):
            test_id = extract_json_id(json_files[i])
            if test_id:
                result[test_id]["json"] = json_files[i]

        yield from result.items()

def extract_xodr_id(filepath):
    for event, elem in ET.iterparse(filepath, events=("end",)):
        if elem.tag == "sdc_test_info" and "test_id" in elem.attrib:
            return filepath, elem.attrib["test_id"]
        elem.clear()
    return None, None


def extract_json_id(filepath):
    json_id_regex = re.compile(r'"test_id"\s*:\s*"([^"]+)"')
    with open(filepath, "r", encoding="utf-8") as f:
        head = f.read(128)
        match = json_id_regex.search(head)
        if match:
            return filepath, match.group(1)

        f.seek(0)
        data = json.load(f)
        return filepath, data.get("test_id")


def index(directory:str):

    indice = defaultdict(dict)
    xodr_files = [str(xodr_file) for xodr_file in Path(directory).rglob('*.xodr')]
    json_files = [str(sim) for sim in sorted(Path(directory).rglob('*.json'))]

    with ThreadPoolExecutor(max_workers=1) as executor:
        for filepath, test_id in executor.map(extract_xodr_id, xodr_files):
            indice[test_id]["xodr"] = filepath
    with ThreadPoolExecutor(max_workers=1) as executor:
        for filepath, test_id in executor.map(extract_json_id, json_files):
            indice[test_id]["json"] = filepath

    return indice



if __name__ == '__main__':
    # ds = MyIterableDataset("D:\\SensoDat\\mt_set_2")
    # result = dict(DataLoader(ds, num_workers=2, collate_fn=lambda batch: batch))
    # print(result)
    # for mtset in Path("D:\\SensoDat").iterdir():
    #     print(f"{mtset}: {len(index(mtset))}")
    # print(len(index("D:\\SensoDat")))
    ...
