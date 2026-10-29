import sys
from xml.etree import ElementTree
from json import load
from pathlib import Path
from typing import Iterable
from tqdm import tqdm
import argparse

#TODO convert to command line tool

class SensoDatPropertyException(Exception):
    def __init__(self, message: str, property_name: str = None):
        super().__init__(message)
        self.message = message

    def __str__(self):
        return self.message


# ---------------------------------------------- Road Properties ---------------------------------------------

def load_road_property(filename: str, road_property):
    if road_property in __road_properties:
        yield from road_property(ElementTree.parse(filename))
    else:
        raise SensoDatPropertyException(f"Not a road property: {road_property.__name__}")

def reference_line(tree: ElementTree.ElementTree):
    for point in tree.getroot()[1][1]:
        yield float(point.attrib['x']), float(point.attrib['y'])

def test_duration(tree: ElementTree.ElementTree):
    yield float(tree.getroot()[0][0].attrib['test_duration'])
# TODO fix test_outcome to include 'NOT_EXECUTED'
def test_outcome(tree: ElementTree.ElementTree):
    yield int(tree.getroot()[0][0].attrib['test_outcome'] == 'PASS')

def test_validity(tree: ElementTree.ElementTree):
    yield int(tree.getroot()[0][0].attrib['is_valid'] == 'True')

def test_id(test: ElementTree.ElementTree):
    yield test.getroot()[0][0].attrib['test_id']



__road_properties = [reference_line, test_duration, test_outcome, test_validity, test_id]

# -------------------------------------------- Behaviour Properties ------------------------------------------

def load_behaviour_property(filename:str, behaviour_property):
    if behaviour_property in __behaviour_properties:
        with open(filename) as fj:
            simdata = load(fj)
        yield from trajectory(simdata)
    else:
        raise SensoDatPropertyException(f"Not a behaviour property: {behaviour_property.__name__}")

def trajectory(simdata:dict):
    for timeframe in simdata['simulation_data']:
        yield timeframe['position'][0], timeframe['position'][1]

def time_stamps(simdata:dict):
    for timeframe in simdata['simulation_data']:
        yield timeframe['time']

def simulation_id(simdata:dict):
    yield simdata['test_id']



__behaviour_properties = [trajectory, time_stamps, simulation_id]

# ------------------------------------------ Higher Level Methods --------------------------------------------

# TODO refactor higher level methods
# TODO load all TCs with simdata whether available or not
# TODO fix loader to handle not executed tests

def load_properties(xodr_file, json_file, *properties):
    tree = []
    simdata = []
    for get_property in properties:
        if get_property in __road_properties:
            if tree == []:
                tree = ElementTree.parse(xodr_file)
            yield get_property(tree)
        elif get_property in __behaviour_properties:
            if simdata == []:
                with open(json_file) as fj:
                    simdata = load(fj)
            yield get_property(simdata)
        else:
            raise SensoDatPropertyException(f"Unknown or not implemented property: {get_property.__name__}")

def load_test_suite(directory:str):
    json_files = [str(p) for p in sorted(Path(directory).rglob('*.json'))]
    xodr_files = []
    for json_file in json_files:
        xodr_files.append(json_file[:-33]+json_file[-24:-14]+'.xodr')
    for xodr_file, json_file in zip(xodr_files, json_files):
        yield load_properties(xodr_file, json_file, reference_line, test_outcome, test_duration, trajectory)

def load_roads(directory:str):
    xodr_files = [str(xodr_file) for xodr_file in Path(directory).rglob('*.xodr')]
    for xodr_file in xodr_files:
        yield load_properties(xodr_file, None, test_id, reference_line, test_outcome, test_duration)

def icst_syntax(roads, outcomes, durations=None):
    json_obj = []
    for index, road in enumerate(roads):
        obj = {
            "_id": {
                "$oid": str(index)
            },
            "meta_data": {
                "test_info": {
                    "test_outcome": "FAIL" if outcomes[index][0] == 0 else "PASS",
                    "test_duration": 0.0 if durations is None else durations[index][0]
                },
                "run_config": {
                    "rf": 1.5,
                    "oob": 0.5,
                    "max_speed": 120
                }
            },
            "road_points": [ {"x": x1, "y" : y1} for x1, y1 in road ]
        }
        json_obj.append(obj)
    return json_obj

def convert(test_suite:Iterable, verbose=True):
    test_suite = list(test_suite)
    listed_test_suite = []
    with tqdm(range(len(test_suite)), unit="tests", mininterval=0, disable=(not verbose)) as bar:
        for i in bar:
            listed_test_suite.append([list(column) for column in test_suite[i]])
    return list(zip(*listed_test_suite))

def __show_missing_sims(sensodat_directory:str):
    for p in Path(sensodat_directory).glob('mt_set_*\\mt_set_*\\*'):
        xodr_files = sorted(p.rglob('*.xodr'))
        json_files = sorted(p.rglob('*.json'))
        if len(xodr_files) != len(json_files):
            print(str(p), len(xodr_files), len(json_files))

if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser(description="SensoDat Loader "
                                                     "loads SensoDat (or any of its subdirectories) and converts it "
                                                     "into JSON format. By default, the result is printed on the "
                                                     "standard output by column, "
                                                     "which means that the resulting JSON object is a list of lists "
                                                     "where each row represents a property, and each column represents"
                                                     " a test case. Otherwise, the -r (--byrow) option can be used to "
                                                     "print the result by row (each row a testcase). "
                                                     "List of available properties: "
                                                     "reference_line (the road points), test_outcome (1 if PASS, 0 if "
                                                     "FAIL, else -1 ),  test_duration, test_validity, test_id, "
                                                     "trajectory (as road points), time_stamps (one for each point in "
                                                     "trajectory). "
                                                     "By default the selected properties are: test_id, reference_line, "
                                                     "test_outcome, test_duration, trajectory.",
                                         )
    arg_parser.add_argument('directory', help="Directory containing SensoDat or parts of it")
    arg_parser.add_argument('-r', '--byrow', action='store_true', help='Flag. If set, prints the resulting'
                                                                       ' JSON object by row (each row a testcase, each '
                                                                       'column a property)')
    arg_parser.add_argument('-i', '--icst-syntax', action='store_true', help="Flag. If set, the output is "
                                                                             "a JSON object with the same structure as "
                                                                             "that used in SDC Tool Competition data "
                                                                             "example, the same needed to use it in the"
                                                                             "Competition Tool Evaluator")
    arg_parser.add_argument('-p', '--property', nargs='+',
                            help="The list of properties to include in the resulting JSON object")
    arg_parser.add_argument('-o', '--output-file', nargs='?', help="If provided, the file to write the "
                                                                   "resulting JSON object. If not, prints to standard"
                                                                   "output. ")
    args = arg_parser.parse_args()

    road_properties = []
    behaviour_properties = []
    local_namespace = locals()
    if args.property is not None:
        for p in args.property:
            if p in local_namespace:
                if local_namespace[p] in __road_properties:
                    road_properties.append(p)
                elif local_namespace[p] in __behaviour_properties:
                    behaviour_properties.append(p)
                else:
                    print(f"Unknown or not implemented property: {p}\nTo list all supported properties, use the -h option",
                          file=sys.stderr)
                    exit(1)
            else:
                print(f"Unknown or not implemented property: {p}\nTo list all supported properties, use the -h option", file=sys.stderr)
                exit(1)



    # sensodat = convert(load_roads(argv[1]))
    # with open(argv[2], 'w') as f:
    #     dump(sensodat, f)