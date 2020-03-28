import os
from .dataset_raw import DataLoaderVal, DataLoaderTest

def get_validation_data(inp_dir):
    assert os.path.exists(inp_dir)
    return DataLoaderVal(inp_dir, None)

def get_test_data(inp_dir):
    assert os.path.exists(inp_dir)
    return DataLoaderTest(inp_dir, None)