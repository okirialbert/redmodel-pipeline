import os
import glob
import pandas as pd

from infer import _infer_example


cwd = os.getcwd()
filenames = glob.glob(cwd + "/*.csv")


def _load_csv_files(file_path):
    df_list = []
    for filename in file_path:
        df_list.merge(pd.read_csv(filename, names=["text"]))

    _combined_ds = pd.concat(df_list, ignore_index=True)
    print("Copying files: ",df_list)

    return _combined_ds

def _make_batched_inference(files):
    _series_txt = _load_csv_files(filenames)
    _tensor_list = []
    for row in _series_txt:
        _tensor_list.append(_series_txt[row])
    print(_tensor_list)

    return _infer_example(_tensor_list)

_make_batched_inference(filenames)



