import os
import glob
import pandas as pd
import json

from infer import _infer_example



cwd = os.getcwd()
filenames = glob.glob(cwd + "/*.csv")


def _load_csv_files(file_path):
    df_list = []
    print("Copying files: ",file_path)
    for filename in file_path:
        df_list.append(pd.read_csv(filename, usecols=['text']))

    _combined_ds = pd.concat(df_list, ignore_index=True)

    return _combined_ds

def _make_batched_inference(files):
    _series_txt = _load_csv_files(filenames)

    _pred_list = []

    for value in _series_txt.values:
        pred = _infer_example(str(value))
        _pred_list.append(pred)
    prediction = json.dumps(_pred_list, indent=4)
    return prediction

result = _make_batched_inference(filenames)
print(result.predictions.mean())
