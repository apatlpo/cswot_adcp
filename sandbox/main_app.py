import argparse
import ui_helper as ui
from glob import glob

parser = argparse.ArgumentParser()
#main argument : filename
parser.add_argument("file")
#add optional argument (for colorbars)
for key, value in ui.DisplayParameter()._asdict().items():
    parser.add_argument(f"--{key}", required=False, default=value)

#parse parameters
args = parser.parse_args()
dict_values = vars(args)
args_float = ["amplitude_min", "amplitude_max", 
              "direction_min", "direction_max", 
              "correlation_min", "correlation_max", 
              "map_expansion",
              ]
# enforce float types
for a in args_float:
    if a in dict_values:
        dict_values[a] = float(dict_values[a])
# get file name
file_name = dict_values.pop('file')
file_names = sorted(glob(file_name))

# !!! cswot: filter out values with bottom track
extract_id = lambda f: int(f.split("-")[-1].split("_")[0])

#file_names = [f for f in file_names if extract_id(f)>377 and extract_id(f)<380]
#file_names = [f for f in file_names if extract_id(f)>380]
#print(file_names)

options = ui.DisplayParameter(**dict_values)
# file_name = "ADCP_DriX__20220922T202647_018_000000.STA"
# file_name = "C:\data\datasets\ADCP\EC150\Data_selection_NetCDF\HYDROMOMAR-D20200905-T041447.nc"

ui = ui.UIDesc(filename_list=file_names, display_parameter=options)
ui.to_standalone()
