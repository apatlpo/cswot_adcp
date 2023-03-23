import argparse
import ui_helper as ui

parser = argparse.ArgumentParser()
#main argument : filename
parser.add_argument("file")
#add optional argument (for colorbars)
for key, value in ui.DisplayParameter()._asdict().items():
    parser.add_argument(f"--{key}", required=False, default=value)

#parse parameters
args = parser.parse_args()
dict_values = vars(args)
file_name = dict_values.pop('file')
options = ui.DisplayParameter(**dict_values)
# file_name = "ADCP_DriX__20220922T202647_018_000000.STA"
# file_name = "C:\data\datasets\ADCP\EC150\Data_selection_NetCDF\HYDROMOMAR-D20200905-T041447.nc"

ui = ui.UIDesc(filename_list=[file_name], display_parameter=options)
ui.to_standalone()
