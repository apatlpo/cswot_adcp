from cswot_adcp import data_loader as loader
import ui_helper as ui


file_name = "ADCP_DriX__20220922T202647_018_000000.STA"
file_name = "C:\data\datasets\ADCP\EC150\Data_selection_NetCDF\HYDROMOMAR-D20200905-T041447.nc"
#sta = loader.round_time_values(sta)
ui = ui.UIDesc(filename_list=[file_name])
ui.to_standalone()



