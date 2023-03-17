from cswot_adcp import data_loader as loader
import ui_helper as ui


file_name = "ADCP_DriX__20220922T202647_018_000000.STA"

sta = loader.read_data(file_name)

#sta = loader.round_time_values(sta)
ui = ui.UIDesc(data=sta)
ui.declare_widgets()
ui.to_standalone()



