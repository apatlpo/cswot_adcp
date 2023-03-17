from pathlib import Path

import pandas as pd
import xarray as xr
import cswot_adcp.adcp as ad



def read_data(file:Path):
    # STA file reading
    sta = ad.read_WH300(file)
    # navigation compensation
    sta = ad.ADCPcompNav(sta)

    use_std_time(sta)
    sta = update_datastructure(sta)

    return sta


def use_std_time(data:xr.Dataset):
    """
    override time_gps as it seems to be not reliable may introduce inaccurate positions
    """
    for v in data.reset_coords():
        if "time_gps" in data[v].dims and v!="time_gps":
            data[v] = data[v].rename(time_gps="time").assign_coords(time=data.time)

def round_time_values(data:xr.Dataset, round_value = "500ms"):
    """round time to values"""
    data["time"] = data.time.dt.round(round_value)
    data = data.assign_coords(time_date=data.time)
    data["time"] = (data.time - data.time[0]) / pd.Timedelta("1s")
    return data

def update_datastructure(data:xr.Dataset):
    """Update data structure to match our expectation"""
    # split velocity field dir dimension into multiple variables

    long_names = dict(E="Eastward velocity", N="Northward velocity",
                      U="Upward velocity", err="Error velocity",
                      Mag="Velocity magnitude", Dir="Velocity Direction",
                     )
    units = dict(E="m/s", N="m/s", U="m/s", err="m/s", Mag="m/s", Dir="degrees")
    dirs = data.dir.values

    def split_speed(da):
        ds = xr.merge([da.sel(dir=d).rename(da.name+"_"+str(d))
                       .assign_attrs(units=units[d], long_name=long_names[d]) for d in dirs],
                      compat="override")
        del ds.attrs["units"]
        return ds.assign_attrs(long_name="compensated velocity fields")
    data = data.rename_vars({"vel comp Nav":"compensated"})

    ds_vel = split_speed(data["compensated"])


    return xr.merge([data,ds_vel])