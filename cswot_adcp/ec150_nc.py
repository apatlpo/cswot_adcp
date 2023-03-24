
import numpy as np
import xarray as xr

from cswot_adcp.xarray_model import Names


def vlen_to_matrix(dataarray: xr.DataArray):
    max_samples = 0
    for sub_array in dataarray.values:  # parse data to retrieve maximum number of samples
        max_samples = max(max_samples, len(sub_array))
    # fill ping with data\n",
    matrix = np.full(
        (dataarray.shape[0], max_samples),  # only for 1D vlen
        dtype="float32",
        fill_value=float(np.nan),
    )
    for bnr in range(dataarray.shape[0]):
        # Warning, auto scale if set by default ping[bnr][:count] = sample_amplitude[start:stop]
        count = len(dataarray.values[bnr])
        matrix[bnr][:count] = dataarray.values[bnr]
    return matrix.transpose()



def read_ec150(file):
    """Read ec150 adcp format and return values in a compatible xarray format """

    da_mean = xr.load_dataset(filename_or_obj=file, group="Sonar/Beam_group1/ADCP/Mean_current", decode_times=False)
    xr.set_options(keep_attrs=True)
    da_mean.coords["mean_time"] = da_mean.mean_time // 1_000_000  # convert to ms
    da_mean.mean_time.attrs["units"] = "milliseconds since 1601-01-01 00:00:00Z"
    da_mean = xr.decode_cf(da_mean)
    da_adcp = xr.load_dataset(filename_or_obj=file, group="Sonar/Beam_group1/ADCP/", decode_times=False)
    # time for correlation variable is defined in a higher subgroup
    north = vlen_to_matrix(da_mean.current_velocity_geographical_north)
    east = vlen_to_matrix(da_mean.current_velocity_geographical_east)
    magnitude = np.sqrt(north ** 2 + east ** 2)
    direction = np.mod(360 + np.rad2deg(np.arctan2(north ** 2, east)), 360)

    #compute correlation matrix
    matrix_list = []
    for beam in da_adcp.beam:
        values = da_adcp.correlation.isel(beam=beam)

        matrix_list.append(vlen_to_matrix(values))
    correlations = np.dstack(matrix_list)

    #retrieve range values to create a coordinate variable
    dataarray =da_adcp.slant_range_to_bottom.isel(beam=0) #TODO should be corrected from all beam direction and have a mean value
    max_range= np.max(dataarray.data)

    final = xr.Dataset(
        data_vars={
            Names.compensated_dir: ([ "range","time"], direction),
            Names.compensated_N: (["range","time" ], north),
            Names.compensated_E: (["range","time" ], east),
            Names.compensated_magnitude: (["range","time" ], magnitude),
            Names.correlation: (["range","time", "beam"],correlations),
            Names.elatitude_gps: (["time" ], da_mean.mean_platform_latitude.data),
            Names.elongitude_gps: (["time"], da_mean.mean_platform_longitude.data),
            Names.ship_heading: (["time"], np.zeros(da_mean.mean_platform_longitude.data.shape)),
        },
        coords={

            Names.time: (["time"], da_mean.mean_time.data),
            Names.range: (["range"],np.arange(0,max_range,max_range/direction.shape[0]))
        }
    )
    return final

if __name__ == "__main__":
    read_ec150("C:\data\datasets\ADCP\EC150\Data_selection_NetCDF\HYDROMOMAR-D20200905-T041447.nc")
