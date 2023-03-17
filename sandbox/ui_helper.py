from typing import Tuple

import cartopy.crs as ccrs
import holoviews as hv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import panel as pn
import xarray as xr
import hvplot.pandas
import hvplot.xarray

import cswot_adcp.maps as mp


# pn.extension('ipywidgets')


def get_display_extent(STA, buffer=.01):
    """ compute horizontal extent of the STA, add a marging on bounding box for display purpose"""
    lon, lat = STA["elongitude_gps"], STA["elatitude_gps"]
    lon_scale = 1 / np.cos(np.pi / 180 * lat.mean())
    extent = [lon.min() - buffer * lon_scale,
              lon.max() + buffer * lon_scale,
              lat.min() - buffer,
              lat.max() + buffer,
              ]
    extent = [float(e) for e in extent]
    return extent


HEIGHT = 200
HEIGHT_MAP = 400

RANGE_SLICE = "Range"
TIME_SLICE = "Time"


class UIDesc:
    def __init__(self, data: xr.Dataset):
        self.data = data
        self.range_selection = True  # indicate if we use a time based selection or range type

    def declare_time_slider(self):
        self.frame_slider = pn.widgets.IntSlider(name='Frame Index', start=0, end=self.data.time.shape[0] - 1, value=0)

    def declare_range_slider(self):
        self.range_slider = pn.widgets.IntSlider(name='Range Index', start=0, end=self.data.range.shape[0] - 1, value=0)

    def update_time_label(self, frame):
        value = self.data.time.isel(time=self.frame_slider.value)
        self.time_label = pn.widgets.StaticText(name='Current date', value=f"{value.values} (frame {frame})",
                                                sizing_mode='stretch_width')
        return self.time_label

    def update_range_label(self, range_index):
        value = self.data.range.isel(range=self.range_slider.value)
        self.range_label = pn.widgets.StaticText(name='Current range',
                                                 value=f"{value.values}m (range_index {range_index})",
                                                 sizing_mode='stretch_width')
        return self.range_label

    def get_total_amplitude(self):
        return (self.data["compensated_Mag"]
                .hvplot(x="time", y="range", responsive=True, clim=(0, 1), height=HEIGHT, cmap="inferno")
                .opts(invert_yaxis=True, title="velocity magnitude")
                )

    def get_total_dir(self):
        return (self.data["compensated_Dir"]
                .hvplot(x="time", y="range", responsive=True, clim=(-180, 180), height=HEIGHT, cmap="hsv")
                .opts(invert_yaxis=True, title="velocity direction")
                )

    def get_total_corr(self):
        return (self.data["corr"].mean("beam")
                .hvplot(x="time", y="range", responsive=True, clim=(0, 100), height=HEIGHT, cmap="hot", )
                .opts(invert_yaxis=True, title="correlation")
                )

    def __get_data_slice(self, frame: int, range_index: int, slice: str, variable_name: str, title: str,
                         ylim: Tuple[int, int]):
        if slice == TIME_SLICE:
            selected_data = self.data[variable_name].isel(time=frame)
            if len(selected_data.shape) > 1:
                # correlation data, we reduce selected data
                selected_data = selected_data.mean("beam")
            return selected_data.hvplot.line(y=variable_name, x="range", responsive=True, height=HEIGHT, ylim=ylim) \
                .opts(invert_axes=True, invert_yaxis=True, title=title)

        else:
            selected_data = self.data[variable_name].isel(range=range_index)
            if len(selected_data.shape) > 1:
                # correlation data, we reduce selected data
                selected_data = selected_data.mean("beam")
            return selected_data.hvplot.line(x="time", y=variable_name, responsive=True, height=HEIGHT, ylim=ylim) \
                       .opts(invert_axes=False, invert_yaxis=False, title=title) * self.get_vline(frame=frame)

    def get_profile_amplitude(self, frame: int, range_index: int, slice: str = RANGE_SLICE):
        return self.__get_data_slice(frame=frame, range_index=range_index, slice=slice, variable_name="compensated_Mag",
                                     title="velocity magnitude profile", ylim=(0, 1))

    def get_profile_direction(self, frame: int, range_index: int, slice: str = RANGE_SLICE):
        return self.__get_data_slice(frame=frame, range_index=range_index, slice=slice, variable_name="compensated_Dir",
                                     title="velocity direction profile", ylim=(-180, 180))

    def get_profile_correlation(self, frame: int, range_index: int, slice: str = RANGE_SLICE):
        return self.__get_data_slice(frame=frame, range_index=range_index, slice=slice, variable_name="corr",
                                     title="correlation profile", ylim=(0, 200))

    def get_trajectory(self, frame):
        subset = self.data[['elongitude_gps', 'elatitude_gps']]
        _df = subset.to_dataframe()
        extent = get_display_extent(self.data, buffer=0.1)
        base = _df.hvplot.points('elongitude_gps', 'elatitude_gps', geo=True, color='gray', alpha=0.2,
                                 xlim=(extent[0], extent[1]), ylim=(extent[2], extent[3]), tiles='EsriNatGeo',
                                 # width=500
                                 # width=500,
                                 height=HEIGHT_MAP,
                                 )
        selected = subset.isel(time=frame)
        pos_selected = pd.DataFrame(
            data={'elongitude_gps': [float(selected.elongitude_gps)], 'elatitude_gps': [float(selected.elatitude_gps)]})
        focus = pos_selected.hvplot.points('elongitude_gps', 'elatitude_gps', geo=True, color='red',
                                           alpha=1., height=HEIGHT_MAP)
        return base * focus

    def get_quiver(self, range_index=0):
        selected_range = self.data.isel(range=range_index)
        fig = plt.figure(figsize=(10, 10))
        extent = get_display_extent(self.data, buffer=.01)
        _lon_central = (extent[0] + extent[1]) * 0.5
        _lat_central = (extent[2] + extent[3]) * 0.5
        aspect = (extent[1] - extent[0]) / (extent[3] - extent[2])
        # used to be ccrs.Orthographic(...)
        proj = ccrs.LambertAzimuthalEqualArea(
            central_longitude=_lon_central,
            central_latitude=_lat_central,
        )
        ax = fig.add_subplot(111, projection=proj)
        x, y = selected_range["elongitude_gps"].values, selected_range["elatitude_gps"].values

        ax.plot(x, y, color="b", transform=mp.crs)
        sampled = selected_range.isel(time=slice(0, None, 10))
        x, y, u, v = sampled["elongitude_gps"].values, sampled["elatitude_gps"].values, sampled["compensated_E"].values, \
                     sampled["compensated_N"].values

        q = ax.quiver(x=x, y=y, u=u, v=v, transform=mp.crs, pivot="tail", scale=2,
                      # width=1e-2,
                      )
        uref = 0.5
        ax.quiverkey(q, 0.3, 0.1, uref, f'{uref} m/s', transform=mp.crs, color="blue", labelcolor="blue",
                     labelpos='N', coordinates='axes')
        mpl_pane = pn.pane.Matplotlib(fig, tight=True, interactive=False, dpi=255, width=int(aspect * HEIGHT_MAP),
                                      height=HEIGHT_MAP)
        return mpl_pane

    def get_vline(self, frame):
        # retrieve time index
        time = self.data.time[frame].values
        return hv.VLine(time).opts(color="red")

    def get_hline(self, range_index):
        # retrieve time index
        range = self.data.range[range_index].values
        return hv.HLine(range).opts(color="red")

    def declare_widgets(self):
        self.declare_time_slider()
        self.declare_range_slider()

        self.slice_selector = pn.widgets.RadioBoxGroup(name='Slice selector', options=[TIME_SLICE, RANGE_SLICE],
                                                       value=RANGE_SLICE,
                                                       inline=True)

        self.pamplitude_dmap = pn.bind(
            lambda frame, range_index, slice: self.get_profile_amplitude(frame=frame, range_index=range_index,
                                                                         slice=slice),
            frame=self.frame_slider, range_index=self.range_slider, slice=self.slice_selector)

        self.pdirection_dmap = pn.bind(
            lambda frame, range_index, slice: self.get_profile_direction(frame=frame, range_index=range_index,
                                                                         slice=slice),
            frame=self.frame_slider, range_index=self.range_slider, slice=self.slice_selector)
        self.pcorrelation_dmap = pn.bind(
            lambda frame, range_index, slice: self.get_profile_correlation(frame=frame, range_index=range_index,
                                                                           slice=slice),
            frame=self.frame_slider, range_index=self.range_slider, slice=self.slice_selector)

        self.ptime_label = pn.bind(self.update_time_label, frame=self.frame_slider)
        self.prange_label = pn.bind(self.update_range_label, range_index=self.range_slider)

        pamplitude_dline = pn.bind(
            lambda frame, range_index: self.get_total_amplitude() * self.get_vline(frame) * self.get_hline(range_index),
            frame=self.frame_slider, range_index=self.range_slider)
        self.dd_amp = hv.DynamicMap(pamplitude_dline)

        plot_corr_dline = pn.bind(
            lambda frame, range_index: self.get_total_corr() * self.get_vline(frame) * self.get_hline(range_index),
            frame=self.frame_slider, range_index=self.range_slider)
        self.dd_corr = hv.DynamicMap(plot_corr_dline)

        plot_dir_dline = pn.bind(
            lambda frame, range_index: self.get_total_dir() * self.get_vline(frame) * self.get_hline(range_index),
            frame=self.frame_slider, range_index=self.range_slider)
        self.dd_dir = hv.DynamicMap(plot_dir_dline)

        self.trajectory_dmap = pn.bind(lambda frame: self.get_trajectory(frame), frame=self.frame_slider)

        self.quiver_map = pn.bind(lambda range_index: self.get_quiver(range_index=range_index),
                                  range_index=self.range_slider)

    def to_notebook(self):
        self.controls = pn.Column(
            #  trajectory_dmap,
            pn.Row(
                pn.WidgetBox("Frame selector",
                             self.frame_slider,
                             self.ptime_label, sizing_mode='stretch_width'
                             ),
                pn.WidgetBox("Range selector",
                             self.range_slider,
                             self.prange_label, sizing_mode='stretch_width'
                             )
            ),
            pn.Row(
                pn.WidgetBox("Select slice data on",
                             self.slice_selector
                             )
            ),
        )
        self.maps = pn.Column(
            pn.Row(
            self.trajectory_dmap,
            self.quiver_map,
            sizing_mode='stretch_width'
        )
        )

        self.graphs = pn.Column(
            pn.Row(
                self.dd_amp,
                self.pamplitude_dmap,
            ),
            pn.Row(
                self.dd_corr,
                self.pcorrelation_dmap,
            ),

            pn.Row(
                self.dd_dir,
                self.pdirection_dmap,
            ),
            sizing_mode='stretch_width'
        )
        return self.controls, self.maps, self.graphs

    def to_standalone(self):
        bootstrap = pn.template.BootstrapTemplate(title='ADCP STA Data viewer')

        bootstrap.sidebar.append(
            pn.Column(
                #  trajectory_dmap,
                pn.Row(
                    pn.WidgetBox("Frame selector",
                                 self.frame_slider,
                                 self.ptime_label, sizing_mode='stretch_width'
                                 ),
                ),
                pn.Row(
                    pn.WidgetBox("Range selector",
                                 self.range_slider,
                                 self.prange_label, sizing_mode='stretch_width'
                                 )
                ),
                pn.Row(
                    pn.WidgetBox("Select slice data on",
                                 self.slice_selector
                                 )
                ),
            )
        )
        bootstrap.main.append(
            pn.Column(
                pn.Row(
                    self.trajectory_dmap,
                    self.quiver_map,
                    sizing_mode='stretch_width'
                ),

                pn.Row(
                    self.dd_amp,
                    self.pamplitude_dmap,
                ),
                pn.Row(
                    self.dd_corr,
                    self.pcorrelation_dmap,
                ),

                pn.Row(
                    self.dd_dir,
                    self.pdirection_dmap,
                ),
                sizing_mode='stretch_width'
            )
        )
        bootstrap.servable()
        bootstrap.show()
