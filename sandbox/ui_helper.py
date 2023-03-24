from collections import namedtuple
from ctypes import Array
from pathlib import Path
from typing import Tuple, NamedTuple

import cartopy.crs as ccrs
import cartopy.feature
import holoviews as hv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import panel as pn
import xarray as xr
import hvplot.pandas
import hvplot.xarray

import cswot_adcp.maps as mp
from cswot_adcp import data_loader as loader
from cswot_adcp.xarray_model import Names as names


# pn.extension('ipywidgets')


def get_display_extent(STA, buffer=.01):
    """ compute horizontal extent of the STA, add a marging on bounding box for display purpose"""
    lon, lat = STA[names.elongitude_gps], STA[names.elatitude_gps]
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

RANGE_SLICE = names.range
TIME_SLICE = names.time

DisplayParameter = namedtuple("DisplayParameter",
                              ["amplitude_min", "amplitude_max", "direction_min", "direction_max", "correlation_min",
                               "correlation_max", "amplitude_cmap", "direction_cmap", "correlation_cmap"],
                              defaults=[0, 1, -180, 180, 0, 100, "inferno", "hsv", "hot"])

class UIDesc:
    def __init__(self, filename_list: Array[str], display_parameter=DisplayParameter()):
        self.file_name = None
        self.display = display_parameter
        if len(filename_list) != 1:
            raise Exception("Only one single file is supported")
        self.__load_file(file_name=filename_list[0])
        self.range_selection = True  # indicate if we use a time based selection or range type

    def declare_time_slider(self):
        self.frame_slider = pn.widgets.IntSlider(name='Frame Index', start=0, end=self.data.time.shape[0] - 1, value=0)

    def declare_range_slider(self):
        self.range_slider = pn.widgets.IntSlider(name='Range Index', start=0, end=self.data.range.shape[0] - 1, value=0)

    def update_time_label(self, frame):
        """Update datetime label, convert frame index to readable date"""
        value = self.data.time.isel(time=self.frame_slider.value)
        self.time_label = pn.widgets.StaticText(name='Current date', value=f"{value.values} (frame {frame})",
                                                sizing_mode='stretch_width')
        return self.time_label

    def update_range_label(self, range_index):
        """ Display range in meter"""
        value = self.data.range.isel(range=self.range_slider.value)
        self.range_label = pn.widgets.StaticText(name='Current range',
                                                 value=f"{value.values}m (range_index {range_index})",
                                                 sizing_mode='stretch_width')
        return self.range_label

    def get_total_amplitude(self):
        """Return graph with magnitude on the whole file"""
        return (self.data[names.compensated_magnitude]
                .hvplot(x=names.time, y=names.range, responsive=True,
                        clim=(self.display.amplitude_min, self.display.amplitude_max), height=HEIGHT,
                        cmap=self.display.amplitude_cmap)
                .opts(invert_yaxis=True, title="velocity magnitude")
                )

    def get_total_dir(self):
        """return graph with direction on the whole file"""
        return (self.data[names.compensated_dir]
                .hvplot(x=names.time, y=names.range, responsive=True,
                        clim=(self.display.direction_min, self.display.direction_max), height=HEIGHT,
                        cmap=self.display.direction_cmap)
                .opts(invert_yaxis=True, title="velocity direction")
                )

    def get_total_corr(self):
        """return graph with correlation on the whole file"""
        return (self.data[names.correlation].mean(names.beam_dimension)
                .hvplot(x=names.time, y=names.range, responsive=True,
                        clim=(self.display.correlation_min, self.display.correlation_max), height=HEIGHT,
                        cmap=self.display.correlation_cmap, )
                .opts(invert_yaxis=True, title="correlation")
                )

    def __get_data_slice(self, frame: int, range_index: int, slice: str, variable_name: str, title: str,
                         ylim: Tuple[int, int]):
        """Utility function creating a slice (on time or range) on a data set"""
        if slice == TIME_SLICE:
            selected_data = self.data[variable_name].isel(time=frame)
            if len(selected_data.shape) > 1:
                # correlation data, we reduce selected data
                selected_data = selected_data.mean(names.beam_dimension)
            return selected_data.hvplot.line(y=variable_name, x=names.range, responsive=True, height=HEIGHT, ylim=ylim) \
                .opts(invert_axes=True, invert_yaxis=True, title=title)

        else:
            selected_data = self.data[variable_name].isel(range=range_index)
            if len(selected_data.shape) > 1:
                # correlation data, we reduce selected data
                selected_data = selected_data.mean(names.beam_dimension)
            return selected_data.hvplot.line(x=names.time, y=variable_name, responsive=True, height=HEIGHT, ylim=ylim) \
                       .opts(invert_axes=False, invert_yaxis=False, title=title) * self.get_vline(frame=frame)

    def get_profile_amplitude(self, frame: int, range_index: int, slice: str = RANGE_SLICE):
        """return amplitude slice"""
        return self.__get_data_slice(frame=frame, range_index=range_index, slice=slice,
                                     variable_name=names.compensated_magnitude,
                                     title="velocity magnitude profile", ylim=(0, 1))

    def get_profile_direction(self, frame: int, range_index: int, slice: str = RANGE_SLICE):
        """return direction slice"""
        return self.__get_data_slice(frame=frame, range_index=range_index, slice=slice,
                                     variable_name=names.compensated_dir,
                                     title="velocity direction profile", ylim=(-180, 180))

    def get_profile_correlation(self, frame: int, range_index: int, slice: str = RANGE_SLICE):
        """return correlation slice"""
        return self.__get_data_slice(frame=frame, range_index=range_index, slice=slice, variable_name=names.correlation,
                                     title="correlation profile", ylim=(0, 200))

    def get_trajectory(self, frame):
        """get a interactive map for navigation display"""
        subset = self.data[[names.elongitude_gps, names.elatitude_gps]]
        _df = subset.to_dataframe()
        extent = get_display_extent(self.data, buffer=0.1)
        base = _df.hvplot.points(names.elongitude_gps, names.elatitude_gps, geo=True, color='gray', alpha=0.2,
                                 xlim=(extent[0], extent[1]), ylim=(extent[2], extent[3]), tiles='EsriNatGeo',
                                 # width=500
                                 # width=500,
                                 height=HEIGHT_MAP,
                                 )
        selected = subset.isel(time=frame)
        pos_selected = pd.DataFrame(
            data={names.elongitude_gps: [float(selected.elongitude_gps)],
                  names.elatitude_gps: [float(selected.elatitude_gps)]})
        focus = pos_selected.hvplot.points(names.elongitude_gps, names.elatitude_gps, geo=True, color='red',
                                           alpha=1., height=HEIGHT_MAP)
        return base * focus

    def get_quiver(self, range_index=0):
        """return vectorial map (quiver) dfor a given range"""
        selected_range = self.data.isel(range=range_index)
        fig = plt.figure(figsize=(10, 10))
        extent = get_display_extent(self.data, buffer=.1)
        _lon_central = (extent[0] + extent[1]) * 0.5
        _lat_central = (extent[2] + extent[3]) * 0.5
        #aspect = (extent[1] - extent[0]) / (extent[3] - extent[2])
        # used to be ccrs.Orthographic(...)
        proj = ccrs.LambertAzimuthalEqualArea(
            central_longitude=_lon_central,
            central_latitude=_lat_central,
        )
        ax = fig.add_subplot(111, projection=proj)

        #remove if not needed
        ax.stock_img() # water background (often only blue color)
        ax.coastlines() #coastline, only if near coast

        #add grid lines
        gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                          linewidth=2, color='gray', alpha=0.5, linestyle='--')

        #add navigation
        x, y = selected_range[names.elongitude_gps].values, selected_range[names.elatitude_gps].values
        ax.plot(x, y, color="b", transform=mp.crs)
        sampled = selected_range.isel(time=slice(0, None, 10))
        x, y, u, v = sampled[names.elongitude_gps].values, sampled[names.elatitude_gps].values, sampled[
            names.compensated_E].values, \
                     sampled[names.compensated_N].values

        #add quiver speed vectors
        q = ax.quiver(x=x, y=y, u=u, v=v, transform=mp.crs, pivot="tail", scale=2,
                      # width=1e-2,
                      )

        #add quiver key
        uref = 0.5
        ax.quiverkey(q, 0.3, 0.1, uref, f'{uref} m/s', transform=mp.crs, color="blue", labelcolor="blue",
                     labelpos='N', coordinates='axes')

        #set map extent
        ax.set_extent(extents=tuple(extent))
        mpl_pane = pn.pane.Matplotlib(fig, tight=True, interactive=False,
                                      #width=int(aspect * HEIGHT_MAP),
                                      height=HEIGHT_MAP,
                                      )
        return mpl_pane

    def get_vline(self, frame):
        """Draw a vertical line (time cursor)"""
        # retrieve time index
        time = self.data.time[frame].values
        return hv.VLine(time).opts(color="red")

    def get_hline(self, range_index):
        """Draw a horizontal line (range cursor)"""
        # retrieve time index
        range = self.data.range[range_index].values
        return hv.HLine(range).opts(color="red")


    ######## CODE TO MODIFY TO CREATE A NEW GRAPHIC
    def get_basic_time_graph(self):
        """return a basic graph aimed to be modified if need be, by default will be the ship heading displayed  """
        #CREATE A 1D graphic depending on time based axis
        return (self.data[names.ship_heading]
                .hvplot(x=names.time, responsive=True,
                        height=HEIGHT)
                .opts(invert_yaxis=True, title="ship heading (deg)")
                )


    def create_widgets(self):
        """Create all widgets"""
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

        ######## CODE TO MODIFY TO CREATE A NEW GRAPHIC
        #add basic graph : make it react on time slider and add time bar on it
        self.basic_time_plot =  pn.bind(lambda frame: self.get_basic_time_graph() * self.get_vline(frame),
                                  frame=self.frame_slider)

    def __load_file(self, file_name: str):
        if self.file_name != file_name:
            if not Path(file_name).is_file():
                raise Exception(f"{file_name} does not exist or is not a file")
            self.data = loader.read_data(file_name)
            self.file_name = file_name
            self.create_widgets()

    def __get_map_widget(self):
        """Retrieve map widget, """
        return pn.Column(pn.Row(
            self.trajectory_dmap,
            self.quiver_map,
            sizing_mode='stretch_width'
        ), sizing_mode='stretch_width')

    def __get_graph_widget(self):
        graphs = pn.Column(
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
            ######## CODE TO MODIFY TO CREATE A NEW GRAPHIC
            #add the plot below
            pn.Row(
                self.basic_time_plot
            ),
            sizing_mode='stretch_width'
        )
        return graphs

    def __get_control_widget(self):
        control_widget = pn.Column(
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

        return control_widget

    def to_notebook(self):
        """Get the list of widget for display in a jupyter notebook
         return controls, maps, graphs the list of widget control
        """
        self.controls = self.__get_control_widget()
        self.maps = self.__get_map_widget()
        self.graphs = self.__get_graph_widget()
        return self.controls, self.maps, self.graphs

    def to_standalone(self):
        bootstrap = pn.template.BootstrapTemplate(title='ADCP STA Data viewer')

        side_bar = pn.Column(
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
            ), sizing_mode='stretch_width'
        )

        main_widgets = pn.Column(

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
            ######## CODE TO MODIFY TO CREATE A NEW GRAPHIC
            # add the plot below
            pn.Row(
                self.basic_time_plot
            ),
            sizing_mode='stretch_width'
        )

        bootstrap.sidebar.append(
            pn.Column(side_bar, sizing_mode='stretch_width')
        )
        bootstrap.main.append(
            pn.Column(main_widgets, sizing_mode='stretch_width'),

        )
        bootstrap.servable()
        bootstrap.show()
