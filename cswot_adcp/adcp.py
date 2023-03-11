# -*- coding: utf-8 -*-
"""
Created on Thu Nov 24 16:04:56 2022

@author: nlebouff
"""
import numpy as np
import xarray as xr

import datetime
import pytz

from pyproj import Geod

g = Geod(ellps="WGS84")

# local code
from .WH300_File import WH300_File

import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)


def read_WH300(file_name):
    data = WH300_File(file_name, 2020)

    utc_tz = pytz.utc  # define UTC timezone

    dataWH = xr.Dataset()
    if "Vel" in list(data.V):
        vel = np.array(data.V["Vel"].T)
        Mag = np.sqrt(vel[0, :, :] ** 2 + vel[1, :, :] ** 2)
        Dir = np.mod(360 + np.rad2deg(np.arctan2(vel[0, :, :], vel[1, :, :])), 360)

        dataWH["vel"] = xr.DataArray(
            np.append(vel, [Mag, Dir], axis=0),
            dims=("dir", "range", "time"),
            attrs={
                "units": "meter/second for speed and degree for direction",
                "long_name": "East/North/Up/Err/Magnitude/Direction velocity components measured by ADCP",
            },
        )
        dataWH["amp"] = xr.DataArray(
            np.array(data.V["Amp"].T),
            dims=("beam", "range", "time"),
            attrs={
                "units": "RDI count",
                "long_name": "Received signal strength measured by ADCP in each beam",
            },
        )
        dataWH["corr"] = xr.DataArray(
            np.array(data.V["Cor"].T),
            dims=("beam", "range", "time"),
            attrs={
                "units": "RDI count",
                "long_name": "Correlation coefficient for data quality in each beam",
            },
        )
        dataWH["prcnt_gd"] = xr.DataArray(
            np.array(data.V["PGd"].T),
            dims=("beam", "range", "time"),
            attrs={"units": "pct", "long_name": "Percent Good pings for u,v"},
        )
        dataWH["Ens"] = xr.DataArray(
            data.V["VL"]["ENum"],
            dims=("time"),
            attrs={"units": "index", "long_name": "ADCP Ensemble index"},
        )
        dataWH["pitch"] = xr.DataArray(
            data.V["VL"]["Pitch"],
            dims=("time"),
            attrs={"units": "degree", "long_name": "ADCP pitch"},
        )
        dataWH["roll"] = xr.DataArray(
            data.V["VL"]["Roll"],
            dims=("time"),
            attrs={"units": "degree", "long_name": "ADCP roll"},
        )
        dataWH["heading"] = xr.DataArray(
            data.V["Nav"]["Heading"],
            dims=("time"),
            attrs={"units": "degree", "long_name": "ADCP heading"},
        )
        dataWH["Tsd_depth"] = xr.DataArray(
            data.V["VL"]["TransducerDepth"],
            dims=("time"),
            attrs={"units": "meters", "long_name": "Transducer depth"},
        )
        dataWH["c_sound"] = xr.DataArray(
            data.V["VL"]["SoundSpeed"],
            dims=("time"),
            attrs={"units": "meter/second", "long_name": "sound speed"},
        )
        dataWH["salinity"] = xr.DataArray(
            data.V["VL"]["Salinity"],
            dims=("time"),
            attrs={"units": "ppm", "long_name": "salinity"},
        )
        dataWH["temp"] = xr.DataArray(
            data.V["VL"]["Temperature"],
            dims=("time"),
            attrs={
                "units": "degree Celsius",
                "long_name": "ADCP transducer temperature",
            },
        )
        dataWH["route"] = xr.DataArray(
            data.V["Nav"]["CMG"],
            dims=("time_gps"),
            attrs={"units": "degree", "long_name": "vessel route"},
        )
        dataWH["vessel_speed"] = xr.DataArray(
            data.V["Nav"]["SMG"],
            dims=("time_gps"),
            attrs={"units": "meter/second", "long_name": "vessel speed"},
        )
        dataWH["Ens_gps"] = xr.DataArray(
            data.V["Nav"]["ENum"],
            dims=("time_gps"),
            attrs={"units": "index", "long_name": "ADCP Ensemble index"},
        )
        dataWH["flatitude_gps"] = xr.DataArray(
            data.V["Nav"]["FirstLat"],
            dims=("time_gps"),
            attrs={
                "units": "decimal degree",
                "long_name": "Latitude from ship nav before ADCP ensemble",
            },
        )
        dataWH["flongitude_gps"] = xr.DataArray(
            data.V["Nav"]["FirstLon"],
            dims=("time_gps"),
            attrs={
                "units": "decimal degree",
                "long_name": "Longitude from ship nav before ADCP ensemble",
            },
        )
        dataWH["elatitude_gps"] = xr.DataArray(
            data.V["Nav"]["LastLat"],
            dims=("time_gps"),
            attrs={
                "units": "decimal degree",
                "long_name": "Latitude from ship nav after ADCP ensemble",
            },
        )
        dataWH["elongitude_gps"] = xr.DataArray(
            data.V["Nav"]["LastLon"],
            dims=("time_gps"),
            attrs={
                "units": "decimal degree",
                "long_name": "Longitude from ship nav after ADCP ensemble",
            },
        )

        time = np.full(0, np.datetime64("2020"))
        for time_i in data.V["VL"]["time"]:
            time = np.append(
                time, np.datetime64(datetime.datetime.fromtimestamp(time_i, utc_tz))
            )
        dataWH.coords["time"] = time
        dataWH.coords["time"].attrs["long name"] = "ADCP ping time"

        time_gps = np.full(0, np.datetime64("2020"))
        for time_i in data.V["Nav"]["EPCTime"]:
            time_gps = np.append(
                time_gps, np.datetime64(datetime.datetime.fromtimestamp(time_i, utc_tz))
            )
        dataWH.coords["time_gps"] = time_gps
        dataWH.coords["time_gps"].attrs["long name"] = "GPS time"
        dataWH.coords["range"] = (
            data.V["FL"]["Bin1Dstnc"]
            + np.arange(0, data.V["FL"]["NCells"]) * data.V["FL"]["CellDepth"]
        )
        dataWH.coords["range"].attrs = {"units": "meter", "long name": "ADCP bin depth"}
        dataWH.coords["dir"] = ["E", "N", "U", "err", "Mag", "Dir"]
        dataWH.coords["dir"].attrs = {
            "units": "",
            "long_name": "East/North/Up/Err/Magnitude/Direction",
        }
        dataWH.coords["beam"] = [1, 2, 3, 4]
        dataWH.coords["beam"].attrs = {"units": "index", "long_name": "ADCP beam index"}

        if "BT" in list(data.V):
            vel = np.array(data.V["BT"]["Vel"].T)
            Mag = np.sqrt(vel[0, :] ** 2 + vel[1, :] ** 2)
            Dir = np.mod(360 + np.rad2deg(np.arctan2(vel[0, :], vel[1, :])), 360)

            dataWH["BT_vel"] = xr.DataArray(
                np.append(vel, [Mag, Dir], axis=0),
                dims=("dir", "time"),
                attrs={
                    "units": "meter/second for speed and degree for direction",
                    "long_name": "East/North/Up/Err/Magnitude/Direction velocity components from Bottom Track",
                },
            )
            dataWH["BT_Range"] = xr.DataArray(
                np.array(data.V["BT"]["Range"].T),
                dims=("beam", "time"),
                attrs={
                    "units": "meters",
                    "long_name": "Bottom Range measured by Bottom Track each beam",
                },
            )
            dataWH["BT_Corr"] = xr.DataArray(
                np.array(data.V["BT"]["Corr"].T),
                dims=("beam", "time"),
                attrs={
                    "units": "meters",
                    "long_name": "Correlation from Bottom Track each beam",
                },
            )
            dataWH["BT_Amp"] = xr.DataArray(
                np.array(data.V["BT"]["Amp"].T),
                dims=("BT amp dim unkown", "time"),
                attrs={
                    "units": "meters",
                    "long_name": "Bottom Amplitude from Bottom Track?",
                },
            )

    else:
        print("no velocity data in ENX")

    return dataWH


def ADCPcompNav(dataADCP, compType=None):
    # par défaut, compensation BT si présente, Nav sinon
    # Nav forcée si compType='Nav'

    E_rel = dataADCP.vel.sel(dir="E")
    N_rel = dataADCP.vel.sel(dir="N")

    E_nav = np.sin(np.deg2rad(dataADCP.route)) * dataADCP.vessel_speed
    N_nav = np.cos(np.deg2rad(dataADCP.route)) * dataADCP.vessel_speed

    if ("BT_vel" in list(dataADCP)) & (compType != "Nav"):
        E_BT = -dataADCP.BT_vel.sel(dir="E").data
        N_BT = -dataADCP.BT_vel.sel(dir="N").data
        E_nav = np.where(E_BT == E_BT, E_BT.data, E_nav.data)
        N_nav = np.where(N_BT == N_BT, N_BT.data, N_nav.data)

    E = E_rel.data + E_nav.data
    N = N_rel.data + N_nav.data
    MagC = np.sqrt(E * E + N * N)
    DirC = np.rad2deg(np.arctan2(E, N))

    dataADCP["vel comp Nav"] = xr.DataArray(
        np.array(
            [E, N, dataADCP.vel.sel(dir="U"), dataADCP.vel.sel(dir="err"), MagC, DirC]
        ),
        dims=("dir", "range", "time"),
        attrs={
            "units": "meter/second for speed and degree for direction",
            "long_name": "East/North/Up/Err/Magnitude/Direction velocity components compensated from ship navigation",
        },
    )
    dataADCP = dataADCP.set_coords(
        ["flongitude_gps", "flatitude_gps", "elongitude_gps", "elatitude_gps"]
    )
    

    return dataADCP


def ENX2STA(ENX, dt_sta_s):

    # time for ENX data seems not robust. GPS time is taken as coordinate
    # time slices for sta
    t = [ENX.time_gps[0].data]
    while (t[-1] + np.timedelta64(dt_sta_s, "s")) < ENX.time_gps[-1]:
        t = np.append(t, t[-1] + np.timedelta64(dt_sta_s, "s"))

    # sta with vessel motion
    nbin = len(ENX.coords["range"])
    timesta = t
    nsta = len(timesta)

    Esta = np.full((nbin, nsta), np.nan)
    Nsta = np.full((nbin, nsta), np.nan)
    nbeam = np.shape(ENX.amp.data)[0]
    ampsta = np.full((nbeam, nbin, nsta), np.nan)
    corrsta = np.full((nbeam, nbin, nsta), np.nan)
    prcnt_gdsta = np.full((nbeam, nbin, nsta), np.nan)
    PctEnsAveraged = np.full((nbin, nsta), np.nan)
    pitchsta = np.full((nsta,), np.nan)
    rollsta = np.full((nsta,), np.nan)
    headingsta = np.full((nsta,), np.nan)
    c_soundsta = np.full((nsta,), np.nan)
    salinitysta = np.full((nsta,), np.nan)
    tempsta = np.full((nsta,), np.nan)
    routesta = np.full((nsta,), np.nan)
    vessel_speedsta = np.full((nsta,), np.nan)
    time_gpssta = np.full((nsta,), np.nan)
    flatitude_gpssta = np.full((nsta,), np.nan)
    flongitude_gpssta = np.full((nsta,), np.nan)
    elatitude_gpssta = np.full((nsta,), np.nan)
    elongitude_gpssta = np.full((nsta,), np.nan)
    latitude_gpssta = np.full((nsta,), np.nan)
    longitude_gpssta = np.full((nsta,), np.nan)

    for it in range(nsta - 1):

        ind0 = np.where(ENX.time_gps.data >= timesta[it])[0][0]
        ind1 = np.where(ENX.time_gps.data >= timesta[it + 1])[0][0]

        Esta[:, it] = np.nanmean(ENX.vel.sel(dir="E").data[:, ind0:ind1], axis=1)
        Nsta[:, it] = np.nanmean(ENX.vel.sel(dir="N").data[:, ind0:ind1], axis=1)
        val = ENX.vel.sel(dir="N").data[:, ind0:ind1]
        PctEnsAveraged[:, it] = (
            np.sum(np.where(val == val, 1, 0), axis=1) / np.shape(val)[1] * 100
        )

        ampsta[:, :, it] = np.nanmean(ENX.amp.data[:, :, ind0:ind1], axis=2)
        corrsta[:, :, it] = np.nanmean(ENX.corr.data[:, :, ind0:ind1], axis=2)
        prcnt_gdsta[:, :, it] = np.nanmean(ENX.prcnt_gd.data[:, :, ind0:ind1], axis=2)

        pitchsta[it] = np.nanmean(ENX.pitch.data[ind0:ind1])
        rollsta[it] = np.nanmean(ENX["roll"].data[ind0:ind1])
        headingsta[it] = np.rad2deg(
            np.arctan2(
                np.nanmean(np.sin(np.deg2rad(ENX.heading.data[ind0:ind1]))),
                np.nanmean(np.cos(np.deg2rad(ENX.heading.data[ind0:ind1]))),
            )
        )
        c_soundsta[it] = np.nanmean(ENX.c_sound.data[ind0:ind1])
        salinitysta[it] = np.nanmean(ENX.salinity.data[ind0:ind1])
        tempsta[it] = np.nanmean(ENX.temp.data[ind0:ind1])

        time_gpssta[it] = np.mean(ENX.time_gps[ind0:ind1])
        routesta[it] = np.rad2deg(
            np.arctan2(
                np.nanmean(np.sin(np.deg2rad(ENX.route[ind0:ind1]))),
                np.nanmean(np.cos(np.deg2rad(ENX.route[ind0:ind1]))),
            )
        )
        vessel_speedsta[it] = np.nanmean(ENX.vessel_speed[ind0:ind1])
        flatitude_gpssta[it] = ENX.flatitude_gps[ind0]
        flongitude_gpssta[it] = ENX.flongitude_gps[ind0]
        elatitude_gpssta[it] = ENX.elatitude_gps[ind1]
        elongitude_gpssta[it] = ENX.elongitude_gps[ind1]
        latitude_gpssta[it] = (
            np.nanmean(ENX.flatitude_gps[ind0:ind1] + ENX.elatitude_gps[ind0:ind1]) / 2
        )
        longitude_gpssta[it] = (
            np.nanmean(ENX.flongitude_gps[ind0:ind1] + ENX.elongitude_gps[ind0:ind1])
            / 2
        )

    MagCsta = np.sqrt(Esta**2 + Nsta**2)
    DirCsta = np.mod(360 + np.rad2deg(np.arctan2(Esta, Nsta)), 360)

    STA = xr.Dataset()
    STA["vel"] = xr.DataArray(
        np.array([Esta, Nsta, MagCsta, DirCsta]),
        dims=("dir", "range", "time"),
        attrs=ENX["vel"].attrs,
    )
    STA.coords["time"] = timesta + np.timedelta64(dt_sta_s * 500, "ms")
    STA.coords["time"].attrs = ENX.coords["time"].attrs
    STA.coords["range"] = ENX.coords["range"].data
    STA.coords["range"].attrs = ENX.coords["range"].attrs
    STA.coords["beam"] = ENX.coords["beam"]
    STA.coords["dir"] = ["E", "N", "Mag", "Dir"]
    STA.coords["dir"].attrs = ENX.coords["dir"].attrs
    STA.coords["time_gps"] = time_gpssta
    STA.coords["time_gps"].attrs = STA.coords["time_gps"].attrs
    STA["timedelta"] = xr.DataArray(
        dt_sta_s, attrs={"units": "second", "long name": "averaged period"}
    )
    STA["amp"] = xr.DataArray(
        ampsta, dims=("beam", "range", "time"), attrs=ENX["amp"].attrs
    )
    STA["corr"] = xr.DataArray(
        corrsta, dims=("beam", "range", "time"), attrs=ENX["corr"].attrs
    )
    STA["prcnt_gd"] = xr.DataArray(
        prcnt_gdsta, dims=("beam", "range", "time"), attrs=ENX["prcnt_gd"].attrs
    )
    STA["PctEnsAveraged"] = xr.DataArray(
        PctEnsAveraged,
        dims=("range", "time"),
        attrs={
            "units": "%",
            "long name": "proportion of True Flagged data in ensemble average",
        },
    )
    STA["pitch"] = xr.DataArray(pitchsta, dims=("time"), attrs=ENX["pitch"].attrs)
    STA["roll"] = xr.DataArray(rollsta, dims=("time"), attrs=ENX["roll"].attrs)
    STA["heading"] = xr.DataArray(headingsta, dims=("time"), attrs=ENX["heading"].attrs)
    STA["c_sound"] = xr.DataArray(c_soundsta, dims=("time"), attrs=ENX["c_sound"].attrs)
    STA["salinity"] = xr.DataArray(
        salinitysta, dims=("time"), attrs=ENX["salinity"].attrs
    )
    STA["temp"] = xr.DataArray(tempsta, dims=("time"), attrs=ENX["temp"].attrs)
    STA["Ens"] = xr.DataArray(np.arange(0, nsta), dims=("time"), attrs=ENX["Ens"].attrs)
    STA["route"] = xr.DataArray(routesta, dims=("time_gps"), attrs=ENX["route"].attrs)
    STA["vessel_speed"] = xr.DataArray(
        vessel_speedsta, dims=("time_gps"), attrs=ENX["vessel_speed"].attrs
    )
    STA["latitude_gps"] = xr.DataArray(
        latitude_gpssta, dims=("time_gps"), attrs=ENX["flatitude_gps"].attrs
    )
    STA["longitude_gps"] = xr.DataArray(
        longitude_gpssta, dims=("time_gps"), attrs=ENX["flongitude_gps"].attrs
    )
    STA["latitude_gps"].attrs = {
        "units": "decimal degree",
        "long_name": "medium latitude from ship nav during ADCP ensemble",
    }
    STA["longitude_gps"].attrs = {
        "units": "decimal degree",
        "long_name": "medium longitude from ship nav during ADCP ensemble",
    }
    STA["flatitude_gps"] = xr.DataArray(
        flatitude_gpssta, dims=("time_gps"), attrs=ENX["flatitude_gps"].attrs
    )
    STA["flongitude_gps"] = xr.DataArray(
        flongitude_gpssta, dims=("time_gps"), attrs=ENX["flongitude_gps"].attrs
    )
    STA["elatitude_gps"] = xr.DataArray(
        elatitude_gpssta, dims=("time_gps"), attrs=ENX["elatitude_gps"].attrs
    )
    STA["elongitude_gps"] = xr.DataArray(
        elongitude_gpssta, dims=("time_gps"), attrs=ENX["elongitude_gps"].attrs
    )

    try:  # vessel motion compensation

        Esta = np.full((nbin, nsta), np.nan)
        Nsta = np.full((nbin, nsta), np.nan)
        for it in range(nsta - 1):
            ind0 = np.where(ENX.time_gps.data >= timesta[it])[0][0]
            ind1 = np.where(ENX.time_gps.data >= timesta[it + 1])[0][0]
            Esta[:, it] = np.nanmean(
                ENX["vel comp Nav"].sel(dir="E").data[:, ind0:ind1], axis=1
            )
            Nsta[:, it] = np.nanmean(
                ENX["vel comp Nav"].sel(dir="N").data[:, ind0:ind1], axis=1
            )
        MagCsta = np.sqrt(Esta**2 + Nsta**2)
        DirCsta = np.mod(360 + np.rad2deg(np.arctan2(Esta, Nsta)), 360)
        STA["vel comp Nav"] = xr.DataArray(
            np.array([Esta, Nsta, MagCsta, DirCsta]),
            dims=("dir", "range", "time"),
            attrs={
                "units": "meter/second for speed and degree for direction",
                "long_name": "East/North/Magnitude/Direction velocity components compensated from ship navigation",
            },
        )

    except (KeyError):
        print("no vessel motion compensation in ENX")

    return STA


def process_adcp(file_name, t0, t1, dt_sta, file_out):

    STA_Flag = xr.Dataset()

    # read full ENX
    print("read file")
    ENX = read_WH300(file_name)

    # #ss-ech
    # ENX=ENX.isel(time=slice(0,-1,2))
    # ENX=ENX.isel(time_gps=slice(0,-1,2))

    if len(ENX.data_vars) > 0:
        # select time period
        # time for ENX data seems not robust. GPS time is taken as coordinate
        if (ENX.time_gps.data[0] > np.datetime64(t1)) | (
            ENX.time_gps.data[-1] < np.datetime64(t0)
        ):
            print("no data in specified time period")
            return

        if ENX.time.data[0] < np.datetime64(t0):
            ind0 = 0
        else:
            ind0 = np.where(ENX.time_gps.data >= np.datetime64(t0))[0][0]

        if ENX.time.data[-1] < np.datetime64(t1):
            ind1 = len(ENX.time)
        else:
            ind1 = np.where(ENX.time_gps.data <= np.datetime64(t1))[0][-1]

        ENX = ENX.isel(time=slice(ind0, ind1))
        ENX = ENX.isel(time_gps=slice(ind0, ind1))

        # #compensate for vessel navigation
        # ENX=ADCPcompNav(ENX)

        # #compute STA
        # STA=ENX2STA(ENX,30)

        # filter ENX data on quality parameters
        ####################################
        print("compute filter")
        # set -1000 value if not used
        Ship_speed_max = 10 * 1852 / 3600  # m/s
        Ship_speed_min = 1 * 1852 / 3600  # m/s
        PG4thr = 90  #% criteria DT INSU - non effectif sur des ENX (val 0 ou 100)
        AmpBotThr = 250  # bottom amplitude rejection (after TVG)
        AmpMin = 80  # minimum signal amplitude
        CorThr = 100  # correlation quality
        PitchMax = 15  # °
        RollMax = 15  # °
        PGav = 30  #% min averaged values

        # TVG=20*np.log10(np.repeat([ENX.range.data],len(ENX.time.data),axis=0)).T
        TVG = (
            210
            - 5
            * 20
            * np.log10(
                np.repeat([ENX.range.data], len(ENX.time.data), axis=0)
                / ENX.range.data[0]
            ).T
        )
        TVG = TVG - TVG[0, 0]
        # create individual masks
        nbins = len(ENX.coords["range"])
        mask_VspeedMin = np.repeat(
            [ENX["vessel_speed"] > Ship_speed_min], nbins, axis=0
        )
        mask_VspeedMax = np.repeat(
            [ENX["vessel_speed"] < Ship_speed_max], nbins, axis=0
        )
        mask_PG4 = ENX["prcnt_gd"].sel(beam=4) > PG4thr
        mask_AmpMin = ENX["amp"].mean(dim="beam") > AmpMin
        # mask_AmpBotThr=(ENX["amp"].mean(dim='beam')<AmpBotThr)
        mask_AmpBotThr = ENX["amp"].mean(dim="beam") - TVG < AmpBotThr
        mask_Corr = ENX["corr"].mean(dim="beam") > CorThr
        mask_Pitch = np.repeat([(np.abs(ENX["pitch"]) < PitchMax)], nbins, axis=0)
        mask_Roll = np.repeat([(np.abs(ENX["roll"]) < RollMax)], nbins, axis=0)

        # filtre lobes secondaires du fond
        npg = len(ENX.coords["time"])
        for ip in range(0, npg):
            ibt = np.where(~mask_AmpBotThr[:, ip])
            if len(ibt[0]) > 0:
                ilob = np.ceil((1 - 0.90) * ibt[0][0])
                mask_AmpBotThr[int(ibt[0][0] - ilob) :, ip] = False

        # combine masks
        FlagQual = np.full(np.shape(mask_VspeedMin), True)
        FlagQual = (FlagQual & mask_VspeedMin) if (Ship_speed_max != -1000) else None
        FlagQual = (FlagQual & mask_VspeedMax) if (Ship_speed_max != -1000) else None
        FlagQual = (FlagQual & mask_PG4) if (Ship_speed_max != -1000) else None
        FlagQual = (FlagQual & mask_AmpMin) if (Ship_speed_max != -1000) else None
        FlagQual = (FlagQual & mask_AmpBotThr) if (Ship_speed_max != -1000) else None
        FlagQual = (FlagQual & mask_Corr) if (Ship_speed_max != -1000) else None
        FlagQual = (FlagQual & mask_Pitch) if (Ship_speed_max != -1000) else None
        FlagQual = (FlagQual & mask_Roll) if (Ship_speed_max != -1000) else None

        # create flaged array
        FlagQual6 = np.repeat([FlagQual], 6, axis=0)

        ENX_Flag = ENX.copy(deep=True)
        ENX_Flag["vel"] = ENX_Flag["vel"].astype("float")
        ENX_Flag["amp"] = ENX_Flag["amp"].astype("float")
        ENX_Flag["corr"] = ENX_Flag["corr"].astype("float")
        ENX_Flag["prcnt_gd"] = ENX_Flag["prcnt_gd"].astype("float")
        ENX_Flag["vel"][:, :, :] = np.where(FlagQual6, ENX_Flag.vel.data, np.nan)
        ENX_Flag["amp"][:, :, :] = np.where(
            FlagQual6[0:4, :, :], ENX_Flag.amp.data, np.nan
        )
        ENX_Flag["corr"][:, :, :] = np.where(
            FlagQual6[0:4, :, :], ENX_Flag.corr.data, np.nan
        )
        ENX_Flag["prcnt_gd"][:, :, :] = np.where(
            FlagQual6[0:4, :, :], ENX_Flag.prcnt_gd.data, np.nan
        )

        # compensate for vessel navigation
        print("compensate for vessel motion")
        ENX_Flag = ADCPcompNav(ENX_Flag)

        # compute STA
        print("compute filtered STA")
        STA_Flag = ENX2STA(ENX_Flag, dt_sta)
        STA_Flag["Filt_Thr"] = xr.DataArray(
            np.array(
                [
                    Ship_speed_min,
                    Ship_speed_max,
                    PG4thr,
                    AmpBotThr,
                    AmpMin,
                    CorThr,
                    PitchMax,
                    RollMax,
                    PGav,
                ]
            ),
            dims=("filter"),
            attrs={
                "units": "",
                "long_name": "thresholds used for data filtering. -1000 value if not used",
            },
        )
        STA_Flag.coords["filter"] = [
            "Ship_speed_min",
            "Ship_speed_max",
            "PG4thr",
            "AmpBotThr",
            "AmpMin",
            "CorThr",
            "PitchMax",
            "RollMax",
            "Pct_av",
        ]
        STA_Flag.coords["filter"].attrs = {
            "units": "m/s, m/s, %, RDI count, RDI count, RDI count, degree, degree, %"
        }

        # keep data sufficiently averaged
        FlagNb = STA_Flag.PctEnsAveraged.data > PGav
        FlagNb = np.repeat([FlagNb], 4, axis=0)
        STA_Flag["vel"][:, :, :] = np.where(FlagNb, STA_Flag.vel.data, np.nan)
        STA_Flag["vel comp Nav"][:, :, :] = np.where(
            FlagNb, STA_Flag["vel comp Nav"].data, np.nan
        )
        if len(file_out) > 0:
            STA_Flag.to_netcdf(file_out)

    return STA_Flag
