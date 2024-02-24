# Generate LMA, radar scan locations, and aircraft track movies for TRACER.

# Lightning data in data/Houston/LIGHTNING/6sensor_minimum/
# PX/RAX/SKYLER scan times in data/PX1000.csv, data/RAXPOL.csv, data/SKYLER.csv
# Aircraft tracks (non public) in data/Houston/AircraftTracks/Convair/ and data/Houston/AircraftTracks/Learjet/
# NEXRAD data from KGHX, cached in nexradawscache/
# clone https://github.com/AdamTheisen/cell-track-stats to data/Houston/CSAPR-TRACER-cell-track-stats


# Usage: python quicklook.py YYYYMMDDHHMMSS (start) YYYYMMDDHHMMSS (end) [--overview (for overview only)]
# This probably won't be used outside of TTU lightning research group, so if this is hard to use, shoot me a slack message. /shrug

import warnings
warnings.filterwarnings(action='ignore')

import numpy as np
import datetime
import xarray as xr
from act.io import read_icartt 
import pandas as pd
from pyxlma.plot.xlma_base_plot import BlankPlot, FractionalSecondFormatter
from pyxlma.plot.interactive import InteractiveLMAPlot, event_space_time_limits
import sys
import nexradaws
import pyart
conn = nexradaws.NexradAwsInterface()
from pathlib import Path
import os
import re
from cartopy import crs as ccrs
from pyxlma.coords import RadarCoordinateSystem

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import colors as pltcolors
from matplotlib.dates import AutoDateLocator


import cv2
from PIL import Image
import pytesseract


starttime = datetime.datetime.strptime(sys.argv[1], '%Y%m%d%H%M%S')
endtime = datetime.datetime.strptime(sys.argv[2], '%Y%m%d%H%M%S')

req_date = starttime.replace(hour=0, minute=0, second=0, microsecond=0)
tomorrow_date = req_date + datetime.timedelta(days=1)

ltgpath = 'data/Houston/LIGHTNING/6sensor_minimum/'
convairpath = 'data/Houston/AircraftTracks/Convair/'
learpath = 'data/Houston/AircraftTracks/Learjet/'

lma_filenames = [os.path.join(ltgpath, file) for file in os.listdir(ltgpath) if req_date.strftime('%y%m%d') in file]
tomorrow_lma = [os.path.join(ltgpath, file) for file in os.listdir(ltgpath) if tomorrow_date.strftime('%y%m%d') in file]
lma_filenames.extend(tomorrow_lma)

if len(lma_filenames) == 0:
    print("No LMA files found.")
    exit()

# From DOE / Adam Theisen repo
csapr_scan_csv_filenames =  []
if os.path.exists(f'data/Houston/CSAPR-TRACER-cell-track-stats/data/houcsapr.{req_date.strftime("%Y%m%d")}.csv'):
    csapr_scan_csv_filenames.append(f'data/Houston/CSAPR-TRACER-cell-track-stats/data/houcsapr.{req_date.strftime("%Y%m%d")}.csv')
if os.path.exists(f'data/Houston/CSAPR-TRACER-cell-track-stats/data/houcsapr.{tomorrow_date.strftime("%Y%m%d")}.csv'):
    csapr_scan_csv_filenames.append(f'data/Houston/CSAPR-TRACER-cell-track-stats/data/houcsapr.{tomorrow_date.strftime("%Y%m%d")}.csv')

# ESCAPE aircraft tracks, from a non-public dataset.
# Two research flights on this day. 
convair_track_filenames = [os.path.join(convairpath, file) for file in os.listdir(convairpath) if req_date.strftime('%Y%m%d') in file]
tomorrow_convair = [os.path.join(convairpath, file) for file in os.listdir(convairpath) if tomorrow_date.strftime('%Y%m%d') in file]
convair_track_filenames.extend(tomorrow_convair)

lear_track_filenames = [os.path.join(learpath, file) for file in os.listdir(learpath) if req_date.strftime('%Y%m%d') in file]
tomorrow_lear = [os.path.join(learpath, file) for file in os.listdir(learpath) if tomorrow_date.strftime('%Y%m%d') in file]
lear_track_filenames.extend(tomorrow_lear)


px_1k_csv_filename = 'data/PX1000.csv'
raxpol_csv_filename = 'data/RaXPol.csv'
skyler_csv_filename = 'data/SKYLER.csv'

animation_output_directory = f'./figures/{req_date.strftime("%Y%m%d")}/'

def np_strftime(d, fmt):
    "Have to convert to not nanoseconds, e.g. d.astype('M8[s]')"
    import datetime
    return d.astype(datetime.datetime).strftime(fmt)

### Load data
#### LMA data

alt_data = np.array([], dtype='float32')
lon_data = np.array([], dtype='float32')
lat_data = np.array([], dtype='float32')
time_data = np.array([], dtype='datetime64[ns]')
chi_data = np.array([], dtype='float32')
station_data = np.array([], dtype='float32')

for file in lma_filenames:
    ds = xr.open_dataset(file)
    alt_data = np.append(alt_data, ds.event_altitude.data)
    lon_data = np.append(lon_data, ds.event_longitude.data)
    lat_data = np.append(lat_data, ds.event_latitude.data)
    time_data = np.append(time_data, ds.event_time.data)
    chi_data = np.append(chi_data, ds.event_chi2.data)
    station_data = np.append(station_data, ds.event_stations.data)

    stat_code_data = ds.station_code.data
    stat_lon_data = ds.station_longitude.data
    stat_lat_data = ds.station_latitude.data
    stat_alt_data = ds.station_altitude.data

num_evts_data = np.arange(alt_data.shape[0], dtype='int64')

ds = xr.Dataset(
    data_vars=dict(
        event_altitude=('number_of_events', alt_data),
        event_longitude=('number_of_events', lon_data),
        event_latitude=('number_of_events', lat_data),
        event_time=('number_of_events', time_data),
        event_chi2=('number_of_events', chi_data),
        event_stations=('number_of_events', station_data),
        station_altitude=('station_code', stat_alt_data),
        station_longitude=('station_code', stat_lon_data),
        station_latitude=('station_code', stat_lat_data)
    ),
    coords=dict(
        number_of_events=('number_of_events', num_evts_data),
        station_code=('station_code', stat_code_data),
    )
)

print(f'{len(ds.number_of_events.data)} events between {ds.event_time.data.min()} and {ds.event_time.data.max()}')

## Learjet data
lear_datasets = []
lear_flights_dirs = [f for f in sorted(os.listdir('data/Houston/LEAR_data/')) if starttime.strftime('%Y%m%d') in f]
for dir in lear_flights_dirs:
    file_to_read = [f for f in sorted(os.listdir(f'data/Houston/LEAR_data/{dir}')) if f.endswith('.ict') and 'Page0' in f][0]
    lear_ds = read_icartt(f'data/Houston/LEAR_data/{dir}/{file_to_read}')
    lear_datasets.append(lear_ds)
if len(lear_datasets) > 0:
    lear_datasets = xr.concat(lear_datasets, dim='time')
    temp_vmin = np.min(lear_datasets.Dew.data)
    temp_vmax = np.max(lear_datasets.Temp.data)
else:
    lear_datasets = None




#### Learjet video
video_file_idx = 0
video_seek_idx = 0
last_image_dt = None
last_success_idx = None
lear_video_files = [f for f in sorted(os.listdir('data/Houston/NRC_aerosol_videos/')) if starttime.strftime('%Y-%m-%d') in f and f.endswith('.avi')]
lear_videos = []
if len(lear_video_files) > 0:
    for f in lear_video_files:
        lear_videos.append(cv2.VideoCapture(os.path.join('data/Houston/NRC_aerosol_videos/', f)))

#### Convair UHSAS data
uhsas_cabin_files = [f for f in os.listdir('data/Houston/NRC_aerosol_data/') if f.startswith(starttime.strftime('%Y%m%d')) and f.endswith('UHSAS_c.nc')]
uhsas_cabin_datasets = []
if len(uhsas_cabin_files) > 0:
    for f in uhsas_cabin_files:
        this_uhsas = xr.open_dataset(os.path.join('data/Houston/NRC_aerosol_data/', f))
        uhsas_cabin_datasets.append(this_uhsas)
    uhsas_cabin_datasets = xr.concat(uhsas_cabin_datasets, dim='time')
else:
    uhsas_cabin_datasets = None


uhsas_wing_files = [f for f in os.listdir('data/Houston/NRC_aerosol_data/') if f.startswith(starttime.strftime('%Y%m%d')) and f.endswith('UHSAS_w.nc')]
uhsas_wing_datasets = []
if len(uhsas_wing_files) > 0:
    for f in uhsas_wing_files:
        this_uhsas = xr.open_dataset(os.path.join('data/Houston/NRC_aerosol_data/', f))
        uhsas_wing_datasets.append(this_uhsas)
    uhsas_wing_datasets = xr.concat(uhsas_wing_datasets, dim='time')
else:
    uhsas_wing_datasets = None

uhsas_vmax = np.max([uhsas_cabin_datasets.Nuhsas_c.max(), uhsas_wing_datasets.Nuhsas_w.max()])
#### Research radar scans

csapr_lon, csapr_lat = -95.283893, 29.531782

def load_csapr_scan_table(csapr_scan_csv_filename):
    csapr_arm = pd.read_csv(csapr_scan_csv_filename, 
        parse_dates=['time']).drop('Unnamed: 0', axis='columns').rename(
        {'azimith_max':'azimuth_max'}, axis='columns')
    csapr_arm['rhi/ppi'] = csapr_arm['scan_mode']
    csapr_arm['date_time_start'] = csapr_arm['time']

    # Use the start of the next time window as the end of the current time window.
    # Needs a minus sign due to periods=-1.
    csapr_arm['date_time_end'] = csapr_arm['date_time_start'] - csapr_arm['time'].diff(periods=-1)

    csapr_arm['longitude'] = csapr_lon
    csapr_arm['latitude'] = csapr_lat
    
    csapr_arm_final = []
    for mode, table in  csapr_arm.groupby('scan_mode'):
        d_az = (table.azimuth_max-table.azimuth_min).abs().max()
        d_el = (table.elevation_max-table.elevation_min).abs().max()
        print(f"Max azimuth difference of {d_az} for {mode}")
        print(f"Max elevation difference of {d_el} for {mode}")
        constant_angle = 'n/a'
        sweep_angle = 'n/a'
        if mode == 'rhi':
            print('rhi found')
            # Also handles hsrhi
            if d_az < 1.0:
                constant_angle = table.azimuth_max.astype('str')
            else:
                constant_angle = csapr_arm.azimuth_min.map('{:3.2f}'.format) + ',' + csapr_arm.azimuth_max.map('{:3.2f}'.format)
            sweep_angle = csapr_arm.elevation_min.map('{:3.2f}'.format) + '-' + csapr_arm.elevation_max.map('{:3.2f}'.format)   
        if mode == 'ppi':
            print('ppi found')
            sweep_angle = csapr_arm.azimuth_min.map('{:3.2f}'.format) + '-' + csapr_arm.azimuth_max.map('{:3.2f}'.format)
            constant_angle = csapr_arm.elevation_min.map('{:3.2f}'.format) + '-' + csapr_arm.elevation_max.map('{:3.2f}'.format)        

        table['constant angle'] = constant_angle
        table['sweep angle range'] = sweep_angle

        csapr_arm_final.append(table)
    csapr_arm_final = pd.concat(csapr_arm_final, ignore_index=True)
    return csapr_arm_final

if len(csapr_scan_csv_filenames) > 0:
    csapr = [load_csapr_scan_table(csapr_scan_csv_filename) for csapr_scan_csv_filename in csapr_scan_csv_filenames]
    csapr = pd.concat(csapr, ignore_index=True)
else:
    csapr = None

if px_1k_csv_filename is not None:
    px1k = pd.read_csv(px_1k_csv_filename, parse_dates=['date_time_start', 'date_time_end'])
    if 'sweep angle range' not in px1k.columns:
        px1k['sweep angle range'] =  pd.Series(['n/a']*len(px1k))
else:
    px1k = None

if raxpol_csv_filename is not None:
    raxpol = pd.read_csv(raxpol_csv_filename, parse_dates=['date_time_start', 'date_time_end'])
    if 'sweep angle range' not in raxpol.columns:
        raxpol['sweep angle range'] =  pd.Series(['n/a']*len(raxpol))
else:
    raxpol = None

if skyler_csv_filename is not None:
    skyler = pd.read_csv(skyler_csv_filename, parse_dates=['date_time_start', 'date_time_end'])
    if 'sweep angle range' not in skyler.columns:
        skyler['sweep angle range'] =  pd.Series(['n/a']*len(skyler))
else:
    skyler = None

def load_lear_track(lear_track_filename):
    lear_track_df = pd.read_csv(lear_track_filename, sep='\s+', 
                       parse_dates={'time_raw':['yyyy', 'month', 'day', 'hh', 'mm','ss']})
    # Make the columns match the convair dataframe format.
    lear_track_df['time'] = pd.to_datetime(lear_track_df.time_raw, format='%Y %m %d %H %M %S')
    lear_track_df['AltM']=lear_track_df['Alt(ft)']/3.28084
    lear_track_df['Lon']=lear_track_df['Longitude(deg)']
    lear_track_df['Lat']=lear_track_df['Latitude(deg)']
    return lear_track_df

def load_convair_track(convair_track_filename):
    convair_track_df = pd.read_csv(convair_track_filename,
        parse_dates=['Time']).rename(
        columns={'Time':'time','lon':'Lon','lat':'Lat','alt':'AltM'}
    )
    return convair_track_df


if len(lear_track_filenames) > 0:
    lear_track_dfs = [load_lear_track(lear_track_filename) for lear_track_filename in lear_track_filenames]
    lear_track_df = pd.concat(lear_track_dfs)
else:
    lear_track_df = None

if len(convair_track_filenames) > 0:
    convair_track_dfs = [load_convair_track(convair_track_filename) for convair_track_filename in convair_track_filenames]
    convair_track_df = pd.concat(convair_track_dfs)
else:
    convair_track_df = None

def overlapping_date_range(d_start, d_end, t_start, t_end):
    """
    d is data, t valid time range.
    ------------------------
    (ts > rs) & (ts < de) & (te > ds) & (te > de)
    d |      |
    t    |       |
    ------------------------
    (ts > ds) & (ts < de) & (te > ds) & (te > de)
    d    |      |
    t |       |
    ------------------------
    (ts > de)
    d    |      |
    t              |       |
    ------------------------
    (te < ds)
    d              |       |
    t    |      |
    ------------------------
    
    Not either of the last two should be sufficent.
    """
    in_range = ~((t_start > d_end) | (t_end < d_start))
    return in_range

def swap_angle_basis(angle):
    """ Switch from geographic to math angle, or vice versa, using this one weird trick."""
    return (450.0 - angle) % 360.0

def radar_ray_coords(radar_ctr, ray_angle, ray_distance_km=30):
    """ Returns an estimated ([lon0,lon_end], [lat0, lat_end]) in degrees
        for the given radar position, angle of the ray in degrees,  and distance 
    """
    ray_length = ray_distance_km / 111.0 # km/(km/deg)
    ray = np.deg2rad(ray_angle)
    return ([radar_ctr[0], radar_ctr[0] + ray_length*np.sin(ray)], 
            [radar_ctr[1], radar_ctr[1] + ray_length*np.cos(ray)])


tlim = pd.to_datetime(starttime).to_pydatetime(), pd.to_datetime(endtime).to_pydatetime()

class myBlankPlot(BlankPlot):
    def __init__(self, stime, bkgmap, **kwargs):
        super().__init__(stime, bkgmap, **kwargs)

    def plot(self, **kwargs):
        super().plot(**kwargs)
        FIG_WIDTH = 24.5
        FIG_HEIGHT = 13.75

        self.fig.set_size_inches(FIG_WIDTH, FIG_HEIGHT)

        self.ax_th.set_position([.85/FIG_WIDTH, (FIG_HEIGHT-1.4)/FIG_HEIGHT, 7.055/FIG_WIDTH, 1.1/FIG_HEIGHT])
        self.ax_lon.set_position([.85/FIG_WIDTH, (FIG_HEIGHT-3.05)/FIG_HEIGHT, 5.525/FIG_WIDTH, 1.1/FIG_HEIGHT])
        self.ax_hist.set_position([6.8/FIG_WIDTH, (FIG_HEIGHT-3.05)/FIG_HEIGHT, 1.105/FIG_WIDTH, 1.1/FIG_HEIGHT])
        self.ax_plan.set_position([.85/FIG_WIDTH, (FIG_HEIGHT-9.1)/FIG_HEIGHT, 5.525/FIG_WIDTH, 5.5/FIG_HEIGHT])
        self.ax_lat.set_position([6.8/FIG_WIDTH, (FIG_HEIGHT-9.1)/FIG_HEIGHT, 1.105/FIG_WIDTH, 5.5/FIG_HEIGHT])
        

        self.ax_cam = self.fig.add_axes([8.6/FIG_WIDTH, (FIG_HEIGHT-5.62)/FIG_HEIGHT, 7.055/FIG_WIDTH, 5.32/FIG_HEIGHT])
        self.ax_cam.set_title('SPEC Learjet Data')
        self.ax_cam.axis('off')

        self.ax_lear_td = self.fig.add_axes([8.6/FIG_WIDTH, (FIG_HEIGHT-7.35)/FIG_HEIGHT, 7.055/FIG_WIDTH, 1.1/FIG_HEIGHT])
        self.ax_lear_td.xaxis.set_major_formatter(FractionalSecondFormatter(self.ax_lear_td))
        self.ax_lear_td.xaxis.set_major_locator(AutoDateLocator())
        self.ax_lear_td.minorticks_on()
        self.ax_lear_td.set_xlabel('Time (UTC)')
        self.ax_lear_td.set_ylabel('Temperature ($\\degree$C)')

        
        self.ax_lear_cwc = self.fig.add_axes([8.6/FIG_WIDTH, (FIG_HEIGHT-9.35)/FIG_HEIGHT, 7.055/FIG_WIDTH, 1.1/FIG_HEIGHT])
        self.ax_lear_cwc.xaxis.set_major_formatter(FractionalSecondFormatter(self.ax_lear_cwc))
        self.ax_lear_cwc.xaxis.set_major_locator(AutoDateLocator())
        self.ax_lear_cwc.minorticks_on()
        self.ax_lear_cwc.set_xlabel('Time (UTC)')
        self.ax_lear_cwc.set_ylabel('Water Content ($\\frac{g}{m^{3}}$)')

        self.ax_cabin_uhsas = self.fig.add_axes([16.5/FIG_WIDTH, (FIG_HEIGHT-7.35)/FIG_HEIGHT, 7.055/FIG_WIDTH, 1.1/FIG_HEIGHT])
        self.ax_cabin_uhsas.xaxis.set_major_formatter(FractionalSecondFormatter(self.ax_cabin_uhsas))
        self.ax_cabin_uhsas.xaxis.set_major_locator(AutoDateLocator())
        self.ax_cabin_uhsas.minorticks_on()
        self.ax_cabin_uhsas.set_xlabel('Time (UTC)')
        self.ax_cabin_uhsas.set_ylabel('Diameter ($\\mu$m)')
        self.ax_cabin_uhsas.set_title('UHSAS (Aircraft Cabin)')

        
        self.ax_wing_uhsas = self.fig.add_axes([16.5/FIG_WIDTH, (FIG_HEIGHT-9.35)/FIG_HEIGHT, 7.055/FIG_WIDTH, 1.1/FIG_HEIGHT])
        self.ax_wing_uhsas.xaxis.set_major_formatter(FractionalSecondFormatter(self.ax_wing_uhsas))
        self.ax_wing_uhsas.xaxis.set_major_locator(AutoDateLocator())
        self.ax_wing_uhsas.minorticks_on()
        self.ax_wing_uhsas.set_xlabel('Time (UTC)')
        self.ax_wing_uhsas.set_ylabel('Diameter ($\\mu$m)')
        self.ax_wing_uhsas.set_title('UHSAS (Aircraft Wing)')

        self.cax_1 = self.fig.add_axes([1.1/FIG_WIDTH, (FIG_HEIGHT-13.25)/FIG_HEIGHT, 5.525/FIG_WIDTH, 0.01])
        self.cax_2 = self.fig.add_axes([16.5/FIG_WIDTH, (FIG_HEIGHT-13.25)/FIG_HEIGHT, 7.055/FIG_WIDTH, 0.01])

class AnnotatedLMAPlot(InteractiveLMAPlot):
    def __init__(self, ds, xlim=None, ylim=None, zlim=None, tlim=None, **kwargs):
        xlim_ds, ylim_ds, zlim_ds, tlim_ds = event_space_time_limits(ds)
        if xlim is None: xlim = xlim_ds
        if ylim is None: ylim = ylim_ds
        if zlim is None: zlim = zlim_ds
        if tlim is None: tlim = tlim_ds

        self.xlim = xlim
        self.ylim = ylim
        self.zlim = zlim
        self.tlim = tlim
        super().__init__(ds, **kwargs)

    # @output.capture()
    def make_plot(self):
        tlim = self.bounds['t']
        tlim_sub = pd.to_datetime(tlim[0]), pd.to_datetime(tlim[1])
        tstring = 'LMA {}-{}'.format(tlim_sub[0].strftime('%H%M'),
                                     tlim_sub[1].strftime('%H%M UTC %d %B %Y '))
        self.lma_plot = myBlankPlot(pd.to_datetime(tlim_sub[0]), bkgmap=True,
                      xlim=self.xlim, ylim=self.ylim, zlim=self.zlim, tlim=self.tlim, title=tstring)
        super(AnnotatedLMAPlot, self).make_plot()
        self.lma_plot.ax_th.set_title(tstring)
        # For longer durations, successive overplots of PPIs will 
        # completely hide the LMA data.
        plot_duration_sec = (tlim[1]-tlim[0]).total_seconds()
        if plot_duration_sec > 10*60:
            radar_alpha = 0.0 # transparent
        else:
            radar_alpha = 0.1 # slight shading
        
                
        for radar_name, radar_df, radar_color in zip(('PX1000', 'CSAPR', 'RaXPol', 'SKYLER'),
                (px1k, csapr, raxpol, skyler), ((.5,.5,.5), (.4,.4,.4), (.5,.5,.5), (.5,.5,.5))):

            # Skip any radar we don't have.
            if radar_df is None: continue
                        
            in_range = overlapping_date_range(radar_df['date_time_start'], radar_df['date_time_end'],
                                              self.bounds['t'][0], self.bounds['t'][1])
            radar_in_time = radar_df[in_range]
            # print(radar_in_time)
            radar_ctr = radar_in_time['longitude'].mean(), radar_in_time['latitude'].mean()
            ra, = self.lma_plot.ax_plan.plot(radar_ctr[0], radar_ctr[1], color=radar_color, marker='o', ms=4)
            radar_artists = [ra]
            for scan_type, scan_angle, sweep_angle_range in zip(radar_in_time['rhi/ppi'], radar_in_time['constant angle'],
                                             radar_in_time['sweep angle range']):
                # print(radar_name, scan_type, scan_angle, sweep_angle_range)
                if scan_type.lower() == 'rhi':
                    if scan_angle != 'n/a':
                        scan_angles = list(map(float, scan_angle.split(',')))
                        for ray in scan_angles:
                            # FIXME: do a proper map projection (or geod transform)
                            ray_lons, ray_lats = radar_ray_coords(radar_ctr, ray)
                            ra, = self.lma_plot.ax_plan.plot(ray_lons, ray_lats,
                                                       color=radar_color, linewidth=.5)
                            radar_artists.append(ra)
                if scan_type.lower() == 'ppi':
                    if sweep_angle_range != 'n/a':
                        geog_angle_start_end = tuple(map(float, sweep_angle_range.split('-')))
                        # Switch order, because the winding is CCW instead of CW
                        angle_end, angle_start = map(swap_angle_basis, geog_angle_start_end)
                        # if radar_name == 'CSAPR':
                            # print(geog_angle_start_end, (angle_start, angle_end))
                        ppi_range_km = 60.0
                        km_per_deg = 111.0
                        ra = matplotlib.patches.Wedge(radar_ctr, ppi_range_km/km_per_deg, angle_start, angle_end,
                                                         facecolor=radar_color+(radar_alpha,), edgecolor=radar_color)
                        self.lma_plot.ax_plan.add_artist(ra)
                    else:
                        ra, = self.lma_plot.ax_plan.plot(radar_ctr[0], radar_ctr[1],
                                                   markerfacecolor="None", marker='o', ms=16,
                                                   markeredgecolor=radar_color, markeredgewidth=.5)
                    radar_artists.append(ra)
            self.data_artists.extend(radar_artists)
                    
                    
        for track_df, track_highlight_color, whole_track_color in zip(
                (convair_track_df, lear_track_df), ('red', 'blue'), ((.9,.8,.8), (.8,.8,.9))):
            # Skip any aircraft we don't have.
            if track_df is None: continue

            # time_pad = pd.to_timedelta('3 min')
            time_pad = pd.to_timedelta('3 sec')
            track_in_time = ((track_df.time >= (self.bounds['t'][0]-time_pad)) & 
                             (track_df.time <= (self.bounds['t'][1]+time_pad)) &
                             (track_df.Lon >= self.bounds['x'][0]) & (track_df.Lon <= self.bounds['x'][1]) &
                             (track_df.Lat >= self.bounds['y'][0]) & (track_df.Lat <= self.bounds['y'][1])
                            )
        
            current_track = track_df[track_in_time]
            c0, = self.lma_plot.ax_plan.plot(track_df.Lon, track_df.Lat, color=whole_track_color, zorder=-10)
            c1, = self.lma_plot.ax_plan.plot(current_track.Lon, current_track.Lat, color=track_highlight_color)
            c2, = self.lma_plot.ax_th.plot(track_df.time, track_df.AltM/1000.0, color=whole_track_color, zorder=-10)
            c3, = self.lma_plot.ax_th.plot(current_track.time, current_track.AltM/1000.0, color=track_highlight_color)
            c4, = self.lma_plot.ax_lon.plot(track_df.Lon, track_df.AltM/1000.0, color=whole_track_color, zorder=-10)
            c5, = self.lma_plot.ax_lon.plot(current_track.Lon, current_track.AltM/1000.0, color=track_highlight_color)
            c6, = self.lma_plot.ax_lat.plot(track_df.AltM/1000.0, track_df.Lat, color=whole_track_color, zorder=-10)
            c7, = self.lma_plot.ax_lat.plot(current_track.AltM/1000.0, current_track.Lat, color=track_highlight_color)
            track_artists = [c0,c1,c2,c3,c4,c5,c6,c7]
            self.data_artists.extend(track_artists)
        
        time_pad_nexrad = pd.to_timedelta('5 min')
        scans = conn.get_avail_scans_in_range(tlim[0]-time_pad_nexrad, tlim[1], 'KHGX')
        scans = [scan for scan in scans if "_MDM" not in scan.filename]
        if len(scans) > 0:
            if os.path.exists(os.path.join('nexradawscache', scans[-1].filename)):
                radar = pyart.io.read(os.path.join('nexradawscache', scans[-1].filename)).extract_sweeps([0])
            else:
                res = conn.download(scans[-1], 'nexradawscache')
                radar = res.success[0].open_pyart().extract_sweeps([0])
            reflec = radar.get_field(0, 'reflectivity')
            if hasattr(self, 'last_radar_filename') and (self.last_radar_filename == scans[-1].filename) and hasattr(self, 'radar_x'):
                pass
            else:
                rcs = RadarCoordinateSystem(radar.latitude['data'][0], radar.longitude['data'][0], radar.altitude['data'][0])
                rng, el = np.meshgrid(radar.range['data'], radar.fixed_angle['data'][0]*np.ones_like(radar.azimuth['data']))
                az = np.meshgrid(radar.range['data'], radar.azimuth['data'])[1]
                rx, ry, _ = rcs.toLonLatAlt(rng, az, el)
                self.radar_x = np.array(rx).reshape(reflec.shape)
                self.radar_y = np.array(ry).reshape(reflec.shape)
                self.last_radar_filename = scans[-1].filename
            pcm = self.lma_plot.ax_plan.pcolormesh(self.radar_x, self.radar_y, reflec, vmin=-10, vmax=80, cmap='pyart_ChaseSpectral', zorder=0, transform=ccrs.PlateCarree())
            self.lma_plot.ax_plan.set_title(pyart.util.datetime_from_radar(radar).strftime('KGHX Lowest Elevation Reflectivity %H%M%S UTC'))
            self.data_artists.append(pcm)
            self.lma_plot.fig.colorbar(pcm, cax=self.lma_plot.cax_1, orientation='horizontal', label='Reflectivity (dBZ)')
        if uhsas_cabin_datasets is not None:
            tminnumpy = np.array([tlim[0]]).astype('datetime64[s]')[0]
            tmaxnumpy = np.array([tlim[1]]).astype('datetime64[s]')[0]
            times_to_select = (uhsas_cabin_datasets.time.data > tminnumpy) & (uhsas_cabin_datasets.time.data < tmaxnumpy)
            this_aerosol_data = uhsas_cabin_datasets.isel(time=times_to_select)
            if this_aerosol_data.Nuhsas_c.shape[0] > 0:
                if len(this_aerosol_data.bin_lower_uhsas_c.data) > 0 and np.all(this_aerosol_data.bin_lower_uhsas_c.data[0, 1:] == this_aerosol_data.bin_upper_uhsas_c.data[0, :-1]):
                    bin_sequence = np.append(this_aerosol_data.bin_lower_uhsas_c.data[0, :], this_aerosol_data.bin_upper_uhsas_c.data[0, -1])
                    time_edges = np.append(this_aerosol_data.time.data, this_aerosol_data.time.data[-1] + np.diff(this_aerosol_data.time.data)[-1])
                    dsdhandle = self.lma_plot.ax_cabin_uhsas.pcolormesh(time_edges, bin_sequence, this_aerosol_data.Nuhsas_c.data, cmap='viridis', norm=pltcolors.LogNorm(vmin=1, vmax=uhsas_vmax))
                    self.lma_plot.fig.colorbar(dsdhandle, cax=self.lma_plot.cax_2, orientation='horizontal', label='Count per bin')
                else:
                    dsdhandle = self.lma_plot.ax_cabin_uhsas.text(0.5, 0.5, 'No UHSAS data', ha='center', va='center', transform=self.lma_plot.ax_cabin_uhsas.transAxes)
            else:
                dsdhandle = self.lma_plot.ax_cabin_uhsas.text(0.5, 0.5, 'No UHSAS data', ha='center', va='center', transform=self.lma_plot.ax_cabin_uhsas.transAxes)
            self.data_artists.append(dsdhandle)
        self.lma_plot.ax_cabin_uhsas.set_xlim(tlim[0], tlim[1])
        self.lma_plot.ax_cabin_uhsas.set_ylim(0, 1)
        if uhsas_wing_datasets is not None:
            tminnumpy = np.array([tlim[0]]).astype('datetime64[s]')[0]
            tmaxnumpy = np.array([tlim[1]]).astype('datetime64[s]')[0]
            times_to_select = (uhsas_wing_datasets.time.data > tminnumpy) & (uhsas_wing_datasets.time.data < tmaxnumpy)
            this_aerosol_data = uhsas_wing_datasets.isel(time=times_to_select)
            if this_aerosol_data.Nuhsas_w.shape[0] > 0:
                if len(this_aerosol_data.bin_lower_uhsas_w.data) > 0 and np.all(this_aerosol_data.bin_lower_uhsas_w.data[0, 1:] == this_aerosol_data.bin_upper_uhsas_w.data[0, :-1]):
                    bin_sequence = np.append(this_aerosol_data.bin_lower_uhsas_w.data[0, :], this_aerosol_data.bin_upper_uhsas_w.data[0, -1])
                    time_edges = np.append(this_aerosol_data.time.data, this_aerosol_data.time.data[-1] + np.diff(this_aerosol_data.time.data)[-1])
                    dsdhandle = self.lma_plot.ax_wing_uhsas.pcolormesh(time_edges, bin_sequence, this_aerosol_data.Nuhsas_w.data, cmap='viridis', norm=pltcolors.LogNorm(vmin=1, vmax=uhsas_vmax))
                    self.lma_plot.fig.colorbar(dsdhandle, cax=self.lma_plot.cax_2, orientation='horizontal', label='Count per bin')
                else:
                    dsdhandle = self.lma_plot.ax_wing_uhsas.text(0.5, 0.5, 'No UHSAS data', ha='center', va='center', transform=self.lma_plot.ax_wing_uhsas.transAxes)
            else:
                dsdhandle = self.lma_plot.ax_wing_uhsas.text(0.5, 0.5, 'No UHSAS data', ha='center', va='center', transform=self.lma_plot.ax_wing_uhsas.transAxes)
            self.data_artists.append(dsdhandle)
        self.lma_plot.ax_wing_uhsas.set_xlim(tlim[0], tlim[1])
        self.lma_plot.ax_wing_uhsas.set_ylim(0, 1)

        self.lma_plot.ax_cam.get_children().clear()
        if len(lear_videos) > 0 and tlim[1] - tlim[0] < datetime.timedelta(minutes=6):
            good_frames = []
            good_times = []
            global video_file_idx
            global video_seek_idx
            global last_image_dt
            global last_success_idx
            if lear_videos[video_file_idx].isOpened() and video_file_idx != -1 and video_seek_idx != -1:
                while video_seek_idx < lear_videos[video_file_idx].get(cv2.CAP_PROP_FRAME_COUNT):
                    if last_image_dt is not None and last_image_dt > tlim[1]:
                        break
                    lear_videos[video_file_idx].set(cv2.CAP_PROP_POS_FRAMES, video_seek_idx)
                    ret, frame = lear_videos[video_file_idx].read()
                    if ret:
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        ocr_rgb = frame_rgb.copy()[:58, :360, :]
                        greenmask = (ocr_rgb[:, :, 1] > 175) & (ocr_rgb[:, :, 0] < 150) & (ocr_rgb[:, :, 2] < 150)
                        ocr_rgb[:, :, 1][~greenmask] = 0
                        ocr_rgb[:, :, 1][greenmask] = 255
                        ocr_rgb[:, :, 0] = ocr_rgb[:, :, 1]
                        ocr_rgb[:, :, 2] = ocr_rgb[:, :, 1]
                        pil_image = Image.fromarray(ocr_rgb)
                        string_lines_in_image = pytesseract.image_to_string(pil_image).split('\n')
                        image_dt = None
                        for line in string_lines_in_image:
                            if 'UTC' in line:
                                try:
                                    line = re.sub(r'[^A-Za-z0-9]+', '', line).replace('UTC', '000UTC')
                                    image_dt = datetime.datetime.strptime(line, '%Y%m%d%H%M%S%fUTC')
                                    if image_dt.year != tlim[0].year and image_dt.year != tlim[1].year:
                                        image_dt = image_dt.replace(year=tlim[0].year)
                                    if image_dt.month != tlim[0].month and image_dt.month != tlim[1].month:
                                        image_dt = image_dt.replace(month=tlim[0].month)
                                    if image_dt.day != tlim[0].day and image_dt.day != tlim[1].day:
                                        image_dt = image_dt.replace(day=tlim[0].day)
                                    if last_image_dt is not None and last_success_idx is not None:
                                        diff_in_times = (image_dt - last_image_dt).total_seconds()
                                        if diff_in_times < 0 or diff_in_times > 60+((video_seek_idx - last_success_idx)/30):
                                            image_dt = None
                                            break
                                    last_image_dt = image_dt
                                    last_success_idx = video_seek_idx
                                except ValueError as e:
                                    image_dt = None
                                break
                        if image_dt is not None:
                            if image_dt > tlim[0] and image_dt < tlim[1]:
                                good_frames.append(frame_rgb)
                                good_times.append(image_dt)
                            if image_dt > tlim[1]:
                                break
                        video_seek_idx += 150
                else:
                    video_seek_idx = 0
                    last_success_idx = None
                    last_image_dt = None
                    video_file_idx += 1
                    if video_file_idx >= len(lear_videos):
                        video_file_idx = -1
                        video_seek_idx = -1
            if len(good_frames) > 0:
                plane_image = self.lma_plot.ax_cam.imshow(good_frames[-1])
            else:
                plane_image = self.lma_plot.ax_wing_uhsas.text(0.5, 0.5, 'Plane not in flight', ha='center', va='center', transform=self.lma_plot.ax_cam.transAxes)
        if lear_datasets is not None:
            this_time_lear_data = lear_datasets.sel(time=slice(tlim[0], tlim[1]))
            if this_time_lear_data.time.shape[0] > 0:
                lear_temp = self.lma_plot.ax_lear_td.plot(this_time_lear_data.time.data, this_time_lear_data.Temp.data, color='red')
                lear_dew = self.lma_plot.ax_lear_td.plot(this_time_lear_data.time.data, this_time_lear_data.Dew.data, color='lime')
                self.lma_plot.ax_lear_td.set_xlim(tlim[0], tlim[1])
                self.lma_plot.ax_lear_td.set_ylim(temp_vmin, temp_vmax)
                self.data_artists.extend(lear_temp)
                self.data_artists.extend(lear_dew)
            else:
                lear_temp = self.lma_plot.ax_lear_td.text(0.5, 0.5, 'No Learjet data', ha='center', va='center', transform=self.lma_plot.ax_lear_td.transAxes)
                self.data_artists.append(lear_temp)

interactive_lma = AnnotatedLMAPlot(ds, tlim=tlim)

if os.path.exists(os.path.join(animation_output_directory, 'latlon.txt')):
    ll = np.genfromtxt(os.path.join(animation_output_directory, 'latlon.txt'), delimiter=',')
    interactive_lma.lma_plot.ax_plan.set_xlim(ll[0], ll[1])
    interactive_lma.lma_plot.ax_plan.set_ylim(ll[2], ll[3])

Path(animation_output_directory).mkdir(parents=True, exist_ok=True)


interactive_lma.lma_plot.fig.savefig(os.path.join(animation_output_directory, 
                                                  "LMA_overview_"+tlim[0].strftime('%Y%m%d')+'.png'))

if "--overview" in sys.argv:
    exit()

animation_output_directory = animation_output_directory + '/anim/'
Path(animation_output_directory).mkdir(parents=True, exist_ok=True)
Path(animation_output_directory.replace('anim/', 'loop_FED/')).mkdir(parents=True, exist_ok=True)

if animation_output_directory is not None:

    dt = pd.to_timedelta('60 seconds')
    tspan = pd.to_timedelta('5 minutes')
    t0, tf = tlim[0], tlim[1]

    n_frames=int((tf-t0)/dt)
    time_limits = [(t0+iframe*dt, t0+tspan+iframe*dt) for iframe in range(n_frames)]
    for iframe, tlimi in enumerate(time_limits):
        ti0 = tlimi[0].strftime('%Y%m%d_%H%M%S')
        filename = os.path.join(f"{animation_output_directory}", f"LMA_aircraft_radar_{ti0}.png")
        interactive_lma.lma_plot.ax_th.set_xlim(tlimi)
        interactive_lma.lma_plot.fig.canvas.draw()
        interactive_lma.lma_plot.fig.savefig(filename)
        if (iframe % 25) == 0:
            print(iframe)
            print(filename)

