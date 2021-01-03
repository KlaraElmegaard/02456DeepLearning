import numpy as np
import pandas as pd
import pickle

#from torch.utils.data import Dataset

#requirements for random pick of users
import random


#######################################
#PLOT
#requrements for compute_roc() and other plots
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import clear_output


#SKLEARN
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import roc_curve, auc

scaler = MinMaxScaler()
#scaler = StandardScaler()

#=======================================
# PICKLE COMPRESSED
#=======================================
import bz2
import pickle
import _pickle as cPickle
#=======================================

#=======================================
#PLOT ON MAPS
# It doesn't work on Colab
#=======================================
#requirements for plot_filtered()
from datashader.bokeh_ext import InteractiveImage
import bokeh.plotting as bp
#import datashader.transfer_functions as dtf
from colorcet import fire, glasbey_warm, glasbey_dark, glasbey_light, bkr
import datashader as ds
#import holoviews as hv
#import geoviews as gv
#from holoviews.operation.datashader import datashade
#hv.extension('bokeh')
#import bokeh.plotting as bp
from bokeh.models.tiles import WMTSTileSource

#=======================================
# PANDAS ON GPU from redis - requires conda
# To install coDF on Colab, 
# Follow instructions at https://stackoverflow.com/questions/60188071/installing-cudf-cuml-into-colab-with-rapids-ai-version-0-11
#=======================================
#import cudf
import math as Math
#=======================================

def scale_by_ID(df, ixx):
    """
    """
    #ixxStd = [xx+'std' for xx in ixx]
    #df[ixxStd]=0
    for ID in df['user'].unique():
        #df.loc[ df['user'][(df['user']==ID)].index, (ixxStd)] = scaler.fit_transform(df[ixx][(df['user']==ID)])
        df.loc[ df['user'][(df['user']==ID)].index, (ixx)] = scaler.fit_transform(df[ixx][(df['user']==ID)])
        
    return df

def load_user_data(user, data_directory = '/mnt/sdb1/data_valse/DeepLearning2020/Pickle', load_Dummies=False):
    """
    """
    
    
    
    ts = pd.to_datetime(decompress_pickle(f'{data_directory}/TS_{user}.pbz2'))      
    
    
    labels2 = np.array(decompress_pickle(f'{data_directory}/label2_{user}.pbz2')).astype(int)
   
    
    labelsM = np.array(decompress_pickle(f'{data_directory}/labelM_{user}.pbz2')).astype(int)
   
    
    labelsP = np.array(decompress_pickle(f'{data_directory}/labelP_{user}.pbz2')).astype(int)
   
    
    pos_webm = np.array(decompress_pickle(f'{data_directory}/pos_xW_yW_{user}.pbz2'))
    
    if load_Dummies:
        dummy_array = decompress_pickle(f'{data_directory}/dummies_array_{user}.pbz2')
    else:
        dummy_array = None
    
                            
    return user, ts, pos_webm, dummy_array, labels2, labelsM, labelsP

def create_dummies_pickle_from_images(user, data_directory = '/mnt/sdb1/data_valse/DeepLearning2020/Pickle'):
    """
    """
    image_data = decompress_pickle(f'{data_directory}/images_list_{user}.pbz2')
        
    dummy = np.stack([np.stack([image_data[j,:,:k].sum() for k in range(0,image_data.shape[3])], axis = 0 ) for j in range(0,image_data.shape[0])], axis = 0 )
    dummy = np.where(dummy > 0, 1, 0)
    compress_pickle(f'{data_directory}/dummies_array_{user}', dummy)
    
def create_data_frame(user, ts, pos_webm, dummy_array, labels2, labelsM, labelsP, segmentation = False, seq_cutoff_time = 300, seq_cutoff_speed = 42, Standardize = False):
    """
    """
    df = pd.DataFrame({
        'user': user,
        'ts': ts,
        'image_ix': np.arange(0, ts.shape[0]),
        'x': pos_webm[:,0],
        'y': pos_webm[:,1],
        'x_web': pos_webm[:,0],
        'y_web': pos_webm[:,1],
        'label2': labels2, 
        'labelP': labelsP, 
        'labelM': labelsM
    })
    
    if dummy_array is not None:
        
        df['f_highway_motorway'] = dummy_array[:,0]
        df['f_traffic_signals'] = dummy_array[:,1]
        df['f_bus_stops'] = dummy_array[:,2]
        df['f_landuse_meadow'] = dummy_array[:,3]
        df['f_landuse_residential'] = dummy_array[:,4]
        df['f_landuse_industrial'] = dummy_array[:,5]
        df['f_landuse_commercial'] = dummy_array[:,6]
        df['f_shop'] = dummy_array[:,7]
        df['f_railways'] = dummy_array[:,8]
        df['f_railways_station'] = dummy_array[:,9]
        df['f_subway'] = dummy_array[:,10]
        
    
    df.sort_values('ts', inplace = True)
    
    df['delta_t'] = np.concatenate([[0], (df['ts'].values[1:] - df['ts'].values[:-1]) / pd.to_timedelta('1s')], axis = 0)    
    df['delta_d'] = np.concatenate([[0], np.linalg.norm(df[['x', 'y']].values[1:] - df[['x', 'y']].values[:-1], axis = 1)], axis = 0)    
    df['bearing'] = np.concatenate([[0], np.arctan2(df[['y']].values[1:] - df[['y']].values[:-1], df[['x']].values[1:] - df[['x']].values[:-1]).reshape(-1)], axis = 0)
    df['speed'] = df['delta_d'] / df['delta_t'] 
    
    cut_labels_6 = [0, 1, 2, 3, 4]
    cut_bins = [0, 6, 10, 14, 18, 24]
    df['tod'] = pd.cut(df['ts'].dt.hour, bins=cut_bins, labels=cut_labels_6, right = False).astype(int)
    
    df = df[lambda x: x['delta_t'] > 0].copy()
    
    seq_bins = np.cumsum((df['delta_t'] >= seq_cutoff_time) | (df['speed'] > seq_cutoff_speed))
    seq_bin_ids, seq_bin_counts = np.unique(seq_bins, return_counts=True)
    
    if segmentation:
        df['segment_id'] = seq_bins
        df['segment_ix'] = np.concatenate([np.arange(seq_bin_counts[i]) for i in range(len(seq_bin_counts))])
        df['segment_point_count'] = np.repeat(seq_bin_counts, seq_bin_counts)
    
    if Standardize:
        df = scale_by_ID(df, ['delta_d','bearing','speed'])
    
    return df

def train_test_data_split(u=12, Random = False, k=9):
    """
    """
    if Random:        
        #Pick random users
        users = range(0,u)
        #Split Test/Train according to random pick
        train_val = random.sample(population=users, k=k) 
        train = train_val[:-1]
        val = train_val[-1:]
        test  = [x for x in users if x not in train_val]
    elif k==9:
        #this sequence comes form a random sampling
        #to make it consistent across multiple contributors, and installaitons, 
        #the seed is not enough. Hence we store it as a statc variable.
        train, val, test = ([8, 6, 4, 5, 9, 1, 11, 7], [2], [0, 3, 10])
    
    elif k==10:
        train, val, test = ([3, 1, 7, 5, 4, 9, 8, 10, 11], [2], [0, 6])
    
    return train, val, test



# Utility visualization functions
def hex_to_rgb(hex):
    """
    """
    hex = hex.lstrip('#')
    hlen = len(hex)
    return tuple(int(hex[i:i+hlen//3], 16) for i in range(0, hlen, hlen//3))

# Default plot ranges:
def create_image_wrap(fdf, col, w=1000, h=900, x_range = (1373757.1102773394, 1412506.1502695908), y_range = (7478418.9895278225, 7520786.118694777), background = 'black'):
    """
    """
    def create_image(x_range=x_range, y_range=y_range, w=w, h=h):
        """
        """
        cvs = ds.Canvas(x_range=x_range, y_range=y_range, plot_height=h, plot_width=w)
    
        if len(fdf[col].unique())>10:
            colormap = fire
        else:
            colormap = bkr
        agg = cvs.points(fdf, 'x_web', 'y_web', agg=ds.mean(col))
        image = dtf.shade(agg, cmap=colormap)
        ds.utils.export_image(image,filename=col+'.png')
        return dtf.dynspread(image, threshold=0.75, max_px=8)

    return create_image

def base_plot(tools='pan,wheel_zoom,reset', w=1000, h=900, x_range = (1373757.1102773394, 1412506.1502695908), y_range = (7478418.9895278225, 7520786.118694777), background = 'black'):
    """
    """
    p = bp.figure(tools=tools
                  , plot_width=int(w)
                  , plot_height=int(h)
                  , x_range=x_range, y_range=y_range
                  , outline_line_color=None,
        min_border=0, min_border_left=0, min_border_right=0,
        min_border_top=0, min_border_bottom=0,
                 x_axis_type="mercator", y_axis_type="mercator")
    p.xgrid.grid_line_color = None
    p.ygrid.grid_line_color = None
    return p

def plot_filtered(col, fdf, background = 'black', Toner=True, x_range = (1373757.1102773394, 1412506.1502695908), y_range = (7478418.9895278225, 7520786.118694777)):
    """
    
    """
    p = base_plot(x_range = x_range, y_range = y_range)
    if Toner:
        url="http://tile.stamen.com/toner-background/{Z}/{X}/{Y}.png"
    else:
        url="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{Z}/{Y}/{X}.png"
    tile_renderer = p.add_tile(WMTSTileSource(url=url))
    tile_renderer.alpha=1.0 if background == "black" else 0.15
    return InteractiveImage(p, create_image_wrap(col, fdf, x_range = x_range, y_range = y_range))

def compute_roc(y_true, y_pred, phase_name, plot=False):
    """
    TODO
    :param y_true: ground truth
    :param y_pred: predictions
    :param plot:
    :return:
    """
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    auc_score = auc(fpr, tpr)
    if plot:
        plt.figure(figsize=(7, 6))
        plt.plot(fpr, tpr, color='blue',
                 label='ROC (AUC = %0.4f)' % auc_score)
        plt.legend(loc='lower right')
        plt.title(f"ROC Curve {phase_name}")
        plt.xlabel("FPR")
        plt.ylabel("TPR")
        plt.show()

    return fpr, tpr, auc_score 

# Load any compressed pickle file
def decompress_pickle(file):
    data = bz2.BZ2File(file, 'rb')
    data = cPickle.load(data)
    return data

# Pickle a file and then compress it into a file with extension 
def compress_pickle(title, data):
    with bz2.BZ2File(title + '.pbz2', 'w') as f: 
        cPickle.dump(data, f)
        

def to_EPSG3857_4(lon, lat, x_web, y_web):
    a = 6378137.0
    n = lon.shape[0]
    for i in range(n):
        x_web[i] = a * np.pi * lon[i] / 180.0
        y_web[i] = a * math.log(math.tan(np.pi * (0.25 + lat[i] / 360.0)))
        
'''
import multiprocessing
from joblib import Parallel, delayed
import pickle
def convertWebM(loc): 
    return list(project_webM(*loc))

num_cores = multiprocessing.cpu_count()
inputs = [loc for loc in zip(df.lon, df.lat)]
#webM = pd.DataFrame(Parallel(n_jobs=num_cores)(delayed(convertWebM)(i) for i in inputs), columns=['x_web', 'y_web'])
webM = pd.DataFrame([list(project_webM(*loc)) for loc in zip(df.lon, df.lat)], columns=['x_web', 'y_web'])
'''
def convert_to_WebMercator(df):
        
    # create a GPU dataframe from the Pandas dataframe
    gpudf = cudf.DataFrame.from_pandas(df[['lon','lat']])
    gpudf = gpudf.apply_rows(
            to_EPSG3857_4,
            incols=['lon', 'lat'],
            outcols=dict(x_web=np.float64, y_web=np.float64),
            kwargs=dict())

    webM = gpudf[['x_web', 'y_web']].to_pandas()
    
    return webM



        
def to_EPSG25832(lon, lat, x, y):
    n = lon.shape[0]
    for i in range(n):
        Zone= int(Math.floor(lon[i]/6+31))
        x[i] = 0.5*Math.log((1+Math.cos(lat[i]*Math.pi/180)\
               *Math.sin(lon[i]*Math.pi/180-(6*Zone-183)*Math.pi/180))\
               /(1-Math.cos(lat[i]*Math.pi/180)*Math.sin(lon[i]*Math.pi\
               /180-(6*Zone-183)*Math.pi/180)))*0.9996*6399593.62\
               /Math.pow((1+Math.pow(0.0820944379, 2)\
               *Math.pow(Math.cos(lat[i]*Math.pi/180), 2)), 0.5)\
               *(1+ Math.pow(0.0820944379,2)/2\
               *Math.pow((0.5*Math.log((1+Math.cos(lat[i]*Math.pi/180)\
               *Math.sin(lon[i]*Math.pi/180-(6*Zone-183)*Math.pi/180))\
               /(1-Math.cos(lat[i]*Math.pi/180)*Math.sin(lon[i]*Math.pi/180-(6*Zone-183)\
               *Math.pi/180)))),2)*Math.pow(Math.cos(lat[i]*Math.pi/180),2)/3)+500000
        x[i] = round(x[i]*100)*0.01
        y[i] = (Math.atan(Math.tan(lat[i]*Math.pi/180)/Math.cos((lon[i]*Math.pi/180-(6*Zone -183)\
               *Math.pi/180)))-lat[i]*Math.pi/180)*0.9996*6399593.625\
               /Math.sqrt(1+0.006739496742*Math.pow(Math.cos(lat[i]*Math.pi/180),2))\
               *(1+0.006739496742/2*Math.pow(0.5*Math.log((1+Math.cos(lat[i]*Math.pi/180)\
               *Math.sin((lon[i]*Math.pi/180-(6*Zone -183)*Math.pi/180)))\
               /(1-Math.cos(lat[i]*Math.pi/180)*Math.sin((lon[i]*Math.pi/180-(6*Zone -183)*Math.pi/180)))),2)\
               *Math.pow(Math.cos(lat[i]*Math.pi/180),2))+0.9996*6399593.625\
               *(lat[i]*Math.pi/180-0.005054622556*(lat[i]*Math.pi/180+Math.sin(2*lat[i]*Math.pi/180)/2)\
               +4.258201531e-05*(3*(lat[i]*Math.pi/180+Math.sin(2*lat[i]*Math.pi/180)/2)\
               +Math.sin(2*lat[i]*Math.pi/180)*Math.pow(Math.cos(lat[i]*Math.pi/180),2))\
               /4-1.674057895e-07*(5*(3*(lat[i]*Math.pi/180+Math.sin(2*lat[i]*Math.pi/180)/2)\
               +Math.sin(2*lat[i]*Math.pi/180)*Math.pow(Math.cos(lat[i]*Math.pi/180),2))\
               /4+Math.sin(2*lat[i]*Math.pi/180)*Math.pow(Math.cos(lat[i]*Math.pi/180),2)\
               *Math.pow(Math.cos(lat[i]*Math.pi/180),2))/3)
        if (lat[i]<0):
            y[i] += 10000000
        y[i] = round(y[i]*100)*0.01
    
def convert_to_UTM(df):
    
    # create a GPU dataframe from the Pandas dataframe
    gpudf = cudf.DataFrame.from_pandas(df)
    gpudf = gpudf.apply_rows(to_EPSG25832
                             ,incols=['lon', 'lat']
                             , outcols=dict(x=np.float64, y=np.float64)
                             ,kwargs=dict())

    UTM32 = gpudf[['x', 'y']].to_pandas()
    
    return UTM32

    