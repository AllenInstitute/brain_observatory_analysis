# processing
import numpy as np
import pandas as pd
import os
from scipy.signal import savgol_filter
from scipy import signal
from tqdm import tqdm
from allensdk.brain_observatory import sync_dataset
import json
import glob

# plotting
import matplotlib.pyplot as plt
from brain_observatory_analysis.utilities import image_utils
import cv2
import seaborn as sns


# PATHS
output_base_dir = '//allen/programs/mindscope/workgroups/learning/'
behavioral_video_plots_dir = os.path.join(output_base_dir, 'behavioral_video_plots')
behavioral_video_annotation_dir = os.path.join(output_base_dir, 'behavioral_video_annotation')

# FUNCTIONS


def get_h5file(path):
    """
    Get the h5 file from a given directory, which includes specific view folder

    Parameters
    ----------
    path : str
        full path to the directory containing the h5 file

    Returns
    -------
    h5file : str
        full path to the h5 file
    """

    # TO DO: eye folder has 2 h5 files, need to figure out which one to use
    h5files = glob.glob(os.path.join(path, '*.h5'))
    if len(h5files) == 1:
        h5file = h5files[0]
    elif len(h5files) > 1:
        print(f'found {len(h5files)} h5 files')
        h5file = h5files
    else:
        print('no dlc output file found in this directory')
    return h5file


def get_mp4file(path, folder=None):
    """
    Get the mp4 files from a given directory, if folder is provided, then only return the mp4 file for that camera view

    Parameters
    ----------
    path : str
        full path to the directory containing the mp4 files
    folder : str
        folder that can be used in get_moviename_from_folder to identify the camera view, default is None

    Returns
    -------
    mp4file : str
        full path to the mp4 file
    """

    movies = glob.glob(os.path.join(path, '*.mp4'))
    if folder is not None:
        moviename = get_moviename_from_folder(folder)
        mp4file = [m for m in movies if moviename in m][0]
    else:
        print('camera view was not provided, returning all mp4 files')
        mp4file = movies
    print('mp4 file: {}'.format(mp4file))
    return mp4file


def read_DLC_h5file(h5file):
    """
    Read in a h5 file from DLC and return a dataframe with the DLC data wiht clean column names

    Parameters
    ----------
    h5file : str
        full path to the h5 file"""
    df = pd.read_hdf(h5file, key='df_with_missing')
    junk = df.keys()[0][0]  # this is a hack to get the right column names
    df = df[junk]
    df_dlc = pd.DataFrame(df.unstack())  # long format shape
    df_dlc = df_dlc.reset_index()
    df_dlc = df_dlc.rename(columns={0: 'value', 'level_2': 'frame_number'})  # rename columns to clear names
    print('loaded DLC data')
    return df_dlc


def extract_lost_frames_from_json(cam_json):
    '''
    Get the indices of the lost frames (not written to disk) from the camera
    json file. Note that these indices are for the DATA frames (ie 0 corresponds
    to the first data frame and NOT the prepended metadata frame)

    Parameters:
    ------------
    cam_json: dict
        json file from MVR

    Returns:
    ------------
    lost_frames: list
        list of lost frames indices
    '''

    lost_count = cam_json['RecordingReport']['FramesLostCount']
    if lost_count == 0:
        return []

    lost_string = cam_json['RecordingReport']['LostFrames'][0]
    lost_spans = lost_string.split(',')

    lost_frames = []
    for span in lost_spans:

        start_end = span.split('-')
        if len(start_end) == 1:
            lost_frames.append(int(start_end[0]))
        else:
            lost_frames.extend(np.arange(int(start_end[0]), int(start_end[1]) + 1))

    return np.array(lost_frames) - 1  # you have to subtract one since the json starts indexing at 1 according to Totte


def get_frame_exposure_times(cam_label):
    '''
    Returns the sync line label for the camera exposure times
    Parameters:
    ------------
    cam_label: str
        camera label from MVR json file
    Returns:
    ------------
    sync_line: str
        sync line label for the camera exposure times
    '''

    exposure_sync_line_label_dict = {
        'Eye': 'eye_cam_exposing',
        'Face': 'face_cam_exposing',
        'Behavior': 'beh_cam_exposing'}
    return exposure_sync_line_label_dict[cam_label]


def get_frame_index(frame_file_path):

    frame_file_base = os.path.basename(frame_file_path)
    frame_file_index = frame_file_base.replace('img', '').replace('.png', '')
    return int(frame_file_index)


def get_moviename_from_folder(folder):
    """
    Get the movie name from the folder name

    Parameters
    ----------
    folder : str
        full path to the folder containing the movie

    Returns
    -------
    moviename : str
        name of the movie
    """
    folder_to_moviename_mapping = {'side_tracking': 'Behavior',
                                   'eye_tracking': 'Eye',
                                   'face_tracking': 'Face'}
    moviename = folder_to_moviename_mapping[folder]
    return moviename


def read_json(json_file):
    """
    Read in a json file and return a dictionary

    Parameters
    ----------
    json_file : str
        full path to the json file

    Returns
    -------
    data : dict
        dictionary containing the json data
    """
    with open(json_file) as f:
        data = json.load(f)
    return data


def get_jsonfile(path, folder=None):
    """
    Get the json files from a given directory, if folder is provided, then only return the json file for that camera view

    Parameters
    ----------
    path : str
        full path to the directory containing the json files
    folder : str
        folder that can be used in get_moviename_from_folder to identify the camera view, default is None

    Returns
    -------
    jsonfile : str
        full path to the json file
    """
    jsonfiles = glob.glob(os.path.join(path, '*.json'))
    if folder is not None:
        moviename = get_moviename_from_folder(folder)
        jsonfile = [j for j in jsonfiles if moviename in j][0]
    else:
        print('camera view was not provided, returning all json files')
        jsonfile = jsonfiles
    print('json file: {}'.format(jsonfile))
    return jsonfile


def get_syncfile(path):
    """
    Get the sync file from a given directory

    Parameters
    ----------
    path : str
        full path to the directory containing the sync file

    Returns
    -------
    syncfile : str
        full path to the sync file
    """
    syncfile = glob.glob(os.path.join(path, '*.h5'))[0]
    print('sync file: {}'.format(syncfile))
    return syncfile


def get_sync_dataset(syncfile):
    """
    Get the sync dataset from a given sync file. Use get_syncfile to get the sync file.

    Parameters
    ----------
    syncfile : str
        full path to the sync file

    Returns
    -------
    sync_dataset : allensdk.brain_observatory.sync_dataset.SyncDataset
        sync dataset
    """
    sync_dataset_object = sync_dataset.Dataset(syncfile)
    print('successfully loaded sync dataset object')
    return sync_dataset_object


def get_cam_timestamps(sync_dataset, cam_json, account_for_metadata_frame=True):
    '''
    Returns the experiment timestamps for the frames recorded in an MVR video

    Parameters:
    ------------
    sync_dataset : allensdk.brain_observatory.sync_dataset.SyncDataset
        sync dataset object from a given experiment, use get_sync_dataset to get this object
    cam_json : str or full path to json file saved for each camera view. Use get_jsonfile to get this file
        json file from MVR.
    account_for_metadata_frame : True/False, if TRUE prepend a NaN to the exposure times for the
    metadata frame that is prepended to the MVR video
    '''

    if isinstance(cam_json, str):
        cam_json_data = read_json(cam_json)

    total_frames_recorded = cam_json_data['RecordingReport']['FramesRecorded']

    cam_label = cam_json_data['RecordingReport']['CameraLabel']
    sync_line = get_frame_exposure_times(cam_label)

    exposure_times = sync_dataset.get_rising_edges(sync_line, units='seconds')

    lost_frames = extract_lost_frames_from_json(cam_json_data)

    # filter out lost frames from the sync exposure times
    frame_times = [e for ie, e in enumerate(exposure_times) if ie not in lost_frames]

    # cut off extra exposure times that didn't make it into the video
    frame_times = frame_times[:total_frames_recorded]  # hopefully this becomes obsolete after MVR stop sequence changes

    # add a NaN to the beginning for the metadata frame
    if account_for_metadata_frame:
        frame_times = np.insert(frame_times, 0, np.nan)

    return np.array(frame_times)


def plot_DLC_points(frame, points_to_plot, bodyparts=None):
    """
    Plots the DLC points on a frame.
    """
    if bodyparts is None:
        bodyparts = points_to_plot['bodyparts'].unique()
    bodycolors = image_utils.generate_distinct_colors(len(bodyparts))
    for b, bodypart in enumerate(bodyparts):
        one_point = points_to_plot[points_to_plot.bodyparts == bodypart]
        x = one_point[one_point.coords == 'x']['value'].values[0]
        y = one_point[one_point.coords == 'y']['value'].values[0]
        cv2.circle(frame, (int(x), int(y)), 5, bodycolors[b], -1)

    return frame


def process_video(output_path, mp4file, df_dlc, start_frame=1000, end_frame=2000):
    """
    Process a video by plotting the DLC points on each frame, saving the movie and
    the frames as pngs.

    Parameters
    ----------
    output_path : str
        full path to the output directory
    mp4file : str
        full path to the mp4 file
    df_dlc : pandas dataframe
        dataframe containing the DLC data. Use read_DLC_h5file to get this dataframe.
    start_frame : int
        frame number to start plotting from, default is 1000
    end_frame : int
        frame number to end plotting at, default is 2000

    """
    # Load the video file
    movie_fullname_processed = os.path.join(output_path, os.path.basename(mp4file))
    video = cv2.VideoCapture(mp4file)
    # Get video properties
    fps = video.get(cv2.CAP_PROP_FPS)
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Create VideoWriter object to save the modified frames
    if os.path.isdir(output_path) is False:
        os.makedirs(output_path)
        print('making dir')
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # Change codec as per your requirement
    output_video = cv2.VideoWriter(movie_fullname_processed, fourcc, fps, (width, height))
    print(f"Processing video {mp4file} with {fps} fps at {width}x{height}...")

    frames_to_plot = np.sort(df_dlc.frame_number.unique())
    print(f'found #{len(frames_to_plot)} frames to plot')

    for this_frame in tqdm(frames_to_plot[start_frame:end_frame]):
        video.set(cv2.CAP_PROP_POS_FRAMES, this_frame)
        ret, frame = video.read()

        points_to_plot = df_dlc[df_dlc.frame_number == this_frame]
        # Process the frame
        processed_frame = plot_DLC_points(frame, points_to_plot)
        plt.imshow(processed_frame)
        if os.path.exists(os.path.join(output_path, 'frames')) is False:
            os.makedirs(os.path.join(output_path, 'frames'))
        plt.savefig(os.path.join(output_path, 'frames', f'{this_frame}.png'))
        plt.close('all')
        output_video.write(processed_frame)

    # Release the video capture and writer objects
    video.release()
    output_video.release()

    print("Video processing complete.")


def plot_smoothed_traces(df_dlc, output_path=None):
    '''
    Plot the smoothed traces for each bodypart and save the figure to the output_path
    '''
    bodyparts = df_dlc['bodyparts'].unique()
    for bodypart in bodyparts:
        fig, ax = plt.subplots(1, 1, figsize=(10, 3))
        coords = df_dlc[df_dlc.bodyparts == bodypart]['coords'].unique()
        for coord in coords:
            data = df_dlc[(df_dlc.bodyparts == bodypart) &
                          (df_dlc.coords == coord)]
            x = df_dlc[(df_dlc.bodyparts == bodypart) &
                       (df_dlc.coords == coord)]['frame_number'].values
            filt_data = savgol_filter(data['value'].values, window_length=50, polyorder=1)
            if coord == 'likelihood':
                ax2 = ax.twinx()
                ax2.plot(x, filt_data, 'g')
            else:
                ax.plot(x, filt_data)
        ax.set_xlabel('frame')
        ax.set_ylabel('x,y coords')
        ax2.set_ylabel('likelihood')
        ax.set_title(bodypart)
        if output_path is not None:
            if os.path.isdir(output_path) is False:
                os.makedirs(output_path)
            fig.savefig(os.path.join(output_path, f'{bodypart}_trace.png'))
            plt.close('all')


def plot_downsampled_scatterplot(df_dlc, output_path, window_length=50, polyorder=1):
    bodyparts = df_dlc['bodyparts'].unique()
    for bodypart in bodyparts:
        # figure
        fig, ax = plt.subplots(1, 1, figsize=(7, 7))

        # traces
        coords = df_dlc[df_dlc.bodyparts == bodypart]['coords'].unique()
        trace = df_dlc[(df_dlc.bodyparts == bodypart) &
                       (df_dlc.coords == coords[0])][0].values
        x = savgol_filter(trace, window_length=window_length, polyorder=polyorder)
        trace = df_dlc[(df_dlc.bodyparts == bodypart) &
                       (df_dlc.coords == coords[1])][0].values
        y = savgol_filter(trace, window_length=window_length, polyorder=polyorder)
        likelihood = df_dlc[(df_dlc.bodyparts == bodypart) &
                            (df_dlc.coords == coords[2])][0]
        time = df_dlc[(df_dlc.bodyparts == bodypart) &
                      (df_dlc.coords == coords[2])]['level_2'].values

        # resample
        sample = 2000
        x_new = signal.resample(x, sample)
        y_new = signal.resample(y, sample)
        time_new = np.linspace(0, max(time), sample, endpoint=False)

        # plot
        sns.scatterplot(x=x_new, y=y_new, hue=time_new, markers='o', alpha=0.5, legend=False, ax=ax)
        ax.set_xlabel('x coord')
        ax.set_ylabel('y coord')
        ax.set_title(bodypart)

        # save figure
        if os.path.isdir(output_path) is False:
            os.makedirs(output_path)
        fig.savefig(os.path.join(output_path, f'{bodypart}_scatter.png'))
        plt.close('all')
