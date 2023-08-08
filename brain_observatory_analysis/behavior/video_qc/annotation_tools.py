# processing
import numpy as np
import pandas as pd
import os
from scipy.signal import savgol_filter
from scipy import signal
import h5py
from tqdm import tqdm

# plotting
import matplotlib.pyplot as plt
from brain_observatory_analysis.ophys.experiment_loading import start_lamf_analysis
from brain_observatory_analysis.utilities import image_utils
import cv2


# PATHS
base_dir = '//allen/programs/mindscope/workgroups/learning/behavioral_video_plots'


def extract_lost_frames_from_json(cam_json):
    '''
    Get the indices of the lost frames (not written to disk) from the camera
    json file. Note that these indices are for the DATA frames (ie 0 corresponds
    to the first data frame and NOT the prepended metadata frame)
    '''
    
    lost_count = cam_json['RecordingReport']['FramesLostCount']
    if lost_count == 0:
        return []
    
    lost_string = cam_json['RecordingReport']['LostFrames'][0]
    lost_spans = lost_string.split(',')
    
    lost_frames = []
    for span in lost_spans:
        
        start_end = span.split('-')
        if len(start_end)==1:
            lost_frames.append(int(start_end[0]))
        else:
            lost_frames.extend(np.arange(int(start_end[0]), int(start_end[1])+1))
    
    return np.array(lost_frames)-1 #you have to subtract one since the json starts indexing at 1 according to Totte


def get_frame_exposure_times(cam_label):
    
    exposure_sync_line_label_dict = {
            'Eye': 'eye_cam_exposing',
            'Face': 'face_cam_exposing',
            'Behavior': 'beh_cam_exposing'}
    return exposure_sync_line_label_dict[cam_label]


def get_frame_index(frame_file_path):
    
    frame_file_base = os.path.basename(frame_file_path)
    frame_file_index = frame_file_base.replace('img', '').replace('.png', '')
    return int(frame_file_index)


def get_frame_exposure_times(sync_dataset, cam_json, account_for_metadata_frame=True):
    '''
    Returns the experiment timestamps for the frames recorded in an MVR video
    
    sync_dataset should be a Dataset object built from the sync h5 file
    cam_json is the json that MVR writes to accompany each mp4
    
    account_for_metadata_frame: if TRUE prepend a NaN to the exposure times for the
    metadata frame that is prepended to the MVR video
    '''
    
    if isinstance(cam_json, str):
        cam_json = read_json(cam_json)
     
    total_frames_recorded = cam_json['RecordingReport']['FramesRecorded']
    
    cam_label =  cam_json['RecordingReport']['CameraLabel']
    sync_line = get_frame_exposure_times(cam_label)
    
    exposure_times = sync_dataset.get_rising_edges(sync_line, units='seconds')
    
    lost_frames = extract_lost_frames_from_json(cam_json)
    
    #filter out lost frames from the sync exposure times
    frame_times = [e for ie, e in enumerate(exposure_times) if ie not in lost_frames]
    
    #cut off extra exposure times that didn't make it into the video
    frame_times = frame_times[:total_frames_recorded] #hopefully this becomes obsolete after MVR stop sequence changes
    
    #add a NaN to the beginning for the metadata frame
    if account_for_metadata_frame:
        frame_times = np.insert(frame_times, 0, np.nan)

    return np.array(frame_times)


def plot_DLC_points(frame, points_to_plot, bodyparts = None):
    """
    Plots the DLC points on a frame
    """
    if bodyparts is None:
        bodyparts = points_to_plot['bodyparts'].unique()
    bodycolors = image_utils.generate_distinct_colors(len(bodyparts))
    for b, bodypart in enumerate(bodyparts):
        one_point = points_to_plot[points_to_plot.bodyparts==bodypart]
        x = one_point[one_point.coords=='x'][0].values[0]
        y = one_point[one_point.coords=='y'][0].values[0]
        cv2.circle(frame, (int(x), int(y)), 5, bodycolors[b], -1)
        
    return frame

def process_video(input_path, output_path, mfile, df_dlc):
    # Load the video file
    movie_fullname = os.path.join(input_path, mfile)
    movie_fullname_processed = os.path.join(output_path, mfile)
    video = cv2.VideoCapture(movie_fullname)
    print(movie_fullname_processed)
    # Get video properties
    fps = video.get(cv2.CAP_PROP_FPS)
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    

    # Create VideoWriter object to save the modified frames
    if os.path.isdir(output_path)==False:
        os.makedirs(output_path)
        print('making dir')
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # Change codec as per your requirement
    output_video = cv2.VideoWriter(movie_fullname_processed, fourcc, fps, (width, height))
    
    
   
    frame_count = 0
    frames_to_plot = np.sort(df_dlc.level_2.unique())
    print(f'found #{len(frames_to_plot)} frames to plot')
    
    for this_frame in tqdm(frames_to_plot[1700:2000]):
        video.set(cv2.CAP_PROP_POS_FRAMES, this_frame)
        ret, frame = video.read()
        
        points_to_plot = df_dlc[df_dlc.level_2==this_frame]
        # Process the frame
        processed_frame = plot_DLC_points(frame, points_to_plot)
        plt.imshow(processed_frame)
        plt.savefig(os.path.join(output_path, 'frames', f'{this_frame}.png'))
        plt.close('all')
        output_video.write(processed_frame)
        

    # Release the video capture and writer objects
    video.release()
    output_video.release()

    print("Video processing complete.")


def plot_smoothed_traces(df_dlc, output_path=''):
    bodyparts = df_dlc['bodyparts'].unique()
    for bodypart in bodyparts:
        fig, ax = plt.subplots(1,1, figsize = (10,3))
        coords = df_dlc[df_dlc.bodyparts==bodypart]['coords'].unique()
        for coord in coords:
            data = df_dlc[(df_dlc.bodyparts==bodypart) &
                      (df_dlc.coords==coord)]
            x = df_dlc[(df_dlc.bodyparts==bodypart) &
                      (df_dlc.coords==coord)]['level_2'].values
            filt_data = savgol_filter(data[0].values, window_length = 50, polyorder = 1)
            if coord == 'likelihood':
                ax2 = ax.twinx()
                ax2.plot(x, filt_data, 'g')
            else:
                ax.plot(x, filt_data)
        ax.set_xlabel('frame')
        ax.set_ylabel('x,y coords')
        ax2.set_ylabel('likelihood')
        ax.set_title(bodypart)
        if os.path.isdir(output_path)==False:
            os.makedirs(output_path)
        fig.savefig(os.path.join(output_path, f'{bodypart}_trace.png'))
        plt.close('all')


def plot_downsampled_scatterplot(df_dlc, output_path):
    bodyparts = df_dlc['bodyparts'].unique()
    for bodypart in bodyparts:
        # figure
        fig, ax = plt.subplots(1,1, figsize = (7,7))

        # traces
        coords = df_dlc[df_dlc.bodyparts==bodypart]['coords'].unique()
        trace = df_dlc[(df_dlc.bodyparts==bodypart) &
                      (df_dlc.coords==coords[0])][0].values
        x = savgol_filter(trace,window_length = 50, polyorder = 1)
        trace = df_dlc[(df_dlc.bodyparts==bodypart) &
                      (df_dlc.coords==coords[1])][0].values
        y = savgol_filter(trace, window_length = 50, polyorder = 1)
        likelihood = df_dlc[(df_dlc.bodyparts==bodypart) & 
                      (df_dlc.coords==coords[2])][0]
        time = df_dlc[(df_dlc.bodyparts==bodypart) &
                      (df_dlc.coords==coords[2])]['level_2'].values

        # resample
        sample = 2000 
        x_new = signal.resample(x, sample)
        y_new = signal.resample(y, sample)
        time_new = np.linspace(0, max(time), sample, endpoint=False)

        # plot
        sns.scatterplot(x=x_new, y=y_new, hue=time_new, markers = 'o', alpha = 0.5, legend=False, ax=ax)
        ax.set_xlabel('x coord')
        ax.set_ylabel('y coord')
        ax.set_title(bodypart)
        
        # save figure
        if os.path.isdir(output_path)==False:
            os.makedirs(output_path)
        fig.savefig(os.path.join(output_path, f'{bodypart}_scatter.png'))
        plt.close('all')

