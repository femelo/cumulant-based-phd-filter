# Provide parameters to various services, editable in settings.json
import sys
import json
from os import path, popen, mkdir
import copy as cp

VIDEO_EXTENSIONS = ['mp4', 'avi', 'mts']

# Distances for string comparison
def memoize(func):
    mem = {}
    def memoizer(*args, **kwargs):
        key = str(args) + str(kwargs)
        if key not in mem:
            mem[key] = func(*args, **kwargs)
        return mem[key]
    return memoizer

@memoize
def levenshtein_dist(s, t):
    if s == "":
        return len(t)
    if t == "":
        return len(s)
    if s[-1] == t[-1]:
        cost = 0
    else:
        cost = 1
       
    res = min([levenshtein_dist(s[:-1], t)+1,
               levenshtein_dist(s, t[:-1])+1, 
               levenshtein_dist(s[:-1], t[:-1]) + cost])
    return res

def indicator_function(s, t):
    word_list_s = [w.strip().replace(',', '') for w in s.split(' ')]
    word_list_t = [w.strip().replace(',', '') for w in t.split(' ')]
    if len(word_list_s) > len(word_list_t):
        base_list = word_list_t
        seek_list = word_list_s
    else:
        base_list = word_list_s
        seek_list = word_list_t
    
    score = 0
    for word in base_list:
        if word in seek_list:
            score -= len(word)
    return score

def get_min_dist(s, stringList):
    # First try
    if len(stringList) == 0 or len(s) == 0:
        return float('inf'), None
    dist = [indicator_function(s,t) for t in stringList]
    min_dist = min(dist)
    if min_dist == 0:
        dist = [levenshtein_dist(s,t) for t in stringList]
        min_dist = min(dist)
    ind = dist.index(min_dist)
    #if min_dist < 0:
    #    min_dist = -min_dist
    return min_dist, ind

def _input(message):
    if not isinstance(message, str):
        raise ValueError('Message for \'input\' method must be a string.')
    if sys.version_info[0] < 3:
        return raw_input(message)
    else:
        return input(message)

def get_subfolders(ref_dir):
    subfolders = popen("ls " + ref_dir).read().split('\n')
    for item in subfolders[::-1]:
        if item == '' or not path.isdir(ref_dir + '/' + item):
            del subfolders[subfolders.index(item)]
    return subfolders

def get_files(ref_dir):
    files = popen("ls " + ref_dir).read().split('\n')
    for item in files[::-1]:
        if item == '' or not path.isfile(ref_dir + '/' + item):
            del files[files.index(item)]
    return files

def get_all_subfolders(ref_dir):
    if ref_dir[-1] == '/':
        ref_dir = ref_dir[:-1]
    last_dir = ref_dir.split('/')[-1]
    path_list = ref_dir.split('/')[:-1]
    base_dir = ''
    for level in path_list[1:]:
        base_dir += '/' + level
    list_of_subfolders = []
    list_of_base_dirs = []
    sList = [last_dir]
    bDirs = [base_dir]
    while len(sList) > 0:
        newSList = []
        newBDirs = []
        for i, subfolder in enumerate(sList):
            d = path.join(bDirs[i],subfolder)
            l = get_subfolders(d)
            newSList += l
            newBDirs += [d for _ in range(len(l))]
        list_of_subfolders += newSList
        list_of_base_dirs += newBDirs
        sList = cp.copy(newSList)
        bDirs = cp.copy(newBDirs)
    return list_of_subfolders, list_of_base_dirs

def get_all_files(ref_dir):
    subfolders, base_dirs = get_all_subfolders(ref_dir)
    files = get_files(ref_dir)
    files_dirs = [ref_dir for _ in files]
    #files = []
    #files_dirs = []
    for i, subfolder in enumerate(subfolders):
        folder = path.join(base_dirs[i], subfolder)
        l = get_files(folder)
        files += l
        files_dirs += [folder for _ in l]
    return files, files_dirs

class TrackingParams():
    def __init__(self, settings):
        self.filter = str(settings["filter"])
        self.declutter = str(settings["declutter"]).lower() == "yes"
        self.camera_params_file = str(settings["camera_params_file"])
        self.motion_model = str(settings["motion_model"])
        self.sigma_q = settings["sigma_q"]
        self.sigma_r = settings["sigma_r"]
        self.prob_survival = settings["prob_survival"]
        self.prob_detection = settings["prob_detection"]
        self.prob_false_alarm = settings["prob_false_alarm"]
        self.false_alarm_rate = settings["false_alarm_rate"]
        self.pruning_threshold = settings["pruning_threshold"]
        self.N_scan = settings["pruning_horizon"]
        self.N_miss = settings["miss_horizon"]
        self.track_min_length = settings["track_min_length"]
        self.prob_gating = settings["gating_confidence"]
        self.iou_threshold = settings["iou_threshold"]
        self.reid_threshold = settings["reid_threshold"]
        self.use_saved_dets = settings["use_saved_dets"].lower() == "yes"
        self.fps = settings["default_fps"]
        dir_settings = settings["directories"]
        self.databases_base_dir = str(dir_settings["databases_base_dir"])
        if self.databases_base_dir[-1] == '/':
            self.databases_base_dir = self.databases_base_dir[:-1]
        self.detections_base_dir = str(dir_settings["detections_base_dir"])
        if self.detections_base_dir[-1] == '/':
            self.detections_base_dir = self.detections_base_dir[:-1]
        self.datasetsIDs = [str(item).lower() for item in dir_settings["datasets"]]
        self.datasets_folders = [str(item) for item in dir_settings["datasets_folders"]]
        self.input_images_dir = str(dir_settings["input_images_dir"])
        self.input_videos_dir = str(dir_settings["input_videos_dir"])
        self.video_file_format = str(dir_settings["video_file_format"])
        self.output_videos_dir = str(dir_settings["output_videos_dir"])
        self.dets_dir = str(dir_settings["detections_dir"])
        self.det_file_ext = str(dir_settings["detection_file_extension"])
        self.results_dir = str(dir_settings["results_dir"])
        self.result_file_format  = str(dir_settings["result_file_format"])

class Parameters():
    def __init__(self, settings_file, datasetID = None, videoID = None):
        self.settings = json.load(open(settings_file)) 
        self.tracking = TrackingParams(self.settings["tracking"])
        if datasetID is not None and videoID is not None:
            self.set_file_names(datasetID, videoID)
        else:
            self.input_video_file = None
            self.output_video_file = None
            self.result_file = None
            self.dets_file = None

    def set_file_names(self, datasetID, videoID):
        if not datasetID.lower() in self.tracking.datasetsIDs:
            print('Dataset not set or non-existent.')
            exit()
        dataset_folder = self.tracking.datasets_folders[self.tracking.datasetsIDs.index(datasetID.lower())]

        # Set videos base dir
        if not path.isdir(self.tracking.databases_base_dir):
            print(self.tracking.databases_base_dir + ' does not exist.')
            exit()
        base_dir = path.realpath(self.tracking.databases_base_dir)
        
        base_subfolders = get_subfolders(base_dir)
        
        base_subdir = None
        structuredByDataset = False
        if dataset_folder in base_subfolders:
            dataset_subfolders = get_subfolders(path.join(base_dir, dataset_folder))
            if self.tracking.input_videos_dir in dataset_subfolders:
                structuredByDataset = True
                base_subdir = path.join(base_dir, dataset_folder)
        
        # Set input video file name
        if not structuredByDataset:
            base_subdir = path.join(base_dir, self.tracking.input_videos_dir)
            if not path.isdir(base_subdir):
                print('Could not find any folder with input videos.')
                exit()
            input_video_file = self.find_video_file(base_subdir, videoID)
            if input_video_file is None:
                input_video_file = self.find_video_file(base_subdir, videoID)
                if input_video_file is None:
                    print('Video \'{:s}\' not found.'.format(videoID))
                    exit()
                
        if base_subdir is None:
            print('No input video directory defined.')
            exit()

        if structuredByDataset:
            input_videos_dir = path.join(base_subdir, self.tracking.input_videos_dir)
            if not path.isdir(input_videos_dir):
                print(input_videos_dir + ' does not exist.')
                exit()
            input_video_file = self.find_video_file(input_videos_dir, videoID)
            if input_video_file is None:
                print('Video \'{:s}\' not found.'.format(videoID))
                exit()
            
        # Set output video file name
        if structuredByDataset:
            output_videos_dir = path.join(base_subdir, self.tracking.output_videos_dir)
        else:
            output_videos_dir = path.join(base_dir, self.tracking.output_videos_dir)
        if not path.isdir(output_videos_dir):
            mkdir(output_videos_dir)
        file_name = input_video_file.split('/')[-1]
        file_name_no_ext = file_name[:-4]
        output_video_file = path.join(output_videos_dir, file_name_no_ext+'.mp4')

        # Set result file name
        if structuredByDataset:
            results_dir = path.join(base_subdir, self.tracking.results_dir)
        else:
            results_dir = path.join(base_dir, self.tracking.results_dir)
        if not path.isdir(results_dir):
            mkdir(results_dir)
        file_name = self.tracking.result_file_format.format(self.tracking.filter, videoID)
        if len(file_name.split('.')[-1]) <= 3:
            file_name_complete = file_name
        else:
            if file_name[-1] == '.':
                file_name = file_name[:-1]
            file_name_complete = file_name + '.txt'
        result_file = path.join(results_dir, file_name_complete)

        self.input_video_file = input_video_file
        self.output_video_file = output_video_file
        self.result_file = result_file
        if not self.tracking.use_saved_dets:
            return
        
        # Optional
        # Set detections base dir
        if not path.isdir(self.tracking.detections_base_dir):
            print(self.tracking.detections_base_dir + ' does not exist.')
            exit()
        base_dir = path.realpath(self.tracking.detections_base_dir)

        # Set detection file name
        base_subfolders = get_subfolders(base_dir)
        
        base_subdir = None
        structuredByDataset = False
        if dataset_folder in base_subfolders:
            dataset_subfolders = get_subfolders(path.join(base_dir, dataset_folder))
            if self.tracking.dets_dir in dataset_subfolders:
                structuredByDataset = True
                base_subdir = path.join(base_dir, dataset_folder)
        
        # Set detections and features file name
        skip = False
        dets_file = None
        if not structuredByDataset:
            base_subdir, dets_file = self.find_detections_file(base_dir, videoID, skip)
                    
        if base_subdir is None:
            print('No detections directory defined.')
            exit()

        skip = False
        if structuredByDataset:
            base_subdir, dets_file = self.find_detections_file(base_subdir, videoID, skip)

        if dets_file is None:
            print('Detections file required for \'{:s}\' but not found.'.format(videoID))
            exit()
                
        self.dets_file = dets_file
        return

    def get_file_names(self, datasetID, videoID):
        if self.input_video_file is None or \
            self.output_video_file is None or \
            self.result_file is None:
            self.set_file_names(datasetID, videoID)
        return self.input_video_file, self.output_video_file, self.result_file

    def find_video_file(self, base_subdir, videoID):
        files, files_base_dirs = get_all_files(base_subdir)
        indexes = [i for i, item in enumerate(files) \
            if item[-3:] in VIDEO_EXTENSIONS or item[-3:].lower() in VIDEO_EXTENSIONS]
        videosIDs = [files[i][:-4] for i in indexes]
        videosIDs_lower = [item.lower() for item in videosIDs]
        videos_files = [files[i] for i in indexes]
        videos_base_dirs = [files_base_dirs[i] for i in indexes]
        if videoID in videosIDs:
            ind = videosIDs.index(videoID)
            base_subdir = videos_base_dirs[ind]
            input_video_file = path.join(base_subdir, videos_files[ind])
        else:
            min_dist, ind = get_min_dist(videoID.lower(), videosIDs_lower)
            if ind is None or min_dist > abs(len(videosIDs[ind])-len(videoID)):
                return None
            base_subdir = videos_base_dirs[ind]
            input_video_file = path.join(base_subdir, videos_files[ind])
        return input_video_file

    def find_detections_file(self, base_dir, videoID, skip = False):
        base_subdir = path.join(base_dir, self.tracking.dets_dir)
        dets_file = None
        if not path.isdir(base_subdir):
            print('Could not find any folder with detections.')
            answer = _input('Continue anyway? [\033[0;1my\033[0m/n] ')
            if len(answer) != 0 and answer[0].lower() != 'y':
                print('Detections folder not found.')
                exit()
            skip = True
        if not skip:
            #subfolders = get_subfolders(base_subdir)
            #subfolders_lower = [item.lower() for item in subfolders]
            files, files_base_dirs = get_all_files(base_subdir)
            det_file_ext = self.tracking.det_file_ext
            video_file_ext = self.input_video_file.split('.')[-1]
            if det_file_ext[0] == '.':
                det_file_ext = det_file_ext[1:]
            if det_file_ext[:3] == '{:}' or \
                det_file_ext[:4] == '{:s}' or \
                det_file_ext[:5] == '{0:s}':
                det_file_ext = det_file_ext.format(video_file_ext)
            n_ext = len(det_file_ext)
            indexes = [i for i, item in enumerate(files) \
                if item[-n_ext:] == det_file_ext or item[-n_ext:].lower() == det_file_ext]
            videosIDs = [files[i][:-n_ext-1] for i in indexes]
            videosIDs_lower = [item.lower() for item in videosIDs]
            dets_files = [files[i] for i in indexes]
            dets_base_dirs = [files_base_dirs[i] for i in indexes]
            if videoID in videosIDs:
                ind = videosIDs.index(videoID)
                base_subdir = dets_base_dirs[ind]
                dets_file = path.join(base_subdir, dets_files[ind])
            else:
                min_dist, ind = get_min_dist(videoID.lower(), videosIDs_lower)
                if ind is None or min_dist > abs(len(videosIDs[ind])-len(videoID)):
                    print('No detection file for \'{:s}\' was found.'.format(videoID))
                    answer = _input('Continue anyway? [\033[0;1my\033[0m/n] ')
                    if len(answer) != 0 and answer[0].lower() != 'y':
                        print('Detection file not found.')
                        exit()
                    base_subdir = None
                    dets_file = None
                else:
                    base_subdir = dets_base_dirs[ind]
                    dets_file = path.join(base_subdir, dets_files[ind])
        return base_subdir, dets_file



        
