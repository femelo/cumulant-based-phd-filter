# import sys
# from os import path
# sys.path.append( path.dirname( path.dirname( path.abspath(__file__) ) ) )
from .MultiObjectFilter import MultiObjectFilter, cos_dist, euc_dist, iou_dist, split_m_dist, out_of_bounds
from .ComponentsManagement import prune, cap
from scipy.linalg import block_diag
try:
    from scipy.misc import logsumexp
except:
    from scipy.special import logsumexp
from scipy.stats import chi2
import numpy as np
from copy import copy, deepcopy
from itertools import product
from termcolor import colored

# Constants
val_min = np.spacing(1)
log_val_min = np.log(val_min)
real_min = 2.0 ** -1022.0

class CumulantFilter:
    def __init__(self, parameters, img_dim=(1920, 1080), num_frames=500, debug=False):
        # Set model
        self.parameters = parameters
        self.imgDim = img_dim
        self.numOfFrames = num_frames
        self.debug = debug
        # Set model
        self.set_model()

        self.declutter = parameters.declutter
        self.n_x = self.model['x_dim']
        self.n_z = self.model['z_dim']
        self.obs_indexes = self.model['obs_indexes']
        # Set the multi-object filter structure
        self.filter = MultiObjectFilter(self.model, self.debug)
        self.motionModel = self.model['motion_model']
        self.N_miss = self.model['N_miss']
        # Initialize estimates
        self.estimates = {'states': [], 'covariances': [], 'boxes': [], 'velocities': [], 'labels': [], \
        'N': 0, 'var_N': 0, 'times': [], 'misses': [], 'inactive': []}
        self.ended_tracks = {'states': [], 'covariances': [], 'velocities': [], 'labels': [], \
        'N': 0, 'offtimes': [], 'times': [], 'inactive': []}
        self.marks_estimates = {'l': [], 't': []}
        # Current intensity and features
        self.intensity = {'w': [], 'l': [], 'm': [], 'P': [], 't': []}
        self.marks = {'l': [], 't': [], 't_miss': []}
            
        self.birth_intensity = {'w': [], 'l': [], 'm': [], 'P': [], 't': []}
        self.birth_marks = {'l': [], 't': []}

        self.k = 0 # Initial time stamp
        # Cardinality distribution parameters
        self.c_1 = 0.0 # first-order factorial cumulant
        self.c_2 = val_min # second-order factorial cumulant
        self.c_2_clutter = self.model['var_c'] - self.model['mu_c']
        self.var_N = 0
        self.N = 0
        # Other parameters
        self.l_dictionary = [] # list of labels used in dictionary
        self.time = [] # track times
        self.misses = [] # number of detection misses
    
    ## Initialize model
    def set_model(self):
        img_dim = self.imgDim
        num_frames = self.numOfFrames
        parameters = self.parameters

        model = {}
        model['img_width'] = img_dim[0]
        model['img_height'] = img_dim[1]

        # Model depends on whether the tracking coordinates are in 3D
        if parameters is None:
            motion_model = 'RW'
        else:
            if parameters.motion_model.lower() in ['rw', 'cv']:
                motion_model = parameters.motion_model
            else:
                if self.debug:
                    print(colored('Invalid motion model: \'{:s}\'. Setting to default.'.format(parameters.motion_model), 'yellow'))
                motion_model = 'RW'

        model['motion_model'] = motion_model

        # Basic parameters
        if motion_model.lower() == 'cv':
            model['x_dim'] = 6   # dimension of state vector (x,y,vx,vy,w,h) ---- Constant velocity (CV)
            model['obs_indexes'] = [0,1,4,5]
        elif motion_model.lower() == 'rw':
            model['x_dim'] = 4  # dimension of state vector (x,y,w,h) --------- Random walk (RW)
            model['obs_indexes'] = list(range(4))

        model['z_dim'] = 4   # dimension of observation vector (x,y,w,h)
        
        # Dynamical model parameters (CV model or RW model)
        T = 1.0
        model['T'] = T # sampling period
        if motion_model.lower() == 'cv':
            if parameters is not None:
                alpha_q_mat = np.zeros((6, 4))
                I2 = np.eye(2)
                alpha_q_mat[np.ix_([0,1], [0,1])] = I2
                alpha_q_mat[np.ix_([2,3], [0,1])] = I2
                alpha_q_mat[np.ix_([4,5], [2,3])] = I2
                alpha_q = np.diag(alpha_q_mat.dot(np.array(parameters.sigma_q)))
                alpha_r = np.diag(parameters.sigma_r)
            else:
                # WAMI 01
                alpha_q = np.diag([3.00, 3.00, 3.00, 3.00, 3.0, 3.0])
                alpha_r = np.diag([10.0, 10.0, 1.0, 1.0])
                # WAMI 02
                # alpha_q = np.diag([1.00, 1.00, 1.00, 1.00, 5.0, 5.0])
                # alpha_r = np.diag([10.0, 10.0, 10.0, 10.0])
        else:
            if parameters is not None:
                alpha_q = np.diag(parameters.sigma_q)
                alpha_r = np.diag(parameters.sigma_r)
            else:
                alpha_q = np.diag([2.5, 2.5, 0.1, 0.1])
                alpha_r = np.diag([15.0, 15.0, 1.00, 1.00])

        if motion_model.lower() == 'cv':
            # Set transition matrix
            F1 = np.array([[1.0, T], [0.0, 1.0]]) # transition matrix per centroid coordinate
            F = np.eye(6) # initialise complete transition matrix
            F[np.ix_([0,2], [0,2])] = F1
            F[np.ix_([1,3], [1,3])] = F1
            model['F'] = F
            # Set process noise covariance matrix
            Q1 = np.array([[T**3/3, T**2/2], [T**2/2, T]]) # covariance matrix per centroid coordinate, continuous
            # Q1 = np.array([[T**4/4, T**3/2], [T**3/2, T**2]])  # covariance matrix per centroid coordinate, discrete
            Q = np.zeros((6,6))
            Q[np.ix_([0,2], [0,2])] = Q1
            Q[np.ix_([1,3], [1,3])] = Q1
            Q[4,4] = 1.0
            Q[5,5] = 1.0
            model['Q'] = alpha_q.dot(Q)
        elif motion_model.lower() == 'rw':
            F1 = np.eye(2) # transition matrix for the centroid and size
            F2 = np.eye(2)
            model['F'] = block_diag(F1,F2)
            # Set process noise covariance matrix
            Q = np.zeros((4,4))
            Q1 = T*np.eye(2)
            Q[np.ix_([0,1], [0,1])] = Q1
            Q[np.ix_([2,3], [2,3])] = Q1
            model['Q'] = alpha_q.dot(Q)

        # Observation model parameters
        I1 = np.eye(2)
        O1 = np.zeros((2,2))
        O2 = np.zeros((2,2))
        I2 = np.eye(2)
        if motion_model.lower() == 'cv':
            # Observation matrix
            model['H'] = np.concatenate((np.concatenate((I1, O1, O2), axis=1),
                np.concatenate((O2.T, O2.T, I2), axis=1)), axis=0)
            # Transpose observation matrix
            H_t = np.transpose(model['H'])
            model['H_t'] = H_t
        elif motion_model.lower() == 'rw':
            # Observation matrix
            model['H'] = np.eye(4)
            # Transpose observation matrix
            H_t = np.transpose(model['H'])
            model['H_t'] = H_t
        # Measurement noise
        model['R'] = alpha_r.dot(block_diag(I1, I2))

        if parameters is None:
            model['N_miss'] = 10
            # Gating probability
            model['p_g'] = 0.9999
        else:
            model['N_miss'] = parameters.N_miss
            # Gating probability
            model['p_g'] = parameters.prob_gating
        if self.debug:
            print(colored('Number of allowed misses: {0:d}'.format(model['N_miss']), 'magenta'))

        # Gating threshold
        model['d_g'] = 2
        model['gamma_thr'] = chi2.ppf(model['p_g'], model['d_g'])
        if parameters is None:
            model['min_IOU'] = 0.00
            model['reid_thr'] = 0.10
        else:
            model['min_IOU'] = parameters.iou_threshold
            model['reid_thr'] = parameters.reid_threshold

        # Survival/death parameters
        if parameters is not None:
            p_s = parameters.prob_survival
        else:
            p_s = 0.9999
        model['p_s'] = p_s
        model['q_s'] = 1 - p_s
        model['log_p_s'] = np.log(p_s)
        model['log_q_s'] = np.log(1 - p_s)

        # Birth intensity (per component)
        model['mu_birth'] = 1.0 / num_frames
        model['sigma_v_birth'] = 0.5

        # Physical constraint
        model['delta_p_max'] = 25

        # Detection and clutter rate parameter
        if parameters is not None:
            p_d = parameters.prob_detection
            lambda_c = parameters.false_alarm_rate
        else:
            p_d = 0.90
            lambda_c = 0.01

        # Detection parameters
        model['p_d'] = p_d # probability of detection
        model['q_d'] = 1.0 - p_d # probability of missed detection
        model['log_p_d'] = np.log(p_d)
        model['log_q_d'] = np.log(1.0 - p_d)

        # Clutter (false alarm) parameters
        model['mu_c'] = lambda_c
        model['var_c'] = lambda_c
        # Poisson average rate of uniform clutter (per scan)
        model['range_c'] = np.array([[0, img_dim[0]],[0, img_dim[1]]]) # uniform clutter region
        model['pdf_c'] = 1.0 / np.prod(model['range_c'][:,1]-model['range_c'][:,0]) # uniform clutter density

        model['log_mu_c'] = np.log(lambda_c)
        model['log_pdf_c'] = np.log(model['pdf_c'])

        # Reference number of frames (half-life of the envelope) for the time constant
        model['K_ref'] = 20.0
        # Time constant
        model['tau'] = model['K_ref']/np.log(2.0)

        # Maximum number of intensity components
        model['L'] = 7
        model['L_max'] = 350
        if parameters is None:
            # Threhsold to prune neglectable intensity components
            model['prune_thr'] = parameters.pruning_threshold
        else:
            model['prune_thr'] = 1.0e-4
            # Threshold to merge similar intensity components
        # model['merge_thr'] = 0.80

        # For decluttering track display only
        model['velocity_sup'] = 12.34 # corresponds to 200 km/h = 55.55 m/s = 12.34 pixels/frame
        model['velocity_inf'] = 0.75
        model['acceleration_sup'] = 0.1

        # Save model
        self.model = model

    ## Prediction step.
    def predict(self, detections):
        n_x = self.n_x
        n_z = self.n_z
        # Get parameters
        p_s = self.model['p_s']
        # c_1_upd = self.c_1
        c_2_upd = self.c_2

        indexes = self.obs_indexes

        # Predict intensity and marks
        intensity_prd, marks_prd = \
            self.filter.predict(self.intensity, self.marks)
        # Incorporate newborn components
        intensity_prd, marks_prd = \
            self.filter.add_birth(intensity_prd, marks_prd, self.birth_intensity, self.birth_marks)
            
        # Cumulants prediction
        c_1_prd = np.sum(intensity_prd['w'])
        c_2_prd = (p_s**2) * c_2_upd

        # Set intensity, marks and cardinality parameters
        self.intensity = intensity_prd
        self.marks = marks_prd
        self.c_1 = c_1_prd
        self.c_2 = c_2_prd
        self.N = c_1_prd
        self.var_N = c_1_prd + c_2_prd
        self.filter.p_s = p_s
    
    ## Gating.
    def gate(self, detections):
        if len(detections['z']) > 0:
            m = detections['z'].shape[1]
        else:
            m = 0
        if m > 0:
            g_measurements, ng_measurements = \
                self.filter.gate(detections, self.intensity, self.marks)
        else:
            # Set empty outputs
            g_measurements = {'z': [], 'S': [], 'sqrt_S': [], 'inv_S': [], 'nu': []}
            ng_measurements = {'z': []}

        return g_measurements, ng_measurements

    ## Data update step.
    def update(self, g_measurements, ng_measurements):
        # Get parameters
        intensity = self.intensity
        w_prd, l_prd, m_prd, P_prd, t_prd = \
            intensity['w'], intensity['l'], intensity['m'], \
            intensity['P'], intensity['t']
        marks = self.marks
        l_f_prd, t_f_prd, t_miss = marks['l'], marks['t'], marks['t_miss']

        c_1_prd = self.c_1
        c_2_prd = self.c_2
        c_2_clutter = self.c_2_clutter
        # mu_prd = self.N
        p_d = self.model['p_d']
        # q_d = self.model['q_d']
        log_p_d = self.model['log_p_d']
        log_q_d = self.model['log_q_d']
        mu_c = self.model['mu_c']
        pdf_c = self.model['pdf_c']
        # log_pdf_c = self.model['log_pdf_c']
        self.filter.p_d = p_d
        
        # Number of gated measurements
        if len(g_measurements['z']) > 0:
            m = g_measurements['z'].shape[1]
        else:
            m = 0

        # Pre-calculation for Kalman update parameters
        if m > 0:
            # Get variables from hypotheses
            hypotheses = \
                self.filter.update(self.intensity, g_measurements)
            hyp_log_q, hyp_m, hyp_P = \
                hypotheses['log_q_b'], hypotheses['m_b'], hypotheses['P_b']
        else:
            hyp_log_q, hyp_m, hyp_P = [], [], []

        # Pre-calculation of elementary symmetric functions
        if len(l_prd) > 0:
            log_w_prd = np.log(w_prd)
        else:
            log_w_prd = []
        
        # Intensity update
        # Compute l-factors
        alpha = (p_d*c_1_prd +mu_c)**2/(c_2_prd +c_2_clutter)
        # beta = (p_d*c_1_prd +mu_c)/(c_2_prd +c_2_clutter)
        log_factor_num = np.log(complex(alpha + m))
        log_factor_den = np.log(complex(alpha + mu_c + p_d*c_1_prd))
        log_l1_q = log_factor_num-log_factor_den
        log_l2_q = log_factor_num-2*log_factor_den

        # Missed detection components
        if len(l_prd) > 0:
            log_w_upd = np.real(log_l1_q +log_q_d) +log_w_prd
            log_w_upd_2 = log_l2_q + 2.0*log_w_upd
            t_upd = t_prd+1
        else:
            log_w_upd = []
            log_w_upd_2 = []
            t_upd = []

        m_upd = m_prd
        P_upd = P_prd
        l_upd = l_prd

        if len(l_f_prd) > 0:
            inds = list(range(len(l_f_prd)))
            t_miss_upd = np.ones((len(inds),))
        else:
            t_miss_upd = []
        
        if m > 0:
            for ell in range(m):
                log_w = log_p_d + hyp_log_q[:,ell] +log_w_prd
                sum_w = np.sum(np.exp(log_w))
                log_w = log_w -np.log(mu_c*pdf_c +sum_w)

                log_w_upd = np.concatenate((log_w_upd,log_w))
                log_w_upd_2 = np.concatenate((log_w_upd_2, 2.0*log_w.astype(complex) +np.pi*1j))

                m_upd = np.concatenate((m_upd,hyp_m[:,:,ell]), axis=1)
                P_upd = np.concatenate((P_upd,hyp_P[:,:,:,ell]), axis=2)
                l_upd = np.concatenate((l_upd,l_prd))
                t_upd = np.concatenate((t_upd,t_prd+1))

                t_miss_upd = np.concatenate((t_miss_upd,t_miss))
        
        # Get intensity and hypothetical feature weights
        if len(log_w_upd) > 0:
            w_upd = np.exp(log_w_upd)
            w_upd_2 = np.real(np.exp(log_w_upd_2))
        else:
            w_upd = []
            w_upd_2 = []

        # Cardinality update
        # Updated number of targets
        c_1_upd = np.sum(w_upd)
        c_2_upd = np.sum(w_upd_2)
        mu_upd = c_1_upd
        if np.isnan(c_1_upd):
            c_1_upd = np.nansum(w_upd)
            c_2_upd = np.nansum(w_upd_2)
            mu_upd = c_1_upd
            if c_1_upd == 0:
                w_upd = []
                w_upd_2 = []
                m_upd = []
                P_upd = []
                l_upd = []
                t_upd = []
                t_miss_upd = []
            else:
                n_nan_idx = np.logical_not(np.isnan(w_upd))
                w_upd = w_upd[n_nan_idx]
                w_upd_2 = w_upd_2[n_nan_idx]
                m_upd = m_upd[:,n_nan_idx]
                P_upd = P_upd[:,:,n_nan_idx]
                l_upd = l_upd[n_nan_idx]
                t_upd = t_upd[n_nan_idx]
                t_miss_upd = t_miss_upd[n_nan_idx]
        
        # Estimate variance
        var_upd = max(c_2_upd +c_1_upd, np.spacing(1))
        if np.isnan(var_upd):
            var_upd = 0.0

        # Pruning, merging, capping
        intensity_upd = {'w': w_upd, 'l': l_upd, 'm': m_upd, 'P': P_upd, 't': t_upd}
        hyp_marks_upd = {'l': l_upd, 't': t_upd, 't_miss': t_miss_upd}
        intensity_upd, hyp_marks_upd = prune(intensity_upd, hyp_marks_upd, self.model['prune_thr'], \
            t_miss_upd if len(t_miss_upd) > 0 else [], self.N_miss)
        intensity_upd, hyp_marks_upd = cap(intensity_upd, hyp_marks_upd, \
            min(self.model['L_max'], int(mu_upd*self.model['L'])), np.round(mu_upd), \
            self.estimates, self.model['prune_thr'])

        # Update parameters after pruning, merging and capping
        l_upd = intensity_upd['l']
        L_upd = len(l_upd)
        if L_upd <= 0:
            c_1_upd = 0.0
            c_2_upd = 1.0
            mu_upd = 0.0
            var_upd = 1.0

        # Introduce a new set of weights to avoid unrealistic motion (physical constraint)
        #w_con = self.set_constraint(intensity_upd)
        w_con = np.ones((len(l_upd),))
        
        # Update estimates
        estimates, marks_estimates, ended_tracks, w_cmb = self.get_estimates(intensity_upd, hyp_marks_upd, w_con, mu_upd, var_upd)
        # Set new birth model and re-update the intensity and features accordingly
        marks_upd = hyp_marks_upd
        birth_intensity, birth_marks, intensity_upd, marks_upd = \
        self.filter.set_birth_model(ng_measurements, intensity_upd, marks_upd, estimates, ended_tracks)

        # Assign outputs
        self.estimates = estimates
        self.ended_tracks = ended_tracks
        self.intensity = intensity_upd
        self.birth_intensity = birth_intensity
        self.marks = marks_upd
        self.marks_estimates = marks_estimates
        self.birth_marks = birth_marks
        self.k += 1 # Iterate
        # Cardinality distribution parameters
        self.c_1 = c_1_upd # alpha
        self.c_2 = c_2_upd # beta
        self.N = mu_upd
        self.var_N = var_upd
        self.l_dictionary = marks_upd['l']
        self.time = marks_upd['t']
        self.misses = marks_upd['t_miss']

    ## Set physical constraint.
    def set_constraint(self, intensity_upd):
        # Get data
        l_upd = intensity_upd['l']
        m_upd = intensity_upd['m']
        # Allocate vector
        w_con = np.ones((len(l_upd),))

        boxes = self.estimates['boxes']

        if self.motionModel.lower() == 'cv':
            pos_ind, siz_ind = [0,2], [4,5]
        elif self.motionModel.lower() == 'rw':
            pos_ind, siz_ind = [0,1], [2,3]

        if len(boxes) > 0:
            # Get other data
            velocities = self.estimates['velocities']
            labels = self.estimates['labels']
            widths = boxes[2,:]
            areas = np.prod(boxes[[2,3],:], axis=0)
            #delta_p_max = self.model['delta_p_max']
            delta_p_max = 2*(self.model['Cov_w1'] * widths) ** 2
            T = self.model['T']
            # Make a vector of zeros
            n_labels = len(labels)
            for i_l in range(n_labels):
                lbl = labels[i_l]
                inds_lbl = l_upd == lbl
                n_inds = np.sum(inds_lbl.astype('int'))
                if n_inds > 0:
                    p_km1 = boxes[[0,1],i_l].reshape(2,1)
                    v_km1 = velocities[[0,1],i_l].reshape(2,1)
                    p_k_km1 = p_km1 + v_km1*T
                    p_l = np.zeros((2,n_inds))
                    p_l[0,:] = m_upd[pos_ind[0],inds_lbl]
                    p_l[1,:] = m_upd[pos_ind[1],inds_lbl]
                    a_l = m_upd[siz_ind[0],inds_lbl] * m_upd[siz_ind[1],inds_lbl]
                    ratio_a = np.maximum(a_l/areas[i_l], areas[i_l]/a_l)
                    dist = np.sqrt(np.sum((p_l-p_k_km1)**2,axis=0))
                    w_con[inds_lbl] = np.logical_and(dist < delta_p_max[i_l], ratio_a < 1.10).astype('float')
        return w_con
        
    ## Get estimates.
    def get_estimates(self, *args):
        intensity_upd, hyp_marks_upd, w_constraint, N_upd, var_N_upd = args
        w_upd, l_upd,  m_upd, P_upd, t_upd = \
        intensity_upd['w'], intensity_upd['l'], intensity_upd['m'], \
        intensity_upd['P'], intensity_upd['t']
        t_miss_upd = hyp_marks_upd['t_miss']
        
        H = self.model['H']
        n_z = self.model['z_dim']

        # Check for components
        L_upd = len(w_upd)
        if L_upd == 0:
            w_cmb = []
            estimates = {'states': [], 'covariances': [], 'boxes': [], 'velocities': [], 'labels': [], \
            'times': [], 'N': 0, 'var_N': 0, 'misses': [], 'inactive': [], \
            'on_track_states': [], 'on_track_covariances': []}
            ended_tracks = {'states': [], 'covariances': [], 'boxes': [], 'velocities': [], 'labels': [], \
            'offtimes': [], 'times': [], 'N': 0, 'inactive': [], \
            'on_track_states': [], 'on_track_covariances': []}
            marks_estimates = {'l': [], 't': []}
            return estimates, marks_estimates, ended_tracks, w_cmb

        # Estimates of previous tracks
        estimates = deepcopy(self.estimates)
        m_trk, P_trk, v_trk, l_trk, t_trk, N_trk, var_N_trk, N_miss_trk, inact_trk, m_trk_on, P_trk_on  = \
            estimates['states'], estimates['covariances'], estimates['velocities'], \
            estimates['labels'], estimates['times'], \
            estimates['N'], estimates['var_N'], estimates['misses'], estimates['inactive'], \
            estimates['on_track_states'], estimates['on_track_covariances']
        ended_tracks = deepcopy(self.ended_tracks)
        m_end, P_end, v_end, l_end, ot_end, t_end, N_end, inact_end, m_end_on, P_end_on  = \
            ended_tracks['states'], ended_tracks['covariances'], ended_tracks['velocities'], \
            ended_tracks['labels'], ended_tracks['offtimes'], ended_tracks['times'], \
            ended_tracks['N'], ended_tracks['inactive'], \
            estimates['on_track_states'], estimates['on_track_covariances']

        # Get cardinality estimates
        l_unq = list(set(l_upd))

        # Calculate association mask
        if self.motionModel.lower() == 'rw':
            inds1 = [0,1,2,3]
            inds2 = []
        elif self.motionModel.lower() == 'cv':
            inds1 = [0,1,4,5]
            inds2 = [2,3]
        obs_ind = self.obs_indexes
        vel_ind = inds2
        pos_ind = [obs_ind[0],obs_ind[1]]
        siz_ind = [obs_ind[2],obs_ind[3]]
        assoc_mask = np.zeros((L_upd,))
        for l in l_unq:
            ind1 = [i for i in range(L_upd) if l_upd[i] == l]
            if l not in l_trk:
                assoc_mask[ind1] = 1.0
                # assoc_mask[ind1] = 0.5
            else:
                ind2 = ind1
                ind_trk = [i for i in range(len(l_trk)) if l_trk[i] == l]
                N2 = len(ind2)
                if N2 > 0:
                    S_sel = np.einsum('ij...,jk->ik...', H.dot(P_trk[:,:,ind_trk] +P_upd[:,:,ind2]), H.T)
                    m_dist_pos, m_dist_siz, det_S_sel = split_m_dist(H.dot(m_trk[:,ind_trk]), H.dot(m_upd[:,ind2]), S_sel)
                    dist_b = (m_dist_pos < self.model['gamma_thr']).astype(float)
                    dist_s = 0.5*n_z*np.log(2*np.pi) +0.5*np.log(det_S_sel) +0.5*m_dist_pos + 0.5*m_dist_siz
                    assoc_mask[ind2] = dist_b * np.exp(-dist_s)

        keep_comp_ind = np.logical_or(assoc_mask.astype(bool), w_upd > self.model['prune_thr'])
        keep_ind = np.logical_and(keep_comp_ind, t_miss_upd <= 1)
        keep_comp_ind = keep_ind
        assoc_mask = assoc_mask[keep_ind]
        w_upd_red = copy(w_upd[keep_ind])
        l_upd_red = copy(l_upd[keep_ind])
        m_upd_red = copy(m_upd[:,keep_ind])
        P_upd_red = copy(P_upd[:,:,keep_ind])
        t_upd_red = copy(t_upd[keep_ind])

        t_miss_upd_red = copy(t_miss_upd[keep_ind])

        w_const_red = copy(w_constraint[keep_ind])
        
        intensity_upd['l'] = l_upd[keep_comp_ind]
        intensity_upd['w'] = w_upd[keep_comp_ind]
        intensity_upd['m'] = m_upd[:,keep_comp_ind]
        intensity_upd['P'] = P_upd[:,:,keep_comp_ind]
        intensity_upd['t'] = t_upd[keep_comp_ind]
        
        # Get cardinality estimates
        l_unq = list(set(l_upd_red))
        N_est = np.round(min([N_upd,len(l_unq)])).astype('int')
        var_N_est = var_N_upd

        # Combined weights
        w_cmb = (1.0-np.exp(-(t_upd_red-t_miss_upd_red)/(self.numOfFrames/2.0))) * w_const_red * assoc_mask * w_upd_red
        # Get indexes of tracks
        tracks_inds = np.argsort(-w_cmb)

        n_x = m_upd.shape[0]
        n_z = self.n_z

        l_est = -np.ones((N_est,)).astype('int')
        m_est = np.zeros((n_x,N_est))
        P_est = np.zeros((n_x,n_x,N_est))
        if n_x > n_z:
            v_est = np.zeros((n_x-n_z,N_est))
        else:
            v_est = []
        t_est = -np.ones((N_est,)).astype('int')
        t_miss_est = np.zeros(t_est.shape)

        i = 0
        j = 0
        while any(l_est == -1):
            if not any(l_est == l_upd_red[tracks_inds[j]]):
                l_est[i] = l_upd_red[tracks_inds[j]]
                m_est[:,i] = m_upd_red[:,tracks_inds[j]]
                P_est[:,:,i] = P_upd_red[:,:,tracks_inds[j]]
                if len(inds2) > 0:
                    v_est[:,i] = m_upd_red[inds2,tracks_inds[j]]
                t_est[i] = t_upd_red[tracks_inds[j]]
                t_miss_est[i] = t_miss_upd_red[tracks_inds[j]]
                i += 1
            j += 1

        if len(v_est) == 0:
            v_est = np.zeros((2,m_est.shape[1]))

        # Update ended tracks
        if len(l_end) > 0:
            ot_end += 1
            if len(l_est) > 0:
                ind1 = np.array([not l_end[i] in l_est for i in range(l_end.size)])
            else:
                ind1 = np.ones(l_end.shape).astype(bool)
            ind2 = ot_end < self.N_miss+1
            # Ended tracks inside bounds
            ind_ib = np.logical_not(out_of_bounds(m_end[pos_ind,:], v_end, self.imgDim[0], self.imgDim[1]))
            ind = np.logical_and(np.logical_and(ind1, ind2), ind_ib)
            # ind = np.logical_and(ind1, ind2)
            l_end = l_end[ind]
            len_ind = np.sum(ind.astype(int))
            if len_ind > 0:
                F = self.model['F']
                Q = self.model['Q']
                Q_block = np.repeat(Q[:,:,np.newaxis], len_ind, axis=2)
                m_end = F.dot(m_end[:,ind])
                P_Ft = np.swapaxes(np.einsum('ij,jkl->ikl', F, P_end[:,:,ind]), 0, 1)
                P_end = np.einsum('ij,jkl->ikl', F, P_Ft) + Q_block
                
                v_end = v_end[:,ind]
                ot_end = ot_end[ind]
                t_end = t_end[ind]
                inact_end = inact_end[ind]

        # Verify against previous tracks
        if len(l_trk) > 0:
            # Verify new appearing tracks
            ind = [i for i in range(N_trk) if not l_trk[i] in l_est]
            if len(ind) > 0:
                N_miss_trk[ind] += 1
                # Remove tracks with more than N_miss misses
                miss_limits = (self.N_miss + 1)*np.ones(N_miss_trk.shape)
                ind = N_miss_trk < miss_limits
                # Tracks inside bounds
                ind_ob = out_of_bounds(m_trk[pos_ind,:], v_trk, self.imgDim[0], self.imgDim[1])
                ind_ib = np.logical_not(ind_ob)
                # Indexes of tracks that must be terminated
                n_ind = np.logical_or(np.logical_not(ind), np.logical_and(N_miss_trk == 1, ind_ob))
                ind = np.logical_not(n_ind)
                # Add newly terminated tracks to the list of ended tracks
                if np.any(n_ind):
                    if len(l_end) > 0:
                        l_end = np.concatenate((l_end, l_trk[n_ind]))
                        m_end = np.concatenate((m_end, m_trk[:,n_ind]), axis=1)
                        P_end = np.concatenate((P_end, P_trk[:,:,n_ind]), axis=2)
                        v_end = np.concatenate((v_end, v_trk[:,n_ind]), axis=1)
                        ot_end = np.concatenate((ot_end, np.zeros(len(n_ind),)))
                        t_end = np.concatenate((t_end, t_trk[n_ind]))
                        inact_end = np.concatenate((inact_end, inact_trk[n_ind]))
                        m_end_on = np.concatenate((m_end_on, m_trk_on[:,n_ind]), axis=1)
                        P_end_on = np.concatenate((P_end_on, P_trk_on[:,:,n_ind]), axis=2)
                    else:
                        l_end = l_trk[n_ind]
                        m_end = m_trk[:,n_ind]
                        P_end = P_trk[:,:,n_ind]
                        v_end = v_trk[:,n_ind]
                        t_end = t_trk[n_ind]
                        ot_end = np.zeros(np.sum(n_ind.astype(int)),)
                        inact_end = inact_trk[n_ind]
                        m_end_on = m_trk_on[:,n_ind]
                        P_end_on = P_trk_on[:,:,n_ind]
                # Tracks still alive
                l_trk = l_trk[ind]
                m_trk = m_trk[:,ind]
                P_trk = P_trk[:,:,ind]
                v_trk = v_trk[:,ind]
                t_trk = t_trk[ind]
                N_miss_trk = N_miss_trk[ind]
                inact_trk = inact_trk[ind]
                m_trk_on = m_trk_on[:,ind]
                P_trk_on = P_trk_on[:,:,ind]

            # Update tracks with new estimates
            ind_src = [i for i in range(N_est) if l_est[i] in l_trk]
            l_lst = l_trk.tolist()
            ind_dst = [l_lst.index(l_est[i]) for i in range(N_est) if l_est[i] in l_trk]
            ind_rem = list(set(range(len(l_trk)))-set(ind_dst))
            if len(ind_rem) > 0:
                F = self.model['F']
                Q = self.model['Q']
                Q_block = np.repeat(Q[:,:,np.newaxis], len(ind_rem), axis=2)
                m_trk[:,ind_rem] = F.dot(m_trk[:,ind_rem])
                P_Ft = np.swapaxes(np.einsum('ij,jkl->ikl', F, P_trk[:,:,ind_rem]), 0, 1)
                P_trk[:,:,ind_rem] = np.einsum('ij,jkl->ikl', F, P_Ft) + Q_block
            l_trk[ind_dst] = l_est[ind_src]
            m_trk[:,ind_dst] = m_est[:,ind_src]
            P_trk[:,:,ind_dst] = P_est[:,:,ind_src]
            # v_trk[:,ind_dst] = v_est[:,ind_src]
            t_trk[ind_dst] = t_est[ind_src]
            N_miss_trk[ind_dst] = 0
            # Update on-track estimates
            m_trk_on[:,ind_dst] = m_est[:,ind_src]
            P_trk_on[:,:,ind_dst] = P_est[:,:,ind_src]
            
            a_max = self.model['acceleration_sup']
            v_max = self.model['velocity_sup']
            
            d_v = v_est[:,ind_src]-v_trk[:,ind_dst]
            delta_v_mag = np.sqrt(np.sum(d_v**2, axis=0))
            delta_v_mag[delta_v_mag == 0] = val_min
            d_v_max = a_max * (1.0 + self.N_miss)
            coef = (np.minimum(d_v_max, delta_v_mag)/delta_v_mag)*(t_est[ind_src] >= 0.2*self.N_miss).astype(int) + \
                (t_est[ind_src] < 0.2*self.N_miss).astype(int)
            delta_v = coef*d_v
            v_trk[:,ind_dst] += delta_v
            m_trk[np.ix_(vel_ind,ind_dst)] = v_trk[:,ind_dst]
            
            # Include new tracks
            ind = list(set(range(0,N_est))-set(ind_src))
            if len(ind) > 0:
                l_trk = np.concatenate((l_trk, l_est[ind]))
                m_trk = np.concatenate((m_trk, m_est[:,ind]), axis=1)
                P_trk = np.concatenate((P_trk, P_est[:,:,ind]), axis=2)
                v_trk = np.concatenate((v_trk, v_est[:,ind]), axis=1)
                t_trk = np.concatenate((t_trk, t_est[ind]))
                N_miss_trk = np.concatenate((N_miss_trk, np.zeros(len(ind),)))
                m_trk_on = np.concatenate((m_trk_on, m_est[:,ind]), axis=1)
                P_trk_on = np.concatenate((P_trk_on, P_est[:,:,ind]), axis=2)
                if self.declutter:
                    start_inact = 1.0
                else:
                    start_inact = 0.0
                inact_trk = np.concatenate((inact_trk, start_inact+np.zeros(len(ind),)))
            N_trk = len(l_trk)
        else: # no previous tracks found
            l_trk = l_est
            m_trk = m_est
            P_trk = P_est
            v_trk = v_est
            t_trk = t_est
            N_trk = N_est
            m_trk_on = copy(m_est)
            P_trk_on = copy(P_est)
            N_miss_trk = np.zeros(l_est.shape)
            if self.declutter:
                start_inact = 1.0
            else:
                start_inact = 0.0
            inact_trk = start_inact+np.zeros(l_est.shape)

        # Sort by labels
        inds = np.argsort(l_trk)
        l_trk = l_trk[inds]
        m_trk = m_trk[:,inds]
        P_trk = P_trk[:,:,inds]
        v_trk = v_trk[:,inds]
        t_trk = t_trk[inds]
        inact_trk = inact_trk[inds]
        m_trk_on = m_trk_on[:,inds]
        P_trk_on = P_trk_on[:,:,inds]

        # Sort by labels
        if len(l_end) > 0:
            inds = np.argsort(l_end)
            l_end = l_end[inds]
            m_end = m_end[:,inds]
            P_end = P_end[:,:,inds]
            v_end = v_end[:,inds]
            ot_end = ot_end[inds]
            t_end = t_end[inds]
            inact_end = inact_end[inds]
            m_end_on = m_end_on[:,inds]
            P_end_on = P_end_on[:,:,inds]

        N_end = len(l_end)

        # Update activity status
        if self.declutter:
            vel_ub = self.model['velocity_sup']
            vel_lb = self.model['velocity_inf']
            delta = 2.0 / (self.N_miss)
            if len(v_trk) > 0:
                v_mag = np.sqrt(np.sum(v_trk**2, axis=0))
                mov_flag = 1.0*np.logical_or(v_mag < vel_lb, v_mag > vel_ub).astype(int) \
                    -1.0*np.logical_and(v_mag >= vel_lb, v_mag <= vel_ub).astype(int)
                inact_trk = np.maximum(np.minimum(inact_trk +delta*mov_flag, 1.0), 0.0)
                inact_trk = np.minimum(np.maximum(inact_trk, delta*N_miss_trk), 1.0)
            if len(v_end) > 0:
                v_mag = np.sqrt(np.sum(v_end**2, axis=0))
                mov_flag = 1.0*np.logical_or(v_mag < vel_lb, v_mag > vel_ub).astype(int) \
                    -1.0*np.logical_and(v_mag >= vel_lb, v_mag <= vel_ub).astype(int)
                inact_end = np.maximum(np.minimum(inact_end + delta*mov_flag, 1.0), 0.0)
        else:
            inact_trk *= 0
            inact_end *= 0
            delta = 2.0 / (self.N_miss)
            if len(v_trk) > 0:
                inact_trk = np.minimum(np.maximum(inact_trk, delta*N_miss_trk), 1.0)

        # Set outputs
        m_trk_b = m_trk[pos_ind+siz_ind,:]
        m_trk_b[[0,1],:] = m_trk_b[[0,1],:]-m_trk_b[[2,3],:]/2.0
        estimates = {'states': m_trk, 'covariances': P_trk, 'boxes': m_trk_b, 'velocities': v_trk, 'labels': l_trk, \
            'times': t_trk, 'N': N_trk, 'var_N': var_N_est, 'misses': N_miss_trk, 'inactive': inact_trk, \
            'on_track_states': m_trk_on, 'on_track_covariances': P_trk_on}
        ended_tracks = {'states': m_end, 'covariances': P_end, 'velocities': v_end, 'labels': l_end, \
            'offtimes': ot_end, 'times': t_end, 'N': N_end, 'inactive': inact_end, \
            'on_track_states': m_end_on, 'on_track_covariances': P_end_on}
        marks_estimates = {'l': l_trk, 't': t_trk}
        return estimates, marks_estimates, ended_tracks, w_cmb

    ## Filter iteration.
    def iterate(self, detections_input, features_input=None):
        z = detections_input
        detections = {'z': z}
        # Predict intensity, cardinality and marks
        self.predict(detections)
        # Gate measurements and pre-compute parameters for update
        g_measurements, ng_measurements = self.gate(detections)
        # Update intensity, cardinality, marks and estimates
        self.update(g_measurements, ng_measurements)
