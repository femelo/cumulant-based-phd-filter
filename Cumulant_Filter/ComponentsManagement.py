import numpy as np
import copy as cp
from .MultiObjectFilter import iou_dist
N_MISS_DET = 10
val_min = np.spacing(0)

def prune(intensity_in, markers_in, threshold, t_miss_all, N_miss):
    # Get components
    w, l, m, P, t = \
    intensity_in['w'], intensity_in['l'], intensity_in['m'], \
    intensity_in['P'], intensity_in['t']
    m_keys = markers_in.keys()
    m_params = [markers_in[key] for key in m_keys]

    if len(w) > 0:
        op1 = (w < threshold) & (t_miss_all == 0)
        op3 = np.zeros(w.shape).astype(bool)
        n_inds = np.logical_or(op1, op3)
        inds = np.logical_not(n_inds)
    else:
        inds = np.array([False])

    if np.sum(inds.astype('int')) > 0:
        w = w[inds]
        l = l[inds]
        m = m[:,inds]
        P = P[:,:,inds]
        t = t[inds]
        m_params_out = cp.copy(m_params)
        for i in range(len(m_params)):
            if len(m_params[i].shape) > 1:
                m_params_out[i] = m_params[i][:,inds]
            else:
                m_params_out[i] = m_params[i][inds]
    else:
        w = []
        l = []
        m = []
        P = []
        t = []
        m_params_out = [[] for key in m_keys]

    # Set output
    intensity_out = {'w': w, 'l': l, 'm': m, 'P': P, 't': t}
    markers_out = cp.copy(markers_in)
    for i, key in enumerate(m_keys):
        markers_out[key] = m_params_out[i]
    return intensity_out, markers_out

# Auxiliary functions
def w_sum_vec(w,vec):
    n_x = vec.shape[0]
    w_vec = np.matlib.repmat(w.T,n_x,1)
    return np.sum(w_vec * vec,axis=1)

def w_sum_mat(w,mat):
    n_c = mat.shape[2]
    w_mat = mat
    for i in range(n_c):
        w_mat[:,:,i] = w[i] * mat[:,:,i]
    return np.sum(w_mat,axis=2)

def w_max(w,wei,vec):
    i_max = np.argmax(w)
    wei = wei[i_max]
    vec = vec[:,i_max]
    return wei, vec

def cap(intensity_in, markers_in, max_num, min_num, tracks=None, threshold=0):
    # Get components
    w, l, m, P, t = \
    intensity_in['w'], intensity_in['l'], intensity_in['m'], \
    intensity_in['P'], intensity_in['t']
    m_keys = markers_in.keys()
    m_params = [markers_in[key] for key in m_keys]
    if tracks is not None:
        l_trk = tracks['labels']
        N_trk = len(l_trk)
        min_num = max(min_num, N_trk)
    else:
        l_trk = None

    if len(w) > max_num:
        inds = np.argsort(-w)
        l_out = l[inds[0:max_num]]
        l_unq = list(set(l_out))
        while len(l_unq) < min_num and len(inds) < max_num:
            max_num += 1
        
        inds_sel = inds[0:max_num]

        if l_trk is not None:
            inds_ext = inds[max_num:].tolist()
            l_ext = l[inds_ext].tolist()
            w_ext = w[inds_ext]
            l_unq_ext_s = set(l_ext)
            l_int_s = l_unq_ext_s.intersection(set(l_trk)) # verify labels that are in the list of tracks
            l_rem = list(set(l_int_s)-set(l_unq)) # verify those that are not in l_unq
            inds_rem = [inds_ext[l_ext.index(l_i)] for l_i in l_rem if w_ext[l_ext.index(l_i)] > threshold]
            inds_sel = np.append(inds_sel, inds_rem).astype(int)
        
        w_out = w[inds_sel]
        w_out = w_out * np.sum(w)/np.sum(w_out)
        l_out = l[inds_sel]
        m_out = m[:,inds_sel]
        P_out = P[:,:,inds_sel]
        t_out = t[inds_sel]
        m_params_out = cp.copy(m_params)
        for i in range(len(m_params)):
            if len(m_params[i].shape) > 1:
                m_params_out[i] = m_params[i][:,inds_sel]
            else:
                m_params_out[i] = m_params[i][inds_sel]
    else:
        w_out = w
        l_out = l
        m_out = m
        P_out = P
        t_out = t
        m_params_out = cp.copy(m_params)
    # Set output
    intensity_out = {'w': w_out, 'l': l_out, 'm': m_out, 'P': P_out, 't': t_out}
    markers_out = cp.copy(markers_in)
    for i, key in enumerate(m_keys):
        markers_out[key] = m_params_out[i]

    return intensity_out, markers_out

        

