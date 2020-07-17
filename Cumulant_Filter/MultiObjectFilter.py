import numpy as np
from numpy import matlib as ml
from numpy.linalg import solve, inv, cholesky, det
from scipy.linalg import solve_triangular
import copy as cp
from munkres import Munkres
from termcolor import colored

val_min = np.spacing(1)
log_val_min = np.log(val_min)

# Function for updating the intensity components
def intensity_update_log(m, P, z, nu, H, S, sqrt_S, inv_S):
    n_x = m.shape[0]
    n_z = z.shape[0]
    L_m = z.shape[1]

    #K = solve(S.T,P.dot(H.T).T).T
    K = P.dot(np.dot(H.T,inv_S))
    nan_idx = np.isnan(nu[0,:])
    n_nan_idx = np.logical_not(nan_idx)
    # Log of expected likelihood
    log_norm_const = -1*np.sum(np.log(np.diag(sqrt_S[0:2,0:2]))) -0.5*(n_z/2)*np.log(2*np.pi)
    dist = -0.5*np.sum(nu[0:2,n_nan_idx] * (inv_S[0:2,0:2].dot(nu[0:2,n_nan_idx])), axis=0)
    log_q_upd = log_val_min*np.ones((L_m,))
    log_q_upd[n_nan_idx] = (log_norm_const + dist).ravel()

    nu[:,nan_idx] = 0 # does not update the prediction if corresponding detection has not been gated
    # Updated components
    m_upd = m.reshape(n_x,1) +K.dot(nu)
    P_upd = np.zeros((n_x,n_x,L_m))
    for i in range(L_m):
        if nan_idx[i]:
            P_upd[:,:,i] = P
        else:
            P_upd[:,:,i] = P -K.dot(H.dot(P))
        # Ensure symmetry
        P_upd[:,:,i] = 0.5 * (P_upd[:,:,i] + np.swapaxes(P_upd[:,:,i], 0, 1))
    
    return log_q_upd, m_upd, P_upd

def cos_dist(u, v):
    n, N_u = u.shape
    N_v = v.shape[1]    
    dist_u_v = np.zeros((N_u,N_v))
    norm_u = np.sqrt(np.sum(u**2, axis=0))
    norm_v = np.sqrt(np.sum(v**2, axis=0))
    for i in range(N_u):
        dist_u_v[i,:] = np.dot(u[:,i],v)/(norm_u[i]*norm_v)
    return dist_u_v

def euc_dist(u, v, sc_w, sc_h):
    n, N_u = u.shape
    N_v = v.shape[1]
    dist_u_v = np.zeros((N_u,N_v))
    if isinstance(sc_w,np.ndarray):
        W = sc_w
    elif isinstance(sc_w,int) or isinstance(sc_w,float):
        W = sc_w*np.ones((N_u,))
    if isinstance(sc_h,np.ndarray):
        H = sc_h
    elif isinstance(sc_h,int) or isinstance(sc_h,float):
        H = sc_h*np.ones((N_u,))
    for i in range(N_u):
        d_x = (u[0,i]-v[0,:])/W[i]
        d_y = (u[1,i]-v[1,:])/H[i]
        dist_u_v[i,:] = np.sqrt(d_x**2 + d_y**2)
    return dist_u_v

def iou_dist(u, v):
    n1 = u.shape[1]
    n2 = v.shape[1]
    x_a1 = u[[0,1],:]-u[[2,3],:]/2.0
    x_a2 = v[[0,1],:]-v[[2,3],:]/2.0
    x_b1 = u[[0,1],:]+u[[2,3],:]/2.0
    x_b2 = v[[0,1],:]+v[[2,3],:]/2.0
    x1 = np.concatenate((x_a1[0,:].reshape(1,n1), x_a1[1,:].reshape(1,n1), x_b1[0,:].reshape(1,n1), x_b1[1,:].reshape(1,n1)), axis=0)
    x2 = np.concatenate((x_a2[0,:].reshape(1,n2), x_a2[1,:].reshape(1,n2), x_b2[0,:].reshape(1,n2), x_b2[1,:].reshape(1,n2)), axis=0)

    X1 = ml.repmat(x1,1,n2)
    X2 = np.kron(x2, np.ones((1,n1)))

    X = np.concatenate((np.maximum(X1[[0,1],:],X2[[0,1],:]),np.minimum(X1[[2,3],:],X2[[2,3],:])),axis=0)
    WH1 = np.concatenate(((X1[2,:]-X1[0,:]).reshape(1,n1*n2),(X1[3,:]-X1[1,:]).reshape(1,n1*n2)),axis=0)
    WH2 = np.concatenate(((X2[2,:]-X2[0,:]).reshape(1,n1*n2),(X2[3,:]-X2[1,:]).reshape(1,n1*n2)),axis=0)
    WH_int = np.concatenate((np.maximum(0,X[2,:]-X[0,:]).reshape(1,n1*n2),np.maximum(0,X[3,:]-X[1,:]).reshape(1,n1*n2)),axis=0)
    A1 = np.prod(WH1,axis=0).reshape(n2,n1)
    A2 = np.prod(WH2,axis=0).reshape(n2,n1)
    A_int = np.prod(WH_int,axis=0).reshape(n2,n1)
    den = A1+A2-A_int
    dist = np.zeros(A_int.shape)
    ind = np.logical_and(np.isfinite(den), den > 0)
    dist[ind] = 1.0-(A_int[ind]/(den[ind]))

    return dist

def col_m_dist(u, v, S_u):
    S_inv = inv(S_u)
    x_diff = u-v.reshape(-1,1)
    m_dist = np.sum(x_diff*np.dot(S_inv, x_diff), axis=0)
    return m_dist

def split_m_dist(u, v, S_uv):
    ind1 = [0,1]
    ind2 = [2,3]
    if len(u.shape) <= 1:
        u = u[:,np.newaxis]
    if len(v.shape) <= 1:
        v = v[:,np.newaxis]
    N_v = v.shape[1]
    det_S_uv = np.array([det(S_uv[:,:,i]) for i in range(N_v)])
    S_uv1 = S_uv[np.ix_(ind1,ind1,list(range(N_v)))]
    S_uv2 = S_uv[np.ix_(ind2,ind2,list(range(N_v)))]

    # L1_i_list = list(map(lambda P: inv(cholesky(P)), \
    #     [S_uv1[:,:,i] for i in range(N_v)]))
    # m_pos = list(map(lambda L_i, x: np.sum((L_i.dot(x-u[ind1,:]))**2, axis=0), L1_i_list, \
    #     [v[ind1,i].reshape(-1,1) for i in range(N_v)]))
    # L2_i_list = list(map(lambda P: inv(cholesky(P)), \
    #     [S_uv2[:,:,i] for i in range(N_v)]))
    # m_siz = list(map(lambda L_i, x: np.sum((L_i.dot(x-u[ind2,:]))**2, axis=0), L2_i_list, \
    #     [v[ind2,i].reshape(-1,1) for i in range(N_v)]))
    # if len(m_pos) > 1:
    #     m_dist_pos = np.stack(m_pos, axis=0).T
    #     m_dist_siz = np.stack(m_siz, axis=0).T
    # else:
    #     m_dist_pos = m_pos[0].reshape(-1,1)
    #     m_dist_siz = m_siz[0].reshape(-1,1)

    # if N_v > 1:
    #     d_dist1 = np.concatenate([col_m_dist(u[ind1,:], v[ind1,i], S_uv1[:,:,i])[:,np.newaxis] for i in range(N_v)], axis=1)
    #     d_dist2 = np.concatenate([col_m_dist(u[ind2,:], v[ind2,i], S_uv2[:,:,i])[:,np.newaxis] for i in range(N_v)], axis=1)
    try:
        L_uv1 = np.concatenate([cholesky(S_uv1[:,:,i])[:,:,np.newaxis] for i in range(N_v)], axis=2)
    except:
        L_uv1 = np.concatenate([np.sqrt(np.diag(np.diag(S_uv1[:,:,i])))[:,:,np.newaxis] for i in range(N_v)], axis=2)
    L_uv2 = np.sqrt(S_uv2)
    m_dist_pos = np.concatenate([np.sum(solve(L_uv1[:,:,i], u[ind1,:]-v[np.ix_(ind1,[i])])**2, axis=0)[:,np.newaxis] for i in range(N_v)], axis=1)
    m_dist_siz = np.concatenate([np.sum(solve(L_uv2[:,:,i], u[ind2,:]-v[np.ix_(ind2,[i])])**2, axis=0)[:,np.newaxis] for i in range(N_v)], axis=1)

    return m_dist_pos, m_dist_siz, det_S_uv

def out_of_bounds(m, v, img_width, img_height, margin=0.05):
    l_margin = margin
    u_margin = 1.0-l_margin
    ind_l = np.logical_and(m[0,:] < l_margin*img_width, v[0,:] < 0)
    ind_r = np.logical_and(m[0,:] > u_margin*img_width, v[0,:] > 0)
    ind_u = np.logical_and(m[1,:] < l_margin*img_height, v[1,:] < 0)
    ind_b = np.logical_and(m[1,:] > u_margin*img_height, v[1,:] > 0)
    ind_h = np.logical_or(ind_l, ind_r)
    ind_v = np.logical_or(ind_u, ind_b)
    ind = np.logical_or(ind_h, ind_v)
    return ind

def pred_cone(m_ini, P_ini, m_cur, P_cur, t, indexes=([0, 1], [2, 3])):
    pos_ind, vel_ind = indexes
    ind = list(range(m_cur.shape[1]))
    # Get mean direction
    v_mean = m_cur[vel_ind, :]
    P1v_l = list(np.swapaxes(np.swapaxes(P_cur[np.ix_(vel_ind,vel_ind,ind)], 0, 1), 0, 2))
    sqrt_P1v_l = list(map(lambda _P: cholesky(_P), P1v_l))
    sqrt_v = np.concatenate([np.diag(sqrt_P_i)[:,np.newaxis] for sqrt_P_i in sqrt_P1v_l], axis=1)
    U = v_mean * t + np.sign(v_mean)*sqrt_v
    Theta = np.arctan2(U[1, :], U[0, :])
    c_Theta = np.cos(Theta)
    s_Theta = np.sin(Theta)
    # P0_l = list(np.swapaxes(np.swapaxes(P_ini[np.ix_(pos_ind,pos_ind,ind)], 0, 1), 0, 2))
    # P1_l = list(np.swapaxes(np.swapaxes(P_cur[np.ix_(pos_ind,pos_ind,ind)], 0, 1), 0, 2))
    # sqrt_P0_l = list(map(lambda _P: cholesky(_P), P0_l))
    # sqrt_P1_l = list(map(lambda _P: cholesky(_P), P1_l))
    # sqrt_P0 = np.concatenate([sqrt_P_i[:,:,np.newaxis] for sqrt_P_i in sqrt_P0_l], axis=2)
    # sqrt_P1 = np.concatenate([sqrt_P_i[:,:,np.newaxis] for sqrt_P_i in sqrt_P1_l], axis=2)
    P0 = P_ini[np.ix_(pos_ind,pos_ind,ind)]
    P1 = P_cur[np.ix_(pos_ind,pos_ind,ind)]
    R_l = [np.array([[[c_Theta[i]], [+s_Theta[i]]],[[-s_Theta[i]], [+c_Theta[i]]]]) for i in range(len(Theta))]
    R = np.concatenate(R_l, axis=2)
    # sqrt_P0_u = np.einsum('ij...,jk...->ik...', R, sqrt_P0)
    # sqrt_P1_u = np.einsum('ij...,jk...->ik...', R, sqrt_P1)
    P0_u = np.einsum('ij...,jk...->ik...', R, np.swapaxes(np.einsum('ij...,jk...->ik...', R, P0), 0, 1))
    P1_u = np.einsum('ij...,jk...->ik...', R, np.swapaxes(np.einsum('ij...,jk...->ik...', R, P1), 0, 1))
    P0_u_l = list(np.swapaxes(np.swapaxes(P0_u, 0, 1), 0, 2))
    P1_u_l = list(np.swapaxes(np.swapaxes(P1_u, 0, 1), 0, 2))
    sqrt_P0_u_l = list(map(lambda _P: cholesky(_P), P0_u_l))
    sqrt_P1_u_l = list(map(lambda _P: cholesky(_P), P1_u_l))
    sqrt_P0_u = np.concatenate([sqrt_P_i[:,:,np.newaxis] for sqrt_P_i in sqrt_P0_u_l], axis=2)
    sqrt_P1_u = np.concatenate([sqrt_P_i[:,:,np.newaxis] for sqrt_P_i in sqrt_P1_u_l], axis=2)
    # Get width
    W0 = sqrt_P0_u[1, 1, :]
    W1 = sqrt_P1_u[1, 1, :]
    delta_U = np.zeros(U.shape)
    delta_W = W1 - W0
    ind_w = delta_W <= 0
    W0[ind_w] = 0.5 * W1[ind_w]
    delta_W[ind_w] = 0.5 * W1[ind_w]
    delta_U = U * (W0 / delta_W)
    U += delta_U
    O = m_ini[pos_ind, :] -delta_U
    L_inf = np.linalg.norm(delta_U, axis=0)
    L_sup = np.linalg.norm(U, axis=0)
    Psi = np.pi*np.ones((len(ind),))
    ind_l = L_sup > 0
    Psi[ind_l] = 0.5 * np.arctan(W1[ind_l] / L_sup[ind_l])
    cone = np.vstack((O, L_inf, L_sup, Theta, Psi))
    return cone

def is_in_cone(boxes, cones):
    n_b = boxes.shape[1]
    n_c = cones.shape[1]
    corners1 = boxes[[0, 1], :] - 0.5*boxes[[2, 3], :]
    corners2 = boxes[[0, 1], :] + 0.5*boxes[[2, 3], :]
    in_rng_flag1 = np.zeros((n_b, n_c)).astype(bool)
    in_ang_flag1 = np.zeros((n_b, n_c)).astype(bool)
    in_rng_flag2 = np.zeros((n_b, n_c)).astype(bool)
    in_ang_flag2 = np.zeros((n_b, n_c)).astype(bool)
    for i in range(n_b):
        r1_i = corners1[:,i].reshape(-1,1)-cones[[0,1],:]
        r2_i = corners2[:,i].reshape(-1,1)-cones[[0,1],:]
        rng1 = np.linalg.norm(r1_i, axis=0)
        rng2 = np.linalg.norm(r2_i, axis=0)
        ang1 = np.arctan2(r1_i[1, :], r1_i[0, :])
        ang2 = np.arctan2(r2_i[1, :], r2_i[0, :])
        in_rng_flag1[i, :] = np.logical_or(np.logical_and(rng1 >= cones[2, :], rng1 <= cones[3, :]), cones[3,:] == 0)
        in_ang_flag1[i, :] = np.abs(ang1 - cones[4, :]) < cones[5, :]
        in_rng_flag2[i, :] = np.logical_or(np.logical_and(rng2 >= cones[2, :], rng2 <= cones[3, :]), cones[3,:] == 0)
        in_ang_flag2[i, :] = np.abs(ang2 - cones[4, :]) < cones[5, :]
    in_cone_flag1 = np.logical_and(in_rng_flag1, in_ang_flag1)
    in_cone_flag2 = np.logical_and(in_rng_flag2, in_ang_flag2)
    in_cone_flag = np.logical_or(in_cone_flag1, in_cone_flag2)
    return in_cone_flag

## MultiObject filter class.
class MultiObjectFilter:
    def __init__(self, model, debug=False):
        self.n_x = model['x_dim']
        self.n_z = model['z_dim']
        self.img_width = model['img_width']
        self.img_height = model['img_height']
        self.obs_indexes = model['obs_indexes']
        self.vel_indexes = sorted(list(set(range(self.n_x))-set(self.obs_indexes)))
        self.p_s = model['p_s']
        self.F = model['F']
        self.Q = model['Q']
        self.p_d = model['p_d']
        self.H = model['H']
        self.H_t = model['H_t']
        self.R = model['R']
        self.motionModel = model['motion_model']
        self.sigma_v = model['sigma_v_birth']
        self.mu_birth = model['mu_birth']
        self.gamma_thr = model['gamma_thr']
        self.REID_THRESHOLD = model['reid_thr']
        self.MIN_IOU = model['min_IOU'] # set to a value >= 0 to configure gating specifically for the dynamic birth 
        # (if dynamic birth is missing too many objects)
        self.v_x = 0
        self.v_y = 0
        self.v_z = 0
        self.munkres = Munkres()
        self.label_max = -1
        self.N_miss = model['N_miss']
        self.debug = debug

    ## Predict.
    def predict(self, *args):
        intensity_km1, marks_km1 = args
        # Get parameters
        p_s = self.p_s
        F = self.F
        Q = self.Q
        # Get intensity components
        w_km1, l_km1, m_km1, P_km1, t_km1 = \
            intensity_km1['w'], intensity_km1['l'], intensity_km1['m'], \
            intensity_km1['P'], intensity_km1['t']
        
        # Predict intensity components
        intensity_k_km1 = {'w': [], 'l': [], 'm': [], 'P': [], 't': []}
        if len(l_km1) > 0:
            n_x, L_km1 = m_km1.shape
            w = p_s * w_km1
            l = l_km1
            t = t_km1
            m = np.zeros((n_x,L_km1))
            P = np.zeros((n_x,n_x,L_km1))
            for i in range(L_km1):
                m[:,i] = F.dot(m_km1[:,i])
                P[:,:,i] = F.dot(np.dot(P_km1[:,:,i],F.T)) +Q
            # Save intensity components (weights, labels, mean vectors, covariance matrices)
            intensity_k_km1 = {'w': w, 'l': l, 'm': m, 'P': P, 't': t}
        
        # Predict features for maintained components
        marks_k_km1 = cp.copy(marks_km1)
        return intensity_k_km1, marks_k_km1
    
    ## Gating.
    def gate(self, *args):
        detections_k, intensity_km1, marks_km1 = args
        # Get parameters
        H = self.H
        R = self.R
        
        indexes = self.obs_indexes

        # Get intensity components
        w, l, m, P = \
        intensity_km1['w'], intensity_km1['l'], intensity_km1['m'], intensity_km1['P']
        l_f, t_miss = marks_km1['l'], marks_km1['t_miss']
        z_boxes = detections_k['z']

        # Number of detections and components
        n_x = self.n_x
        n_z = self.n_z
        if len(l) > 0:
            L_b = m.shape[1]
            L_f = L_b
        else:
            L_b = 0
            L_f = 0
        if len(z_boxes) > 0:
            L_m = z_boxes.shape[1]
        else:
            L_m = 0

        # Allocate variables
        S = []
        sqrt_S = []
        inv_S = []
        nu = []
        dist_iou = []

        valid_meas = []
        valid_b_mask = []
        
        # Gate per boxes
        valid_inds = []
        valid_inds_nbirth = [] # keep a separate set of indexes to drive how birth components will appear
        valid_meas_matrix = np.zeros((L_b,L_m))
        for i in range(L_b):
            S_i = R + H.dot(np.dot(P[:,:,i],H.T))
            sqrt_S_i = cholesky(S_i)
            inv_sqrt_S_i = solve_triangular(sqrt_S_i, np.eye(n_z), lower=True)
            inv_S_i = inv_sqrt_S_i.T.dot(inv_sqrt_S_i)
            nu_i = z_boxes-H.dot(m[:,i].reshape(n_x,1))
            m_dist_i = np.sum((inv_sqrt_S_i[0:2, 0:2].dot(nu_i[0:2,:]))**2, axis=0)
            
            valid_mask_i = m_dist_i < self.gamma_thr
                
            valid_meas_matrix[i,:] = valid_mask_i.astype('float')
            valid_meas_i = [j for (j, val) in enumerate(valid_mask_i) if val]
            valid_inds = valid_inds + valid_meas_i # concatenate
            nu_i[:,np.logical_not(valid_mask_i)] = np.nan
            # Save
            S.append(S_i)
            sqrt_S.append(sqrt_S_i)
            inv_S.append(inv_S_i)
            nu.append(nu_i)
            # Set valid indexes for detections that cannot be considered births
            dist_l = 1.0-iou_dist(m[indexes,i].reshape(n_z,1), z_boxes).ravel()
            valid_nbirth_mask_i = np.logical_and(valid_mask_i, dist_l > self.MIN_IOU)
            # valid_nbirth_mask_i = valid_mask_i
            valid_meas_nbirth_i = [j for (j, val) in enumerate(valid_nbirth_mask_i) if val]
            valid_inds_nbirth = valid_inds_nbirth + valid_meas_nbirth_i # concatenate
            dist_iou.append(np.maximum(dist_l,val_min))
        
        # Check for instants of missed detections (zero if any detection appeared)
        for ell in range(L_f):
            if np.sum(valid_meas_matrix[l == l_f[ell], :]) > 0:
                t_miss[ell] = 0
            else:
                t_miss[ell] = t_miss[ell]+1

        # Apply the measurement gating
        if len(valid_inds) > 0:
            valid_inds = np.array(list(set(valid_inds))) # unique indexes of gated measurements
            z = z_boxes[:,valid_inds]
        else:
            z = []

        nu_out = []
        dist_iou_out = []
        for i in range(L_b):
            nu_i = nu[i]
            dist_iou_i = dist_iou[i]
            nu_out.append(nu_i[:,valid_inds])
            dist_iou_out.append(dist_iou_i[valid_inds])
        nu = nu_out
        dist_iou = dist_iou_out
        
        all_inds = [i for i in range(L_m)]
        if len(valid_inds) > 0:
            n_valid_inds = np.array(sorted(list(set(all_inds)-set(valid_inds)))).astype('int')
        else:
            if L_m > 0:
                n_valid_inds = np.array(all_inds).astype('int')
            else:
                n_valid_inds = None
        # Identify non-gated indexes according to the second criteria
        if len(valid_inds_nbirth) > 0:
            n_valid_inds_nbirth = np.array(sorted(list(set(all_inds)-set(valid_inds_nbirth)))).astype('int')
        else:
            if L_m > 0:
                n_valid_inds_nbirth = np.array(all_inds).astype('int')
            else:
                n_valid_inds_nbirth = None
            
        z_ng = z_boxes[:,n_valid_inds_nbirth]

        # Set outputs
        gated_measurements_k = \
        {'z': z, 'S': S, 'sqrt_S': sqrt_S, 'inv_S': inv_S, 'nu': nu, 'dist_iou': dist_iou}
        non_gated_measurements_k = {'z': z_ng}
        
        return gated_measurements_k, non_gated_measurements_k

    ## Update.
    def update(self, *args):
        intensity_k_km1, measurements_k = args
        # Get parameters
        H = self.H
        
        # Get parameters of measurements (detected boxes and features)
        z, S, sqrt_S, inv_S, nu, dist_iou = \
            measurements_k['z'], measurements_k['S'], measurements_k['sqrt_S'], \
            measurements_k['inv_S'], measurements_k['nu'], measurements_k['dist_iou']
        # Get components
        w_k_km1, l_k_km1, m_k_km1, P_k_km1 = \
            intensity_k_km1['w'], intensity_k_km1['l'], \
            intensity_k_km1['m'], intensity_k_km1['P']

        # Length of components (boxes, features, measurements)
        n_x = self.n_x
        n_z = self.n_z
        if len(l_k_km1) > 0:
            L_b = m_k_km1.shape[1]
            #L_f = m_f_k_km1.shape[1]
            L_f = L_b
        else:
            L_b = 0
            L_f = 0
        if len(z) > 0:
            L_m = z.shape[1]
        else:
            L_m = 0
        # Allocate variables for hypotheses
        # Boxes
        hyp_log_q_b = np.zeros((L_b,L_m)).astype('float64') # log expected likelihood
        hyp_m_b = np.zeros((n_x,L_b,L_m)) # updated mean vectors
        hyp_P_b = np.zeros((n_x,n_x,L_b,L_m)) # update covariance matrices
        
        # Update intensity components (boxes) and assign hypotheses
        for i in range(L_b):
            log_q_i, m_i, P_i = intensity_update_log(m_k_km1[:,i], P_k_km1[:,:,i], z, cp.copy(nu[i]), H, S[i], sqrt_S[i], inv_S[i])
            # Allocate hypotheses
            hyp_log_q_b[i,:] = log_q_i
            hyp_m_b[:,i,:] = m_i
            hyp_P_b[:,:,i,:] = P_i
        
        # Compose output
        hypotheses_k = {'log_q_b': hyp_log_q_b, 'm_b': hyp_m_b, 'P_b': hyp_P_b}
            
        return hypotheses_k

    ## Concatenate birth components with those predicted.
    def add_birth(self, *args):
        intensity_in, marks_in, birth_intensity, birth_marks = args
        # Get intensity components
        w, l, m, P, t = \
            intensity_in['w'], intensity_in['l'], intensity_in['m'], \
            intensity_in['P'], intensity_in['t']
        # Get components of birth
        w_birth, l_birth, m_birth, P_birth, t_birth = \
            birth_intensity['w'], birth_intensity['l'], birth_intensity['m'], \
            birth_intensity['P'], birth_intensity['t']
        l_f, t_f, t_miss = marks_in['l'], marks_in['t'], marks_in['t_miss']
        l_f_birth, t_f_birth = birth_marks['l'], birth_marks['t']
        
        # Append birth components to the intensity components
        if len(l) > 0:
            l_f = np.concatenate((l_f_birth, l_f),axis=0)
            t_f = np.concatenate((t_f_birth, t_f),axis=0)
            t_miss = np.concatenate((t_f_birth, t_miss),axis=0)
            w = np.concatenate((w_birth, w),axis=0)
            l = np.concatenate((l_birth, l),axis=0)
            m = np.concatenate((m_birth, m),axis=1)
            P = np.concatenate((P_birth, P),axis=2)
            t = np.concatenate((t_birth, t),axis=0)
        else:
            l_f = l_f_birth
            t_f = t_f_birth
            t_miss = t_f_birth
            w = w_birth
            l = l_birth
            m = m_birth
            P = P_birth
            t = t_birth

        # Set output variables
        intensity_out = {'w': w, 'l': l, 'm': m, 'P': P, 't': t}
        marks_out = {'l': l_f, 't': t_f, 't_miss': t_miss}
        
        return intensity_out, marks_out

    ## Set birth model for the next prediction step.
    def set_birth_model(self, *args):
        ng_measurements_k, intensity_k, marks_k, tracks, ended_tracks = args
        F = self.F
        H = self.H
        Q = self.Q
        R = self.R
        n_z = self.n_z
        H_t = self.H_t
        v_x = self.v_x
        v_y = self.v_y
        v_z = self.v_z
        sigma_v = self.sigma_v
        mu_birth = self.mu_birth
            
        # Get intensity components
        w, l, m, P, t = \
            intensity_k['w'], intensity_k['l'], intensity_k['m'], \
            intensity_k['P'], intensity_k['t']
        # Detected boxes
        z_boxes =  ng_measurements_k['z']
        l_f, t_f, t_miss = marks_k['l'], marks_k['t'], marks_k['t_miss']

        m_trk, P_trk, m_trk_b, v_trk, l_trk, t_trk, N_trk, N_miss_trk, inact_trk, m_trk_on, P_trk_on = \
            tracks['states'], tracks['covariances'], tracks['boxes'], tracks['velocities'], \
            tracks['labels'], tracks['times'], tracks['N'], tracks['misses'], tracks['inactive'], \
            tracks['on_track_states'], tracks['on_track_covariances']

        m_end, P_end, v_end, l_end, ot_end, t_end, N_end, inact_end, m_end_on, P_end_on = \
            ended_tracks['states'], ended_tracks['covariances'], ended_tracks['velocities'], \
            ended_tracks['labels'], ended_tracks['offtimes'], ended_tracks['times'], ended_tracks['N'], \
            ended_tracks['inactive'], \
            ended_tracks['on_track_states'], ended_tracks['on_track_covariances']

        # Dimensions
        n_x = self.n_x
        if len(l) > 0:
            L_boxes = m.shape[1]
        else:
            L_boxes = 0
        
        if len(z_boxes) > 0:
            L_birth = z_boxes.shape[1]
        else:
            L_birth = 0

        if self.motionModel.lower() == 'cv':
            indexes = [2,3]
            m_v = np.zeros((6,))
            V = np.zeros((6,6))
            m_v[indexes] = np.array([v_x, v_y])
            V[np.ix_(indexes,indexes)] = sigma_v**2
        elif self.motionModel.lower() == 'rw':
            m_v = np.zeros((4,))
            V = np.zeros((4,4))

        # Get newborn components
        w_birth = mu_birth * np.ones((L_birth,)).astype('float64')
        t_birth = np.zeros((L_birth,))
        m_birth = np.zeros((n_x,L_birth))
        P_birth = np.zeros((n_x,n_x,L_birth))

        P_birth_0 = H_t.dot(R.dot(H_t.T)) + V
        P_birth_total = F.dot(P_birth_0.dot(F.T)) +Q
        # P_birth_total = P_birth_0

        for i in range(L_birth):
            m_birth[:,i] = H_t.dot(z_boxes[:,i]) +m_v
            P_birth[:,:,i] = P_birth_total

        # t_f_birth = np.zeros((L_birth,))
        t_f_birth = t_birth

        # Get non-active labels
        l_birth = -np.ones((L_birth,)).astype('int')
        
        # Append tracks with missed detections to the list of non-active objects
        if len(l_trk) > 0:
            ind_miss = [i for i in range(len(l_trk)) if N_miss_trk[i] > 0]
            l_trk_miss = l_trk[ind_miss]
            t_trk_miss = t_trk[ind_miss]
            m_trk_miss = m_trk[:, ind_miss]
            P_trk_miss = P_trk[:,:,ind_miss]
            n_trk_miss = N_miss_trk[ind_miss]
            inact_trk_miss = inact_trk[ind_miss]
            m_trk_on_miss = m_trk_on[:, ind_miss]
            P_trk_on_miss = P_trk_on[:,:,ind_miss]
        else:
            l_trk_miss = []
        if len(l_trk_miss) > 0 and len(l_end) > 0:
            l_off = np.concatenate((l_end, l_trk_miss))
            t_off = np.concatenate((t_end, t_trk_miss))
            m_off = np.concatenate((m_end, m_trk_miss), axis=1)
            P_off = np.concatenate((P_end, P_trk_miss), axis=2)
            n_off = np.concatenate((self.N_miss+ot_end, n_trk_miss))
            inact_off = np.concatenate((inact_end, inact_trk_miss))
            m_off_ini = np.concatenate((m_end_on, m_trk_on_miss), axis=1)
            P_off_ini = np.concatenate((P_end_on, P_trk_on_miss), axis=2)
            N_off = l_off.size
        elif len(l_end) > 0:
            l_off = l_end
            t_off = t_end
            m_off = m_end
            P_off = P_end
            n_off = ot_end
            inact_off = inact_end
            m_off_ini = m_end_on
            P_off_ini = P_end_on
            N_off = l_off.size
        elif len(l_trk_miss) > 0:
            l_off = l_trk_miss
            t_off = t_trk_miss
            m_off = m_trk_miss
            P_off = P_trk_miss
            n_off = n_trk_miss
            inact_off = inact_trk_miss
            m_off_ini = m_trk_on_miss
            P_off_ini = P_trk_on_miss
            N_off = l_off.size
        else:
            l_off = []
            N_off = 0

        if L_birth > 0 and len(l_off) > 0:
            obs_ind = self.obs_indexes
            pos_ind = [obs_ind[0], obs_ind[1]]
            # siz_ind = [obs_ind[2], obs_ind[3]]
            vel_ind = self.vel_indexes
            if self.motionModel.lower() == 'cv':
                # Calculate prediction cones
                cones_off = pred_cone(m_off_ini, P_off_ini, m_off, P_off, t_off, indexes=(pos_ind, vel_ind))
                # Verify if any boxes are inside the cone
                in_cone_flag = is_in_cone(z_boxes, cones_off)
            else:
                in_cone_flag = np.ones((L_birth, N_off))
            # log_D2 = -euc_dist(m_off[pos_ind,:], z_boxes[[0,1],:], m_off[siz_ind[0],:], m_off[siz_ind[1],:]).T
            log_D2 = -euc_dist(m_off[pos_ind,:], z_boxes[[0,1],:], self.img_width, self.img_height).T
            D2 = np.exp(log_D2)
            N_off = len(l_off)
            H_P = np.einsum('ij,jk...->ik...', H, P_off)
            S_off = np.einsum('ij...,jk->ik...', H_P, H.T) +np.repeat(R[:,:,np.newaxis], N_off, axis=2)
            m_dist_pos, m_dist_siz, det_S_off = split_m_dist(z_boxes, H.dot(m_off), S_off)

            D1_a = ((m_dist_pos < self.gamma_thr).astype(float) * \
                np.exp(-0.5*m_dist_pos-0.5*m_dist_siz))
            
            D1 = D1_a
            # D = 0.5*D1 + 0.5*D2
            D = in_cone_flag.astype(float) * D1 * D2
            # D[D1 <= self.REID_THRESHOLD] = 0
            non_null_columns = np.sum(D, axis=0) > 0
            non_null_rows = np.sum(D, axis=1) > 0
            col_ind = [i for i in range(D.shape[1]) if non_null_columns[i]]
            row_ind = [i for i in range(D.shape[0]) if non_null_rows[i]]
            D = D[np.ix_(row_ind, col_ind)] #.reshape(len(row_ind),len(col_ind))
            del_ind = []
            if D.size > 1:
                C = -D
                C[C == 0] = 1.0/val_min
                indexes = self.munkres.compute(C.tolist())
                for coord in indexes:
                    ind0, ind1 = row_ind[coord[0]], col_ind[coord[1]]
                    del_ind.append(ind1)
                    l_birth[ind0] = l_off[ind1]
                    m_birth[np.ix_(vel_ind, [ind0])] = m_off[np.ix_(vel_ind, [ind1])]
                    P_birth[np.ix_(vel_ind,vel_ind,[ind0])] = P_off[np.ix_(vel_ind,vel_ind,[ind1])]
                    t_birth[ind0] = t_off[ind1]
                    if self.debug:
                        print(colored('Track {:d} reidentified.'.format(l_off[ind1]), 'magenta'))
            elif D.size == 1:
                ind0, ind1 = row_ind[0], col_ind[0]
                del_ind.append(ind1)
                l_birth[ind0] = l_off[ind1]
                m_birth[np.ix_(vel_ind,[ind0])] = m_off[np.ix_(vel_ind, [ind1])]
                P_birth[np.ix_(vel_ind,vel_ind,[ind0])] = P_off[np.ix_(vel_ind,vel_ind,[ind1])]
                t_birth[ind0] = t_off[ind1]
                if self.debug:
                    print(colored('Track {:d} reidentified.'.format(l_off[ind1]), 'magenta'))

            if len(del_ind) > 0:
                # Remove ended tracks that will be reborn from the list of ended tracks
                ind = [i for i in range(N_end) if i not in del_ind]
                if len(ind) > 0:
                    l_end = l_end[ind]
                    m_end = m_end[:,ind]
                    P_end = P_end[:,:,ind]
                    v_end = v_end[:,ind]
                    ot_end = ot_end[ind]
                    t_end = t_end[ind]
                    N_end = len(l_end)
                    inact_end = inact_end[ind]
                    m_end_on = m_end_on[:,ind]
                    P_end_on = P_end_on[:,:,ind]

        missing_labels_inds = l_birth == -1
        numLabels = np.sum(missing_labels_inds.astype('int'))

        label_max = self.label_max
        avail_labels = list(range(label_max+1,label_max+numLabels+1))
        # Update label count
        self.label_max += numLabels
        L_avail = len(avail_labels)

        all_inds = np.array([i for i in range(numLabels)])
        if L_avail >= numLabels:
            if L_avail > 0:
                avail_labels = np.array(avail_labels)
                l_birth[missing_labels_inds] = avail_labels[all_inds]
        
        if L_birth == 0:
            w_birth = []
            t_birth = []
        
        # Set output
        intensity = {'w': w, 'l': l, 'm': m, 'P': P, 't': t}
        birth_intensity = {'w': w_birth, 'l': l_birth, 'm': m_birth, 'P': P_birth, 't': t_birth}
        
        marks = {'l': l_f, 't': t_f, 't_miss': t_miss}
        birth_marks = {'l': l_birth, 't': t_f_birth}
        
        return birth_intensity, birth_marks, intensity, marks
