import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from itertools import product
from scipy import linalg
from scipy.stats import entropy
from discrete_phiid import ig_phiid_discrete

### Stationary distribution options ###

def stationary_lstsq(tpm):
    n = tpm.shape[0]
    A = np.vstack([tpm.T - np.eye(n), np.ones(n)])
    b = np.append(np.zeros(n), 1)
    pi, *_ = np.linalg.lstsq(A, b, rcond=None)
    return pi

def stationary_eigen(tpm):
    """Find left eigenvector corresponding to eigenval nearest 1"""
    eigenvalues, eigenvectors = np.linalg.eig(tpm.T)
    idx = np.argmin(np.abs(eigenvalues - 1))
    pi = np.real(eigenvectors[:, idx])
    pi = pi / np.sum(pi)
    return pi 

def stationary_powerproj(tpm, norm=True, verbose=False):
    """S = sum P_i * u (over all i where λ_i = 1)"""
    
    n = tpm.shape[0]
    u = np.ones(n) / n # uniform
    
    # eigendecomposition
    eigenvalues, left_vecs, right_vecs = linalg.eig(tpm.T, left=True, right=True)
    
    indices = np.where(np.abs(eigenvalues - 1) < 1e-10)[0]
    if verbose:
        print(f"Found {len(indices)} eigenvalue(s) equal to 1")
    
    pi = np.zeros(n, dtype=complex)
    
    for idx in indices:
        v_i = right_vecs[:, idx]  # right eigenvector
        w_i = left_vecs[:, idx]   # left eigenvector
        
        dot_wv = np.dot(np.conj(w_i), v_i)
        dot_wu = np.dot(np.conj(w_i), u)
        
        if verbose:
            print(f"  <w,v>={dot_wv}, <w,u>={dot_wu}")
        
        # Projection operator = 
        projection = v_i * dot_wu / dot_wv
        pi += projection
    
    pi = np.real(pi)
    
    if norm:
        pi = pi / pi.sum()
    
    return pi

def get_statdist(tpm, option='lstsq'):
    """
    Get stationary distribution from TPM

    Parameters
    ----------
    tpm : np.array
        Transition probability matrix
    option : str
        Method to compute stationary distribution
        Default is 'lstsq'.
    
    Returns
    -------
    stationary : np.array
        Stationary distribution
    """
    if option == 'lstsq':
        stationary = stationary_lstsq(tpm)
    elif option == 'eigen':
        stationary = stationary_eigen(tpm)
    elif option == 'powerproj':
        stationary = stationary_powerproj(tpm)
    else:
        raise ValueError(f"Unknown stationary distribution option: {option}")
    return stationary

def get_tdmi(stationary, tpm):
    """
    Time-delayed mutual information of whole system

    Parameters
    ----------
    stationary : np.array
        Stationary distribution over past states
    tpm : np.array
        Transition probability matrix
    
    Returns
    -------
    tdmi : float
        Time-delayed mutual information
    """

    tpm_norm = get_normdist(stationary, tpm)
    future = np.matmul(stationary, tpm)
    H_stationary = entropy(stationary, base=2)
    H_future = entropy(future, base=2)
    H_joint = entropy(tpm_norm.flatten(), base=2)
    tdmi = H_stationary + H_future - H_joint
    return tdmi

def get_m_tdmi(stationary, tpm, element_idx, 
               joint_states=[(0,0), (0,1), (1,0), (1,1)]):
    """
    Time-delayed mutual information of a single element

    Parameters
    ----------
    stationary : np.array
        Stationary distribution over past states
    tpm : np.array
        Transition probability matrix
    element_idx : int
        Index of the element to compute TDMI for
    joint_states : list of tuples
        List of joint states

    Returns
    -------
    m_tdmi : float
        Time-delayed mutual information of the element
    """

    n_elements = len(joint_states[0])
    n_element_states = int(len(joint_states)**(1/n_elements))
    
    tpm_norm = get_normdist(stationary, tpm)
    
    element_past = np.zeros(n_element_states)
    element_future = np.zeros(n_element_states)
    element_joint = np.zeros((n_element_states, n_element_states))
    
    for past_idx, past_state in enumerate(joint_states):
        for future_idx, future_state in enumerate(joint_states):
            prob = tpm_norm[past_idx, future_idx]
            x_i = past_state[element_idx]
            y_i = future_state[element_idx]
            element_past[x_i] += stationary[past_idx]
            element_future[y_i] += prob
            element_joint[x_i, y_i] += prob

    H_past = entropy(element_past, base=2)
    H_future = entropy(element_future, base=2)
    H_joint = entropy(element_joint.flatten(), base=2)
    
    return H_past + H_future - H_joint

def get_am_tdmi(stationary, tpm, 
                joint_states=[(0,0), (0,1), (1,0), (1,1)]):
    """
    Time-delayed mutual information of all elements
    
    Parameters
    ----------
    stationary : np.array
        Stationary distribution over past states
    tpm : np.array
        Transition probability matrix
    joint_states : list of tuples
        List of joint states
    
    Returns
    -------
    element_tdmis : dict
        Dictionary of time-delayed mutual information for each element
    """
    n_elements = len(joint_states[0])
    element_tdmis = {}
    for i in range(n_elements):
        tdmi = get_m_tdmi(stationary, tpm, i, joint_states)
        element_tdmis[f'element_{i}'] = tdmi
    return element_tdmis

def get_wms(stationary, tpm):
    """
    Whole-minus-sum integrated information

    Parameters
    ----------
    stationary : np.array
        Stationary distribution over past states
    tpm : np.array
        Transition probability matrix

    Returns
    -------
    wms : float
        Whole-minus-sum integrated information
    """

    tdmi = get_tdmi(stationary, tpm)
    element_tdmis = get_am_tdmi(stationary, tpm)
    wms = tdmi - sum(element_tdmis.values())
    return wms

def get_normdist(stationary, tpm):
    """
    Get normalized joint distribution

    Parameters
    ----------
    stationary : np.array
        Stationary distribution over past states
    tpm : np.array
        Transition probability matrix
    
    Returns
    -------
    tpm_norm : np.array
        Normalized joint distribution
    """

    tpm_norm = np.vstack([stationary[i]*tpm[i,:] for i in range(tpm.shape[0])])
    tpm_norm = tpm_norm / np.sum(tpm_norm)
    return tpm_norm

def get_dist(tpm, joint_states=[(0,0), (0,1), (1,0), (1,1)],
             stat_dist='stationary'):
    """
    Get formatted joint distribution for PhiID calculation
    
    Parameters
    ----------
    tpm : np.array
        Transition probability matrix
    joint_states : list of tuples
        List of joint states
    stat_dist : str
        Type of stationary distribution to use ('uniform' or 'stationary')

    Returns
    -------
    dist : np.array
        Formatted joint distribution
    """

    n_elements = len(joint_states[0])
    n_element_states = int(len(joint_states)**(1/n_elements))

    if stat_dist == 'uniform':
        stationary = np.ones(len(joint_states)) / len(joint_states)
    else:
        stationary = get_statdist(tpm)

    tpm_norm = get_normdist(stationary, tpm)

    dist = np.zeros((n_element_states,)*n_elements + (n_element_states,)*n_elements)
    for past_idx, past_state in enumerate(joint_states):
        for future_idx, future_state in enumerate(joint_states):
            x1, x2 = past_state
            y1, y2 = future_state
            dist[x1, x2, y1, y2] = tpm_norm[past_idx, future_idx]

    return dist

def compute_phiid(tpm, 
                  joint_states=[(0,0), (0,1), (1,0), (1,1)],
                  stat_dist='stationary'):
    """
    Compute PhiID IG from TPM
    
    Parameters
    ----------
    tpm : np.array
        Transition probability matrix
    joint_states : list of tuples
        List of joint states
    stat_dist : str
        Type of stationary distribution to use ('uniform' or 'stationary')

    Returns
    -------
    phiid_results : dict
        Dictionary of PhiID IG results
    """
    # assert np.allclose(tpm.sum(axis=1), 1), "TPM rows must sum to 1."
    dist = get_dist(tpm, joint_states=joint_states, stat_dist=stat_dist)
    phiid_results = ig_phiid_discrete(dist)
    return phiid_results

def phiid_matrix(phiid_results,
                 atom_order_past = ['r', 'x', 'y', 's'],
                 atom_order_future = ['r', 'a', 'b', 's']):
    """
    Get PhiID IG matrix from results

    Parameters
    ----------
    phiid_results : dict
        Dictionary of PhiID IG results

    Returns
    -------
    phiid_mat : np.array
        PhiID IG matrix
    """
    
    matrix = np.zeros((4,4))
    for j, past_atom in enumerate(atom_order_past):
        for k, future_atom in enumerate(atom_order_future):
            key = f'{past_atom}t{future_atom}'
            matrix[j,k] = phiid_results.get(key, 0)
    return matrix

def get_tpms():
    """
    Get TPMs for various computations

    Returns
    -------
    tpms : dict
        Dictionary of TPMs for various computations
    """
    
    disint_tpm = np.fliplr(np.eye(4))

    int_tpm = disint_tpm.copy() + np.eye(4)
    int_tpm = int_tpm / int_tpm.sum(axis=1, keepdims=True)

    het_tpm = np.abs(np.random.normal(0,1,(4,4)))
    het_tpm = het_tpm / het_tpm.sum(axis=1, keepdims=True)

    downXOR_tpm = np.array([[1,1,0,0],
                            [0,0,1,1],
                            [0,0,1,1],
                            [1,1,0,0]])*.5

    upXOR_tpm = np.array([[1,0,0,1],
                          [1,0,0,1],
                          [0,1,1,0],
                          [0,1,1,0]])*.5

    transfer_tpm = np.array([[1,0,0,0], 
                             [0,0,1,0],
                             [0,1,0,0],
                             [0,0,0,1]])
    
    TBC_tpm = np.eye(4)

    rtr_tpm = np.zeros((4,4))
    rtr_tpm[0,0] = 1
    rtr_tpm[3,3] = 1

    rts_tpm = np.array([[1,0,0,1],
                        [0,0,0,0],
                        [0,0,0,0],
                        [0,1,1,0]])*0.5

    str_tpm = np.array([[1,0,0,0],
                        [0,0,0,1],
                        [0,0,0,1],
                        [1,0,0,0]])

    rtx_tpm = np.array([[1,1,0,0],
                        [0,0,0,0],
                        [0,0,0,0],
                        [0,0,1,1]])*0.5

    xtr_tpm = np.array([[1,0,0,0],
                        [1,0,0,0],
                        [0,0,0,1],
                        [0,0,0,1]])
    
    return {'Disintegrated': disint_tpm,
            'Integrated': int_tpm,
            'Heterogeneous': het_tpm,

            'Downward XOR': downXOR_tpm,
            'Upward XOR': upXOR_tpm,
            'Transfer': transfer_tpm,
            'Two-bit Copy': TBC_tpm,
            'Copy': xtr_tpm,
            'Erasure': rtx_tpm,
            'Oneness': rtr_tpm,
            'Encryption': rts_tpm,
            'Decryption': str_tpm,}