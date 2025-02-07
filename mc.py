import copy
import numpy as np

#def action(phi: np.ndarray, k, l):
 #   return np.sum(-2 * k * phi * (np.roll(phi, 1, 0) + np.roll(phi, 1, 1))+ (1 - 2 * l) * phi**2 + l * phi**4,axis=(1,2))

def get_action(phi, beta):
    #print(phi.shape)
    return -beta*np.sum(np.cos(phi[:,:,0] - phi[:,:,1] + np.roll(phi[:,:,1], -1, 0) - np.roll(phi[:,:,0], -1, 1)),axis=(0,1))

def gauge_cooling(lattice):
    lattice = np.remainder(lattice+np.pi, 2*np.pi) - np.pi
    return lattice

def get_drift(phi, beta):
    phi_d = np.zeros(phi.shape)
    drift1 = beta*np.sin(phi[:,:,0] - phi[:,:,1] + np.roll(phi[:,:,1], -1, 0) - np.roll(phi[:,:,0], -1, 1))
    phi_d[:,:,0] = -drift1+np.roll(drift1,1,axis=1)
    phi_d[:,:,1] = drift1-np.roll(drift1,1,axis=0)
    return phi_d

def get_hamiltonian(chi, action):
    return 0.5 * np.sum(chi**2) + action

def hmc(phi_0, S_0, beta, n_steps=20):
    dt = 1 / n_steps

    phi = copy.deepcopy(phi_0)
    L = phi.shape[0]
    chi = np.random.randn(*phi.shape)
    H_0 = get_hamiltonian(chi, S_0)

    chi += 0.5 * dt * get_drift(phi, beta)
    for i in range(n_steps-1):
        phi += dt * chi
        chi += dt * get_drift(phi, beta)
    phi += dt * chi
    chi += 0.5 * dt * get_drift(phi, beta)
    phi=gauge_cooling(phi)
    S = get_action(phi, beta)
    dH = get_hamiltonian(chi, S) - H_0

    if dH > 0:
        if np.random.rand() >= np.exp(-dH):
            return phi_0, S_0, False
    return phi, S, True


def dm_mc(phi_0, S_0,logq_0, k, l, cfgs_df,logq_df):

    index = np.random.choice(cfgs_df.shape[0])
    phi = cfgs_df[index]
    
    last_logp = - S_0
    last_logq = logq_0
    S = get_action(phi, k, l)
    new_logp = - S
    new_logq = logq_df[index]

    dH = (last_logp - last_logq) - (new_logp - new_logq)

    if dH > 0:
        if np.random.rand() >= np.exp(-dH):
            return phi_0, S_0, logq_0, False
    return phi, S, new_logq, True

