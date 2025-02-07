import random
import torch
from tqdm import tqdm
import numpy as np
# @title Define the ODE sampler

from scipy import integrate


def ode_sampler(score_model,
                marginal_prob_std,
                diffusion_coeff,
                batch_size,
                start,
                end,
                num_steps,
                device,
                z,beta,
                L):
    """Generate samples from score-based models with black-box ODE solvers.

  Args:
    score_model: A PyTorch model that represents the time-dependent score-based model.
    marginal_prob_std: A function that returns the standard deviation 
      of the perturbation kernel.
    diffusion_coeff: A function that returns the diffusion coefficient of the SDE.
    batch_size: The number of samplers to generate by calling this function once.
    atol: Tolerance of absolute errors.
    rtol: Tolerance of relative errors.
    device: 'cuda' for running on GPUs, and 'cpu' for running on CPUs.
    z: The latent code that governs the final sample. If None, we start from p_1;
      otherwise, we start from the given z.
    eps: The smallest time step for numerical stability.
  """
    t = torch.ones(batch_size, device=device)*start
    # Create the latent code
    if z is None:
        init_x =torch.randn(batch_size, 2, L, L, device=device) * (marginal_prob_std(t))[:, None, None, None]
    else:
        init_x = z

    shape = init_x.shape

    def score_eval_wrapper(sample, time_steps):
        """A wrapper of the score-based model for use by the ODE solver."""
        sample = torch.tensor(sample, device=device, dtype=torch.float32).reshape(shape)
        time_steps = torch.tensor(time_steps, device=device, dtype=torch.float32).reshape((sample.shape[0],))
        with torch.no_grad():
            score = score_model(sample, time_steps)
        return score.cpu().numpy().reshape((-1,)).astype(np.float64)

    eval_steps = np.linspace(start, end, num_steps)
    step_size = eval_steps[0] - eval_steps[1]

    def ode_func(t, x):
        """The ODE function for use by the ODE solver."""
        time_step = np.ones((shape[0],)) * t
        g = diffusion_coeff(torch.tensor(t)).cpu().numpy()
        return -0.5 * (g ** 2) * score_eval_wrapper(x, time_step)

    # Run the black-box ODE solver.

    res = integrate.solve_ivp(ode_func, (start, end), init_x.reshape(-1).cpu().numpy(), rtol=1e-6, atol=1e-6, method='RK45')
    print(f"Number of function evaluations: {res.nfev}")
    x = torch.tensor(res.y[:, -1], device=device).reshape(shape)

    return x.detach().cpu().numpy()

def calculate_angle(phi):
    angel = torch.pi *(phi[:,0, :, :] - phi[:,1, :, :] + torch.roll(phi[:,1, :, :], shifts=-1, dims=1)-torch.roll(phi[:,0, :, :], shifts=-1, dims=2))
#    angel = right + up - left - down
    return angel



def calculate_topological_charge(array):
    top_charge = torch.sum(torch.remainder(calculate_angle(array),2*torch.pi)-torch.pi,dim=(1,2))

    return (top_charge / (2 * torch.pi))

def MALA_sampler(score_model,
                 marginal_prob_std,
                 diffusion_coeff,
                 batch_size,
                 num_steps,beta,
                 ratio,alpha,mh_steps,t_mh,
                 L, start, size,
                 device='cuda'):

    def action(phi,beta):
        return beta * (-torch.sum(torch.cos(torch.pi *(phi[:,0, :, :] - phi[:,1, :, :] +
                                             torch.roll(phi[:,1, :, :], shifts=-1, dims=1) -
                                               torch.roll(phi[:,0, :, :], shifts=-1, dims=2))),dim=(1, 2)))
    
    def metropolis_hastings_update(x, y, p, random_a):
      accept_mask = random_a < p  
      updated_x = x.clone()  
      updated_x[accept_mask] = y[accept_mask]  
      acceptance_rate = accept_mask.sum().item() / accept_mask.numel()
      return updated_x,acceptance_rate

    
    
    def M_H(x,y,noise,batch_t,h):
        drift = beta*h*(score_model(x,batch_t) + score_model(y,batch_t))
        p_1 = -(1/h)*0.25*torch.sum((torch.sqrt(2*h)*noise + drift)**2, dim=(1, 2, 3))
        p_2 = 0.5*torch.sum(noise ** 2, dim=(1, 2, 3))
        delta_q = torch.exp(p_1+p_2)
        S_x = action(x,beta)
        S_y = action(y,beta)
        
        delta_pi = torch.exp(S_x-S_y)
        accept_prob = delta_pi * delta_q
        #print(accept_prob)
        random_i =  torch.rand(batch_size)
        return metropolis_hastings_update(x,y,accept_prob,random_i)
    

    indices = torch.arange(num_steps, device=device).float()
    time_steps = torch.pow(ratio, indices) * start
    t = torch.ones(batch_size, device=device)
    init_x = torch.randn(batch_size, 2, L, L, device=device)  * marginal_prob_std(t)[:, None, None, None]
    rate = 0
    rate_traj = []
    s_list = []
    Q_list=[]
    x = init_x
    with torch.no_grad():
        for time_step in tqdm(time_steps):
            MH_step = 1e-9
            rate = 0
            step_size = alpha * (time_step / time_steps[-1]) ** 2
            batch_time_step = torch.ones(batch_size, device=device) * time_step
            if time_step<t_mh:
                size = mh_steps
            for turn in range(size):
                score_t = score_model(x, batch_time_step)
                mean_x = x + beta * score_t * step_size
                randn_noise = torch.randn_like(x)
                x_temp = mean_x + torch.sqrt(2*step_size) * randn_noise
                if time_step<t_mh:
                  x, acc_rate = M_H(x,x_temp,randn_noise,batch_time_step,step_size)
                  MH_step += 1
                  rate += acc_rate
                  #s_list.append(-action(x,1)[0].detach().cpu().numpy()/L**2)
                  #Q_list.append(calculate_topological_charge(x).detach().cpu().numpy())
                else:
                  x = x_temp
            
            #rate_traj.append(rate/MH_step)
            x_cpu = mean_x.detach().cpu().numpy()
        # Do not include any noise in the last sampling step.
    return x_cpu, rate / MH_step , rate_traj,s_list,Q_list

