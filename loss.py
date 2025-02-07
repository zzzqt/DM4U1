#@title Define the loss function (double click to expand or collapse)
import torch
def loss_fn(model, x, marginal_prob_std, eps=1e-5):
  """The loss function for training score-based generative models.

  Args:
    model: A PyTorch model instance that represents a 
      time-dependent score-based model.
    x: A mini-batch of training data.    
    marginal_prob_std: A function that gives the standard deviation of 
      the perturbation kernel.
    eps: A tolerance value for numerical stability.
  """
  random_t = torch.rand(x.shape[0], device=x.device)  # introducing stochasticity for each batch
  random_t = 0.9**(random_t*110)
  #random_t = torch.rand(x.shape[0], device=x.device) * (1. - eps) + eps
  z = torch.randn_like(x)
  std = marginal_prob_std(random_t)
  perturbed_x = x + z * std[:,None,None,None]
  score = model(perturbed_x, random_t)
  loss = torch.mean(torch.sum((score * std[:,None,None,None] + z)**2, dim=(1,2,3)))

  return loss