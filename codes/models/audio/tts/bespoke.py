import random
from time import time
from functools import partial
from tqdm import tqdm
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchdiffeq import odeint

from trainer.networks import register_model

def interpolate(x_values, x, y):
    '''
    x_values: x values whose corresponding y values are unknown
    x: known values
    y: known values
    
    return interpolated_values
    '''
    # Find the indices of the two closest points
    idx = torch.searchsorted(x, x_values)
    # if idx 
    idx = torch.maximum(torch.minimum(idx,torch.ones_like(idx,dtype=int)*(x.shape[0]-1)),torch.zeros_like(idx,dtype=int))

    # Use linear interpolation to compute values between the two points
    x_left = x[idx - 1]
    x_right = x[idx]
    y_left = y[idx - 1]
    y_right = y[idx]

    # Avoid division by zero
    mask = x_right != x_left
    # print(mask)

    x_left = x_left.reshape(-1,1,1,1)
    x_right = x_right.reshape(-1,1,1,1)
    x_values = x_values.reshape(-1,1,1,1)

    # Calculate interpolation
    interpolated_values = torch.zeros_like(y_left, dtype=torch.float)
    interpolated_values = y_left + (x_values - x_left) * (y_right - y_left) / (x_right - x_left)
    return interpolated_values


class BespokeSolver(nn.Module):

    def __init__(
        self,
        num_steps = 5,
        eps = 1e-8
    ):
        super().__init__()
        self.num_steps = num_steps # ODE solver step
        self.theta_t = nn.Parameter(torch.ones(self.num_steps))  #       q(1) ,...,q(n-1) ,q(n)  parameters for time step reparametrization
        self.theta_td = nn.Parameter(torch.ones(self.num_steps)) # q'(0),q'(1),...,q'(n-1) parameters for the derivative of time step w.r.t parameter r
        self.theta_s = nn.Parameter(torch.zeros(self.num_steps)) #       p(1) ,...,p(n-1) ,p(n)  parameters for path transform
        self.theta_sd = nn.Parameter(torch.zeros(self.num_steps))# p'(0),p'(1),...,p'(n-1)
        if isinstance(eps,str):
            eps = float(eps)
        self.eps = torch.tensor(eps)

    def p_sample_flow(self, 
        model,
        x_0=None,
        cond=None,
        t_span = None,
        device=None,
        progress=False,):
        ''' 
        A alternative approach to solve the ode equation with naive euler solver and 
        generate ode path.
        '''

        if device is None:
            device = next(model.parameters()).device

        x_t = x_0
        batch_sz = x_0.shape[0]
        sol = [] # store the samples
        t, _, dt = t_span[0], t_span[-1], t_span[1] - t_span[0]

        for t in tqdm(t_span[:-1], disable=not progress):
            t_batch = torch.tensor([t] * batch_sz, device=device) 
            model_output = model(x_t, t_batch, **cond)
            x_t = x_t + dt * model_output
            sol.append(x_t)
        
        return sol
    
    def forward(self, u, x_0, cond, device=None, gt_steps=40, method = 'euler'):
        '''
        compute losses

        x0: noise (batch, *channels)
        u: vector field, whose input is (x:(batch,*channels), timesteps(batch,), conds(batch, *channel2s))
        cond: additional condition inputs to the vector field model(u)
        gt_steps: how many timesteps are used when generating the ground truth path of x, 
                  the larger this value is, the ground truth path is more realistic, default to be 40.
        method: ode solver used by torchdiffeq, default to be 'euler'

        return Loss
        '''
        if device is None:
            device = next(u.parameters()).device

        loss_fn = nn.MSELoss(reduction='mean')
        

        # compute timesteps from the model parameters
        theta_t = torch.cat((torch.zeros(1).to(device),self.theta_t),dim=0) # q(0), q(1) ,...,q(n-1) ,q(n)
        t_step = torch.cumsum(torch.abs(theta_t),dim=0)/torch.sum(torch.abs(theta_t)) # 0=t0<t1<...<tn=1
        # compute the dt/dr from the model parameters
        t_step_d = torch.abs(self.theta_td) # t'(0), t'(1), ..., t'(n-1)>0

        # compute scaling transform coefficients
        scale = torch.cat((torch.ones(1).to(device),torch.exp(self.theta_s)),dim=0) # 1=s0,s1,...,sn
        # compute ds/dr, the derivative of scaling tranform coefficients with respect to r
        scale_d = self.theta_sd # s'(0), s'(1), ..., s'(n-1)

        # compute the ground truth path of x: x_truth, evaluated at t_step.detach() and the truth vector field: vf_truth, evaluated at t_step.detach()
        u_part = partial(u.forward, **cond) # only accept x and t as parameters
        u_lambda = lambda t, x: u_part(x,t) # change the location of parameters to fit into the custom of the torchdiffeq api
        with torch.no_grad():
            all_step = torch.linspace(0,1,gt_steps+1,device = device)
            x_all = odeint(u_lambda,x_0,all_step.detach(),rtol=1e-6,atol=1e-8,method=method) # (n+1, batch, *channels)
            x_truth = interpolate(t_step.detach(),all_step,x_all)

            del x_all # delete the x_all to save memory
            vf_truth = []
            for i in range(self.num_steps+1):
                vf_truth.append(u(x_truth[i],t_step[i].detach(),**cond))
            vf_truth = torch.stack(vf_truth,dim=0) # (n+1, batch, *channels) 
        
        # compute x_aux to ensure the gradient dx/dt is correct
        x_aux = x_truth + vf_truth*(t_step-t_step.detach()).reshape((-1,1,1,1))# (n+1, batch, *channels)

        del x_truth, vf_truth # delete the x_all to save memory
        torch.cuda.empty_cache()

        # compute lipschitz constant
        h = 1/self.num_steps
        Lu = torch.abs(scale_d)/scale[:-1] + t_step_d # 0,1,...n-1
        Lt = scale[:-1]/scale[1:]*(1+h*Lu) # 0,1,...n-1
        Lt = torch.cat((Lt,torch.ones(1).to(device)),dim=0) # 0,1,...n-1,n
        Lt_cum_prod = torch.cumprod(Lt,dim=0)
        Mt = Lt_cum_prod/ torch.cat((torch.ones(1).to(device),Lt_cum_prod[:-1]))

        # compute loss
        loss = torch.tensor(0.0).to(device)
        self.eps = self.eps.to(device)
        # randomly choose some time steps to compute loss in order to avoid OOM
        trained_timesteps = random.sample(range(self.num_steps),3)
        for i in range(self.num_steps):
            if i not in trained_timesteps:
                continue
            x_i_1 = (scale[i]+h*scale_d[i])/scale[i+1]*x_aux[i]+h*t_step_d[i]*scale[i]/scale[i+1]*u_part(x_aux[i],t_step[i]) # (batch, *channels)
            loss += Mt[i+1]*torch.sqrt(loss_fn(x_i_1,x_aux[i+1])+self.eps)
            # loss.backward()
        terms = {}
        terms["mse"] = loss
        terms["loss"] = loss
        terms["vb"] = torch.tensor(0.0)
        terms["x_start_predicted"]=x_0
        terms["mse_by_batch"]= torch.tensor(0.0)
        # print(f'loss={loss:.3e}')
        return terms

    def inference(self,
        model,
        x_0=None,
        cond=None,
        device=None,
        progress=False):

        if device is None:
            device = next(model.parameters()).device
        batch_sz = x_0.shape[0]

        
        with torch.no_grad():
            # compute timesteps
            theta_t = torch.cat((torch.zeros(1).to(self.device),self.theta_t),dim=0) # q(0), q(1) ,...,q(n-1) ,q(n)
            t_step = torch.cumsum(torch.abs(theta_t),dim=0)/torch.sum(torch.abs(theta_t)) # 0=t0<t1<...<tn=1
            t_step_d = torch.abs(self.theta_td) # t'(0), t'(1), ..., t'(n-1)>0

            # compute scaling transform coefficients
            scale = torch.cat((torch.ones(1).to(self.device),torch.exp(self.theta_s)),dim=0) # 1=s0,s1,...,sn
            scale_d = self.theta_sd # s'(0), s'(1), ..., s'(n-1)


            x_r = x_0*scale[0]  # transform x into x_r
            
            sol = [] # store the samples
            r_span = torch.linspace(0,1,self.num_steps+1,device=device)
            r, _, dr = r_span[0], r_span[-1], r_span[1] - r_span[0]


            for i in tqdm(range(self.num_steps), disable=not progress):
                t_batch = t_step[i].unsqueeze(0).expand(batch_sz)
                u_r = model(x_r/scale[i], t_batch, **cond) * scale[i] * t_step_d[i] + scale_d[i]/scale[i]*x_r
                x_r = x_r + dr * u_r
            
            # transform back to x_t
            x_t = x_r/scale[-1]
        
        return x_t

@register_model
def register_bespoke_solver(opt_net, opt):
    return BespokeSolver(**opt_net['kwargs'])

if __name__ =='__main__':

    device = torch.device('cuda:1')
    bes = BespokeSolver(num_steps=5,device=device).to(device)
    
    x0 = torch.randn((4,100,931)).to(device)

    vf = DiffusionTts().to(device)

    cond = {'precomputed_aligned_embeddings': torch.randn((4,512,931)).to(device)}

    bes.compute_losses(vf,x0,cond)
    state_dict = bes.state_dict()
    for key, param in state_dict.items():
        state_dict[key] = param.cpu()
    print(state_dict)
    torch.save(state_dict,'/home/chong/.cache/tortoise/models/bespoke.pth')



