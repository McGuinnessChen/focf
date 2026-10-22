"""Published EBD reference kernel (Mete Erdogan, MIT license)."""


import torch


import torch.nn as nn


import math


import copy


class LossFullyConnected(nn.Module):
    def __init__(self, args):
        super(LossFullyConnected, self).__init__()
        self.Rehk = torch.zeros((args.out_dim,args.layer_dim), dtype=torch.float64, device='cuda', requires_grad=False)
        self.Rhk = args.R_ini*torch.eye(args.layer_dim , dtype=torch.float64, device='cuda', requires_grad=False)
        self.new_Rhk = torch.zeros((args.layer_dim, args.layer_dim), dtype=torch.float64, device='cuda', requires_grad=False)
        self.new_Rehk = torch.zeros(( args.out_dim,args.layer_dim ), dtype=torch.float64, device='cuda', requires_grad=False)
        nn.init.normal_(self.Rehk, mean=0.0, std=args.Reh_gain)
        self.Rehk0 = copy.deepcopy(self.Rehk)
        self.la_R = args.la_R
        self.la_R2 = args.la_R2
        self.R_eps_weight = args.R_eps_weight
        self.R_eps = self.R_eps_weight*torch.eye(args.layer_dim, dtype=torch.float64, device='cuda', requires_grad=False)
        self.method = args.method

    def forward(self, hk: torch.Tensor, uk:torch.Tensor, er:torch.Tensor,hkm1:torch.Tensor,layer:torch.Tensor):

        # Covariance Update Convex Combination
        la_R = self.la_R
        la_R2 = self.la_R2
        # Transpose of hk: contains hk vectors in its caolumns
        hkT=hk.T
        # Current Layer Output Dimensions: B is batch size,  D is the dimension of the current layer
        B, D = hk.size()
        # all ones vectors with previous layer dimensions
        Bhkm1,Dhkm1 = hkm1.size()
        #oneshkm1=torch.ones(Bhkm1, 1, dtype=torch.float64, device='cuda', requires_grad=False)

        if(self.method == "ebd" or self.method == "dfa2"):
            # Current layer covariance update
            Rhk_update = (hk.T @ hk) / B
            self.new_Rhk = la_R2*(self.Rhk) + (1-la_R2)*(Rhk_update)
            del Rhk_update

        if(self.method == "ebd"):
            # Error vs current layer  cross covariance update
            Rehk_update = (er.T @ hk) / B #/ math.sqrt(D)
            self.new_Rehk = la_R*(self.Rehk) + (1-la_R)*(Rehk_update)
            del Rehk_update

        # Transformed-centralized error
        if (layer>1.0):
            Q=er
        else:
            if(self.method == "dfa1" or self.method == "dfa2"):
                Q=er@self.Rehk0
            else:
                Q=er@self.new_Rehk

        subhk=(torch.sign(hk))
        Fd=(uk>0) # assuming bounded relu
        if (layer>1.0):
            Fd=torch.ones(Fd.shape,dtype=torch.float64, device='cuda', requires_grad=False)
        # Transformed error scaled by the preactivation derivative and nonlinearity derivative
        Z=Fd*Q

        # Gradient for the broadcast-error decorrelation loss
        gradWmse=Z.T@hkm1/B
        gradbmse = 0
        #gradbmse=Z.T@oneshkm1/B

        # preactivation scaled
        #Fdsubhk=Fd*subhk
        # Gradients for the layer entropy
        if(self.method == "ebd" or self.method == "dfa2"):
            hkRhkinvFd=Fd*torch.linalg.solve(self.new_Rhk+self.R_eps, hk+1e-8, left=False)
            gradWcov=-2*hkRhkinvFd.T@hkm1/B/D
        else:
            gradWcov = 0
        gradbcov = 0
        #gradbcov=-2*hkRhkinvFd.T@oneshkm1/B/D

        # Subgradient of layer sparsifying l_1-loss
        if(self.method == "ebd"):
            gradWl1out = subhk.T@hkm1/B/D
        else:
            gradWl1out = 0
        gradbl1out = 0
        #Fdsubhkd=subhk.double()
        #gradbl1out=Fdsubhkd.T@oneshkm1/B/D

        # update the first and second order statistics
        self.Rhk = self.new_Rhk.detach()
        self.Rehk = self.new_Rehk.detach()

        #calculate losses
        NMSEloss=torch.norm(self.Rehk, p='fro')
        Cov_loss = - (torch.logdet(self.new_Rhk + self.R_eps) ) / D
        angle = 180*torch.acos(torch.sum(self.Rehk * self.Rehk0) / (torch.norm(self.Rehk, p='fro') * torch.norm(self.Rehk0, p='fro')))/math.pi
        return NMSEloss,Cov_loss,self.Rehk,self.Rhk,gradWmse,gradbmse, gradWcov,gradbcov,gradWl1out,gradbl1out,angle
