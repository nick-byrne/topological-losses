from multiprocessing import Pool
import cripser as crip
import tcripser as trip
import numpy as np
import torch
import torch.nn.functional as F
import cripser
import copy
from torch.optim import SGD, Adam

def crip_wrapper(X, D):
    return crip.computePH(X, maxdim=D)

def trip_wrapper(X, D):
    return trip.computePH(X, maxdim=D)

def get_roi(X, thresh=0.01):
    true_points = torch.nonzero(X >= thresh)
    corner1 = true_points.min(dim=0)[0]
    corner2 = true_points.max(dim=0)[0]
    roi = [slice(None, None)] + [slice(c1, c2 + 1) for c1, c2 in zip(corner1, corner2)]
    return roi

def get_differentiable_barcode(tensor, barcode):
    '''Makes the barcode returned by CubicalRipser differentiable using PyTorch.
    Note that the critical points of the CubicalRipser filtration reveal changes in sub-level set topology.
    
    Arguments:
        REQUIRED
        tensor  - PyTorch tensor w.r.t. which the barcode must be differentiable
        barcode - Barcode returned by using CubicalRipser to compute the PH of tensor.numpy() 
    '''
    # Identify connected component of ininite persistence (the essential feature)
    inf = barcode[barcode[:, 2] == np.finfo(barcode.dtype).max]
    fin = barcode[barcode[:, 2] < np.finfo(barcode.dtype).max]
    
    # Get birth of infinite feature
    inf_birth = tensor[tuple(inf[:, 3:3+tensor.ndim].astype(np.int64).T)]
    
    # Calculate lifetimes of finite features
    births = tensor[tuple(fin[:, 3:3+tensor.ndim].astype(np.int64).T)]
    deaths = tensor[tuple(fin[:, 6:6+tensor.ndim].astype(np.int64).T)]
    delta_p = (deaths - births)
    
    # Split finite features by dimension
    delta_p = [delta_p[fin[:, 0] == d] for d in range(tensor.ndim)]
    
    # Sort finite features by persistence
    delta_p = [torch.sort(d, descending=True)[0] for d in delta_p]
    
    return inf_birth, delta_p

def multi_class_topological_post_processing(
    inputs, model, prior,
    lr, mse_lambda,
    opt=torch.optim.Adam, num_its=100, construction='0', thresh=None, parallel=True):
    '''Performs topological post-processing.
    
    Arguments:
        REQUIRED
        inputs       - PyTorch tensor - [1, number of classes] + [spatial dimensions (2D or 3D)]
        model        - Pre-trained CNN as PyTorch module (without final activation)
        prior        - Topological prior as dictionary:
                       keys are tuples specifying the channel(s) of inputs
                       values are tuples specifying the desired Betti numbers
        lr           - Learning rate for SGD optimiser
        mse_lambda   - Weighting for similarity constraint
        
        OPTIONAL [default]
        opt          - PyTorch optimiser [torch.optim.Adam]
        num_its      - Iterable of number iterations(s) to run for each scale [100]
        construction - Either '0' (4 (2D) or 6 (3D) connectivity) or 'N' (8 (2D) or 26 (3D) connectivity) ['0']
        thresh       - Threshold at which to define the foreground ROI for topological post-processing
    '''
    
    # Get image properties
    spatial_xyz = list(inputs.shape[2:])
    
    # Get working device
    device = inputs.device
    
    # Get raw prediction
    model.eval()
    with torch.no_grad():
        pred_unet = torch.softmax(model(inputs), 1).detach().squeeze()
        
    # If appropriate, choose ROI for topological consideration
    if thresh:
        roi = get_roi(pred_unet[1:].sum(0).squeeze(), thresh)
    else:
        roi = [slice(None, None)] + [slice(None, None) for dim in range(len(spatial_xyz))]
    
    # Initialise topological model and optimiser
    model_topo = copy.deepcopy(model)
    model_topo.eval()
    optimiser = opt(model_topo.parameters(), lr=lr)
    
    # Inspect prior and convert to tensor
    max_dims = [len(b) for b in prior.values()]
    prior = {torch.tensor(c): torch.tensor(b) for c, b in prior.items()}
    
    # Set mode of cubical complex construction
    PH = {'0': crip_wrapper, 'N': trip_wrapper}

    for it in range(num_its):

        # Reset gradients
        optimiser.zero_grad()

        # Get current prediction
        outputs = torch.softmax(model_topo(inputs), 1).squeeze()
        outputs_roi = outputs[roi]

        # Build class/combination-wise (c-wise) image tensor for prior
        combos = torch.stack([outputs_roi[c.T].sum(0) for c in prior.keys()])

        # Invert probababilistic fields for consistency with cripser sub-level set persistence
        combos = 1 - combos

        # Get barcodes using cripser in parallel without autograd            
        combos_arr = combos.detach().cpu().numpy().astype(np.float64)
        if parallel:
            with torch.no_grad():
                with Pool(len(prior)) as p:
                    bcodes_arr = p.starmap(PH[construction], zip(combos_arr, max_dims))
        else:
            with torch.no_grad():
                bcodes_arr = [PH[construction](combo, max_dim) for combo, max_dim in zip(combos_arr, max_dims)]

        # Get differentiable barcodes using autograd
        max_features = max([bcode_arr.shape[0] for bcode_arr in bcodes_arr])
        bcodes = torch.zeros([len(prior), max(max_dims), max_features], requires_grad=False, device=device)
        for c, (combo, bcode) in enumerate(zip(combos, bcodes_arr)):
            _, fin = get_differentiable_barcode(combo, bcode)
            for dim in range(len(spatial_xyz)):
                bcodes[c, dim, :len(fin[dim])] = fin[dim]

        # Select features for the construction of the topological loss
        stacked_prior = torch.stack(list(prior.values()))
        stacked_prior.T[0] -= 1 # Since fundamental 0D component has infinite persistence
        matching = torch.zeros_like(bcodes).detach().bool()
        for c, combo in enumerate(stacked_prior):
            for dim in range(len(combo)):
                matching[c, dim, slice(None, stacked_prior[c, dim])] = True

        # Find total persistence of features which match (A) / violate (Z) the prior
        A = (1 - bcodes[matching]).sum()
        Z = bcodes[~matching].sum()

        # Get similarity constraint
        mse = F.mse_loss(outputs, pred_unet)

        # Optimisation
        loss = A + Z + mse_lambda * mse
        loss.backward()
        optimiser.step()

    return model_topo
