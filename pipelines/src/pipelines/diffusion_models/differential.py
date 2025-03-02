import torch

def differential_with_dirac(x, sigma, denoise_fn, mixture, source_id=0):
    print("shapes", x.shape, mixture.shape)
    num_sources = x.shape[1]
    x[:, [source_id], :] = mixture - (x.sum(dim=1, keepdim=True) - x[:, [source_id], :])
    score = (x - denoise_fn(x, sigma=sigma)) / sigma
    scores = [score[:, si] for si in range(num_sources)]
    ds = [s - score[:, source_id] for s in scores]
    return torch.stack(ds, dim=1)


def differential_with_gaussian(x, sigma, denoise_fn, mixture, gamma_fn=None):
    # print("shapes", x.shape, mixture.shape)
    # x = x[:, :mixture.shape[1], :]
    import torch.nn.functional as F
    mixture = F.pad(mixture, (0, 0, 0, 2, 0, 0))
    # print("shapes", x.shape, mixture.shape)
    gamma = sigma if gamma_fn is None else gamma_fn(sigma)
    d = (x - denoise_fn(x, sigma=sigma)) / sigma 
    d = d - sigma / (2 * gamma ** 2) * (mixture - x.sum(dim=[1], keepdim=True)) 
    #d = d - 8/sigma * (mixture - x.sum(dim=[1], keepdim=True)) 
    return d