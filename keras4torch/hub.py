import torch

repo_url = 'keras4torch-team/codehub:main'

def list(force_reload=False):
    return torch.hub.list(repo_url, force_reload=force_reload)

def load(model, *args, **kwargs):
    return torch.hub.load(repo_url, model, *args, **kwargs)

def help(model, force_reload=False):
    return torch.hub.help(repo_url, model, force_reload=force_reload)

__all__ = ['list', 'load', 'help']