import scipy.io as sio


def est_load(path):
    h = sio.loadmat(path + '/pcsi.mat')['pcsi']
    hest = sio.loadmat(path + '/ecsi.mat')['ecsi']
    return h, hest
