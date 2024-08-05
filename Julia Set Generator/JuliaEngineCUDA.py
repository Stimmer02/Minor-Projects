import numpy as np
from numba import cuda

@cuda.jit
def cuda_juliaSet(z_initState, z, c, out):
    x = cuda.threadIdx.x
    y = cuda.blockIdx.x

    if abs(z_initState[x][y]) > 2:
        out[x][y] = 0
        return
    z[x][y] = z_initState[x][y]**2 + z_initState[x][y]
    for i in range(1, 256):
        if abs(z[x][y]) > 2:
            out[x][y] = i
            return
        z[x][y] = z[x][y]**2 + c
    out[x][y] = 256

@cuda.jit
def cuda_juliaSetBig(z, c, out, quality):
    x = cuda.threadIdx.x + cuda.blockDim.x * cuda.blockIdx.y
    y = cuda.blockIdx.x + cuda.gridDim.x * cuda.blockIdx.z

    if cuda.threadIdx.x == 0 and cuda.blockIdx.x == 0:
        print(cuda.blockIdx.y, cuda.blockIdx.z)

    if abs(z[x][y]) > 2:
        out[x][y] = 0
        return
    z[x][y] = z[x][y]**2 + z[x][y]
    for i in range(1, quality):
        if abs(z[x][y]) > 2:
            out[x][y] = i
            return
        z[x][y] = z[x][y]**2 + c
    out[x][y] = quality


class JuliaEngineCUDA(object):
    def __init__(self, shape, z, board):
        self.shape = shape

        self.transferZ(z)

        self.cuda_cWork = cuda.device_array_like(self.z)
        self.cuda_out = cuda.device_array_like(board)


    def transferZ(self, z):
        self.z = z
        self.cuda_cInitState = cuda.to_device(self.z)


    def generateImage(self, c, board):
        cuda_juliaSet[self.shape[1], self.shape[0]](self.cuda_cInitState, self.cuda_cWork, c, self.cuda_out)
        self.cuda_out.copy_to_host(board)


def generateBigImage(dim, z, c, board, accuracy):
        cuda_cInitState = cuda.to_device(z)
        cuda_out = cuda.device_array_like(board)
        cuda_juliaSetBig[(1024, dim, dim), (1024)](cuda_cInitState, c, cuda_out, accuracy)
        cuda_out.copy_to_host(board)


