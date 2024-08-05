import numpy as np
import multiprocessing as mp


class JuliaEngineCPU(object):
    def __init__(self, shape, z, board):
        self.threads = mp.cpu_count()

        self.shape = shape
        self.threadBoundries = [(round(i*self.shape[0]/self.threads), round((i+1)*self.shape[0]/self.threads)) for i in range(self.threads)]



    def transferZ(self, z):
        self.z = z


    def generateImage(self, c, board):
        if self.z.shape != board.shape:
            print("ERR dimensions does not match", self.z.shape[0] - board.shape[0], self.z.shape[1] - board.shape[1])
            return

        threadsOutput = mp.SimpleQueue()
        threads = [mp.Process(
            target=JuliaEngineCPU.generatorThread,
            args=(self.threadBoundries[i], threadsOutput, self.z[self.threadBoundries[i][0]:self.threadBoundries[i][1]], c))
            for i in range(self.threads)]

        for t in threads:
            t.start()

        results = [threadsOutput.get() for _ in threads]

        for t in threads:
            t.join()

        for i in results:
            board[i[0][0]:i[0][1]] = i[1]


    def generatorThread(xBoundry, output, zFragment, c):
        output.put((xBoundry, JuliaEngineCPU.generatorThreadCore(zFragment, c)))

    @np.vectorize
    def generatorThreadCore(z, c):
        if abs(z) > 2:
            return 0
        z = z**2 + z
        for i in range(1, 127):
            if abs(z) > 2:
                return i
            z = z**2 + c
        return 127


def generateBigImage(dim, z, c, board, accuracy):
    j = JuliaEngineCPU(z.shape, z, board)
    j.transferZ(z)
    j.generateImage(c, board)

