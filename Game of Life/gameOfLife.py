import random
import matplotlib.pyplot as plt
import matplotlib.colors
import matplotlib.animation as animation
import numpy as np
import pickle
import copy


class GameOfLife():
    def __init__(self):
        self.mapSize = 160
        self.fadeIterations = 30
        self.activePixel = self.fadeIterations*(1+2/3)

        self.fig = plt.figure(figsize=(15,15))
        self.ax = plt.axes(xlim =(0.5, self.mapSize-1.5), ylim =(0.5, self.mapSize-1.5), xticks=[], yticks=[])
        self.cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["#000000","#404030","#00FFAF"])
        with open("screen.pickle", "rb") as infile:
            self.board = np.array(pickle.load(infile))
        self.logicBoard = np.where(self.board > 0, 1, 0)
        self.board = self.boardToShow(self, self.board, self.logicBoard)
        self.im = plt.imshow(self.board, cmap=self.cmap, vmin=0, vmax=self.activePixel, interpolation=None)

        self.cid = self.fig.canvas.mpl_connect('motion_notify_event', self.mouse_move)
        self.cid2 = self.fig.canvas.mpl_connect('button_press_event', self.mouse_press)
        self.cid3 = self.fig.canvas.mpl_connect('button_release_event', self.mouse_release)

        self.cursorClicked = False
        self.pause = True


    def drawAround(self, x, y, value):
        for i in [-1, 0, 1]:
            for j in [-1, 0, 1]:
                self.logicBoard[x+i][y+j] = value

    def mouse_move(self, event):
        if self.cursorClicked and event.xdata != None:
           self.drawAround(round(event.ydata), round(event.xdata), 1 if event.button == 1 else 0)

    def mouse_press(self, event):
        self.pause = False
        self.cursorClicked = True
        if event.xdata != None:
            self.drawAround(round(event.ydata), round(event.xdata), 1 if event.button == 1 else 0)

    def mouse_release(self, event):
        self.cursorClicked = False

    def play(self):
        # plt.gcf().set_facecolor("red")
        self.anim = animation.FuncAnimation(self.fig, self.anim, cache_frame_data=False, interval = 15, blit = True)
        plt.show()

    def neighbours(self, x, y):
        out = 0
        if self.logicBoard[x-1][y-1]:
            out += 1
        if self.logicBoard[x-1][y]:
            out += 1
        if self.logicBoard[x-1][y+1]:
            out += 1
        if self.logicBoard[x][y-1]:
            out += 1
        if self.logicBoard[x][y+1]:
            out += 1
        if self.logicBoard[x+1][y-1]:
            out += 1
        if self.logicBoard[x+1][y]:
            out += 1
        if self.logicBoard[x+1][y+1]:
            out += 1
        return out

    @np.vectorize
    def boardToShow(self, board, logicBoard):
        if logicBoard:
            return self.activePixel
        elif board > self.fadeIterations:
            return self.fadeIterations
        elif board > 0:
            return board -1
        else:
            return 0


    def anim(self, _):
        if self.pause == False:
            newLogicBoard = copy.deepcopy(self.logicBoard)
            for i in range(1, self.mapSize-1):
                for j in range(1, self.mapSize-1):
                    if self.logicBoard[i][j]:
                        if not self.neighbours(i,j) in [2, 3]:
                            newLogicBoard[i][j] = 0
                    elif self.neighbours(i,j) == 3:
                        newLogicBoard[i][j] = 1
            self.logicBoard = newLogicBoard
            self.board = self.boardToShow(self, self.board, self.logicBoard)
        self.im.set_array(self.board)
        return [self.im]




GameOfLife().play()