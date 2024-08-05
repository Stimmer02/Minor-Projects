import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

from collections.abc import Iterable
import os
import matplotlib as mpl

try:
    import JuliaEngineCUDA as jengine
except:
    import JuliaEngineCPU as jengine



class JuliaSet(object):

    def __init__(self, dim, c = 0.285+0.01j):
        self.shape = (256*dim, 256*dim)

        self.zoom = 1
        self.screenX = [-1.71, 0.85]
        self.screenY = [-1.28, 1.28]

        self.c = c
        self.fig = plt.figure(figsize=(20,20))
        self.ax = plt.axes(xticks=[], yticks=[])
        self.board = np.array(np.zeros(self.shape))

        x = np.arange(self.screenX[0]/self.zoom, self.screenX[1]/self.zoom, .01/self.zoom/(self.shape[0]/256))
        y = np.arange(self.screenY[0]/self.zoom, self.screenY[1]/self.zoom, .01/self.zoom/(self.shape[1]/256))
        x, y = np.meshgrid(x, y)
        self.z = x+1j*y

        try:
            self.engine = jengine.JuliaEngineCUDA(self.shape, self.z, self.board)
        except:
            self.engine = jengine.JuliaEngineCPU(self.shape, self.z, self.board)

        self.engine.transferZ(self.z)

        self.engine.generateImage(self.c, self.board)

        self.im = plt.imshow(self.board, cmap=mpl.colormaps["inferno"])

        self.cVector = 0 + 0j
        self.change1 = False
        self.change2 = False
        self.change3 = False

        self.cid1 = self.fig.canvas.mpl_connect('motion_notify_event', self.mouse_move)
        self.cid2 = self.fig.canvas.mpl_connect('button_press_event', self.mouse_press)
        self.cid3 = self.fig.canvas.mpl_connect('button_release_event', self.mouse_release)
        self.cid4 = self.fig.canvas.mpl_connect('scroll_event', self.mouse_scroll)

        self.rendering = False

    def generateZ(self):
        x = np.arange(self.screenX[0]/self.zoom, self.screenX[1]/self.zoom, .01/self.zoom/(self.shape[0]/256))
        y = np.arange(self.screenY[0]/self.zoom, self.screenY[1]/self.zoom, .01/self.zoom/(self.shape[1]/256))
        x, y = np.meshgrid(x, y)
        self.z = x+1j*y
        self.engine.transferZ(self.z)


    def mouse_move(self, event):
        if event.xdata != None:
            if event.button == 3:
                self.cVector = (self.cursor1X - event.xdata)/self.shape[0]/100 + 1j*(self.cursor1Y - event.ydata)/self.shape[1]/100
            elif event.button == 1:
                self.screenX[0] += (self.cursor2X - event.xdata)/self.shape[0]
                self.screenX[1] += (self.cursor2X - event.xdata)/self.shape[0]
                self.screenY[0] += (self.cursor2Y - event.ydata)/self.shape[1]
                self.screenY[1] += (self.cursor2Y - event.ydata)/self.shape[1]
                self.cursor2X = event.xdata
                self.cursor2Y = event.ydata
                self.generateZ()


    def mouse_scroll(self, event):
        self.zoom += self.zoom*(event.step/8)
        if self.zoom == 0:
            self.zoom = 0.1

        self.screenX[0] += (event.xdata-self.shape[0]/2)/self.shape[0]*event.step
        self.screenX[1] += (event.xdata-self.shape[0]/2)/self.shape[0]*event.step
        self.screenY[0] += (event.ydata-self.shape[0]/2)/self.shape[1]*event.step
        self.screenY[1] += (event.ydata-self.shape[0]/2)/self.shape[1]*event.step

        self.generateZ()
        self.change3 = True


    def mouse_press(self, event):
        if event.button == 3:
            self.cursor1X = event.xdata
            self.cursor1Y = event.ydata
            self.change1 = True
        elif event.button == 1:
            self.cursor2X = event.xdata
            self.cursor2Y = event.ydata
            self.change2 = True
        elif event.button == 2:
            if not self.rendering:
                self.rendering = True
                self.anim.event_source.stop()
                print("rendering...")
                saveName = JuliaSet.generateBig(16, self.c, "save", 256, self.zoom, self.screenX, self.screenY)
                print("saved as: "+saveName)
                self.anim.event_source.start()
                self.rendering = False

    def mouse_release(self, event):
        if event.button == 3:
            self.change1 = False
            self.cVector = 0 + 0j
        elif event.button == 1:
            self.change2 = False




    def animation_frame(self, _):
        if self.change1 or self.change2 or self.change3:
            self.c += self.cVector/self.zoom**2
            self.engine.generateImage(self.c, self.board)
            self.im.set_array(self.board)
            self.change3 = False
        return [self.im]

    def play(self):
        self.anim = animation.FuncAnimation(self.fig, self.animation_frame, cache_frame_data=False, interval = 10, blit = True)
        plt.show()
        print(self.c)


    def generateBig(dim, c, filename, accuracy, zoom = None, screenX = None, screenY = None):
        print(zoom, screenX, screenY)
        shape = (1024*dim, 1024*dim)
        if zoom == None:
            zoom = 1
            screenX = [-1.71, 0.85]
            screenY = [-1.28, 1.28]

        x = np.arange(screenX[0]/zoom, screenX[1]/zoom, .01/zoom/(shape[0]/256))
        y = np.arange(screenY[0]/zoom, screenY[1]/zoom, .01/zoom/(shape[1]/256))
        x, y = np.meshgrid(x, y)
        z = x+1j*y

        fig = plt.figure(figsize=(20,20))
        board = np.array(np.zeros(shape))


        jengine.generateBigImage(dim, z, c, board, accuracy)
        plt.axes(xticks=[], yticks=[])
        im = plt.imshow(board, cmap=mpl.colormaps["inferno"])
        i = 0
        while os.path.exists(f"{filename}{i}.png"):
            i += 1
        plt.savefig(f"{filename}{i}.png", bbox_inches="tight", dpi=int(1000*dim/15))
        return f"{filename}{i}.png"



JuliaSet(3,-0.7544812065081824-0.05358000726886422j).play()
# JuliaSet.generateBig(1, 0.2844866793285766+0.009974333966428849j, "save.png", 255)

