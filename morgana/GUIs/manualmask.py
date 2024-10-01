#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 10:57:50 2019

@author: ngritti
"""
from PyQt5 import QtCore
from PyQt5.QtCore import Qt, QElapsedTimer
from PyQt5.QtGui import QKeyEvent
from PyQt5.QtWidgets import QApplication, QVBoxLayout, QDialog, QPushButton, QLabel
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvas
from matplotlib.backends.backend_qt5agg import (
    NavigationToolbar2QT as NavigationToolbar,
)
import numpy as np
import sys, warnings, os
from skimage.io import imread, imsave
from scipy.interpolate import splprep, splev
import cv2
from shapely.geometry import Point, Polygon, LineString
import matplotlib as mpl
from matplotlib.path import Path as MplPath

warnings.filterwarnings("ignore")

class makeManualMask(QDialog):
    MODES = ["snip","add","insert", "drag"]
    def __init__(
        self,
        file_in,
        subfolder="result_segmentation",
        fn=None,
        parent=None,
        wsize=(1000, 1000),
        initial_contour=None,
        stride=10,
    ):
        super(makeManualMask, self).__init__(parent)
        self.setWindowTitle("Manual mask: " + file_in)
        QApplication.setStyle("Material")
        self.setWindowFlag(Qt.WindowCloseButtonHint, False)
        self.file_in = file_in
        self.subfolder = subfolder
        self.fn = fn
        img = imread(file_in)
        input_folder, filename = os.path.split(file_in)
        if initial_contour is None:
            self.coords = np.empty((0,2))
            self.mode = "add"
        else:
            self.mode = "insert"
            if initial_contour=="watershed":
                mask = imread(f"{input_folder}/result_segmentation/{filename.replace('.tif', '_watershed.tif')}")
            elif initial_contour=="classifier":
                mask = imread(f"{input_folder}/result_segmentation/{filename.replace('.tif', '_classifier.tif')}")
            polygons = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
            if len(polygons) >=2:
                raise NotImplementedError("Multiple contours found in initial mask")
            else:
                self.coords = polygons[0][:,0,:]
                self.coords = self.coords[::stride]
        self.previousCoords = []
        if len(img.shape) == 2:
            img = np.expand_dims(img, 0)
        if img.shape[-1] == np.min(img.shape):
            img = np.moveaxis(img, -1, 0)
        self.img = img[0]
        self.lineType = "spline"
        self.snipStart = None
        self.dragPoint = None
        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        self.plotImage()
        self.updateLine(self.coords)
        self.canvas.setFocusPolicy(QtCore.Qt.StrongFocus)
        self.canvas.setFocus()
        self.button = QPushButton("Save mask")
        self.button.clicked.connect(self.saveMask)
        self.updateMessage("press k to see keyboard shorcuts")
        print(self.message)
        self.ax.set_title(self.message)
        self.keyboardShorcutsWindow = None
        layout = QVBoxLayout()
        layout.addWidget(self.canvas)
        layout.addWidget(self.button)   
        self.setLayout(layout)
        self.resize(wsize[0], wsize[1])
        self.__cid_key_press = self.canvas.mpl_connect('key_press_event', self.__key_press_callback)
        self.__cid_button_press = self.canvas.mpl_connect("button_press_event", self.__button_press_callback)
        self.__cid_button_release = self.canvas.mpl_connect("button_release_event", self.__button_release_callback)
        self.__cid_motion = self.canvas.mpl_connect('motion_notify_event', self.__motion_notify_callback)

    def notify(self, receiver, event):
        self.t.start()
        ret = QApplication.notify(self, receiver, event)
        if(self.t.elapsed() > 10):
            print(f"processing event type {event.type()} for object {receiver.objectName()} " 
                  f"took {self.t.elapsed()}ms")
        return ret

    def plotImage(self):
        """plot some random stuff"""
        self.ax = self.figure.add_subplot(111)
        self.ax.clear()
        self.ax.imshow(
            self.img,
            cmap="gray",
            vmin=np.percentile(self.img, 1.0),
            vmax=np.percentile(self.img, 99.0),
        )
        self.line = self.ax.plot(*self.coords.T, "r")[0]
        self.points = self.ax.plot(*self.coords.T, "or")[0]
        self.snipLine = self.ax.plot([], [], "--k", linewidth=1)[0]
        self.snipPoints = self.ax.plot([], [], "ok")[0]
        self.canvas.draw()

    def saveMask(self):
        ny, nx = np.shape(self.img)
        # poly_verts = np.concatenate([self.coords[0].reshape(1,2), self.coords[::-1]])
        poly_verts = np.array(self.line.get_data()).T
        x, y = np.meshgrid(np.arange(nx), np.arange(ny))
        x, y = x.flatten(), y.flatten()
        points = np.vstack((x, y)).T
        roi_path = MplPath(poly_verts)
        mask = 1 * roi_path.contains_points(points).reshape((ny, nx))
        self.updateMessage(f"Saved Mask: Area = {np.sum(mask)}")
        print(self.message)
        self.ax.set_title(self.message)
        folder, filename = os.path.split(self.file_in)
        filename, extension = os.path.splitext(filename)
        if self.fn == None:
            self.fn = filename + "_manual" + extension
        imsave(os.path.join(folder, self.subfolder, self.fn), mask.astype(np.uint16))

    
    def updateLine(self, coords, storePrevious=True):
        if storePrevious:
            self.previousCoords.append(self.coords)
        if len(coords)>1 and np.all(coords[-1]==coords[-2]):
            while np.all(coords[-1]==coords[-2]):
                coords = coords[:-1]
        self.coords = coords.copy()
        self.points.set_data(*self.coords.T)
        if len(self.coords)>0:
            plot_coords = np.vstack([self.coords.copy(), self.coords[0]])
        else:
            plot_coords = self.coords
        if len(self.coords)>3 and self.lineType=="spline":
            tck = splprep(plot_coords.T, s=0, per=True)
            u_new = np.linspace(0, 1, 1000)
            x_new, y_new = splev(u_new, tck[0])
            self.line.set_data(x_new, y_new)
        else:
            self.line.set_data(*plot_coords.T)
        self.canvas.draw()

    def updateMessage(self, message, draw=True):
        self.message = message
        self.ax.set_title(self.message)
        if draw:
            self.canvas.draw()

    def find_closest_edge_indices(self, x, y):
        c = self.coords
        target_point = Point([x,y])
        polygon = Polygon(c)
        if not polygon.is_valid:
            raise ValueError("The given coordinates do not form a valid polygon.")
        distances = [
            (target_point.distance(LineString([c[i], c[(i + 1) % len(c)]])), i, (i + 1) % len(c))
            for i in range(len(c))
        ]
        _, index1, index2 = min(distances, key=lambda x: x[0])
        return (index1, index2)

    def find_outside_points(self, x, y):
        c = self.coords
        ss, se = self.snipStart, np.array([x, y])
        polygon = Polygon(c)
        if not polygon.is_valid:
            raise ValueError("The given coordinates do not form a valid polygon.")
        centroid = np.array([polygon.centroid.x, polygon.centroid.y])
        se = np.array(se)
        snipDirection = se - ss
        np.cross = lambda v1, v2: v1[0] * v2[1] - v1[1] * v2[0]
        centroid_cross = np.cross(snipDirection, centroid - ss)
        oppositeInds = [
            i for i, point in enumerate(c)
            if (np.cross(snipDirection, np.array(point) - ss) * centroid_cross) < 0
            and 0 <= np.dot(np.array(point) - ss, snipDirection) / np.dot(snipDirection, snipDirection) <= 1
        ]
        return oppositeInds

    def __button_press_callback(self, event):
        if event.inaxes == self.ax:
            x, y = int(event.xdata), int(event.ydata)
            n_p = len(self.coords)
            if self.mode == "add":
                if (event.button == 1):
                    self.updateLine(np.vstack((self.coords, [x, y])))
                elif (event.button == 3):
                    if (n_p > 1):
                        self.updateLine(self.coords[:-1])
                    if (n_p == 1):
                        self.updateLine(np.empty((0, 2)))
            elif self.mode == "insert":
                if (event.button == 1):
                    if n_p >=3:
                        try:
                            # find the closest side in the polygon to the clicked position
                            _, index2 = self.find_closest_edge_indices(x, y)
                            self.updateLine(np.insert(self.coords, index2, [x, y], axis=0))
                        except ValueError:
                            self.updateMessage("Invalid Point: The given coordinates do not form a valid polygon.")
                            print(self.message)
                            self.ax.set_title(self.message)     
                    else:
                        self.updateLine(np.vstack((self.coords, [x, y])))
                    

                elif (event.button == 3):
                    if (n_p > 1):
                        # find the closest point and remove it
                        closest = np.argmin(np.linalg.norm(self.coords - [x, y], axis=1))
                        dist = np.linalg.norm(self.coords[closest] - [x, y])
                        if dist < 10:
                            self.updateLine(np.delete(self.coords, closest, axis=0))
                    if (n_p == 1):
                        self.updateLine(np.empty((0, 2)))

                elif (event.button == 3):
                    if (n_p > 1):
                        # find the closest point and remove it
                        closest = np.argmin(np.linalg.norm(self.coords - [x, y], axis=1))
                        dist = np.linalg.norm(self.coords[closest] - [x, y])
                        if dist < 10:
                            self.updateLine(np.delete(self.coords, closest, axis=0))
                    if (n_p == 1):
                        self.updateLine(np.empty((0, 2)))

            elif self.mode == "snip":
                # draw a line and remove all points on the outside of the line
                if (event.button == 1):
                    if self.snipStart is None:
                        self.snipStart = [x, y]
                    else:
                        outside = self.find_outside_points(x, y)
                        self.snipPoints.set_data([], [])
                        self.snipLine.set_data([], [])
                        self.snipStart = None
                        self.updateLine(np.delete(self.coords, outside, axis=0))
            elif self.mode == "drag":
                closest = np.argmin(np.linalg.norm(self.coords - [x, y], axis=1))
                dist = np.linalg.norm(self.coords[closest] - [x, y])
                if dist < 10:
                    self.dragPoint = closest
                    coords = self.coords.copy()
                    coords[self.dragPoint] = [x,y]
                    self.updateLine(coords)

    def __button_release_callback(self, event):
        if event.inaxes == self.ax:
            # x, y = int(event.xdata), int(event.ydata)
            if self.mode == "drag":
                self.updateLine(self.coords)
                self.dragPoint=None

    def __motion_notify_callback(self, event):
        if self.mode == "drag" or self.mode == "snip":
            if event.inaxes == self.ax:
                x, y = int(event.xdata), int(event.ydata)
                if self.mode == "snip" and self.snipStart is not None:
                    self.snipLine.set_data([self.snipStart[0], x], [self.snipStart[1], y])
                    outside = self.find_outside_points(x, y)
                    outsidePoints = self.coords[outside]
                    self.snipPoints.set_data(*outsidePoints.T)
                    self.canvas.draw()
                else:
                    self.snipLine.set_data([], [])
                    self.snipPoints.set_data([], [])
                if self.mode == "drag" and self.dragPoint is not None:
                        coords = self.coords.copy()
                        coords[self.dragPoint] = [x,y]
                        self.updateLine(coords, storePrevious=False)
                

    def __key_press_callback(self, event):
        if event.key == " ":
            self.mode = "snip"
            self.snipStart = None
            self.updateMessage(f"Mode changed to {self.mode}")
        elif event.key == "a":
            self.mode = "add"
            self.updateMessage(f"Mode changed to {self.mode}")
        elif event.key == "i":
            self.mode = "insert"
            self.updateMessage(f"Mode changed to {self.mode}")
        elif event.key == "d":
            self.mode = "drag"
            self.updateMessage(f"Mode changed to {self.mode}")
        elif event.key == "o":
            last = self.lineType
            if self.lineType=="polygon":
                self.lineType = "spline"
            else:
                self.lineType = "polygon"
            self.updateMessage(f"linetype toggled from {last} to {self.lineType}", draw=False)
            self.updateLine(self.coords)
        elif event.key == "c":
            self.updateLine(np.empty((0, 2)))
            self.updateMessage("Cleared points")
        elif event.key == "z":
            if len(self.previousCoords)>0:
                self.updateMessage("Undo", draw=False)
                self.updateLine(self.previousCoords.pop())
                self.previousCoords.pop()
            else:
                self.updateMessage("Nothing to undo.")
        elif event.key == "s":
            self.saveMask()
            self.updateMessage("Mask saved")
        elif event.key == "q":
            self.updateMessage("Quitting")
            self.close()
        elif event.key == "k":
            if self.keyboardShorcutsWindow is None:
                self.keyboardShorcutsWindow = keyboardShortcuts(self)
                self.keyboardShorcutsWindow.setFocusPolicy(Qt.NoFocus)
                self.keyboardShorcutsWindow.show()
                self.keyboardShorcutsWindow.exec()
                self.setFocusPolicy(Qt.StrongFocus)
            else:
                self.keyboardShorcutsWindow.close()
                self.keyboardShorcutsWindow = None
        else:
            pass

class keyboardShortcuts(QDialog):
    def __init__(self, parent=None, wsize=(200,200)):
        self.parent = parent
        super(keyboardShortcuts, self).__init__()
        self.setWindowTitle("Keyboard Shortcuts")
        QApplication.setStyle("Material")
        self.setWindowFlag(Qt.WindowCloseButtonHint, False)
        self.commands = ["a - add mode. Sequentially add and remove points using click and right click",
                         "i - insert mode. Insert and remove points using click and right click",
                         "d - drag mode. Drag a point to a new position",
                         "space - snip mode. Remove points outside the snip line.",
                         "c - clear all points."
                         "z - undo.",
                         "o - toggle linetype from spline to polygon",
                         "s - save mask.",
                         "k - show keyboard shorcuts.",
                         "q - quit",
                         ]
        self.labels = [QLabel(command) for command in self.commands]
        self.layout = QVBoxLayout()
        [self.layout.addWidget(label) for label in self.labels]
        self.setLayout(self.layout)
        self.resize(wsize[0], wsize[1])
        
    
    def keyPressEvent(self, event):
        if event.key == "k":
            self.parent.__key_press_callback(event)
