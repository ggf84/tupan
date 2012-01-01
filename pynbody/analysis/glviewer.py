#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""

"""

from __future__ import (print_function, division)
import OpenGL.GL as gl
import OpenGL.GLU as glu
import OpenGL.GLUT as glut
import numpy as np
import os
import sys
import Image
import subprocess
from matplotlib import cm
from pynbody.lib.utils.timing import (Timer, timings)


__all__ = ['GLviewer']


path = os.path.dirname(__file__)
texture_path = os.path.join(path, 'textures')


ESCAPE = '\033'
FULLSCREEN = False
RECORDSCREEN = False
WINDOW_WIDTH = 768
WINDOW_HEIGHT = 432
WINDOW_TITLE_PREFIX = 'PyNbody Viewer'


ROTINC = 0.0625
ZOOM_FACTOR = 5.0
POINT_SIZE = 64.0
ALPHA = 1.0
CONTRAST = 0.25
COLORMAP = 1
TRACEORBITS = False
COLORMASK = {'r': False, 'g': False, 'b': False}


class GLviewer(object):
    """

    """
    def __init__(self):
        self.window_width = WINDOW_WIDTH
        self.window_height = WINDOW_HEIGHT
        self.window_handle = None
        self.frame_count = 0
        self.textures = {'star': None, 'sph': None, 'blackhole': None}
        self.trans = {'x': 0.0, 'y': 0.0, 'z': 100.05}
        self.rot = {'x': 0.0, 'y': 0.0, 'z': 0.0}
        self.rotate = {'x': 0.0, 'y': 0.0, 'z': 0.0}
        self.mouse_button = None
        self.mouse_state = None
        self.mouse_x = 0
        self.mouse_y = 0

        self.particle = None
        self.exitgl = False
        self.timer = Timer()
        self.timer.start()

        cmdstring = ["ffmpeg", "-y",
                     "-r", "30",
                     "-f", "image2pipe",
                     "-vcodec", "ppm",
                     "-i", "pipe:",
                     "-vcodec", "libx264",
                     "-vpre", "slow",
                     "-qmin", "1",
                     "-qmax", "11",
                     "movie.mp4"]

        self.ffmpeg = subprocess.Popen(cmdstring,
                                       stdin=subprocess.PIPE,
                                       stdout=open(os.devnull, 'w'),
                                       stderr=subprocess.STDOUT)

#        cmdstring = ["mencoder",
#                     "-",
#                     "-demuxer", "rawvideo",
#                     "-rawvideo",
#                     "w={0}:h={1}:fps={2}:format={3}".format(WINDOW_WIDTH,
#                                                             WINDOW_HEIGHT,
#                                                             60, "rgba"),
#                     "-ovc", "lavc",
#                     "-of", "rawvideo",
#                     "-lavcopts", "vcodec=mpeg2video:vbitrate=10000",
#                     "-o", "video.mp4"]

#        self.mencoder = subprocess.Popen(cmdstring,
#                                         stdin=subprocess.PIPE,
#                                         stdout=open(os.devnull, 'w'),
#                                         stderr=subprocess.STDOUT)


    def set_particle(self, particle):
        self.particle = particle


    def enter_main_loop(self):
        if not self.exitgl:
            glut.glutMainLoop()


    def show_event(self, particles):
        if not self.exitgl:
            self.set_particle(particles)
            glut.glutMainLoopEvent()
            glut.glutSetWindow(self.window_handle)
            glut.glutPostRedisplay()


    def initialize(self):
        self.init_window()
        print('OpenGL Version: {0}'.format(gl.glGetString(gl.GL_VERSION)),
              file=sys.stderr)
        self.init_gl()


    def idle(self):
        glut.glutPostRedisplay()


    def resize_func(self, width, height):
        if height == 0:
            height = 1
        self.window_width = width
        self.window_height = height
        gl.glViewport(0, 0, width, height)


    def show_fps(self, secs=0.0):
        elapsed_time = self.timer.elapsed()
        if elapsed_time < secs:
            self.frame_count += 1
        else:
            fmt = "{0}: {1:.1f} fps @ {2} x {3}"
            win_title = fmt.format(WINDOW_TITLE_PREFIX,
                                   self.frame_count / elapsed_time,
                                   self.window_width, self.window_height)
            glut.glutSetWindow(self.window_handle)
            glut.glutSetWindowTitle(win_title)
            self.frame_count = 0
            self.timer.start()


    def adjust_zoom(self):
        ratio = self.window_width / self.window_height

        top = ZOOM_FACTOR
        bottom = -ZOOM_FACTOR
        right = ZOOM_FACTOR * ratio
        left = -ZOOM_FACTOR * ratio
        near = -0.1
        far = -200.0

        gl.glMatrixMode(gl.GL_PROJECTION)
        gl.glLoadIdentity()
        gl.glOrtho(left, right, bottom, top, near, far)
        gl.glMatrixMode(gl.GL_MODELVIEW)


    def adjust_rotation(self, sign=1):
        self.rot['x'] += self.rotate['x'] * ROTINC
        self.rot['y'] += self.rotate['y'] * ROTINC
        self.rot['z'] += self.rotate['z'] * ROTINC
        if self.rot['x'] > 360:
            self.rot['x'] -= 360
        elif self.rot['x'] < 0:
            self.rot['x'] += 360
        if self.rot['y'] > 360:
            self.rot['y'] -= 360
        elif self.rot['y'] < 0:
            self.rot['y'] += 360
        if self.rot['z'] > 360:
            self.rot['z'] -= 360
        elif self.rot['z'] < 0:
            self.rot['z'] += 360
        if sign > 0:
            gl.glRotatef(self.rot['x'], 1.0, 0.0, 0.0)
            gl.glRotatef(self.rot['y'], 0.0, 1.0, 0.0)
            gl.glRotatef(self.rot['z'], 0.0, 0.0, 1.0)
        if sign < 0:
            gl.glRotatef(-self.rot['z'], 0.0, 0.0, 1.0)
            gl.glRotatef(-self.rot['y'], 0.0, 1.0, 0.0)
            gl.glRotatef(-self.rot['x'], 1.0, 0.0, 0.0)


    def keyboard(self, key, x, y):
        global ALPHA
        global COLORMAP
        global CONTRAST
        global FULLSCREEN
        global POINT_SIZE
        global RECORDSCREEN
        global TRACEORBITS
        global ZOOM_FACTOR
        (ps_min, ps_max) = gl.glGetFloatv(gl.GL_ALIASED_POINT_SIZE_RANGE)
        if key == ' ':
            self.rotate['x'] = 0
            self.rotate['y'] = 0
            self.rotate['z'] = 0
        elif key == '+':
            POINT_SIZE += 1
            if POINT_SIZE > ps_max:
                POINT_SIZE = ps_max
        elif key == '-':
            POINT_SIZE -= 1
            if POINT_SIZE < ps_min:
                POINT_SIZE = ps_min
        elif key == '<':
            self.rotate['z'] -= 1
        elif key == '>':
            self.rotate['z'] += 1
        elif key in '0123456789':
            COLORMAP = int(key)
        elif key == 'a':
            ALPHA /= 1.03125
            if ALPHA < 0.0:
                ALPHA = 0.0
        elif key == 'A':
            ALPHA *= 1.03125
            if ALPHA > 1.0:
                ALPHA = 1.0
        elif key == 'c':
            CONTRAST /= 1.015625
            if CONTRAST < 2.0**(-8):
                CONTRAST = 2.0**(-8)
        elif key == 'C':
            CONTRAST *= 1.015625
            if CONTRAST > 16.0:
                CONTRAST = 16.0
        elif key == 'r':
            if not COLORMASK['r']:
                COLORMASK['r'] = True
                COLORMASK['g'] = False
                COLORMASK['b'] = False
            else:
                COLORMASK['r'] = False
        elif key == 'g':
            if not COLORMASK['g']:
                COLORMASK['r'] = False
                COLORMASK['g'] = True
                COLORMASK['b'] = False
            else:
                COLORMASK['g'] = False
        elif key == 'b':
            if not COLORMASK['b']:
                COLORMASK['r'] = False
                COLORMASK['g'] = False
                COLORMASK['b'] = True
            else:
                COLORMASK['b'] = False
        elif key == 'Z':
            ZOOM_FACTOR *= 1.03125
        elif key == 'z':
            ZOOM_FACTOR /= 1.03125
        elif key == 'o' or key == 'O':
            if not TRACEORBITS:
                TRACEORBITS = True
            else:
                TRACEORBITS = False
        elif key == 'f' or key == 'F':
            if not FULLSCREEN:
                glut.glutFullScreen()
                FULLSCREEN = True
            else:
                glut.glutReshapeWindow(WINDOW_WIDTH, WINDOW_HEIGHT)
                FULLSCREEN = False
        elif key == 's' or key == 'S':
            if not RECORDSCREEN:
                RECORDSCREEN = True
            else:
                RECORDSCREEN = False
        elif key == ESCAPE:
            self.exitgl = True
            glut.glutLeaveMainLoop()
            glut.glutHideWindow()

        glut.glutSetWindow(self.window_handle)
        glut.glutPostRedisplay()


    def keyboard_s(self, key, x, y):
        mkey = glut.glutGetModifiers()
        if mkey == glut.GLUT_ACTIVE_CTRL:
            pass
        elif mkey == glut.GLUT_ACTIVE_ALT:
            pass
        else:
            if key == glut.GLUT_KEY_UP:
                self.rotate['x'] += 1
            elif key == glut.GLUT_KEY_DOWN:
                self.rotate['x'] -= 1
            elif key == glut.GLUT_KEY_LEFT:
                self.rotate['y'] -= 1
            elif key == glut.GLUT_KEY_RIGHT:
                self.rotate['y'] += 1
        glut.glutSetWindow(self.window_handle)
        glut.glutPostRedisplay()


    def mouse(self, button, state, x, y):
        global ZOOM_FACTOR
        glut.glutSetWindow(self.window_handle)

        self.mouse_button = button
        self.mouse_state = state
        self.mouse_x = x
        self.mouse_y = y

        if button == 3:
            ZOOM_FACTOR /= 1.03125
            glut.glutPostRedisplay()
        if button == 4:
            ZOOM_FACTOR *= 1.03125
            glut.glutPostRedisplay()


    def mouse_motion(self, x, y):
        glut.glutSetWindow(self.window_handle)

        old_x = self.mouse_x
        old_y = self.mouse_y
        dx = 2 * (x - old_x) / self.window_width
        dy = 2 * (old_y - y) / self.window_height
        aspect_ratio = self.window_width / self.window_height

        if self.mouse_button == glut.GLUT_RIGHT_BUTTON:
            if self.mouse_state == glut.GLUT_DOWN:
                self.trans['x'] += dx * ZOOM_FACTOR * aspect_ratio
                self.trans['y'] += dy * ZOOM_FACTOR
                glut.glutPostRedisplay()
            if self.mouse_state == glut.GLUT_UP:
                pass
        if self.mouse_button == glut.GLUT_LEFT_BUTTON:
            if self.mouse_state == glut.GLUT_DOWN:
                self.rot['y'] -= 90 * dx
                self.rot['x'] -= 90 * dy
#                self.rotate['y'] -= 5 * dx
#                self.rotate['x'] -= 5 * dy
                glut.glutPostRedisplay()
            if self.mouse_state == glut.GLUT_UP:
                pass
        if self.mouse_button == glut.GLUT_MIDDLE_BUTTON:
            if self.mouse_state == glut.GLUT_DOWN:
                pass
            if self.mouse_state == glut.GLUT_UP:
                pass
        self.mouse_x = x
        self.mouse_y = y


    def init_window(self):
        glut.glutInit(sys.argv)
        glut.glutInitDisplayMode(
            glut.GLUT_DOUBLE | glut.GLUT_RGBA | glut.GLUT_DEPTH)
        glut.glutInitWindowPosition(
            (glut.glutGet(glut.GLUT_SCREEN_WIDTH) - self.window_width) // 2,
            (glut.glutGet(glut.GLUT_SCREEN_HEIGHT) - self.window_height) // 2)
        glut.glutSetOption(glut.GLUT_ACTION_ON_WINDOW_CLOSE,
                           glut.GLUT_ACTION_CONTINUE_EXECUTION)
        glut.glutInitWindowSize(self.window_width, self.window_height)
        self.window_handle = glut.glutCreateWindow(WINDOW_TITLE_PREFIX)
        glut.glutDisplayFunc(self.render_func)
        glut.glutReshapeFunc(self.resize_func)
        glut.glutMouseFunc(self.mouse)
        glut.glutMotionFunc(self.mouse_motion)
        glut.glutKeyboardFunc(self.keyboard)
        glut.glutSpecialFunc(self.keyboard_s)
#        glut.glutIdleFunc(self.idle)


    def init_gl(self):
        gl.glClearColor(0.0, 0.0, 0.0, 0.0)
        gl.glDepthMask(gl.GL_TRUE)
        gl.glEnable(gl.GL_DEPTH_TEST)

        gl.glColorMask(gl.GL_TRUE, gl.GL_TRUE, gl.GL_TRUE, gl.GL_FALSE)
        self.textures['star'] = self.load_texture(
                                    os.path.join(texture_path, 'star.png'))
        self.textures['sph'] = self.load_texture(
                                   os.path.join(texture_path, 'sph.png'))
        self.textures['blackhole'] = self.load_texture(
                                         os.path.join(texture_path, 'blackhole.png'))
        self.adjust_zoom()


    def record_screen(self):
        gl.glPixelStorei(gl.GL_PACK_ALIGNMENT, 8)
        screenshot = gl.glReadPixels(0, 0, WINDOW_WIDTH, WINDOW_HEIGHT,
                                     gl.GL_RGBA, gl.GL_UNSIGNED_BYTE)
        im = Image.frombuffer('RGBA', (WINDOW_WIDTH, WINDOW_HEIGHT),
                              screenshot, 'raw', 'RGBA', 0, 1)
        im = im.transpose(Image.FLIP_TOP_BOTTOM)
        im.save(self.ffmpeg.stdin, format='ppm')
#        self.mencoder.stdin.write(im.tostring())


    @timings
    def render_func(self):
        if not TRACEORBITS:
            gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
        else:
            gl.glClear(gl.GL_DEPTH_BUFFER_BIT)
        if COLORMASK['r']:
            gl.glColorMask(gl.GL_TRUE, gl.GL_FALSE, gl.GL_FALSE, gl.GL_FALSE)
        elif COLORMASK['g']:
            gl.glColorMask(gl.GL_FALSE, gl.GL_TRUE, gl.GL_FALSE, gl.GL_FALSE)
        elif COLORMASK['b']:
            gl.glColorMask(gl.GL_FALSE, gl.GL_FALSE, gl.GL_TRUE, gl.GL_FALSE)
        else:
            gl.glColorMask(gl.GL_TRUE, gl.GL_TRUE, gl.GL_TRUE, gl.GL_FALSE)

        gl.glMatrixMode(gl.GL_MODELVIEW)
        gl.glLoadIdentity()

        gl.glTranslatef(self.trans['x'],
                        self.trans['y'],
                        self.trans['z'])

        self.adjust_zoom()
        self.adjust_rotation(1)

        gl.glDisable(gl.GL_DEPTH_TEST)
        self.draw_system()
        gl.glDisable(gl.GL_DEPTH_TEST)

        if RECORDSCREEN:
            self.record_screen()

        self.show_fps(1.0)

        glut.glutSwapBuffers()

        if self.rotate['x'] != 0 or self.rotate['y'] != 0 or self.rotate['z'] != 0:
            glut.glutPostRedisplay()


    def get_colors(self, qty):

        maxqty = qty.max()
        minqty = qty.min()
        qty -= minqty
        qty /= (maxqty - minqty)

        qty = np.power(qty, CONTRAST)

        if COLORMAP == 0:
            rgba = cm.gray(qty, alpha=ALPHA)
        elif COLORMAP == 1:
            rgba = cm.afmhot(qty, alpha=ALPHA)
        elif COLORMAP == 2:
            rgba = cm.hot(qty, alpha=ALPHA)
        elif COLORMAP == 3:
            rgba = cm.gist_heat(qty, alpha=ALPHA)
        elif COLORMAP == 4:
            rgba = cm.copper(qty, alpha=ALPHA)
        elif COLORMAP == 5:
            rgba = cm.gnuplot2(qty, alpha=ALPHA)
        elif COLORMAP == 6:
            rgba = cm.gnuplot(qty, alpha=ALPHA)
        elif COLORMAP == 7:
            rgba = cm.gist_stern(qty, alpha=ALPHA)
        elif COLORMAP == 8:
            rgba = cm.gist_earth(qty, alpha=ALPHA)
        elif COLORMAP == 9:
            rgba = cm.spectral(qty, alpha=ALPHA)

        return rgba


    def draw_system(self):

##        gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ZERO)
        gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE)
##        gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_SRC_COLOR)
##        gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_COLOR)
##        gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_DST_COLOR)
#        gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_DST_COLOR)
##        gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_SRC_ALPHA)
#        gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)
#        gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_DST_ALPHA)
#        gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_DST_ALPHA)
##        gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_CONSTANT_COLOR)
#        gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_CONSTANT_COLOR)
##        gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_CONSTANT_ALPHA)
#        gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_CONSTANT_ALPHA)

        gl.glTexEnvf(gl.GL_TEXTURE_ENV, gl.GL_TEXTURE_ENV_MODE, gl.GL_MODULATE)

        Ntot = self.particle.get_total_num_of_particles()

        blackholes = self.particle['blackhole']
        if blackholes:
            points = blackholes.pos
            colors = self.get_colors(blackholes.mass)
            colors[:, 0].fill(0)
            colors[:, 1].fill(1)
            colors[:, 2].fill(0)
            colors[:, 3].fill(1)
            sizes = np.sqrt(blackholes.mass * Ntot)

            gl.glEnable(gl.GL_BLEND)

            self.draw_points(points, colors, sizes, self.textures['blackhole'])

            gl.glDisable(gl.GL_BLEND)

        bodies = self.particle['body']
        if bodies:
            points = bodies.pos
            colors = self.get_colors(bodies.mass)
            sizes = np.sqrt(bodies.eps2 * Ntot)

            gl.glEnable(gl.GL_BLEND)

            self.draw_points(points, colors, sizes, self.textures['star'])

            gl.glDisable(gl.GL_BLEND)


    def draw_points(self, points, colors, sizes, texture):
        gl.glEnableClientState(gl.GL_VERTEX_ARRAY)
        gl.glVertexPointerd(points)
        gl.glEnableClientState(gl.GL_COLOR_ARRAY)
        gl.glColorPointerd(colors)

        fade_size = 60.0
        att_param = [0.0, 0.0, 0.01]
        (ps_min, ps_max) = gl.glGetFloatv(gl.GL_ALIASED_POINT_SIZE_RANGE)

        gl.glEnable(gl.GL_TEXTURE_2D)
        gl.glBindTexture(gl.GL_TEXTURE_2D, texture)

        gl.glEnable(gl.GL_POINT_SPRITE)

        gl.glPointSize(POINT_SIZE)
        gl.glTexEnvf(gl.GL_POINT_SPRITE, gl.GL_COORD_REPLACE, gl.GL_TRUE)
        gl.glPointParameterfv(gl.GL_POINT_DISTANCE_ATTENUATION, att_param)
        gl.glPointParameterf(gl.GL_POINT_SIZE_MAX, ps_max)
        gl.glPointParameterf(gl.GL_POINT_SIZE_MIN, ps_min)
        gl.glPointParameterf(gl.GL_POINT_FADE_THRESHOLD_SIZE, fade_size)

        ####
        self.draw_arrays(points, colors, sizes, False)
        ####

        gl.glDisable(gl.GL_POINT_SPRITE)
        gl.glDisable(gl.GL_TEXTURE_2D)
        gl.glDisableClientState(gl.GL_VERTEX_ARRAY)
        gl.glDisableClientState(gl.GL_COLOR_ARRAY)


    def draw_arrays(self, points, colors, sizes, individual_sizes):
        if not individual_sizes:
            gl.glDrawArrays(gl.GL_POINTS, 0, len(points))
        else:
            (ps_min, ps_max) = gl.glGetFloatv(gl.GL_ALIASED_POINT_SIZE_RANGE)
            for point, color, size in zip(points, colors, sizes):
                vsize = size * POINT_SIZE
                if vsize < ps_min:
                    vsize = ps_min
                if vsize > ps_max:
                    vsize = ps_max
                gl.glPointSize(vsize)
                gl.glBegin(gl.GL_POINTS)
                gl.glColor4d(color[0], color[1], color[2], color[3])
                gl.glVertex3d(point[0], point[1], point[2])
                gl.glEnd()


    def load_texture(self, name):
        try:
            image = Image.open(name)
        except:
            raise RuntimeError('Failed to load \'{0}\' texture'.format(name))

        ix = image.size[0]
        iy = image.size[1]
        image = image.tostring('raw', 'RGBA', 0, -1)

        # Create Texture
        text_id = gl.glGenTextures(1)
        gl.glBindTexture(gl.GL_TEXTURE_2D, text_id)

        gl.glTexParameteri(gl.GL_TEXTURE_2D,
                           gl.GL_TEXTURE_MAG_FILTER,
                           gl.GL_LINEAR)
        gl.glTexParameteri(gl.GL_TEXTURE_2D,
                           gl.GL_TEXTURE_MIN_FILTER,
                           gl.GL_LINEAR)

        gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGBA, ix, iy, 0,
                        gl.GL_RGBA, gl.GL_UNSIGNED_BYTE, image)

        return text_id


########## end of file ##########
