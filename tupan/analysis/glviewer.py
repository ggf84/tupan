# -*- coding: utf-8 -*-
#

"""
TODO.
"""

from __future__ import (print_function, division)
import os
import sys
import logging
import subprocess
import numpy as np
from PIL import Image
from matplotlib import cm
from ..lib.utils.timing import timings, bind_all, Timer


LOGGER = logging.getLogger(__name__)


try:
    from OpenGL import GL
    from OpenGL import GLUT
    HAS_GL = True
except Exception as exc:
    HAS_GL = False
    LOGGER.exception(str(exc))
    import warnings
    warnings.warn(
        """
        An Exception occurred when trying to import OpenGL module.
        See 'tupan.log' for more details.
        Continuing without GLviewer...
        """,
        stacklevel=1
    )


__all__ = ['GLviewer']


PATH = os.path.dirname(__file__)
TEXTURE_PATH = os.path.join(PATH, 'textures')


ESCAPE = '\033'
FULLSCREEN = False
RECORDSCREEN = False
WINDOW_WIDTH = 768
WINDOW_HEIGHT = 432
WINDOW_TITLE_PREFIX = 'tupan viewer'


ROTINC = 0.0625
ZOOM_FACTOR = 5.0
POINT_SIZE = 64.0
ALPHA = 1.0
CONTRAST = 8.0
COLORMAP = 1
TRACEORBITS = False
COLORMASK = {'r': False, 'g': False, 'b': False}


@bind_all(timings)
class GLviewer(object):
    """

    """
    def __new__(cls, *args, **kwargs):
        if not HAS_GL:
            return None
        return super(GLviewer, cls).__new__(cls, *args, **kwargs)

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

        self.ps = None
        self.exitgl = False
        self.is_initialized = False
        self.timer = Timer()
        self.timer.start()

        self.ffmpeg = None
        self.mencoder = None

    def set_particle_system(self, ps):
        if self.ps is None:
            self.ps = ps.copy()
        else:
            self.ps[np.in1d(self.ps.id, ps.id)] = ps

    def enter_main_loop(self):
        if not self.exitgl:
            GLUT.glutMainLoop()

    def show_event(self, ps):
        if not self.exitgl:
            if not self.is_initialized:
                self.initialize()
            self.set_particle_system(ps)
            GLUT.glutMainLoopEvent()
            GLUT.glutSetWindow(self.window_handle)
            GLUT.glutPostRedisplay()

    def initialize(self):
        self.init_window()
        self.init_gl()
        self.is_initialized = True

    def idle(self):
        GLUT.glutPostRedisplay()

    def resize_func(self, width, height):
        if height == 0:
            height = 1
        self.window_width = width
        self.window_height = height
        GL.glViewport(0, 0, width, height)

    def show_fps(self, secs=0.0):
        elapsed_time = self.timer.elapsed()
        if elapsed_time < secs:
            self.frame_count += 1
        else:
            fmt = "{0}: {1:.1f} fps @ {2} x {3}"
            win_title = fmt.format(WINDOW_TITLE_PREFIX,
                                   self.frame_count / elapsed_time,
                                   self.window_width, self.window_height)
            GLUT.glutSetWindow(self.window_handle)
            GLUT.glutSetWindowTitle(win_title.encode('utf-8'))
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

        GL.glMatrixMode(GL.GL_PROJECTION)
        GL.glLoadIdentity()
        GL.glOrtho(left, right, bottom, top, near, far)
        GL.glMatrixMode(GL.GL_MODELVIEW)

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
            GL.glRotatef(self.rot['x'], 1.0, 0.0, 0.0)
            GL.glRotatef(self.rot['y'], 0.0, 1.0, 0.0)
            GL.glRotatef(self.rot['z'], 0.0, 0.0, 1.0)
        if sign < 0:
            GL.glRotatef(-self.rot['z'], 0.0, 0.0, 1.0)
            GL.glRotatef(-self.rot['y'], 0.0, 1.0, 0.0)
            GL.glRotatef(-self.rot['x'], 1.0, 0.0, 0.0)

    def keyboard(self, key, x, y):
        global ALPHA
        global COLORMAP
        global CONTRAST
        global FULLSCREEN
        global POINT_SIZE
        global RECORDSCREEN
        global TRACEORBITS
        global ZOOM_FACTOR
        (ps_min, ps_max) = GL.glGetFloatv(GL.GL_ALIASED_POINT_SIZE_RANGE)

        key = key.decode('utf-8')

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
            CONTRAST *= 1.015625
            if CONTRAST > 256.0:
                CONTRAST = 256.0
        elif key == 'C':
            CONTRAST /= 1.015625
            if CONTRAST < 0.0625:
                CONTRAST = 0.0625
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
                GLUT.glutFullScreen()
                FULLSCREEN = True
            else:
                GLUT.glutReshapeWindow(WINDOW_WIDTH, WINDOW_HEIGHT)
                FULLSCREEN = False
        elif key == 's' or key == 'S':
            if not RECORDSCREEN:
                RECORDSCREEN = True
            else:
                RECORDSCREEN = False
        elif key == ESCAPE:
            self.exitgl = True
            GLUT.glutLeaveMainLoop()
            GLUT.glutHideWindow()

        GLUT.glutSetWindow(self.window_handle)
        GLUT.glutPostRedisplay()

    def keyboard_s(self, key, x, y):
        mkey = GLUT.glutGetModifiers()
        if mkey == GLUT.GLUT_ACTIVE_CTRL:
            pass
        elif mkey == GLUT.GLUT_ACTIVE_ALT:
            pass
        else:
            if key == GLUT.GLUT_KEY_UP:
                self.rotate['x'] += 1
            elif key == GLUT.GLUT_KEY_DOWN:
                self.rotate['x'] -= 1
            elif key == GLUT.GLUT_KEY_LEFT:
                self.rotate['y'] -= 1
            elif key == GLUT.GLUT_KEY_RIGHT:
                self.rotate['y'] += 1
        GLUT.glutSetWindow(self.window_handle)
        GLUT.glutPostRedisplay()

    def mouse(self, button, state, x, y):
        global ZOOM_FACTOR
        GLUT.glutSetWindow(self.window_handle)

        self.mouse_button = button
        self.mouse_state = state
        self.mouse_x = x
        self.mouse_y = y

        if button == 3:
            ZOOM_FACTOR /= 1.03125
            GLUT.glutPostRedisplay()
        if button == 4:
            ZOOM_FACTOR *= 1.03125
            GLUT.glutPostRedisplay()

    def mouse_motion(self, x, y):
        GLUT.glutSetWindow(self.window_handle)

        old_x = self.mouse_x
        old_y = self.mouse_y
        dx = 2 * (x - old_x) / self.window_width
        dy = 2 * (old_y - y) / self.window_height
        aspect_ratio = self.window_width / self.window_height

        if self.mouse_button == GLUT.GLUT_LEFT_BUTTON:
            if self.mouse_state == GLUT.GLUT_DOWN:
                self.trans['x'] += dx * ZOOM_FACTOR * aspect_ratio
                self.trans['y'] += dy * ZOOM_FACTOR
                GLUT.glutPostRedisplay()
            if self.mouse_state == GLUT.GLUT_UP:
                pass

        if self.mouse_button == GLUT.GLUT_RIGHT_BUTTON:
            if self.mouse_state == GLUT.GLUT_DOWN:
                self.rot['y'] -= 90 * dx
                self.rot['x'] -= 90 * dy
#                self.rotate['y'] -= 5 * dx
#                self.rotate['x'] -= 5 * dy
                GLUT.glutPostRedisplay()
            if self.mouse_state == GLUT.GLUT_UP:
                pass

        if self.mouse_button == GLUT.GLUT_MIDDLE_BUTTON:
            if self.mouse_state == GLUT.GLUT_DOWN:
                pass
            if self.mouse_state == GLUT.GLUT_UP:
                pass

        self.mouse_x = x
        self.mouse_y = y

    def init_window(self):
        GLUT.glutInit(sys.argv)
        GLUT.glutInitDisplayMode(
            GLUT.GLUT_DOUBLE | GLUT.GLUT_RGBA |
            GLUT.GLUT_ALPHA | GLUT.GLUT_DEPTH
        )
        GLUT.glutInitWindowPosition(
            (GLUT.glutGet(GLUT.GLUT_SCREEN_WIDTH) - self.window_width) // 2,
            (GLUT.glutGet(GLUT.GLUT_SCREEN_HEIGHT) - self.window_height) // 2
        )
        GLUT.glutSetOption(
            GLUT.GLUT_ACTION_ON_WINDOW_CLOSE,
            GLUT.GLUT_ACTION_CONTINUE_EXECUTION
        )
        GLUT.glutInitWindowSize(self.window_width, self.window_height)
        win_title = WINDOW_TITLE_PREFIX.encode('utf-8')
        self.window_handle = GLUT.glutCreateWindow(win_title)
        GLUT.glutDisplayFunc(self.render_func)
        GLUT.glutReshapeFunc(self.resize_func)
        GLUT.glutMouseFunc(self.mouse)
        GLUT.glutMotionFunc(self.mouse_motion)
        GLUT.glutKeyboardFunc(self.keyboard)
        GLUT.glutSpecialFunc(self.keyboard_s)
#        GLUT.glutIdleFunc(self.idle)

    def init_gl(self):
        GL.glEnable(GL.GL_DEPTH_TEST)
        GL.glEnable(GL.GL_ALPHA_TEST)
        GL.glClearColor(0.0, 0.0, 0.0, 1.0)
        GL.glColorMask(GL.GL_TRUE, GL.GL_TRUE, GL.GL_TRUE, GL.GL_TRUE)

        self.textures['star'] = self.load_texture(
            os.path.join(TEXTURE_PATH, 'star.png'))
        self.textures['sph'] = self.load_texture(
            os.path.join(TEXTURE_PATH, 'sph.png'))
        self.textures['blackhole'] = self.load_texture(
            os.path.join(TEXTURE_PATH, 'blackhole.png'))

        self.adjust_zoom()

    def record_screen(self):
        if self.ffmpeg is None:
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
#        if self.mencoder is None:
#            cmdstring = ["mencoder",
#                         "-",
#                         "-demuxer", "rawvideo",
#                         "-rawvideo",
#                         "w={0}:h={1}:fps={2}:format={3}".format(
#                             WINDOW_WIDTH,
#                             WINDOW_HEIGHT,
#                             60, "rgba"),
#                         "-ovc", "lavc",
#                         "-of", "rawvideo",
#                         "-lavcopts", "vcodec=mpeg2video:vbitrate=10000",
#                         "-o", "video.mp4"]
#            self.mencoder = subprocess.Popen(cmdstring,
#                                             stdin=subprocess.PIPE,
#                                             stdout=open(os.devnull, 'w'),
#                                             stderr=subprocess.STDOUT)

        GL.glPixelStorei(GL.GL_PACK_ALIGNMENT, 8)
        screenshot = GL.glReadPixels(0, 0, WINDOW_WIDTH, WINDOW_HEIGHT,
                                     GL.GL_RGBA, GL.GL_UNSIGNED_BYTE)
        im = Image.frombuffer('RGBA', (WINDOW_WIDTH, WINDOW_HEIGHT),
                              screenshot, 'raw', 'RGBA', 0, 1)
        im = im.transpose(Image.FLIP_TOP_BOTTOM)
        im.save(self.ffmpeg.stdin, format='ppm')
#        self.mencoder.stdin.write(im.tostring())

    def render_func(self):
        if not TRACEORBITS:
            GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)
        else:
            GL.glClear(GL.GL_DEPTH_BUFFER_BIT)
        if COLORMASK['r']:
            GL.glColorMask(GL.GL_TRUE, GL.GL_FALSE, GL.GL_FALSE, GL.GL_TRUE)
        elif COLORMASK['g']:
            GL.glColorMask(GL.GL_FALSE, GL.GL_TRUE, GL.GL_FALSE, GL.GL_TRUE)
        elif COLORMASK['b']:
            GL.glColorMask(GL.GL_FALSE, GL.GL_FALSE, GL.GL_TRUE, GL.GL_TRUE)
        else:
            GL.glColorMask(GL.GL_TRUE, GL.GL_TRUE, GL.GL_TRUE, GL.GL_TRUE)

        GL.glAlphaFunc(GL.GL_GREATER, 0)

        GL.glMatrixMode(GL.GL_MODELVIEW)
        GL.glLoadIdentity()

        GL.glTranslatef(self.trans['x'],
                        self.trans['y'],
                        self.trans['z'])

        self.adjust_zoom()
        self.adjust_rotation(1)

        GL.glDepthMask(GL.GL_FALSE)
        self.draw_system()
        GL.glDepthMask(GL.GL_TRUE)

        if RECORDSCREEN:
            self.record_screen()

        self.show_fps(1.0)

        GLUT.glutSwapBuffers()

        if (self.rotate['x'] != 0
                or self.rotate['y'] != 0
                or self.rotate['z'] != 0):
            GLUT.glutPostRedisplay()

    def get_colors(self, qty):

        qty = np.power(qty / qty.max(), 1.0/CONTRAST)

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

# #        GL.glBlendFunc(GL.GL_SRC_ALPHA, GL.GL_ZERO)
#         GL.glBlendFunc(GL.GL_SRC_ALPHA, GL.GL_ONE)
# #        GL.glBlendFunc(GL.GL_SRC_ALPHA, GL.GL_SRC_COLOR)
# #        GL.glBlendFunc(GL.GL_SRC_ALPHA, GL.GL_ONE_MINUS_SRC_COLOR)
# #        GL.glBlendFunc(GL.GL_SRC_ALPHA, GL.GL_DST_COLOR)
#         GL.glBlendFunc(GL.GL_SRC_ALPHA, GL.GL_ONE_MINUS_DST_COLOR)
# #        GL.glBlendFunc(GL.GL_SRC_ALPHA, GL.GL_SRC_ALPHA)
#         GL.glBlendFunc(GL.GL_SRC_ALPHA, GL.GL_ONE_MINUS_SRC_ALPHA)
#         GL.glBlendFunc(GL.GL_SRC_ALPHA, GL.GL_DST_ALPHA)
#         GL.glBlendFunc(GL.GL_SRC_ALPHA, GL.GL_ONE_MINUS_DST_ALPHA)
# #        GL.glBlendFunc(GL.GL_SRC_ALPHA, GL.GL_CONSTANT_COLOR)
#         GL.glBlendFunc(GL.GL_SRC_ALPHA, GL.GL_ONE_MINUS_CONSTANT_COLOR)
# #        GL.glBlendFunc(GL.GL_SRC_ALPHA, GL.GL_CONSTANT_ALPHA)
#         GL.glBlendFunc(GL.GL_SRC_ALPHA, GL.GL_ONE_MINUS_CONSTANT_ALPHA)

        GL.glTexEnvf(GL.GL_TEXTURE_ENV, GL.GL_TEXTURE_ENV_MODE, GL.GL_MODULATE)

        ntot = self.ps.n

        if 'bodies' in self.ps.members:
            bodies = self.ps.members['bodies']
            if bodies.n:
                points = bodies.pos
                colors = self.get_colors(bodies.mass)
                sizes = np.sqrt(bodies.eps2 * ntot)

                GL.glEnable(GL.GL_BLEND)
#                GL.glBlendFunc(GL.GL_SRC_ALPHA, GL.GL_ONE_MINUS_SRC_ALPHA)
                GL.glBlendFunc(GL.GL_SRC_ALPHA, GL.GL_ONE)
                self.draw_points(points, colors, sizes, self.textures['star'])
                GL.glDisable(GL.GL_BLEND)

        if 'blackholes' in self.ps.members:
            blackholes = self.ps.members['blackholes']
            if blackholes.n:
                points = blackholes.pos
                colors = self.get_colors(blackholes.mass)
                colors[..., 0].fill(0)
                colors[..., 1].fill(1)
                colors[..., 2].fill(0)
                colors[..., 3].fill(1)
                sizes = np.sqrt(blackholes.mass * ntot)

#                GL.glAlphaFunc(GL.GL_EQUAL, 1)
                GL.glDepthMask(GL.GL_TRUE)
                GL.glEnable(GL.GL_BLEND)
                GL.glBlendFunc(GL.GL_SRC_ALPHA, GL.GL_ZERO)
                self.draw_points(
                    points, colors, sizes, self.textures['blackhole'])
                GL.glDisable(GL.GL_BLEND)
                GL.glDepthMask(GL.GL_FALSE)
#                GL.glAlphaFunc(GL.GL_GREATER, 0)

        if 'stars' in self.ps.members:
            stars = self.ps.members['stars']
            if stars.n:
                points = stars.pos
                colors = self.get_colors(stars.mass)
                sizes = np.sqrt(stars.eps2 * ntot)

                GL.glEnable(GL.GL_BLEND)
#                GL.glBlendFunc(GL.GL_SRC_ALPHA, GL.GL_ONE_MINUS_SRC_ALPHA)
                GL.glBlendFunc(GL.GL_SRC_ALPHA, GL.GL_ONE)
                self.draw_points(points, colors, sizes, self.textures['star'])
                GL.glDisable(GL.GL_BLEND)

        if 'sphs' in self.ps.members:
            sph = self.ps.members['sphs']
            if sph.n:
                points = sph.pos
                colors = self.get_colors(sph.mass)
                sizes = np.sqrt(sph.eps2 * ntot)

                GL.glEnable(GL.GL_BLEND)
                GL.glBlendFunc(GL.GL_SRC_ALPHA, GL.GL_ONE)
                self.draw_points(points, colors, sizes, self.textures['sph'])
                GL.glDisable(GL.GL_BLEND)

    def draw_points(self, points, colors, sizes, texture):
        GL.glEnableClientState(GL.GL_VERTEX_ARRAY)
        GL.glVertexPointerd(points)
        GL.glEnableClientState(GL.GL_COLOR_ARRAY)
        GL.glColorPointerd(colors)

        fade_size = 60.0
        att_param = [0.0, 0.0, 0.01]
        (ps_min, ps_max) = GL.glGetFloatv(GL.GL_ALIASED_POINT_SIZE_RANGE)

        GL.glEnable(GL.GL_TEXTURE_2D)
        GL.glBindTexture(GL.GL_TEXTURE_2D, texture)

        GL.glEnable(GL.GL_POINT_SPRITE)

        GL.glPointSize(POINT_SIZE)
        GL.glTexEnvf(GL.GL_POINT_SPRITE, GL.GL_COORD_REPLACE, GL.GL_TRUE)
        GL.glPointParameterfv(GL.GL_POINT_DISTANCE_ATTENUATION, att_param)
        GL.glPointParameterf(GL.GL_POINT_SIZE_MAX, ps_max)
        GL.glPointParameterf(GL.GL_POINT_SIZE_MIN, ps_min)
        GL.glPointParameterf(GL.GL_POINT_FADE_THRESHOLD_SIZE, fade_size)

        ####
        self.draw_arrays(points, colors, sizes, False)
        ####

        GL.glDisable(GL.GL_POINT_SPRITE)
        GL.glDisable(GL.GL_TEXTURE_2D)
        GL.glDisableClientState(GL.GL_VERTEX_ARRAY)
        GL.glDisableClientState(GL.GL_COLOR_ARRAY)

    def draw_arrays(self, points, colors, sizes, individual_sizes):
        if not individual_sizes:
            GL.glDrawArrays(GL.GL_POINTS, 0, len(points))
        else:
            (ps_min, ps_max) = GL.glGetFloatv(GL.GL_ALIASED_POINT_SIZE_RANGE)
            for point, color, size in zip(points, colors, sizes):
                vsize = size * POINT_SIZE
                if vsize < ps_min:
                    vsize = ps_min
                if vsize > ps_max:
                    vsize = ps_max
                GL.glPointSize(vsize)
                GL.glBegin(GL.GL_POINTS)
                GL.glColor4d(color[0], color[1], color[2], color[3])
                GL.glVertex3d(point[0], point[1], point[2])
                GL.glEnd()

    def load_texture(self, name):
        try:
            image = Image.open(name)
        except:
            raise RuntimeError('Failed to load \'{0}\' texture'.format(name))

        ix = image.size[0]
        iy = image.size[1]
        image = image.tobytes('raw', 'RGBA', 0, -1)

        # Create Texture
        text_id = GL.glGenTextures(1)
        GL.glBindTexture(GL.GL_TEXTURE_2D, text_id)

        GL.glTexParameteri(GL.GL_TEXTURE_2D,
                           GL.GL_TEXTURE_MAG_FILTER,
                           GL.GL_LINEAR)
        GL.glTexParameteri(GL.GL_TEXTURE_2D,
                           GL.GL_TEXTURE_MIN_FILTER,
                           GL.GL_LINEAR)

        GL.glTexParameteri(GL.GL_TEXTURE_2D,
                           GL.GL_TEXTURE_WRAP_S,
                           GL.GL_CLAMP_TO_EDGE)
        GL.glTexParameteri(GL.GL_TEXTURE_2D,
                           GL.GL_TEXTURE_WRAP_T,
                           GL.GL_CLAMP_TO_EDGE)

        GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, GL.GL_RGBA, ix, iy, 0,
                        GL.GL_RGBA, GL.GL_UNSIGNED_BYTE, image)

        return text_id


# -- End of File --
