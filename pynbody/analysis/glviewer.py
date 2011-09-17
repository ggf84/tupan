#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""

"""

from __future__ import print_function
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
import numpy as np
import os
import sys
import math
import time
import string
import Image
import colorsys
import subprocess


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
ZOOM_FACTOR = 1.0
POINT_SIZE = 6.0
CONTRAST = 1.0
TRACEORBITS = False
COLORSCHEME = 1
COLORMASK = {'r': False, 'g': False, 'b': False}


vert = '''
void main() {
    gl_FrontColor = gl_Color;
    gl_Position = ftransform();
}
'''

frag = '''
void main() {
    gl_FragColor = gl_Color;
}
'''




def define_shader():
    shader_program = glCreateProgram()

#    glProgramParameteri(shader_program, GL_GEOMETRY_INPUT_TYPE_EXT, gl.GL_POINTS )
#    glProgramParameteri(shader_program, GL_GEOMETRY_OUTPUT_TYPE_EXT, gl.GL_TRIANGLE_STRIP )
#    glProgramParameteri(shader_program, GL_GEOMETRY_VERTICES_OUT_EXT, 4 )

    vobj = glCreateShader(GL_VERTEX_SHADER)
    glShaderSource(vobj, vert)
    glCompileShader(vobj)
    print(glGetShaderInfoLog(vobj))
    glAttachShader(shader_program, vobj)

#    gobj = glCreateShader(GL_GEOMETRY_SHADER)
#    glShaderSource(gobj, geom)
#    glCompileShader(gobj)
#    print glGetShaderInfoLog(gobj)
#    glAttachShader(shader_program, gobj)

    fobj = glCreateShader(GL_FRAGMENT_SHADER)
    glShaderSource(fobj, frag)
    glCompileShader(fobj)
    print(glGetShaderInfoLog(fobj))
    glAttachShader(shader_program, fobj)

    glLinkProgram(shader_program)
    print(glGetProgramInfoLog(shader_program))

    return shader_program








def gaussian(height, center_x, center_y, width_x, width_y):
    """Returns a gaussian function with the given parameters"""
    width_x = float(width_x)
    width_y = float(width_y)
    return lambda x, y: height*np.exp(-(((center_x-x)/width_x)**2+((center_y-y)/width_y)**2)/2.0)




class Timer(object):
    """

    """
    def __init__(self):
        self.tic = 0.0
        self.toc = 0.0
        self.stopped = False

    def start(self):
        self.stopped = False
        self.tic = time.time()

    def stop(self):
        self.stopped = True
        self.toc = time.time()

    def elapsed(self):
        if not self.stopped:
            self.toc = time.time()
        return self.toc - self.tic




class GLviewer(object):
    """

    """
    def __init__(self):
        self.window_width = WINDOW_WIDTH
        self.window_height = WINDOW_HEIGHT
        self.window_handle = None
        self.frame_count = 0
        self.shader_program = 0
        self.textures = {}
        self.rot_x = 0.0
        self.rot_y = 0.0
        self.rot_z = 0.0
        self.rotate_x = 0
        self.rotate_y = 0
        self.rotate_z = 0
        self.particle = None
        self.exitgl = False
        self.timer = Timer()
        self.timer.start()
        self._body_colors = None

        cmdstring = ["ffmpeg", "-y",
                     "-r", "60",
                     "-b", "10000K",
                     "-f", "image2pipe",
                     "-vcodec", "ppm",
                     "-i", "pipe:",
                     "-vcodec", "libx264",
                     "-vpre", "slow",
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
            glutMainLoop()


    def show_event(self, particles):
        if not self.exitgl:
            self.set_particle(particles)
            glutMainLoopEvent()


    def initialize(self):
        self.init_window()
        print('INFO: OpenGL Version: {0}'.format(glGetString(GL_VERSION)),
              file=sys.stderr)
        self.init_gl()


    def idle(self):
        glutPostRedisplay()


    def resize_func(self, width, height):
        if height == 0:
            height = 1
        self.window_width = width
        self.window_height = height
        glViewport(0, 0, width, height)


    def show_fps(self, secs=0.0):
        elapsed_time = self.timer.elapsed()
        if elapsed_time < secs:
            self.frame_count += 1
        else:
            fmt = "{0}: {1:.1f} fps @ {2} x {3}"
            win_title = fmt.format(WINDOW_TITLE_PREFIX,
                                   self.frame_count / elapsed_time,
                                   self.window_width, self.window_height)
            glutSetWindow(self.window_handle)
            glutSetWindowTitle(win_title)
            self.frame_count = 0
            self.timer.start()


    def adjust_zoom(self):
        global ZOOM_FACTOR
        beta = 0.05
        if (ZOOM_FACTOR > 2.0/beta):
            ZOOM_FACTOR = 2.0/beta
        if (ZOOM_FACTOR < 0.0001/beta):
            ZOOM_FACTOR = 0.0001/beta
        theta = beta * ZOOM_FACTOR
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        ratio = float(self.window_width)/float(self.window_height)
        gluPerspective(theta, ratio, 0.1, 20000.0)
        glMatrixMode(GL_MODELVIEW)


    def rotate(self, sign=1):
        self.rot_x += self.rotate_x * ROTINC
        self.rot_y += self.rotate_y * ROTINC
        self.rot_z += self.rotate_z * ROTINC
        if self.rot_x > 360:
            self.rot_x -= 360
        elif self.rot_x < 0:
            self.rot_x += 360
        if self.rot_y > 360:
            self.rot_y -= 360
        elif self.rot_y < 0:
            self.rot_y += 360
        if self.rot_z > 360:
            self.rot_z -= 360
        elif self.rot_z < 0:
            self.rot_z += 360
        if sign > 0:
            glRotatef(self.rot_x, 1.0, 0.0, 0.0)
            glRotatef(self.rot_y, 0.0, 1.0, 0.0)
            glRotatef(self.rot_z, 0.0, 0.0, 1.0)
        if sign < 0:
            glRotatef(-self.rot_z, 0.0, 0.0, 1.0)
            glRotatef(-self.rot_y, 0.0, 1.0, 0.0)
            glRotatef(-self.rot_x, 1.0, 0.0, 0.0)


    def keyboard(self, key, x, y):
        global POINT_SIZE
        global CONTRAST
        global ZOOM_FACTOR
        if key == ' ':
            self.rotate_x = 0
            self.rotate_y = 0
            self.rotate_z = 0
        elif key == '+':
            POINT_SIZE += 1
        elif key == '-':
            POINT_SIZE -= 1
            if POINT_SIZE < 1:  POINT_SIZE = 1
        elif key == '<':
            self.rotate_z -= 1
        elif key == '>':
            self.rotate_z += 1
        elif key in '0123456789':
            global COLORSCHEME
            COLORSCHEME = int(key)
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
        elif key == 'c':
            CONTRAST *= 2
            if CONTRAST > 2**16:  CONTRAST = 2.0**16
        elif key == 'C':
            CONTRAST /= 2
            if CONTRAST < 2**(-16):  CONTRAST = 2.0**(-16)
        elif key == 'Z':
            ZOOM_FACTOR *= 1.0078125
        elif key == 'z':
            ZOOM_FACTOR /= 1.0078125
        elif key == 'o' or key == 'O':
            global TRACEORBITS
            if not TRACEORBITS:
                TRACEORBITS = True
            else:
                TRACEORBITS = False
        elif key == 'f' or key == 'F':
            global FULLSCREEN
            if not FULLSCREEN:
                glutFullScreen()
                FULLSCREEN = True
            else:
                glutReshapeWindow(WINDOW_WIDTH, WINDOW_HEIGHT)
                FULLSCREEN = False
        elif key == 's' or key == 'S':
            global RECORDSCREEN
            if not RECORDSCREEN:
                RECORDSCREEN = True
            else:
                RECORDSCREEN = False
        elif key == ESCAPE:
            self.exitgl = True
            glutLeaveMainLoop()
            glutHideWindow()


    def keyboard_s(self, key, x, y):
        mkey = glutGetModifiers()
        if mkey == GLUT_ACTIVE_CTRL:
            pass
        elif mkey == GLUT_ACTIVE_ALT:
            pass
        else:
            if key == GLUT_KEY_UP:
                self.rotate_x += 1
            elif key == GLUT_KEY_DOWN:
                self.rotate_x -= 1
            elif key == GLUT_KEY_LEFT:
                self.rotate_y -= 1
            elif key == GLUT_KEY_RIGHT:
                self.rotate_y += 1


    def init_window(self):
        glutInit(sys.argv)
        glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA | GLUT_DEPTH)
        glutInitWindowPosition((glutGet(GLUT_SCREEN_WIDTH)-self.window_width)/2,
                               (glutGet(GLUT_SCREEN_HEIGHT)-self.window_height)/2)
        glutSetOption(GLUT_ACTION_ON_WINDOW_CLOSE,
                      GLUT_ACTION_CONTINUE_EXECUTION)
        glutInitWindowSize(self.window_width, self.window_height)
        self.window_handle = glutCreateWindow(WINDOW_TITLE_PREFIX)
        glutDisplayFunc(self.render_func)
        glutReshapeFunc(self.resize_func)
        glutKeyboardFunc(self.keyboard)
        glutSpecialFunc(self.keyboard_s)
#        glutIdleFunc(self.idle)


    def init_gl(self):
        glEnable(GL_DEPTH_TEST)
        glClearColor(0.0, 0.0, 0.0, 0.0)
        glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_FALSE)
        self.textures['star'] = self.load_texture(os.path.join(texture_path,
                                                               'glow.png'))
        self.adjust_zoom()


    def record_screen(self):
        glPixelStorei(GL_PACK_ALIGNMENT,8)
        screenshot = glReadPixels(0, 0, WINDOW_WIDTH, WINDOW_HEIGHT,
                                  GL_RGBA, GL_UNSIGNED_BYTE)
        im = Image.frombuffer('RGBA', (WINDOW_WIDTH, WINDOW_HEIGHT),
                              screenshot, 'raw', 'RGBA', 0, 1)
        im.save(self.ffmpeg.stdin, format='ppm')
#        self.mencoder.stdin.write(im.tostring())


    def render_func(self):
        if not TRACEORBITS:
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        else:
            glClear(GL_DEPTH_BUFFER_BIT)
        if COLORMASK['r']:
            glColorMask(GL_TRUE, GL_FALSE, GL_FALSE, GL_FALSE)
        elif COLORMASK['g']:
            glColorMask(GL_FALSE, GL_TRUE, GL_FALSE, GL_FALSE)
        elif COLORMASK['b']:
            glColorMask(GL_FALSE, GL_FALSE, GL_TRUE, GL_FALSE)
        else:
            glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_FALSE)

        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        glTranslatef(0.0, 0.0, -10000.05)
        self.adjust_zoom()
        self.rotate(1)

        self.draw_system()

        if RECORDSCREEN:
            self.record_screen()

        self.show_fps(1.0)

        glutSwapBuffers()
        glutPostRedisplay()


    def set_point_size_limits(self):
        global POINT_SIZE
        (ps_min, ps_max) = glGetFloatv(GL_ALIASED_POINT_SIZE_RANGE)
        if (POINT_SIZE > ps_max):
            POINT_SIZE = ps_max
        if (POINT_SIZE < ps_min):
            POINT_SIZE = ps_min


    def get_colors(self, obj):
        if self._body_colors is None:
            m = obj.mass

            r = m**0.5
            g = m
            b = m**2.0

            r /= r.mean()
            g /= g.mean()
            b /= b.mean()

            self._body_colors = (r, g, b)

        (r, g, b) = self._body_colors

        if COLORSCHEME == 0:
            colors = np.ones((len(obj), 3), dtype='f8')
        elif COLORSCHEME == 1:
            colors = (np.vstack((r, g, b)).T)
        elif COLORSCHEME == 2:
            colors = (np.vstack((b, g, r)).T)
        elif COLORSCHEME == 3:
            colors = (np.vstack((g, b, r)).T)
        elif COLORSCHEME == 4:
            colors = (np.vstack((g, r, b)).T)
        elif COLORSCHEME == 5:
            colors = (np.vstack((b, r, g)).T)
        elif COLORSCHEME == 6:
            colors = (np.vstack((r, b, g)).T)
        elif COLORSCHEME == 7:
            colors = (np.vstack((r, r, r)).T)
        elif COLORSCHEME == 8:
            colors = (np.vstack((g, g, g)).T)
        elif COLORSCHEME == 9:
            colors = (np.vstack((b, b, b)).T)

        colors *= np.log10(1.0+CONTRAST)/3
        return colors


    def draw_system(self):
        self.set_point_size_limits()

        bodies = self.particle['body']
        if bodies:
            points = bodies.pos
            colors = self.get_colors(bodies)

#            glPushMatrix()

#            glEnable(GL_TEXTURE_2D)
#            glBindTexture(GL_TEXTURE_2D, self.textures['star'])
#            glEnable(GL_VERTEX_PROGRAM_POINT_SIZE)

#            glUseProgram(self.shader_program)

            glBlendFunc(GL_SRC_ALPHA, GL_ONE)
#            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
#            glBlendFunc(GL_DST_ALPHA, GL_ONE)
#            glBlendFunc(GL_DST_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

            glDepthMask(GL_FALSE)
            glEnable(GL_BLEND)

            self.draw_points(points, colors, self.textures['star'])

            glDisable(GL_BLEND)
            glDepthMask(GL_TRUE)

#            glDisable(GL_VERTEX_PROGRAM_POINT_SIZE)
#            glBindTexture(GL_TEXTURE_2D, 0)
#            glDisable(GL_TEXTURE_2D)

#            glPopMatrix()


        blackholes = self.particle['blackhole']
        if blackholes:
            points = blackholes.pos
            colors = np.zeros((len(blackholes), 3), dtype='f8')
            colors[:,0].fill(1)
            self.draw_points(points, colors, self.textures['star'])



    def draw_points(self, points, colors, texture):
        glEnableClientState(GL_VERTEX_ARRAY)
        glVertexPointerd(points)

        glEnableClientState(GL_COLOR_ARRAY)
        glColorPointerd(colors)

        glEnable(GL_TEXTURE_2D)
        glBindTexture(GL_TEXTURE_2D, texture)
        glEnable(GL_POINT_SPRITE)
        glTexEnvf(GL_POINT_SPRITE, GL_COORD_REPLACE, GL_TRUE)

        glPointSize(POINT_SIZE)
        glDrawArrays(GL_POINTS, 0, len(points))

        glTexEnvf(GL_POINT_SPRITE, GL_COORD_REPLACE, GL_FALSE)
        glDisable(GL_POINT_SPRITE)
        glBindTexture(GL_TEXTURE_2D, 0)
        glDisable(GL_TEXTURE_2D)

        glPointSize(1.0)
        glDrawArrays(GL_POINTS, 0, len(points))

        glColorPointerd(None)
        glDisableClientState(GL_COLOR_ARRAY)

        glVertexPointerd(None)
        glDisableClientState(GL_VERTEX_ARRAY)


    def load_texture(self, name):
#        X, Y = np.mgrid[0.0:129.0, 0.0:129.0]
#        z = gaussian(1.0, 64.0, 64.0, 16.0, 16.0)
#        image = z(X, Y)
#        image = Image.fromarray(image, 'RGBA')

        try:
            image = Image.open(name)
        except:
            raise RuntimeError('Failed to load \'{0}\' texture'.format(name))

        ix = image.size[0]
        iy = image.size[1]
        image = image.tostring('raw', 'RGBA', 0, -1)

        # Create Texture
        id = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, id)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
#        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_GENERATE_MIPMAP, GL_TRUE)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, ix, iy, 0,
                     GL_RGBA, GL_UNSIGNED_BYTE, image)

        image = None
        return id


########## end of file ##########
