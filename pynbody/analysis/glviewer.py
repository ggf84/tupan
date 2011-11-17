#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""

"""

from __future__ import print_function
import OpenGL.GL as gl
import OpenGL.GLU as glu
import OpenGL.GLUT as glut
import numpy as np
import os
import sys
import math
import string
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
ZOOM_FACTOR = 1.0
POINT_SIZE = 6.0
ALPHA = 0.5
COLORMAP = 1
TRACEORBITS = False
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
    shader_program = gl.glCreateProgram()

#    gl.glProgramParameteri(shader_program, gl.GL_GEOMETRY_INPUT_TYPE_EXT, gl.GL_POINTS )
#    glProgramParameteri(shader_program, gl.GL_GEOMETRY_OUTPUT_TYPE_EXT, gl.GL_TRIANGLE_STRIP )
#    glProgramParameteri(shader_program, gl.GL_GEOMETRY_VERTICES_OUT_EXT, 4 )

    vobj = gl.glCreateShader(gl.GL_VERTEX_SHADER)
    gl.glShaderSource(vobj, vert)
    gl.glCompileShader(vobj)
    print(gl.glGetShaderInfoLog(vobj))
    gl.glAttachShader(shader_program, vobj)

#    gobj = gl.glCreateShader(gl.GL_GEOMETRY_SHADER)
#    gl.glShaderSource(gobj, geom)
#    gl.glCompileShader(gobj)
#    print gl.glGetShaderInfoLog(gobj)
#    gl.glAttachShader(shader_program, gobj)

    fobj = gl.glCreateShader(gl.GL_FRAGMENT_SHADER)
    gl.glShaderSource(fobj, frag)
    gl.glCompileShader(fobj)
    print(gl.glGetShaderInfoLog(fobj))
    gl.glAttachShader(shader_program, fobj)

    gl.glLinkProgram(shader_program)
    print(gl.glGetProgramInfoLog(shader_program))

    return shader_program








def gaussian(height, center_x, center_y, width_x, width_y):
    """Returns a gaussian function with the given parameters"""
    width_x = float(width_x)
    width_y = float(width_y)
    return lambda x, y: height*np.exp(-(((center_x-x)/width_x)**2+((center_y-y)/width_y)**2)/2.0)




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


    def initialize(self):
        self.init_window()
        print('INFO: OpenGL Version: {0}'.format(gl.glGetString(gl.GL_VERSION)),
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
        global ZOOM_FACTOR
        beta = 0.05
        if (ZOOM_FACTOR > 2.0/beta):
            ZOOM_FACTOR = 2.0/beta
        if (ZOOM_FACTOR < 0.0001/beta):
            ZOOM_FACTOR = 0.0001/beta
        theta = beta * ZOOM_FACTOR
        gl.glMatrixMode(gl.GL_PROJECTION)
        gl.glLoadIdentity()
        ratio = float(self.window_width)/float(self.window_height)
        glu.gluPerspective(theta, ratio, 0.1, 20000.0)
        gl.glMatrixMode(gl.GL_MODELVIEW)


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
            gl.glRotatef(self.rot_x, 1.0, 0.0, 0.0)
            gl.glRotatef(self.rot_y, 0.0, 1.0, 0.0)
            gl.glRotatef(self.rot_z, 0.0, 0.0, 1.0)
        if sign < 0:
            gl.glRotatef(-self.rot_z, 0.0, 0.0, 1.0)
            gl.glRotatef(-self.rot_y, 0.0, 1.0, 0.0)
            gl.glRotatef(-self.rot_x, 1.0, 0.0, 0.0)


    def keyboard(self, key, x, y):
        global ALPHA
        global COLORMAP
        global FULLSCREEN
        global POINT_SIZE
        global RECORDSCREEN
        global TRACEORBITS
        global ZOOM_FACTOR
        (ps_min, ps_max) = gl.glGetFloatv(gl.GL_ALIASED_POINT_SIZE_RANGE)
        if key == ' ':
            self.rotate_x = 0
            self.rotate_y = 0
            self.rotate_z = 0
        elif key == '+':
            POINT_SIZE += 1
            if POINT_SIZE > ps_max:  POINT_SIZE = ps_max
        elif key == '-':
            POINT_SIZE -= 1
            if POINT_SIZE < ps_min:  POINT_SIZE = ps_min
        elif key == '<':
            self.rotate_z -= 1
        elif key == '>':
            self.rotate_z += 1
        elif key in '0123456789':
            COLORMAP = int(key)
        elif key == 'a':
            ALPHA /= 1.03125
            if ALPHA < 0.0:  ALPHA = 0.0
        elif key == 'A':
            ALPHA *= 1.03125
            if ALPHA > 1.0:  ALPHA = 1.0
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
            ZOOM_FACTOR *= 1.0078125
        elif key == 'z':
            ZOOM_FACTOR /= 1.0078125
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


    def keyboard_s(self, key, x, y):
        mkey = glut.glutGetModifiers()
        if mkey == glut.GLUT_ACTIVE_CTRL:
            pass
        elif mkey == glut.GLUT_ACTIVE_ALT:
            pass
        else:
            if key == glut.GLUT_KEY_UP:
                self.rotate_x += 1
            elif key == glut.GLUT_KEY_DOWN:
                self.rotate_x -= 1
            elif key == glut.GLUT_KEY_LEFT:
                self.rotate_y -= 1
            elif key == glut.GLUT_KEY_RIGHT:
                self.rotate_y += 1


    def init_window(self):
        glut.glutInit(sys.argv)
        glut.glutInitDisplayMode(glut.GLUT_DOUBLE | glut.GLUT_RGBA | glut.GLUT_DEPTH)
        glut.glutInitWindowPosition(
                    (glut.glutGet(glut.GLUT_SCREEN_WIDTH)-self.window_width)/2,
                    (glut.glutGet(glut.GLUT_SCREEN_HEIGHT)-self.window_height)/2)
        glut.glutSetOption(glut.GLUT_ACTION_ON_WINDOW_CLOSE,
                           glut.GLUT_ACTION_CONTINUE_EXECUTION)
        glut.glutInitWindowSize(self.window_width, self.window_height)
        self.window_handle = glut.glutCreateWindow(WINDOW_TITLE_PREFIX)
        glut.glutDisplayFunc(self.render_func)
        glut.glutReshapeFunc(self.resize_func)
        glut.glutKeyboardFunc(self.keyboard)
        glut.glutSpecialFunc(self.keyboard_s)
#        glut.glutIdleFunc(self.idle)


    def init_gl(self):
        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glClearColor(0.0, 0.0, 0.0, 0.0)
        gl.glColorMask(gl.GL_TRUE, gl.GL_TRUE, gl.GL_TRUE, gl.GL_FALSE)
        self.textures['star'] = self.load_texture(os.path.join(texture_path,
                                                               'glow.png'))
        self.adjust_zoom()


    def record_screen(self):
        gl.glPixelStorei(gl.GL_PACK_ALIGNMENT,8)
        screenshot = gl.glReadPixels(0, 0, WINDOW_WIDTH, WINDOW_HEIGHT,
                                     gl.GL_RGBA, gl.GL_UNSIGNED_BYTE)
        im = Image.frombuffer('RGBA', (WINDOW_WIDTH, WINDOW_HEIGHT),
                              screenshot, 'raw', 'RGBA', 0, 1)
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
        gl.glTranslatef(0.0, 0.0, -10000.05)
        self.adjust_zoom()
        self.rotate(1)

        self.draw_system()

        if RECORDSCREEN:
            self.record_screen()

        self.show_fps(1.0)

        glut.glutSwapBuffers()
        glut.glutPostRedisplay()


    def get_colors(self, qty):
        qty = np.sqrt(qty/qty.mean())

        if COLORMAP == 0:
            rgba = cm.gray(qty, alpha=ALPHA)
        elif COLORMAP == 1:
            rgba = cm.copper(qty, alpha=ALPHA)
        elif COLORMAP == 2:
            rgba = cm.hot(qty, alpha=ALPHA)
        elif COLORMAP == 3:
            rgba = cm.autumn(qty, alpha=ALPHA)
        elif COLORMAP == 4:
            rgba = cm.summer(qty, alpha=ALPHA)
        elif COLORMAP == 5:
            rgba = cm.gist_stern(qty, alpha=ALPHA)
        elif COLORMAP == 6:
            rgba = cm.cool(qty, alpha=ALPHA)
        elif COLORMAP == 7:
            rgba = cm.spectral(qty, alpha=ALPHA)
        elif COLORMAP == 8:
            rgba = cm.jet(qty, alpha=ALPHA)
        elif COLORMAP == 9:
            rgba = cm.gist_rainbow(qty, alpha=ALPHA)

        return rgba


    def draw_system(self):
        bodies = self.particle['body']
        if bodies:
            points = bodies.pos
            colors = self.get_colors(bodies.mass)

#            gl.glPushMatrix()

#            gl.glEnable(gl.GL_TEXTURE_2D)
#            gl.glBindTexture(gl.GL_TEXTURE_2D, self.textures['star'])
#            gl.glEnable(gl.GL_VERTEX_PROGRAM_POINT_SIZE)

#            gl.glUseProgram(self.shader_program)

            gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE)
#            gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)
#            gl.glBlendFunc(gl.GL_DST_ALPHA, gl.GL_ONE)
#            gl.glBlendFunc(gl.GL_DST_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)

            gl.glDepthMask(gl.GL_FALSE)
            gl.glEnable(gl.GL_BLEND)

            self.draw_points(points, colors, self.textures['star'])

            gl.glDisable(gl.GL_BLEND)
            gl.glDepthMask(gl.GL_TRUE)

#            gl.glDisable(gl.GL_VERTEX_PROGRAM_POINT_SIZE)
#            gl.glBindTexture(gl.GL_TEXTURE_2D, 0)
#            gl.glDisable(gl.GL_TEXTURE_2D)

#            gl.glPopMatrix()


        blackholes = self.particle['blackhole']
        if blackholes:
            points = blackholes.pos
            colors = np.zeros((len(blackholes), 3), dtype='f8')
            colors[:,0].fill(1)
            self.draw_points(points, colors, self.textures['star'])



    def draw_points(self, points, colors, texture):
        gl.glEnableClientState(gl.GL_VERTEX_ARRAY)
        gl.glVertexPointerd(points)

        gl.glEnableClientState(gl.GL_COLOR_ARRAY)
        gl.glColorPointerd(colors)

        gl.glEnable(gl.GL_TEXTURE_2D)
        gl.glBindTexture(gl.GL_TEXTURE_2D, texture)
        gl.glEnable(gl.GL_POINT_SPRITE)
        gl.glTexEnvf(gl.GL_POINT_SPRITE, gl.GL_COORD_REPLACE, gl.GL_TRUE)

        gl.glPointSize(POINT_SIZE)
        gl.glDrawArrays(gl.GL_POINTS, 0, len(points))

        gl.glTexEnvf(gl.GL_POINT_SPRITE, gl.GL_COORD_REPLACE, gl.GL_FALSE)
        gl.glDisable(gl.GL_POINT_SPRITE)
        gl.glBindTexture(gl.GL_TEXTURE_2D, 0)
        gl.glDisable(gl.GL_TEXTURE_2D)

        gl.glPointSize(1.0)
        gl.glDrawArrays(gl.GL_POINTS, 0, len(points))

        gl.glColorPointerd(None)
        gl.glDisableClientState(gl.GL_COLOR_ARRAY)

        gl.glVertexPointerd(None)
        gl.glDisableClientState(gl.GL_VERTEX_ARRAY)


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
        id = gl.glGenTextures(1)
        gl.glBindTexture(gl.GL_TEXTURE_2D, id)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_S, gl.GL_REPEAT)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_T, gl.GL_REPEAT)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
#        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR_MIPMAP_LINEAR)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_GENERATE_MIPMAP, gl.GL_TRUE)
        gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGBA, ix, iy, 0,
                        gl.GL_RGBA, gl.GL_UNSIGNED_BYTE, image)

        image = None
        return id


########## end of file ##########
