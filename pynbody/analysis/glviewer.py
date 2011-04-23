#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""

"""

from __future__ import print_function
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
from OpenGL.raw import GL
from OpenGL.arrays import vbo
from OpenGL.arrays import ArrayDatatype as ADT
import numpy as np
import os
import sys
import math
import string
import Image


ESCAPE = '\033'
FULLSCREEN = False
RECORD = False
WINDOW_WIDTH = 1024
WINDOW_HEIGHT = 640

WINDOW_TITLE_PREFIX = 'PyNbody Viewer'
#file_path = os.path.dirname(__file__) + os.sep + 'textures'
file_path = 'textures'



ROTINC = 0.05
POINT_SIZE = 10.0
ZOOM_FACTOR = 1.0








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
        self.initialize()
#        glutMainLoop()


    def update(self, particle):
        self.particle = particle
        glutMainLoop()
        print('after close...')


    def initialize(self):
        self.init_window()
        print('INFO: OpenGL Version: {0}'.format(glGetString(GL_VERSION)))
        self.init_gl()


    def idle(self):
        glutPostRedisplay()


    def resize_func(self, width, height):
        if height == 0:
            height = 1
        self.window_width = width
        self.window_height = height
        glViewport(0, 0, width, height)


    def timer_func(self, value):
        if value != 0:
            tmp = ('{0}: {1} fps @ {2} x {3}').format(WINDOW_TITLE_PREFIX,
                                                      self.frame_count * 4,
                                                      self.window_width,
                                                      self.window_height)
            glutSetWindow(self.window_handle)
            glutSetWindowTitle(tmp)
        self.frame_count = 0
        glutTimerFunc(250, self.timer_func, 1)


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
        elif key == 'Z':
            ZOOM_FACTOR *= 1.01
        elif key == 'z':
            ZOOM_FACTOR /= 1.01
        elif key == 'f' or key == 'F':
            global FULLSCREEN
            if not FULLSCREEN:
                glutFullScreen()
                FULLSCREEN = True
            else:
                glutReshapeWindow(WINDOW_WIDTH, WINDOW_HEIGHT)
                FULLSCREEN = False
        elif key == 'r' or key == 'R':
            global RECORD
            if not RECORD:
                RECORD = True
            else:
                RECORD = False
        elif key == ESCAPE:
            glutDestroyWindow(self.window_handle)


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
        glutInit (sys.argv)
        glutInitDisplayMode(GLUT_RGBA | GLUT_DEPTH | GLUT_DOUBLE)
        glutInitWindowPosition((glutGet(GLUT_SCREEN_WIDTH) - self.window_width)/2,
                               (glutGet(GLUT_SCREEN_HEIGHT) - self.window_height)/2)
        glutSetOption(GLUT_ACTION_ON_WINDOW_CLOSE,
                      GLUT_ACTION_GLUTMAINLOOP_RETURNS)
        glutInitWindowSize(self.window_width, self.window_height)
        self.window_handle = glutCreateWindow(WINDOW_TITLE_PREFIX)
        glutDisplayFunc(self.render_func)
        glutReshapeFunc(self.resize_func)
        glutKeyboardFunc(self.keyboard)
        glutSpecialFunc(self.keyboard_s)
        glutIdleFunc(self.idle)
        glutTimerFunc(0, self.timer_func, 0)


    def init_gl(self):
        glEnable(GL_DEPTH_TEST)
        glClearColor(0.0, 0.0, 0.0, 1.0)
        glActiveTexture(GL_TEXTURE0)
        self.textures['star'] = self.load_texture(os.path.join(file_path,
                                                               'glow.png'))
        glEnable(GL_TEXTURE_2D)
        self.adjust_zoom()


    def output_ppm_stream(self):
        screenshot = glReadPixels(0, 0, WINDOW_WIDTH, WINDOW_HEIGHT,
                                  GL_RGBA, GL_UNSIGNED_BYTE)
        im = Image.frombuffer('RGBA', (WINDOW_WIDTH, WINDOW_HEIGHT),
                              screenshot, 'raw', 'RGBA', 0, 1)
#        im.resize((320,240))
        im.save(sys.stdout, format='ppm')


    def render_func(self):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        glTranslatef(0.0, 0.0, -10000.05)
        self.adjust_zoom()
        self.rotate(1)

        self.draw_system()

        glutSwapBuffers()
        glutPostRedisplay()
        if RECORD:
            self.output_ppm_stream()
        self.frame_count += 1


    def draw_system(self):
#        from pynbody.models import (IMF, Plummer)
#        imf = IMF.padoan2007(0.075, 120.0)
#        p = Plummer(2048, imf, epsf=4.0)
#        p.make_plummer()
#        self.particle = p._body

#        points = [[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0],
#                  [0.0, 1.0, 0.0], [0.0, -1.0, 0.0]]

        points = self.particle.pos

        r = self.particle.ekin()/self.particle.mass
        g = np.sqrt((self.particle.acc**2).sum(1))
        b = -self.particle.epot()/self.particle.mass
        r /= r.max()
        g /= g.max()
        b /= b.max()
        a = ((r+g+b)/3.0).max()
        r /= a
        g /= a
        b /= a
        colors = np.vstack((r,g,b)).T


#        print(points.shape, colors.shape)


#        glPushMatrix()

#        glEnable(GL_TEXTURE_2D)
#        glActiveTexture(GL_TEXTURE0)
#        glBindTexture(GL_TEXTURE_2D, self.textures['star'])
#        glEnable(GL_VERTEX_PROGRAM_POINT_SIZE)
#        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE)
        glEnable(GL_BLEND)
        glDepthMask(GL_FALSE)

#        glUseProgram(self.shader_program)

        self.draw_points(points, colors)
#        self.draw_quads(points, colors)

#        glUseProgram(0)

        glDepthMask(GL_TRUE)
        glDisable(GL_BLEND)
#        glDisable(GL_VERTEX_PROGRAM_POINT_SIZE)
#        glBindTexture(GL_TEXTURE_2D, 0)
#        glDisable(GL_TEXTURE_2D)

#        glPopMatrix()



    def set_point_size(self):
        global POINT_SIZE
        (ps_min, ps_max) = glGetFloatv(GL_ALIASED_POINT_SIZE_RANGE)
        if (POINT_SIZE > ps_max):
            POINT_SIZE = ps_max
        if (POINT_SIZE < ps_min):
            POINT_SIZE = ps_min
        glPointSize(POINT_SIZE)


    def draw_points(self, points, colors):
#        texVertices = [(0, 0), (0, 1), (1, 1), (1, 0)]

        self.set_point_size()

        glVertexPointerd(points)
        glEnableClientState(GL_VERTEX_ARRAY)

#        glTexCoordPointerd(texVertices)
#        glEnableClientState(GL_TEXTURE_COORD_ARRAY)

        glColorPointerd(colors)
        glEnableClientState(GL_COLOR_ARRAY)

        glEnable(GL_POINT_SPRITE)
        glTexEnvf(GL_POINT_SPRITE, GL_COORD_REPLACE, GL_TRUE)

        glDrawArrays(GL_POINTS, 0, len(points))

        glTexEnvf(GL_POINT_SPRITE, GL_COORD_REPLACE, GL_FALSE)
        glDisable(GL_POINT_SPRITE)

        glDisableClientState(GL_COLOR_ARRAY)
        glColorPointerd(None)

#        glDisableClientState(GL_TEXTURE_COORD_ARRAY)
#        glTexCoordPointerd(None)

        glDisableClientState(GL_VERTEX_ARRAY)
        glVertexPointerd(None)





    def draw_quads(self, points, colors):
        for p,c in zip(points, colors):
            x = p[0]
            y = p[1]
            z = p[2]
            size = 0.0001 * POINT_SIZE**2

            glMatrixMode(GL_MODELVIEW)
            glPushMatrix()

            glTranslatef(x, y, z)

            glBegin(GL_QUADS)

            glColor3d(c[0], c[1], c[2])

            glTexCoord2d(0, 0)
            glVertex3d(-size, +size, 0.0)
            glTexCoord2d(0, 1)
            glVertex3d(+size, +size, 0.0)
            glTexCoord2d(1, 1)
            glVertex3d(+size, -size, 0.0)
            glTexCoord2d(1, 0)
            glVertex3d(-size, -size, 0.0)

            glEnd()


#            glColor3d(c[0], c[1], c[2])
#            glutSolidSphere(size, 32, 32)


            glPopMatrix()




    def load_texture(self, name):
#        X, Y = np.mgrid[0.0:513.0, 0.0:513.0]
#        z = gaussian(8.0, 256.0, 256.0, 16.0, 16.0)
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
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, ix, iy, 0,
                     GL_RGBA, GL_UNSIGNED_BYTE, image)


#        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP)
#        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP)
#        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
#        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)


#        glTexParameterf(GL_TEXTURE_2D, GL_GENERATE_MIPMAP, GL_TRUE)
#        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER,
#                                       GL_LINEAR_MIPMAP_LINEAR)
#        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)


        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)

#        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
#        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)


#        glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_DECAL)
#        glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_MODULATE)
#        glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE)
#        glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_COMBINE)


        image = None

        return id








from pynbody.models import (IMF, Plummer)
imf = IMF.padoan2007(0.075, 120.0)
p = Plummer(2048, imf, epsf=4.0, seed=1)
p.make_plummer()
#p.show()

viewer = GLviewer()
viewer.update(p._body)

#for i in range(100):
#    viewer.update(p._body)
#    viewer.particle = p._body[:-128]
#    glutMainLoop()




########## end of file ##########
