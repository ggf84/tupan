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

WINDOW_TITLE_PREFIX = 'PyNbody Viewer'
#file_path = os.path.dirname(__file__) + os.sep + 'textures'
file_path = 'textures'



ROTINC = 0.05













class GLviewer(object):
    """

    """
    def __init__(self, window_width=640, window_height=480):
        self.window_width = window_width
        self.window_height = window_height
        self.window_handle = None
        self.frame_number = 0
        self.frame_count = 0
        self.textures = {}
        self.rot_x = 0.0
        self.rot_y = 0.0
        self.rotate_x = 0
        self.rotate_y = 0
        self.particle = None
        self.initialize()
#        glutMainLoop()


    def update(self, particle):
        self.particle = particle
        glutMainLoop()
        print('after close...')


    def keyboard(self, key, x, y):
        key = string.upper(key)
        if key == ' ':
            self.rotate_x = 0
            self.rotate_y = 0
        elif key == ESCAPE:
            glutDestroyWindow(self.window_handle)


    def keyboard_s(self, key, x, y):
        mkey = glutGetModifiers()
        if key == GLUT_KEY_UP:
            if mkey == GLUT_ACTIVE_CTRL:
                self.rotate_x -= 1
            elif mkey == GLUT_ACTIVE_ALT:
                pass
            else:
                pass
        elif key == GLUT_KEY_DOWN:
            if mkey == GLUT_ACTIVE_CTRL:
                self.rotate_x += 1
            elif mkey == GLUT_ACTIVE_ALT:
                pass
            else:
                pass
        elif key == GLUT_KEY_LEFT:
            if mkey == GLUT_ACTIVE_CTRL:
                self.rotate_y -= 1
            elif mkey == GLUT_ACTIVE_ALT:
                pass
            else:
                pass
        elif key == GLUT_KEY_RIGHT:
            if mkey == GLUT_ACTIVE_CTRL:
                self.rotate_y += 1
            elif mkey == GLUT_ACTIVE_ALT:
                pass
            else:
                pass


    def rotate(self, sig):
        self.rot_x += self.rotate_x * ROTINC
        self.rot_y += self.rotate_y * ROTINC
        if self.rot_x > 359.0:
            self.rot_x -= 359.0
        if self.rot_y > 359.0:
            self.rot_y -= 359.0
        if sig > 0:
            glRotatef(self.rot_x, 1.0, 0.0, 0.0)
            glRotatef(self.rot_y, 0.0, 1.0, 0.0)
        if sig < 0:
            glRotatef(-self.rot_y, 0.0, 1.0, 0.0)
            glRotatef(-self.rot_x, 1.0, 0.0, 0.0)


    def idle(self):
        glutPostRedisplay()


    def timer_function(self, value):
        if value != 0:
            tmp = ('{0}: {1} Frames Per'
                   ' Second @ {2} x {3}').format(WINDOW_TITLE_PREFIX,
                                                 self.frame_count * 4,
                                                 self.window_width,
                                                 self.window_height)
            glutSetWindowTitle(tmp)
            tmp = None

        self.frame_count = 0
        glutTimerFunc(250, self.timer_function, 1)


    def init_window(self):
        glutInit (sys.argv)
        glutInitDisplayMode(GLUT_DEPTH | GLUT_DOUBLE | GLUT_RGBA)
        glutSetOption(GLUT_ACTION_ON_WINDOW_CLOSE,
                      GLUT_ACTION_GLUTMAINLOOP_RETURNS)
        glutInitWindowSize(self.window_width, self.window_height)
        self.window_handle = glutCreateWindow(WINDOW_TITLE_PREFIX)
        glutDisplayFunc(self.render_func)
        glutReshapeFunc(self.resize_func)
        glutKeyboardFunc(self.keyboard)
        glutSpecialFunc(self.keyboard_s)
        glutIdleFunc(self.idle)
        glutTimerFunc(0, self.timer_function, 0)


    def init_gl(self):
#        glActiveTexture(GL_TEXTURE0)
        self.textures['star'] = self.load_texture(os.path.join(file_path,
                                                               'star.png'))
#        glEnable(GL_TEXTURE_2D)

#        glActiveTexture(GL_TEXTURE1)
        self.textures['bh'] = self.load_texture(os.path.join(file_path,
                                                             'bh.png'))
#        glEnable(GL_TEXTURE_2D)

        glClearColor(0.0, 0.0, 0.0, 1.0)
        glClearDepth(1.0)
        glDepthMask(GL_FALSE)
        glDepthFunc(GL_LESS)
        glDisable(GL_DEPTH_TEST)
        glDisable(GL_CULL_FACE)
        glEnable(GL_BLEND)
        glShadeModel(GL_SMOOTH)
        glHint(GL_PERSPECTIVE_CORRECTION_HINT, GL_NICEST)
        glHint(GL_POINT_SMOOTH_HINT, GL_NICEST)

#        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        glBlendFunc(GL_SRC_ALPHA, GL_ONE)

        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        ratio = float(self.window_width)/float(self.window_height)
        gluPerspective(0.1, ratio, 0.1, 20000.0)
        glMatrixMode(GL_MODELVIEW)


    def initialize(self):
        self.init_window()
        print('INFO: OpenGL Version: {0}'.format(glGetString(GL_VERSION)))
        self.init_gl()


    def resize_func(self, width, height):
        if height == 0:
            height = 1
        self.window_width = width
        self.window_height = height
        glViewport(0, 0, width, height)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        ratio = float(width)/float(height)
        gluPerspective(0.1, ratio, 0.1, 20000.0)
        glMatrixMode(GL_MODELVIEW)


    def render_func(self):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()
        glTranslatef(0.0, 0.0, -10000.05)

#        glColor4d(0.125, 0.0, 0.25, 1.0)
#        glutSolidSphere(4.0, 32, 2)

        self.rotate(+1)
        self.draw_system()
        glutSwapBuffers()
        glutPostRedisplay()
#        self.take_screenshot()
        self.frame_number += 1
        self.frame_count += 1


    def draw_system(self):
#        from pynbody.models import (IMF, Plummer)
#        imf = IMF.padoan2007(0.075, 120.0)
#        p = Plummer(1024, imf, epsf=4.0, seed=1)
#        p.make_plummer()
#        self.particle = p._body

#        if self.particle is not None:
#            glTexEnvfv(GL_TEXTURE_ENV, GL_TEXTURE_ENV_COLOR, (1, 1, 1, 1))
#            for p in self.particle.pos:
#                self.draw_point(p, 0.0625, GL_TEXTURE0)

#        points = [[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0],
#                  [0.0, 1.0, 0.0], [0.0, -1.0, 0.0]]

        points = self.particle.pos
#        rgb = np.array([[72.0,24.0,8.0]])
        rgb = np.array([[120.0,24.0,6.0]])
        colors = ((self.particle.mass) * rgb.T).T
        colors /= colors.mean()



        glPushMatrix()
        glEnableClientState(GL_VERTEX_ARRAY)
        glEnableClientState(GL_COLOR_ARRAY)

        glVertexPointerd(points)
        glColorPointerd(colors)

        glTexEnvf(GL_POINT_SPRITE_ARB, GL_COORD_REPLACE_ARB, GL_TRUE)
        glEnable(GL_POINT_SPRITE_ARB)
        glEnable(GL_POINT_SMOOTH)
        glEnable(GL_BLEND)
#        glEnable(GL_TEXTURE_2D)
#        glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_COMBINE)
#        glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE)
#        glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_MODULATE)
#        glBindTexture(GL_TEXTURE_2D, self.textures['star'])
        glPointSize(1.0)

#        self.rotate(+1)
        glDrawArrays(GL_POINTS, 0, len(points))

#        glDisable(GL_TEXTURE_2D)
        glDisable(GL_BLEND)
        glDisable(GL_POINT_SMOOTH)
        glDisable(GL_POINT_SPRITE_ARB)
        glTexEnvf(GL_POINT_SPRITE_ARB, GL_COORD_REPLACE_ARB, GL_FALSE)

        glDisableClientState(GL_COLOR_ARRAY)
        glDisableClientState(GL_VERTEX_ARRAY)
        glPopMatrix()



    def draw_point (self, point, size, texture):
        x = point[0];   y = point[1];   z = point[2]

        glMatrixMode(GL_MODELVIEW)
        glPushMatrix()

        self.rotate(+1)
        glTranslatef(x, y, z)
        self.rotate(-1)
        glTranslatef(-x, -y, -z)

        glBegin(GL_QUADS)
        glMultiTexCoord2d(texture, 0, 0); glVertex3f(x-size, y+size, z)
        glMultiTexCoord2d(texture, 0, 1); glVertex3f(x+size, y+size, z)
        glMultiTexCoord2d(texture, 1, 1); glVertex3f(x+size, y-size, z)
        glMultiTexCoord2d(texture, 1, 0); glVertex3f(x-size, y-size, z)
        glEnd()

        glPopMatrix()


    def take_screenshot(self):
        screenshot = glReadPixels(0, 0, 640, 480, GL_RGBA, GL_UNSIGNED_BYTE)
        im = Image.frombuffer("RGBA", (640,480), screenshot, "raw", "RGBA", 0, 0)
        im.save('frame{0}.png'.format(str(self.frame_number).zfill(4)))


    def load_texture(self, name):
        try:
            image = Image.open(name)
        except:
            raise RuntimeError('Failed to load \'{0}\' texture'.format(name))

        ix = image.size[0]
        iy = image.size[1]
        image = image.tostring("raw", "RGBX", 0, -1)

        # Create Texture
        id = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, id)

        glPixelStorei(GL_UNPACK_ALIGNMENT,1)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, ix, iy, 0,
                     GL_RGBA, GL_UNSIGNED_BYTE, image)
#        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP)
#        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP)
#        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
#        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
#        glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_DECAL)

        image = None

        return id








from pynbody.models import (IMF, Plummer)
imf = IMF.padoan2007(0.075, 120.0)
p = Plummer(1024, imf, epsf=4.0, seed=1)
p.make_plummer()
#p.show()

viewer = GLviewer()
viewer.update(p._body)

#for i in range(100):
#    viewer.update(p._body)
#    viewer.particle = p._body[:-128]
#    glutMainLoop()































#            coords = self.particle.pos
#            glBindBuffer(GL_ARRAY_BUFFER, coords)
#            glBufferData(GL_ARRAY_BUFFER,
#                         ADT.arrayByteCount(coords),
#                         ADT.voidDataPointer(coords),
#                         GL_STATIC_DRAW_ARB)
#            glVertexPointer(3, GL_DOUBLE, 0, None)
#            glDrawArrays(GL_QUADS, 0, coords.shape[0])


#            glPointSize(2)
##            glColor3d(0.0, 1.0, 1.0)
#            colors = np.ones(3*len(coords)).reshape((-1, 3)).astype(np.float32)
#            col_vbo = vbo.VBO(data=colors,
#                              usage=GL_DYNAMIC_DRAW,
#                              target=GL_ARRAY_BUFFER)
#            col_vbo.bind()
#            pos_vbo = vbo.VBO(data=coords,
#                              usage=GL_DYNAMIC_DRAW,
#                              target=GL_ARRAY_BUFFER)
#            pos_vbo.bind()
#            glColorPointer(3, GL_FLOAT, 0, col_vbo)
#            glVertexPointer(3, GL_DOUBLE, 0, pos_vbo)
#            glEnableClientState(GL_COLOR_ARRAY)
#            glEnableClientState(GL_VERTEX_ARRAY)
#            glDrawArrays(0, 0, len(coords))
#            glDisableClientState(GL_VERTEX_ARRAY)
#            glDisableClientState(GL_COLOR_ARRAY)


#            glVertexPointerd(coords)
#            glTexEnvfv(GL_TEXTURE_ENV, GL_TEXTURE_ENV_COLOR, (1, 1, 1, 1))
##            glColor3d(0.0, 1.0, 1.0)
#            glEnableClientState(GL_VERTEX_ARRAY)
#            glDrawArrays(GL_POINTS, 0, len(coords))
#            glDisableClientState(GL_VERTEX_ARRAY)



#            vertex_vbo = GLuint(0)
#            glGenBuffers(1, vertex_vbo)
#            glBindBuffer(GL_ARRAY_BUFFER, coords)
#            glBufferData(GL_ARRAY_BUFFER, ADT.arrayByteCount(coords), ADT.voidDataPointer(coords), GL_STATIC_DRAW_ARB)
#            glBindBuffer(GL_ARRAY_BUFFER, vertex_vbo)
#            glVertexPointer(3, GL_DOUBLE, 0, None)
#            glDrawArrays(GL_QUADS, 0, coords.shape[0])




########## end of file ##########
