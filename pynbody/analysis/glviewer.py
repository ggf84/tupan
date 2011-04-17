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
WINDOW_WIDTH = 800
WINDOW_HEIGHT = 600

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
    return lambda x, y: height*np.exp(-(((center_x-x)/width_x)**2+((center_y-y)/width_y)**2)/2)







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
        self.rotate_x = 0
        self.rotate_y = 0
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
        if self.rot_x > 359:
            self.rot_x -= 359
        elif self.rot_x < -359:
            self.rot_x += 359
        if self.rot_y > 359:
            self.rot_y -= 359
        elif self.rot_y < -359:
            self.rot_y += 359
        if sign > 0:
            glRotatef(self.rot_x, 1.0, 0.0, 0.0)
            glRotatef(self.rot_y, 0.0, 1.0, 0.0)
        if sign < 0:
            glRotatef(-self.rot_y, 0.0, 1.0, 0.0)
            glRotatef(-self.rot_x, 1.0, 0.0, 0.0)


    def keyboard(self, key, x, y):
        global POINT_SIZE
        global ZOOM_FACTOR
        if key == ' ':
            self.rotate_x = 0
            self.rotate_y = 0
        elif key == '+':
            POINT_SIZE += 1
        elif key == '-':
            POINT_SIZE -= 1
            if POINT_SIZE < 1:  POINT_SIZE = 1
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
                self.rotate_x -= 1
            elif key == GLUT_KEY_DOWN:
                self.rotate_x += 1
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

        self.textures['star'] = self.load_texture(os.path.join(file_path,
                                                               'star.png'))

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
#        p = Plummer(1024, imf, epsf=4.0, seed=1)
#        p.make_plummer()
#        self.particle = p._body

#        points = [[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0],
#                  [0.0, 1.0, 0.0], [0.0, -1.0, 0.0]]

        points = self.particle.pos

        r = self.particle.ekin()/self.particle.mass
        g = -self.particle.epot()/self.particle.mass
        b = np.sqrt((self.particle.acc**2).sum(1))
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

        glEnable(GL_TEXTURE_2D)
#        glActiveTexture(GL_TEXTURE0)
#        glBindTexture(GL_TEXTURE_2D, self.textures['star'])
#        glEnable(GL_VERTEX_PROGRAM_POINT_SIZE_NV)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE)
        glEnable(GL_BLEND)
        glDepthMask(GL_FALSE)



#        glUseProgram(self.shader_program)

        self.draw_points(points, colors)
#        self.draw_quads(points, colors)

#        glUseProgram(0)

        glDepthMask(GL_TRUE)
        glDisable(GL_BLEND)
#        glDisable(GL_VERTEX_PROGRAM_POINT_SIZE_NV)
#        glBindTexture(GL_TEXTURE_2D, 0)
        glDisable(GL_TEXTURE_2D)

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
        glEnable(GL_POINT_SPRITE)
        glTexEnvi(GL_POINT_SPRITE, GL_COORD_REPLACE, GL_TRUE)

        self.set_point_size()

        glEnableClientState(GL_VERTEX_ARRAY)
        glVertexPointerd(points)

        glEnableClientState(GL_COLOR_ARRAY)
        glColorPointerd(colors)

        glDrawArrays(GL_POINTS, 0, len(points))

        glColorPointerd(None)
        glDisableClientState(GL_COLOR_ARRAY)

        glVertexPointerd(None)
        glDisableClientState(GL_VERTEX_ARRAY)

        glTexEnvi(GL_POINT_SPRITE, GL_COORD_REPLACE, GL_FALSE)
        glDisable(GL_POINT_SPRITE)




    def draw_quads(self, points, colors):
        for p,c in zip(points, colors):
            x = p[0]
            y = p[1]
            z = p[2]
            size = 0.0001 * POINT_SIZE**2 #* (c[0] + c[1] + c[2])

            glMatrixMode(GL_MODELVIEW)
            glPushMatrix()

            glTranslatef(x, y, z)


#            glColor3d(c[0], c[1], c[2])
#            glutSolidSphere(size, 32, 32)


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

            glPopMatrix()




    def load_texture(self, name):
        try:
            image = Image.open(name)
        except:
            raise RuntimeError('Failed to load \'{0}\' texture'.format(name))

        ix = image.size[0]
        iy = image.size[1]
        image = image.tostring('raw', 'RGBX', 0, -1)


#        x = np.linspace(-1.0,1.0,num=32)
#        y = np.linspace(-1.0,1.0,num=32)
#        X, Y = np.meshgrid(x, y)
#        z = gaussian(1.0, 0.0, 0.0, 0.1, 0.1)
#        image = z(X, Y)
#        ix, iy = image.shape
#        image = image.tostring()





        # Create Texture
        id = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, id)

        glTexParameterf(GL_TEXTURE_2D, GL_GENERATE_MIPMAP, GL_TRUE)
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER,
                                       GL_LINEAR_MIPMAP_LINEAR)
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)

#        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
#        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)

#        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
#        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)


        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP)
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP)


#        glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_DECAL)
#        glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_MODULATE)
#        glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE)
#        glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_COMBINE)


        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, ix, iy, 0,
                     GL_RGBA, GL_UNSIGNED_BYTE, image)


        image = None

        return id








from pynbody.models import (IMF, Plummer)
imf = IMF.padoan2007(0.075, 120.0)
p = Plummer(4096, imf, epsf=4.0, seed=1)
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
