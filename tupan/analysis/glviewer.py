# -*- coding: utf-8 -*-
#

"""
TODO.
"""

import logging
import numpy as np
from matplotlib import cm
from vispy import gloo
from vispy import app
from vispy.util.transforms import ortho, translate, scale
from vispy.util.transforms import xrotate, yrotate, zrotate


LOGGER = logging.getLogger(__name__)


VERT_SHADER = """
#version 120
// Uniforms
// ------------------------------------
uniform mat4  u_model;
uniform mat4  u_view;
uniform mat4  u_projection;
uniform float u_psize;

// Attributes
// ------------------------------------
attribute vec3  a_position;
attribute vec4  a_color;
attribute float a_psize;

// Varyings
// ------------------------------------
varying vec4  v_color;
varying float v_psize;

// Main
// ------------------------------------
void main (void) {
    v_color  = a_color;
    v_psize  = a_psize * u_psize;
    gl_Position = u_projection * u_view * u_model * vec4(a_position, 1.0);
    gl_PointSize = v_psize;
}
"""

FRAG_SHADER0 = """
#version 120
// Varyings
// ------------------------------------
varying vec4  v_color;
varying float v_psize;

// Main
// ------------------------------------
void main()
{
    float r = length(gl_PointCoord.xy - vec2(0.5f, 0.5f));
    if (r > 0.5f) discard;  // kill pixels outside circle
    float alpha = exp(-16*r*r);
    float core = exp(-256*r*r);
    gl_FragColor = vec4(v_color.rgb * (1 + core), (alpha + 7*core)/8);
}
"""

FRAG_SHADER1 = """
#version 120
// Varyings
// ------------------------------------
varying vec4  v_color;
varying float v_psize;

// Main
// ------------------------------------
void main()
{
    float r = length(gl_PointCoord.xy - vec2(0.5f, 0.5f));
    if (r > 0.5f) discard;  // kill pixels outside circle
    float alpha = exp(-16*r*r);
    gl_FragColor = vec4(v_color.rgb, alpha);
}
"""

FRAG_SHADER2 = """
#version 120
// Varyings
// ------------------------------------
varying vec4  v_color;
varying float v_psize;

// Main
// ------------------------------------
void main()
{
    float r = length(gl_PointCoord.xy - vec2(0.5f, 0.5f));
    float alpha = 1 - 8 * r;
    if (abs(alpha) > 0.36f) discard;  // kill pixels outside circle
    alpha = exp(-16*alpha*alpha);
    gl_FragColor = vec4(1, 0, 0, alpha);
}
"""


class GLviewer(app.Canvas):

    def __init__(self):
        app.Canvas.__init__(self, keys='interactive')
        width = 768
        self.aspect = 16.0 / 9.0
        self.size = width, int(width / self.aspect)
        self.titlestr = "tupan viewer: {fps:.1f} fps @ {width} x {height}"
        self.measure_fps(callback=lambda x: x)
        self.title = self.titlestr.format(fps=self.fps,
                                          width=self.size[0],
                                          height=self.size[1])

        self.ps = None
        self.data = {}
        self.vdata = {}
        self.program = {}
        self.program['bodies'] = gloo.Program(VERT_SHADER, FRAG_SHADER0)
        self.program['stars'] = gloo.Program(VERT_SHADER, FRAG_SHADER0)
        self.program['sphs'] = gloo.Program(VERT_SHADER, FRAG_SHADER1)
        self.program['blackholes'] = gloo.Program(VERT_SHADER, FRAG_SHADER2)
        self.view = np.eye(4, dtype=np.float32)
        self.model = np.eye(4, dtype=np.float32)
        self.projection = np.eye(4, dtype=np.float32)
        self.psize = 20

        self.translate = [-0.0, -0.0, -0.0]
        translate(self.view, *self.translate)

        for prog in self.program.values():
            prog['u_psize'] = self.psize
            prog['u_model'] = self.model
            prog['u_view'] = self.view

        self.show()

    def on_initialize(self, event):
        gloo.set_state(depth_test=True, blend=True, clear_color='black')

    def on_key_press(self, event):
        if event.key == '+':
            self.psize += 1
        elif event.key == '-':
            self.psize -= 1
        elif event.key == 'Up':
            xrotate(self.model, +1)
        elif event.key == 'Down':
            xrotate(self.model, -1)
        elif event.key == 'Right':
            yrotate(self.model, +1)
        elif event.key == 'Left':
            yrotate(self.model, -1)
        elif event.key == '>':
            zrotate(self.model, +1)
        elif event.key == '<':
            zrotate(self.model, -1)
        elif event.text == 'z':
            scale(self.model, 1/1.03125)
        elif event.text == 'Z':
            scale(self.model, 1*1.03125)
        for prog in self.program.values():
            prog['u_psize'] = self.psize
            prog['u_model'] = self.model
        self.update()

    def on_resize(self, event):
        w, h = event.size
        aspect = w / float(h)
        self.aspect = aspect
        gloo.set_viewport(0, 0, w, h)
        self.projection = ortho(-5*aspect, +5*aspect, -5, +5, -100.0, +100.0)
        for prog in self.program.values():
            prog['u_projection'] = self.projection

    def on_mouse_wheel(self, event):
        if event.delta[1] > 0:
            scale(self.model, 1/1.03125)
        if event.delta[1] < 0:
            scale(self.model, 1*1.03125)
        for prog in self.program.values():
            prog['u_model'] = self.model
        self.update()

    def on_mouse_move(self, event):
        x, y = event.pos
        w, h = self.size
        if event.is_dragging:
            dx = 5 * (2 * x / float(w) - 1) * self.aspect
            dy = 5 * (1 - 2 * y / float(h))
            if event.button == 1:
                self.translate[0] = dx
                self.translate[1] = dy
                self.view = np.eye(4, dtype=np.float32)
                translate(self.view, *self.translate)
                for prog in self.program.values():
                    prog['u_view'] = self.view
            if event.button == 2:
                xrotate(self.model, +dy)
                yrotate(self.model, -dx)
                for prog in self.program.values():
                    prog['u_model'] = self.model
        self.update()

    def on_draw(self, event):
        gloo.clear()

        if 'bodies' in self.data:
            gloo.set_depth_mask(False)
            gloo.set_state(blend_func=('src_alpha', 'one'))
            self.program['bodies'].draw('points')
            gloo.set_depth_mask(True)

        if 'stars' in self.data:
            gloo.set_depth_mask(False)
            gloo.set_state(blend_func=('src_alpha', 'one'))
            self.program['stars'].draw('points')
            gloo.set_depth_mask(True)

        if 'sphs' in self.data:
            gloo.set_depth_mask(False)
            gloo.set_state(blend_func=('src_alpha', 'one'))
            self.program['sphs'].draw('points')
            gloo.set_depth_mask(True)

        if 'blackholes' in self.data:
            gloo.set_state(blend_func=('src_alpha', 'zero'))
            self.program['blackholes'].draw('points')

        self.title = self.titlestr.format(fps=self.fps,
                                          width=self.size[0],
                                          height=self.size[1])

    def init_vertex_buffers(self):
        for member in self.ps.members:
            n = member.n
            name = member.name
            self.data[name] = np.zeros(n, [('a_position', np.float32, 3),
                                           ('a_color',    np.float32, 4),
                                           ('a_psize', np.float32, 1)])
            self.vdata[name] = gloo.VertexBuffer(self.data[name])
            self.program[name].bind(self.vdata[name])

    def show_event(self, ps):
        if self.ps is None:
            self.ps = ps.copy()
            self.init_vertex_buffers()
        else:
            self.ps[np.in1d(self.ps.id, ps.id)] = ps
        for member in self.ps.members:
            name, pos, c, s = member.name, member.pos, member.mass, member.mass
            c = (c/c.max())**(1/3.0)
            s = (s/s.max())**(1/4.0)
            self.data[name]['a_position'][...] = pos.T
            self.data[name]['a_color'][...] = cm.cubehelix(c)
            self.data[name]['a_psize'][...] = s
            self.vdata[name].set_data(self.data[name])
        app.process_events()
        self.update()

    def enter_main_loop(self):
        app.run()


# -- End of File --
