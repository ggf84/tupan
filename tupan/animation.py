# -*- coding: utf-8 -*-
#

"""
TODO.
"""

import logging
import subprocess
import numpy as np
from PIL import Image
from matplotlib import cm
from vispy import gloo
from vispy import app
from vispy.util.transforms import ortho, translate, scale, rotate


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

%s  // declare auxiliary functions

// Main
// ------------------------------------
void main (void) {
    v_color  = a_color;
    v_psize  = a_psize * u_psize;
    gl_Position = u_projection * u_view * u_model * vec4(a_position, 1.0);
    gl_PointSize = v_psize;
}
""" % ""

ALPHA_FUNCTION = """
float get_alpha(float r, float a, float b) {
    return b * (1 - r) / (a + r);
}
"""

FRAG_SHADER0 = """
#version 120
// Varyings
// ------------------------------------
varying vec4  v_color;
varying float v_psize;

%s  // declare auxiliary functions

// Main
// ------------------------------------
void main()
{
    float r = length(2 * gl_PointCoord.xy - vec2(1, 1));
    if (r > 1) discard;  // kill pixels outside circle
    float alpha = get_alpha(r, 0.0625, 0.0625);
    gl_FragColor = vec4(v_color.rgb * (1 + alpha), alpha);
}
""" % ALPHA_FUNCTION

FRAG_SHADER1 = """
#version 120
// Varyings
// ------------------------------------
varying vec4  v_color;
varying float v_psize;

%s  // declare auxiliary functions

// Main
// ------------------------------------
void main()
{
    float r = length(2 * gl_PointCoord.xy - vec2(1, 1));
    if (r > 1) discard;  // kill pixels outside circle
    float alpha = get_alpha(r, 64, 1);
    gl_FragColor = vec4(v_color.rgb, alpha);
}
""" % ALPHA_FUNCTION

FRAG_SHADER2 = """
#version 120
// Varyings
// ------------------------------------
varying vec4  v_color;
varying float v_psize;

%s  // declare auxiliary functions

// Main
// ------------------------------------
void main()
{
    float r = length(2 * gl_PointCoord.xy - vec2(1, 1));
    if (r > 1) discard;  // kill pixels outside ring
    float r1 = r - 0.5;
    float r2 = r - 0.25;
    r = abs(max(r1, r2)) * v_psize / 8;
    float alpha = exp(-8 * r * r);
//    float alpha = get_alpha(3*r, 1, 20);
    gl_FragColor = vec4(alpha, 0, 0, alpha);
}
""" % ALPHA_FUNCTION


class GLviewer(app.Canvas):
    """
    TODO.
    """
    def __init__(self):
        width = 768
        aspect = 16.0 / 9.0
        size = width, int(width / aspect)
        super(GLviewer, self).__init__(keys='interactive')
        self.size = size
        self.aspect = aspect

        self.ps = None
        self.data = {}
        self.vdata = {}
        self.program = {}
        self.program['body'] = gloo.Program(VERT_SHADER, FRAG_SHADER0)
        self.program['star'] = gloo.Program(VERT_SHADER, FRAG_SHADER0)
        self.program['sph'] = gloo.Program(VERT_SHADER, FRAG_SHADER1)
        self.program['blackhole'] = gloo.Program(VERT_SHADER, FRAG_SHADER2)
        self.model = np.eye(4, dtype=np.float32)
        self.projection = np.eye(4, dtype=np.float32)
        self.psize = 20

        self.translate = [-0.0, -0.0, -0.0]
        self.view = translate(self.translate)

        for prog in self.program.values():
            prog['u_psize'] = self.psize
            prog['u_model'] = self.model
            prog['u_view'] = self.view

        def callback(fps):
            titlestr = "tupan viewer: %0.1f fps @ %d x %d"
            self.title = titlestr % (fps, self.size[0], self.size[1])
        self.measure_fps(callback=callback)
        self.show(True)
        self.is_visible = True

        self.ffwriter = None
        self.make_movie = False

    def record_screen(self):
        if not self.ffwriter:
            cmdline = [
                "ffmpeg", "-y",
                "-r", "60",
                "-f", "image2pipe",
                "-vcodec", "ppm",
                "-i", "-",
                "-q", "1",
                "-vcodec", "mpeg4",
                "movie.mp4",
            ]
            self.ffwriter = subprocess.Popen(cmdline,
                                             stdin=subprocess.PIPE,
                                             stdout=subprocess.PIPE,
                                             stderr=subprocess.PIPE)

        im = gloo.read_pixels()
        im = Image.frombuffer('RGBA', self.size, im.tostring(),
                              'raw', 'RGBA', 0, 1)
        im.save(self.ffwriter.stdin, format='ppm')

    def on_initialize(self, event):
        gloo.set_state(depth_test=True, blend=True, clear_color='black')

    def on_key_press(self, event):
        if event.key == '+':
            self.psize += 1
        elif event.key == '-':
            self.psize -= 1
        elif event.key == 'Up':
            self.model = np.dot(self.model, rotate(+1, [1, 0, 0]))
        elif event.key == 'Down':
            self.model = np.dot(self.model, rotate(-1, [1, 0, 0]))
        elif event.key == 'Right':
            self.model = np.dot(self.model, rotate(+1, [0, 1, 0]))
        elif event.key == 'Left':
            self.model = np.dot(self.model, rotate(-1, [0, 1, 0]))
        elif event.key == '>':
            self.model = np.dot(self.model, rotate(+1, [0, 0, 1]))
        elif event.key == '<':
            self.model = np.dot(self.model, rotate(-1, [0, 0, 1]))
        elif event.text == 'm':
            self.make_movie = not self.make_movie
        elif event.text == 'z':
            self.model = np.dot(self.model, scale([1/1.03125]*3))
        elif event.text == 'Z':
            self.model = np.dot(self.model, scale([1*1.03125]*3))
        elif event.key == 'Escape':
            self.is_visible = False
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
            self.model = np.dot(self.model, scale([1/1.03125]*3))
        if event.delta[1] < 0:
            self.model = np.dot(self.model, scale([1*1.03125]*3))
        for prog in self.program.values():
            prog['u_model'] = self.model
        self.update()

    def on_mouse_move(self, event):
        x, y = event.pos
        w, h = self.size
        if event.is_dragging:
            dx = (2 * x / float(w) - 1) * self.aspect
            dy = (1 - 2 * y / float(h))
            if event.button == 1:
                self.translate[0] = 5 * dx
                self.translate[1] = 5 * dy
                self.view = translate(self.translate)
                for prog in self.program.values():
                    prog['u_view'] = self.view
            if event.button == 2:
                self.model = np.dot(self.model,
                                    np.dot(rotate(-dx, [0, 1, 0]),
                                           rotate(+dy, [1, 0, 0])))
                for prog in self.program.values():
                    prog['u_model'] = self.model
        self.update()

    def on_draw(self, event):
        gloo.clear()

        if 'body' in self.data:
            gloo.set_depth_mask(False)
            gloo.set_state(blend_func=('src_alpha', 'one'))
            self.program['body'].draw('points')
            gloo.set_depth_mask(True)

        if 'star' in self.data:
            gloo.set_depth_mask(False)
            gloo.set_state(blend_func=('src_alpha', 'one'))
            self.program['star'].draw('points')
            gloo.set_depth_mask(True)

        if 'sph' in self.data:
            gloo.set_depth_mask(False)
            gloo.set_state(blend_func=('src_alpha', 'one'))
            self.program['sph'].draw('points')
            gloo.set_depth_mask(True)

        if 'blackhole' in self.data:
            gloo.set_depth_mask(False)
            gloo.set_state(blend_func=('dst_alpha', 'one_minus_src_alpha'))
            self.program['blackhole'].draw('points')
            gloo.set_depth_mask(True)

        if self.make_movie:
            self.record_screen()

    def init_vertex_buffers(self):
        for name, member in self.ps.members.items():
            n = member.n
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
            self.ps[np.in1d(self.ps.pid, ps.pid)] = ps
        for name, member in self.ps.members.items():
            pos, c, s = member.pos, member.mass, member.mass

            def f(arg): return np.log(arg)
#            def f(arg): return arg**(1/3.0)

            cmax, cmin = c.max(), c.min()
            c = (f(c / cmin) / f(cmax / cmin)) if cmax > cmin else (c / cmax)

            smax, smin = s.max(), s.min()
            s = (f(s / smin) / f(smax / smin)) if smax > smin else (s / smax)

            self.data[name]['a_position'][...] = pos.T
            self.data[name]['a_color'][...] = cm.cubehelix(c)
            self.data[name]['a_psize'][...] = s
            self.vdata[name].set_data(self.data[name])
        self.app.process_events()
        self.update()

    def enter_main_loop(self):
        if self.is_visible:
            self.app.run()


# -- End of File --
