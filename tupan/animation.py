# -*- coding: utf-8 -*-
#

"""
TODO.
"""

import logging
import subprocess
import numpy as np
from vispy import gloo
from vispy import app
from vispy.util.transforms import ortho, translate, scale, rotate
from .config import cli


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
attribute vec3  a_color;
attribute float a_psize;
attribute int a_pid;

// Varyings
// ------------------------------------
varying vec3  v_color;
varying float v_psize;

// Main
// ------------------------------------
void main(void) {
    v_color  = a_color;
    v_psize  = a_psize * u_psize;
    gl_PointSize = v_psize;
    gl_Position = u_projection * u_view * u_model * vec4(a_position, 1);
}
"""

FRAG_SHADER0 = """
#version 120
// Varyings
// ------------------------------------
varying vec3  v_color;
varying float v_psize;

// Main
// ------------------------------------
void main()
{
    float d = length(2 * gl_PointCoord.xy - vec2(1, 1));
    if (d > 1) discard;
    float n = 3;
    d = n * sqrt(d);
    d = n * (n - d)/(1 + d * d);
    gl_FragColor = vec4(v_color * d, 1);
}
"""

FRAG_SHADER1 = """
#version 120
// Varyings
// ------------------------------------
varying vec3  v_color;
varying float v_psize;

// Main
// ------------------------------------
void main()
{
    float d = length(2 * gl_PointCoord.xy - vec2(1, 1));
    if (d > 1) discard;
    float n = 1;
    d = n * sqrt(d);
    d = n * (n - d)/(1 + d * d);
    gl_FragColor = vec4(v_color * d, 1);
}
"""

FRAG_SHADER2 = """
#version 120
// Varyings
// ------------------------------------
varying vec3  v_color;
varying float v_psize;

// Main
// ------------------------------------
void main()
{
    float d = length(2 * gl_PointCoord.xy - vec2(1, 1));
    if (d > 1) discard;
    d = 2 * d - 2;
    d = max(d - 1, d + 1) * v_psize / 4;
    d = exp(- d * d);
    gl_FragColor = vec4(d, 0, 0, 1);
}
"""


class GLviewer(app.Canvas):
    """
    TODO.
    """
    def __init__(self, *args, **kwargs):
        height = 48 * 9
        aspect = 16.0 / 9
        size = int(height * aspect), height
        super(GLviewer, self).__init__(keys='interactive')
        self.size = size
        self.aspect = aspect

        self.mmin = {}
        self.mmax = {}
        self.data = {}
        self.vdata = {}
        self.program = {}
        self.program['body'] = gloo.Program(VERT_SHADER, FRAG_SHADER0)
        self.program['star'] = gloo.Program(VERT_SHADER, FRAG_SHADER0)
        self.program['sph'] = gloo.Program(VERT_SHADER, FRAG_SHADER1)
        self.program['blackhole'] = gloo.Program(VERT_SHADER, FRAG_SHADER2)
        self.model = np.eye(4, dtype=np.float32)
        self.projection = np.eye(4, dtype=np.float32)
        self.psize = 10

        self.translate = [0.0, 0.0, 0.0]
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

    def record_screen(self):
        try:
            im = gloo.read_pixels()
            self.ffwriter.stdin.write(im.tostring())
        except AttributeError:
            cmdline = [
                "ffmpeg", "-y",
                "-f", "rawvideo",
                "-pix_fmt", "rgba",
                "-s", "{0}x{1}".format(*self.size),
                "-r", "{}".format(cli.record),
                "-i", "-",
                "-an",  # no audio
                "-pix_fmt", "yuv420p",
                "-c:v", "libx265",
                "-b:v", "300000k",
                "movie.mkv",
            ]
            self.ffwriter = subprocess.Popen(
                cmdline,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            self.record_screen()

    def on_initialize(self, event):
        gloo.set_state(
            preset='additive',
            clear_color='black',
        )

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

        for key in self.data:
            self.program[key].draw('points')

        if cli.record:
            self.record_screen()

    def init_vertex_buffers(self, ps):
        for name, member in ps.members.items():
            n = member.n
            pid = member.pid
            mass = member.mass

            self.mmin[name] = mass.min()
            self.mmax[name] = mass.max()

            self.data[name] = np.zeros(n, [('a_position', np.float32, 3),
                                           ('a_color',    np.float32, 3),
                                           ('a_psize', np.float32, 1),
                                           ('a_pid', np.int32, 1)])

            self.data[name]['a_pid'][...] = pid

            self.vdata[name] = gloo.VertexBuffer(self.data[name])
            self.program[name].bind(self.vdata[name])

    def show_event(self, ps):
        def f(arg): return np.log2(1 + arg)

        if not self.data:
            self.init_vertex_buffers(ps)

        for name, member in ps.members.items():
            pid = member.pid
            mass = member.mass
            pos = member.rdot[0].T

            mmin, mmax = self.mmin[name], self.mmax[name]
            a = f(mass / mmin) / f(mmax / mmin)

            s = 1 * a**0.25

            q = 5
            qs = [q**i for i in range(3)]
            r, g, b = [i * a**i for i in qs]
            c = np.array([r, g, b]).T
            c /= sum(qs)

            mask = Ellipsis
            if len(self.data[name]['a_pid']) != len(pid):
                mask = np.in1d(self.data[name]['a_pid'], pid, assume_unique=True)

            self.data[name]['a_position'][mask] = pos
            self.data[name]['a_color'][mask] = c
            self.data[name]['a_psize'][mask] = s
            self.data[name]['a_pid'][mask] = pid

            self.vdata[name].set_data(self.data[name])

        self.app.process_events()
        self.update()

    def __enter__(self):
        LOGGER.debug(type(self).__name__+'.__enter__')
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        LOGGER.debug(type(self).__name__+'.__exit__')
        if self.is_visible:
            self.app.run()
        if hasattr(self, 'ffwriter'):
            self.ffwriter.stdin.close()


# -- End of File --
