# -*- coding: utf-8 -*-
#

"""
TODO.
"""

import logging
import subprocess
import numpy as np
from vispy import app
from vispy import gloo
from vispy import visuals
from vispy.util.transforms import perspective, translate, rotate
from .config import cli


LOGGER = logging.getLogger(__name__)


render_vertex = """
#version 120

attribute vec2 position;
attribute vec2 texcoord;
varying vec2 v_texcoord;

void main()
{
    gl_Position = vec4(position, 0.0, 1.0);
    v_texcoord = texcoord;
}
"""

render_fragment = """
#version 120

uniform sampler2D texture;
varying vec2 v_texcoord;

void main()
{
    vec4 c = texture2D(texture, v_texcoord);
    gl_FragColor = c;
}
"""


VERT_SHADER = """
#version 120
// Uniforms
// ------------------------------------
uniform float u_Mlim;
uniform float u_temp;
uniform float u_psize;
uniform float u_magnitude;
uniform float u_brightness;
uniform vec2  u_viewport;
uniform mat4  u_model;
uniform mat4  u_view;
uniform mat4  u_projection;

// Attributes
// ------------------------------------
attribute float a_temp;
attribute float a_psize;
attribute float a_magnitude;
attribute vec3  a_position;

// Varyings
// ------------------------------------
varying float v_temp;
varying float v_psize;
varying float v_brightness;
varying vec2  v_pcenter;


float log10(float arg)
{
    return log(arg) / log(10);
}


// Main
// ------------------------------------
void main(void)
{
    vec4 projPosition = u_projection * u_view * u_model * vec4(a_position, 1);
    vec2 ndcPosition = projPosition.xy / projPosition.w;
    v_pcenter = 0.5 * u_viewport * (ndcPosition + vec2(1, 1));

    float magnitude = a_magnitude + 5 * (log10(projPosition.w) - 1);

    v_psize = 0.5 * u_psize * a_psize;
//    v_psize = u_psize * sqrt(a_psize);
//    v_psize = u_psize * log(1 + 8 * a_psize) / log(1 + 8);
//    v_psize = 0.25 * u_psize * log10(1 + pow(100, (10 - magnitude) / 5));
//    v_psize = 2 * u_psize * (u_Mlim - magnitude) / u_Mlim;
    v_psize = clamp(v_psize, 0, 255);

    float lower = u_magnitude + u_Mlim;
    float upper = u_magnitude;
    v_brightness = (upper - magnitude) / (lower - upper);
    v_brightness = clamp(v_brightness, 0, 1);
    v_brightness *= u_brightness;

    v_temp = u_temp * a_temp;
    gl_PointSize = v_psize;
    gl_Position = projPosition;
//    if (magnitude > lower) {
//        gl_Position.w = -1;
//    }
}
"""

FRAG_SHADER0 = """
#version 120
// Uniforms
// ------------------------------------
uniform vec2 u_viewport;

// Varyings
// ------------------------------------
varying float v_temp;
varying float v_psize;
varying float v_brightness;
varying vec2  v_pcenter;


#define PI 3.1415927


float xbar31(float wavelength)
{
    float t1 = (wavelength-442.0f)*((wavelength < 442.0f)?0.0624f:0.0374f);
    float t2 = (wavelength-599.8f)*((wavelength < 599.8f)?0.0264f:0.0323f);
    float t3 = (wavelength-501.1f)*((wavelength < 501.1f)?0.0490f:0.0382f);
    return 0.362f*exp(-0.5f*t1*t1) + 1.056f*exp(-0.5f*t2*t2) - 0.065f*exp(-0.5f*t3*t3);
}


float ybar31(float wavelength)
{
    float t1 = (wavelength-568.8f)*((wavelength < 568.8f)?0.0213f:0.0247f);
    float t2 = (wavelength-530.9f)*((wavelength < 530.9f)?0.0613f:0.0322f);
    return 0.821f*exp(-0.5f*t1*t1) + 0.286f*exp(-0.5f*t2*t2);
}


float zbar31(float wavelength)
{
    float t1 = (wavelength-437.0f)*((wavelength < 437.0f)?0.0845f:0.0278f);
    float t2 = (wavelength-459.0f)*((wavelength < 459.0f)?0.0385f:0.0725f);
    return 1.217f*exp(-0.5f*t1*t1) + 0.681f*exp(-0.5f*t2*t2);
}


float black_body(float temperature, float wavelength)
{
    // Calculate the emittance of a black body at given
    // temperature (in kelvin) and wavelength (in metres).
    float c1 = 3.7417718e-16;
    float c2 = 1.43878e-2;
    float wlen = wavelength * 1.0e-9;  // Convert to metres.
    return c1 * pow(wlen, -5.0) / (exp(c2 / (wlen * temperature)) - 1.0);
}


vec3 spectrum_to_xyz(float temperature, int lmin, int lmax)
{
    float X = 0.0;
    float Y = 0.0;
    float Z = 0.0;
    for (int i = 0; i < (lmax-lmin)+1; i += 80) {
        float wavelength = float(lmin + i);
        float I = black_body(temperature, wavelength);
        X += I * xbar31(wavelength);
        Y += I * ybar31(wavelength);
        Z += I * zbar31(wavelength);
    }
    return vec3(X, Y, Z) / (X + Y + Z);
}


float gamma_correction(float c)
{
    if (c > 0.0031308) {
        return (1 + 0.055) * pow(c, 1.0/2.4) - 0.055;
    }
    return 12.92 * c;
}


vec3 xyz_to_rgb(vec3 xyz)
{
    mat3 xyz2rgb = mat3(vec3(+3.2404542, -0.9692660, +0.0556434),
                        vec3(-1.5371385, +1.8760108, -0.2040259),
                        vec3(-0.4985314, +0.0415560, +1.0572252));  // sRGB

    vec3 rgb = xyz2rgb * xyz;

/*
    float w = 0.0;
    w = min(w, rgb.r);
    w = min(w, rgb.g);
    w = min(w, rgb.b);
    if (w < 0.0) {
        rgb -= w;
    }

    float norm = 0.0;
    norm = max(norm, rgb.r);
    norm = max(norm, rgb.g);
    norm = max(norm, rgb.b);
    if (norm > 0.0) {
        rgb /= norm;
    }
*/

    rgb = clamp(rgb, 0, 1);
    rgb.r = gamma_correction(rgb.r);
    rgb.g = gamma_correction(rgb.g);
    rgb.b = gamma_correction(rgb.b);

//    rgb = 1 - exp(-8 * rgb);

    return rgb;
}


float airy(float d)
{
    float s = d;
//    float s = 4.493409 * d;        // 1st minimum
//    float s = 7.725252 * d;        // 2nd minimum
//    float s = 10.904122 * d;       // 3rd minimum
//    float s = 14.066194 * d;       // 4th minimum
//    float s = 17.220755 * d;       // 5th minimum
//    float s = 20.371303 * d;       // 6th minimum
//    float s = 23.519452 * d;       // 7th minimum
//    float s = 26.666054 * d;       // 8th minimum
//    float s = 29.811599 * d;       // 9th minimum
    float j1 = (sin(s) / s - cos(s)) / s;
    float I = 2 * j1 / s;
    I *= 9 * I / 4;             // normalization factor I(0) = 4 / 9
    return I;
}


float gaussian(float r2, float sigma2)
{
    return exp(-0.5 * r2 / sigma2) / (2 * PI * sigma2);
}


float spike(vec2 r)
{
    float s = max(0, 1 - 16 * abs(r.x + r.y)) +  max(0, 1 - 16 * abs(r.x - r.y));
    return s * s;
}


vec3 makeStar(in vec2 r)
{
    float d2 = dot(r, r);
    if (d2 > 1) discard;
    vec3 col = spectrum_to_xyz(v_temp, 380, 780);

    float c = 29.811599;
    float c2 = c * c;
    float d = sqrt(d2);
    float psf = airy(c * d);
    psf += (1 - d2) / (c2 * d2);
    psf *= 1 + c * d2 * spike(r);

    col *= psf;
    col *= v_brightness;
    col = xyz_to_rgb(col);
    return col;
}


// Main
// ------------------------------------
void main()
{
    vec2 r = (gl_FragCoord.xy - v_pcenter) / (0.5 * v_psize);
    gl_FragColor.rgb = makeStar(r);
    gl_FragColor.a = 1;
}
"""

FRAG_SHADER1 = """
#version 120
// Varyings
// ------------------------------------
varying float v_psize;
varying vec2  v_pcenter;

// Main
// ------------------------------------
void main()
{
    vec2 r = (gl_FragCoord.xy - v_pcenter) / (0.5 * v_psize);
    float d = length(r);
    if (d > 1) discard;
    float n = 1;
    d = n * (1 - d) * exp2(-n * d);
    gl_FragColor = vec4(vec3(d), 1);
}
"""

FRAG_SHADER2 = """
#version 120
// Varyings
// ------------------------------------
varying float v_psize;
varying vec2  v_pcenter;

// Main
// ------------------------------------
void main()
{
    vec2 r = (gl_FragCoord.xy - v_pcenter) / (0.5 * v_psize);
    float d = length(r);
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
        super(GLviewer, self).__init__(keys='interactive')

        w, h = 768, 432
        self.size = (w, h)

        self.data = {}
        self.vdata = {}
        self.program = {}
        self.program['body'] = gloo.Program(VERT_SHADER, FRAG_SHADER0)
        self.program['star'] = gloo.Program(VERT_SHADER, FRAG_SHADER0)
        self.program['sph'] = gloo.Program(VERT_SHADER, FRAG_SHADER1)
        self.program['blackhole'] = gloo.Program(VERT_SHADER, FRAG_SHADER2)

        self.bg_alpha = 1.0
        self.translate = [0.0, 0.0, -10.0]

        self.u_temp = 1.0
        self.u_psize = 6.0
        self.u_brightness = 1.0
        self.u_magnitude = 0.0
        self.u_view = translate(self.translate)
        self.u_model = np.eye(4, dtype=np.float32)
        self.u_projection = np.eye(4, dtype=np.float32)

        self.text = visuals.TextVisual('', pos=(9, 16), anchor_x='left')
        self.text.color = 'white'
        self.text.font_size = 9
        self.show_legend = True

        self.trans = visuals.transforms.TransformSystem(self)

        def callback(fps):
            titlestr = "tupan viewer: %0.1f fps @ %d x %d"
            self.title = titlestr % (fps, self.size[0], self.size[1])
        self.measure_fps(callback=callback)
        self.is_visible = True
        self.record = cli.record

        tex = gloo.Texture2D(self.size+(4,), format='rgba')
        self.render = gloo.Program(render_vertex, render_fragment)
        self.render["position"] = [(-1, -1), (-1, +1), (+1, -1), (+1, +1)]
        self.render["texcoord"] = [(0, 0), (0, 1), (1, 0), (1, 1)]
        self.render["texture"] = tex
        self.fbo = gloo.FrameBuffer(self.render["texture"],
                                    gloo.RenderBuffer(self.size))
#        self.fbo.resize((h, w))
#        with self.fbo:
#            self.do_draw()
#            im = self.fbo.read()
#            import matplotlib.pyplot as plt
#            plt.figure(figsize=(self.size[0]/100., self.size[1]/100.), dpi=100)
#            self.fig = plt.imshow(im, interpolation='none')
#            plt.show(block=False)

        self.show(True)

    def record_screen(self, im=None):
        try:
            if im is None:
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

    def set_state(self):
        gloo.set_state(
            blend=True,
            cull_face=False,
            depth_test=False,
            clear_color=(0.0, 0.0, 0.0, self.bg_alpha),
            blend_func=('one', 'one_minus_src_color', 'zero', 'one'),
#            blend_func=('src_alpha', 'one', 'zero', 'one'),
        )

    def on_initialize(self, event):
        self.set_state()

    def on_resize(self, event):
        w, h = self.size
        self.fbo.resize((h, w))
        gloo.set_viewport(0, 0, w, h)
        self.u_projection = perspective(45, w / h, 2**(-53), 100.0)

    def do_draw(self):
        gloo.clear()

        if self.show_legend:
            self.text.draw(self.trans)
            self.set_state()

        for key in self.data:
            prog = self.program[key]
            prog['u_temp'] = self.u_temp
            prog['u_psize'] = self.u_psize
            prog['u_brightness'] = self.u_brightness
            prog['u_magnitude'] = self.u_magnitude
            prog['u_viewport'] = self.size
            prog['u_view'] = self.u_view
            prog['u_model'] = self.u_model
            prog['u_projection'] = self.u_projection
            prog.draw('points')

    def on_draw(self, event):
        with self.fbo:
            self.do_draw()
#            im = self.fbo.read()
#            import matplotlib.pyplot as plt
#            self.fig.set_data(im)
#            plt.draw()

        gloo.clear()
        self.render.draw('triangle_strip')

        if self.record:
            self.record_screen()

    def on_key_press(self, event):
        if event.text == '+':
            self.u_psize *= 1.03125
        elif event.text == '-':
            self.u_psize /= 1.03125
        elif event.text == 'a':
            self.u_brightness /= 1.03125
        elif event.text == 'A':
            self.u_brightness *= 1.03125
        elif event.text == 'b':
            self.bg_alpha -= 0.03125 * int(self.bg_alpha > 0.0)
            self.set_state()
        elif event.text == 'B':
            self.bg_alpha += 0.03125 * int(self.bg_alpha < 1.0)
            self.set_state()
        elif event.text == 'm':
            self.u_magnitude -= 0.25
        elif event.text == 'M':
            self.u_magnitude += 0.25
        elif event.text == 't':
            self.u_temp /= 1.03125
        elif event.text == 'T':
            self.u_temp *= 1.03125
        elif event.text == 'Z':
            self.translate[2] -= 0.03125 * (1 + abs(self.translate[2]))
            self.u_view = translate(self.translate)
        elif event.text == 'z':
            self.translate[2] += 0.03125 * (1 + abs(self.translate[2]))
            self.u_view = translate(self.translate)
        elif event.key == 'Up':
            self.u_model = np.dot(self.u_model, rotate(+1, [1, 0, 0]))
        elif event.key == 'Down':
            self.u_model = np.dot(self.u_model, rotate(-1, [1, 0, 0]))
        elif event.key == 'Right':
            self.u_model = np.dot(self.u_model, rotate(+1, [0, 1, 0]))
        elif event.key == 'Left':
            self.u_model = np.dot(self.u_model, rotate(-1, [0, 1, 0]))
        elif event.key == '>':
            self.u_model = np.dot(self.u_model, rotate(+1, [0, 0, 1]))
        elif event.key == '<':
            self.u_model = np.dot(self.u_model, rotate(-1, [0, 0, 1]))
        elif event.text == 'l':
            self.show_legend = not self.show_legend
        elif event.key == 'Escape':
            self.is_visible = False
        self.update()

    def on_mouse_wheel(self, event):
        delta = 0.03125 * event.delta[1]
        self.translate[2] += delta * (1 + abs(self.translate[2]))
        self.u_view = translate(self.translate)
        self.update()

    def on_mouse_move(self, event):
        x, y = event.pos
        w, h = self.size
        if event.is_dragging:
            dx = (2 * x / w - 1) * (w / h)
            dy = (1 - 2 * y / h)
            if event.button == 1:
                self.translate[0] = 5 * dx
                self.translate[1] = 5 * dy
                self.u_view = translate(self.translate)
            if event.button == 2:
                self.u_model = np.dot(self.u_model,
                                      np.dot(rotate(-dx, [0, 1, 0]),
                                             rotate(+dy, [1, 0, 0])))
        self.update()

    def init_vertex_buffers(self, ps):
        for name, member in ps.members.items():
            n = member.n
            mass = member.mass
            pos = member.rdot[0].T

            four_pi = 4 * np.pi
            sigma = 5.67037e-8
            m = mass / mass.mean()
            L = m**(7/2)
            R = m**(2/3)
            T = (3.828e+26 * L / (four_pi * sigma * (6.957e+8 * R)**2))**(1/4)

#            F = sigma * T**4 * (R/10)**2
#            F = 3.828e+26 * L / (four_pi * (10 * 3.0852823)**2)
#            F = L / (four_pi * 10**2)
#            F0 = sigma * 5772**4 * (1/10)**2
#            F0 = 3.828e+26 / (four_pi * (10 * 3.0852823)**2)
#            F0 = 1 / (four_pi * 10**2)
            M = -2.5 * np.log10(L)
            Msum = -2.5 * np.log10(L.sum())

#            print(m.min(), m.max(), m.mean())
#            print(M.min(), M.max(), M.mean())
#            print(L.min(), L.max(), L.mean())
#            print(R.min(), R.max(), R.mean())
#            print(T.min(), T.max(), T.mean())

#            from numpy.random import seed, shuffle
#            seed(0)
#            shuffle(T)
#            shuffle(R)
#            shuffle(M)

            attributes = [
                ('a_temp', np.float32, 1),
                ('a_psize', np.float32, 1),
                ('a_magnitude', np.float32, 1),
                ('a_position', np.float32, 3),
            ]
            self.data[name] = np.zeros(n, attributes)

            self.data[name]['a_temp'][...] = T
            self.data[name]['a_psize'][...] = R
            self.data[name]['a_magnitude'][...] = M
            self.data[name]['a_position'][...] = pos

            self.vdata[name] = gloo.VertexBuffer(self.data[name])
            self.program[name].bind(self.vdata[name])
            self.program[name]['u_Mlim'] = -Msum

    def show_event(self, ps):
        if not self.data:
            self.init_vertex_buffers(ps)

        for name, member in ps.members.items():
            pid = member.pid
            pos = member.rdot[0].T

            self.data[name]['a_position'][pid] = pos
            self.vdata[name].set_data(self.data[name])

        time = ps.time[0]
        self.text.text = f'T = {time:e}'
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
