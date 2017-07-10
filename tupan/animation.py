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
varying float v_psize;
varying float v_brightness;
varying vec2  v_pcenter;
varying vec3  v_pcolor;


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


float black_body_spectrum(float temperature, float wavelength)
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
        float I = black_body_spectrum(temperature, wavelength);
        X += I * xbar31(wavelength);
        Y += I * ybar31(wavelength);
        Z += I * zbar31(wavelength);
    }
    return vec3(X, Y, Z) / (X + Y + Z);
}


vec3 xyz_to_rgb(vec3 xyz)
{
    mat3 xyz2rgb = mat3(vec3(+3.2404542, -0.9692660, +0.0556434),
                        vec3(-1.5371385, +1.8760108, -0.2040259),
                        vec3(-0.4985314, +0.0415560, +1.0572252));  // sRGB

    vec3 rgb = xyz2rgb * xyz;

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

    rgb = pow(rgb, vec3(1.0/2.2));

    return rgb;
}


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

//    v_psize = u_psize * sqrt(a_psize);
//    v_psize = u_psize * log(1 + 8 * a_psize) / log(1 + 8);
//    v_psize = 0.25 * u_psize * log10(1 + pow(100, (10 - magnitude) / 5));
    v_psize = u_psize * (u_Mlim - magnitude) / u_Mlim;
    v_psize = clamp(v_psize, 0, 255);

    float lower = u_magnitude + u_Mlim;
    float upper = u_magnitude;
    v_brightness = (upper - magnitude) / (lower - upper);
    v_brightness = clamp(v_brightness, 0, 1);
    v_brightness *= u_brightness;

    v_pcolor = xyz_to_rgb(spectrum_to_xyz(u_temp * a_temp, 380, 780));
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
varying float v_psize;
varying float v_brightness;
varying vec2  v_pcenter;
varying vec3  v_pcolor;

#define PI 3.1415927

float airy(float d)
{
//    float s = 4.49341 * d;        // first minimum is at ~4.49341
//    float s = 7.72525 * d;        // second minimum is at ~7.72525
//    float s = 10.90412 * d;       // third minimum is at ~10.90412
    float s = 14.06619 * d;       // fourth minimum is at ~14.06619
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

float rand(int seed, float ray)
{
    return mod(sin(float(seed) * 363.5346 + ray * 674.2454) * 6743.4365, 1.0);
}

vec3 makeStar(in vec2 r)
{
    float d = length(r);
    if (d > 1) discard;
    vec3 col = v_pcolor;

/*
    float ang = atan(r.x, r.y);
    for (int i = 0; i < 10; ++i)
    {
        float ray = float(i + 1);
        float rayang = rand(5, ray) * 100 * length(v_pcolor);
        rayang = mod(rayang, 2.0 * PI);
        if (rayang < ang - PI) {rayang += 2.0 * PI;}
        if (rayang > ang + PI) {rayang -= 2.0 * PI;}
        float brite = 0.6 - abs(ang - rayang);
        brite -= d * 0.4;

        if (brite > 0.0)
        {
            col += vec3(0.3 + 1.7 * rand(8644, ray),
                        0.5 + 1.5 * rand(4567, ray),
                        0.7 + 1.3 * rand(7354, ray)) * brite * 0.125;
        }
    }
*/

    col *= airy(d) + (1 - d) / (64 * d);
    col += 0.5 * d * (1 - d) * spike(r);
    col = clamp(col, 0, 1);
    col *= v_brightness;
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
varying vec3  v_pcolor;

// Main
// ------------------------------------
void main()
{
    vec2 r = (gl_FragCoord.xy - v_pcenter) / (0.5 * v_psize);
    float d = length(r);
    if (d > 1) discard;
    float n = 1;
    d = n * (1 - d) * exp2(-n * d);
    gl_FragColor = vec4(v_pcolor * d, 1);
}
"""

FRAG_SHADER2 = """
#version 120
// Varyings
// ------------------------------------
varying float v_psize;
varying vec2  v_pcenter;
varying vec3  v_pcolor;

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


VERT_SHADER3 = """
#version 120
// Uniforms
// ------------------------------------
uniform vec2 u_viewport;

// Attributes
// ------------------------------------
attribute vec2  a_position;

// Main
// ------------------------------------
void main(void)
{
    gl_Position = vec4(a_position, 0, 1);
}
"""

FRAG_SHADER3 = """
#version 120
// Uniforms
// ------------------------------------
uniform vec2 u_viewport;

// Random float generator
// ------------------------------------
float random(vec2 p)
{
    float a = fract(sin(p.x + p.y * 10000.) * 10000.);
    float b = fract(cos(p.y + p.x * 10000.) * 10000.);
    return (a * a + b * b) / 2.;
}

// Star field generation
// ------------------------------------
vec4 makeStarField(vec2 p)
{
    vec2 seed = floor(p * 512);
    float r = random(seed);
    vec4 starfield = vec4(pow(r, 10));
    return starfield;
}

// Main
// ------------------------------------
void main()
{
    vec2 uv = 2 * gl_FragCoord.xy - u_viewport.xy;
    uv /= max(u_viewport.x, u_viewport.y);
    vec4 starfield = makeStarField(uv);
    gl_FragColor = starfield;
}
"""


class GLviewer(app.Canvas):
    """
    TODO.
    """
    def __init__(self, *args, **kwargs):
        w = 48 * 16
        h = 48 * 9
        super(GLviewer, self).__init__(keys='interactive')
        self.size = (w, h)

        self.data = {}
        self.vdata = {}
        self.program = {}
        self.program['body'] = gloo.Program(VERT_SHADER, FRAG_SHADER0)
        self.program['star'] = gloo.Program(VERT_SHADER, FRAG_SHADER0)
        self.program['sph'] = gloo.Program(VERT_SHADER, FRAG_SHADER1)
        self.program['blackhole'] = gloo.Program(VERT_SHADER, FRAG_SHADER2)
        self.background = gloo.Program(VERT_SHADER3, FRAG_SHADER3)

        self.bg_alpha = 1.0

        self.u_temp = 1.0
        self.psize = 10.0
        self.brightness = 1.0
        self.magnitude = 0.0

        self.translate = [0.0, 0.0, -10.0]
        self.view = translate(self.translate)
        self.model = np.eye(4, dtype=np.float32)
        self.projection = np.eye(4, dtype=np.float32)

        for prog in self.program.values():
            prog['u_temp'] = self.u_temp
            prog['u_psize'] = self.psize
            prog['u_brightness'] = self.brightness
            prog['u_magnitude'] = self.magnitude
            prog['u_model'] = self.model
            prog['u_view'] = self.view
            prog['u_viewport'] = (w, h)

        self.background['u_viewport'] = (w, h)
        self.background['a_position'] = [(-1, -1), (-1, 1), (1, 1),
                                         (-1, -1), (1, 1), (1, -1)]

        self.show_legend = True
        self.text = visuals.TextVisual('', pos=(9, 16), anchor_x='left')
        self.text.color = 'white'
        self.text.font_size = 9

        self.trans = visuals.transforms.TransformSystem(self)

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
        w, h = event.size
        gloo.set_viewport(0, 0, w, h)
        self.projection = perspective(45, w / h, 2**(-53), 100.0)
        for prog in self.program.values():
            prog['u_viewport'] = (w, h)
            prog['u_projection'] = self.projection
#        self.background['u_viewport'] = (w, h)

    def on_draw(self, event):
        gloo.clear()

        if self.show_legend:
            self.text.draw(self.trans)
            self.set_state()

#        self.background.draw()

        for key in self.data:
            self.program[key].draw('points')

        if cli.record:
            self.record_screen()

    def on_key_press(self, event):
        if event.text == '+':
            self.psize *= 1.03125
            for prog in self.program.values():
                prog['u_psize'] = self.psize
        elif event.text == '-':
            self.psize /= 1.03125
            for prog in self.program.values():
                prog['u_psize'] = self.psize
        elif event.text == 'a':
            self.brightness /= 1.03125
            for prog in self.program.values():
                prog['u_brightness'] = self.brightness
        elif event.text == 'A':
            self.brightness *= 1.03125
            for prog in self.program.values():
                prog['u_brightness'] = self.brightness
        elif event.text == 'b':
            self.bg_alpha -= 0.03125 * int(self.bg_alpha > 0.0)
            self.set_state()
            gloo.clear()
        elif event.text == 'B':
            self.bg_alpha += 0.03125 * int(self.bg_alpha < 1.0)
            self.set_state()
            gloo.clear()
        elif event.text == 'm':
            self.magnitude -= 0.25
            for prog in self.program.values():
                prog['u_magnitude'] = self.magnitude
        elif event.text == 'M':
            self.magnitude += 0.25
            for prog in self.program.values():
                prog['u_magnitude'] = self.magnitude
        elif event.text == 't':
            self.u_temp /= 1.03125
            for prog in self.program.values():
                prog['u_temp'] = self.u_temp
        elif event.text == 'T':
            self.u_temp *= 1.03125
            for prog in self.program.values():
                prog['u_temp'] = self.u_temp
        elif event.text == 'Z':
            self.translate[2] -= 0.03125 * (1 + abs(self.translate[2]))
            self.view = translate(self.translate)
            for prog in self.program.values():
                prog['u_view'] = self.view
        elif event.text == 'z':
            self.translate[2] += 0.03125 * (1 + abs(self.translate[2]))
            self.view = translate(self.translate)
            for prog in self.program.values():
                prog['u_view'] = self.view
        elif event.key == 'Up':
            self.model = np.dot(self.model, rotate(+1, [1, 0, 0]))
            for prog in self.program.values():
                prog['u_model'] = self.model
        elif event.key == 'Down':
            self.model = np.dot(self.model, rotate(-1, [1, 0, 0]))
            for prog in self.program.values():
                prog['u_model'] = self.model
        elif event.key == 'Right':
            self.model = np.dot(self.model, rotate(+1, [0, 1, 0]))
            for prog in self.program.values():
                prog['u_model'] = self.model
        elif event.key == 'Left':
            self.model = np.dot(self.model, rotate(-1, [0, 1, 0]))
            for prog in self.program.values():
                prog['u_model'] = self.model
        elif event.key == '>':
            self.model = np.dot(self.model, rotate(+1, [0, 0, 1]))
            for prog in self.program.values():
                prog['u_model'] = self.model
        elif event.key == '<':
            self.model = np.dot(self.model, rotate(-1, [0, 0, 1]))
            for prog in self.program.values():
                prog['u_model'] = self.model
        elif event.text == 'l':
            self.show_legend = not self.show_legend
        elif event.key == 'Escape':
            self.is_visible = False
        self.update()

    def on_mouse_wheel(self, event):
        delta = 0.03125 * event.delta[1]
        self.translate[2] += delta * (1 + abs(self.translate[2]))
        self.view = translate(self.translate)
        for prog in self.program.values():
            prog['u_view'] = self.view
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
