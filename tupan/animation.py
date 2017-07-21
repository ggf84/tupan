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

float gamma_correction(float c)
{
    return (c <= 0.0031308)
                ? (12.92 * c)
                : ((1 + 0.055) * pow(c, (1.0/2.4)) - 0.055);
}

vec3 gamma_correction(vec3 c)
{
    c.r = gamma_correction(c.r);
    c.g = gamma_correction(c.g);
    c.b = gamma_correction(c.b);
    return c;
//    return (c <= 0.0031308)
//                ? (12.92 * c)
//                : ((1 + 0.055) * pow(c, vec3(1.0/2.4)) - 0.055);
}

vec3 xyz_to_rgb(vec3 c)
{
    // luminance adjustment
    vec3 C = c;
    c /= (c.x + c.y + c.z);
    C.y = log(1 + 16 * C.y) / log(1 + pow(16, 4));
    C.x = C.y * (c.x / c.y);
    C.z = C.y * ((1 - (c.x + c.y)) / c.y);
    c = C;

    mat3 xyz2rgb = mat3(vec3(+3.2404542, -0.9692660, +0.0556434),
                        vec3(-1.5371385, +1.8760108, -0.2040259),
                        vec3(-0.4985314, +0.0415560, +1.0572252));  // sRGB
    c = xyz2rgb * c;
    c = clamp(c, 0, 1);
    c = gamma_correction(c);
    return c;
}

void main()
{
    vec4 c = texture2D(texture, v_texcoord);
    c.rgb = xyz_to_rgb(c.xyz);
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
uniform float u_bolometric_flux;
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

    v_temp = a_temp + u_temp;
    float magnitude = a_magnitude + 5 * (log10(projPosition.w) - 1);

//    v_psize = u_psize;
//    v_psize = u_psize * pow(a_psize, 3.0/4.0);
    v_psize = 0.5 * u_psize * a_psize;
//    v_psize = 2 * u_psize * sqrt(a_psize);
//    v_psize = 2.5 * u_psize * log(1 + 4 * a_psize) / log(1 + 4);
//    v_psize = 0.25 * u_psize * log10(1 + pow(100, (10 - magnitude) / 5));
//    v_psize = 2 * u_psize * (u_Mlim - magnitude) / u_Mlim;
    v_psize = clamp(v_psize, 0, 255);

//    float lower = u_magnitude + u_Mlim;
//    float upper = u_magnitude;
//    v_brightness = (upper - magnitude) / (lower - upper);
//    v_brightness = clamp(v_brightness, 0, 1);
    v_brightness = u_brightness / u_bolometric_flux;
//    v_brightness = 10 * u_brightness / (5.67037e-8 * pow(v_temp, 4));

    gl_PointSize = v_psize;
    gl_Position = projPosition;
//    if (magnitude > lower) {
//        gl_Position.w = -1;
//    }
}
"""

FRAG_SHADER0 = """
#version 120

#define PI 3.1415927

// Uniforms
// ------------------------------------
uniform vec2 u_viewport;

// Varyings
// ------------------------------------
varying float v_temp;
varying float v_psize;
varying float v_brightness;
varying vec2  v_pcenter;


const float offset = 1.0 / 512;

const vec2 offsets[8] = vec2[](
    vec2(-offset,  offset), // top-left
    vec2( 0.0f,    offset), // top-center
    vec2( offset,  offset), // top-right
    vec2(-offset,  0.0f),   // center-left
//    vec2( 0.0f,    0.0f),   // center-center
    vec2( offset,  0.0f),   // center-right
    vec2(-offset, -offset), // bottom-left
    vec2( 0.0f,   -offset), // bottom-center
    vec2( offset, -offset)  // bottom-right
);


const vec4 xyzbar31[81] = vec4[](
//    vec4(360,0.000129900000,0.000003917000,0.000606100000),
//    vec4(365,0.000232100000,0.000006965000,0.001086000000),
//    vec4(370,0.000414900000,0.000012390000,0.001946000000),
//    vec4(375,0.000741600000,0.000022020000,0.003486000000),
    vec4(380,0.001368000000,0.000039000000,0.006450001000),
    vec4(385,0.002236000000,0.000064000000,0.010549990000),
    vec4(390,0.004243000000,0.000120000000,0.020050010000),
    vec4(395,0.007650000000,0.000217000000,0.036210000000),
    vec4(400,0.014310000000,0.000396000000,0.067850010000),
    vec4(405,0.023190000000,0.000640000000,0.110200000000),
    vec4(410,0.043510000000,0.001210000000,0.207400000000),
    vec4(415,0.077630000000,0.002180000000,0.371300000000),
    vec4(420,0.134380000000,0.004000000000,0.645600000000),
    vec4(425,0.214770000000,0.007300000000,1.039050100000),
    vec4(430,0.283900000000,0.011600000000,1.385600000000),
    vec4(435,0.328500000000,0.016840000000,1.622960000000),
    vec4(440,0.348280000000,0.023000000000,1.747060000000),
    vec4(445,0.348060000000,0.029800000000,1.782600000000),
    vec4(450,0.336200000000,0.038000000000,1.772110000000),
    vec4(455,0.318700000000,0.048000000000,1.744100000000),
    vec4(460,0.290800000000,0.060000000000,1.669200000000),
    vec4(465,0.251100000000,0.073900000000,1.528100000000),
    vec4(470,0.195360000000,0.090980000000,1.287640000000),
    vec4(475,0.142100000000,0.112600000000,1.041900000000),
    vec4(480,0.095640000000,0.139020000000,0.812950100000),
    vec4(485,0.057950010000,0.169300000000,0.616200000000),
    vec4(490,0.032010000000,0.208020000000,0.465180000000),
    vec4(495,0.014700000000,0.258600000000,0.353300000000),
    vec4(500,0.004900000000,0.323000000000,0.272000000000),
    vec4(505,0.002400000000,0.407300000000,0.212300000000),
    vec4(510,0.009300000000,0.503000000000,0.158200000000),
    vec4(515,0.029100000000,0.608200000000,0.111700000000),
    vec4(520,0.063270000000,0.710000000000,0.078249990000),
    vec4(525,0.109600000000,0.793200000000,0.057250010000),
    vec4(530,0.165500000000,0.862000000000,0.042160000000),
    vec4(535,0.225749900000,0.914850100000,0.029840000000),
    vec4(540,0.290400000000,0.954000000000,0.020300000000),
    vec4(545,0.359700000000,0.980300000000,0.013400000000),
    vec4(550,0.433449900000,0.994950100000,0.008749999000),
    vec4(555,0.512050100000,1.000000000000,0.005749999000),
    vec4(560,0.594500000000,0.995000000000,0.003900000000),
    vec4(565,0.678400000000,0.978600000000,0.002749999000),
    vec4(570,0.762100000000,0.952000000000,0.002100000000),
    vec4(575,0.842500000000,0.915400000000,0.001800000000),
    vec4(580,0.916300000000,0.870000000000,0.001650001000),
    vec4(585,0.978600000000,0.816300000000,0.001400000000),
    vec4(590,1.026300000000,0.757000000000,0.001100000000),
    vec4(595,1.056700000000,0.694900000000,0.001000000000),
    vec4(600,1.062200000000,0.631000000000,0.000800000000),
    vec4(605,1.045600000000,0.566800000000,0.000600000000),
    vec4(610,1.002600000000,0.503000000000,0.000340000000),
    vec4(615,0.938400000000,0.441200000000,0.000240000000),
    vec4(620,0.854449900000,0.381000000000,0.000190000000),
    vec4(625,0.751400000000,0.321000000000,0.000100000000),
    vec4(630,0.642400000000,0.265000000000,0.000049999990),
    vec4(635,0.541900000000,0.217000000000,0.000030000000),
    vec4(640,0.447900000000,0.175000000000,0.000020000000),
    vec4(645,0.360800000000,0.138200000000,0.000010000000),
    vec4(650,0.283500000000,0.107000000000,0.000000000000),
    vec4(655,0.218700000000,0.081600000000,0.000000000000),
    vec4(660,0.164900000000,0.061000000000,0.000000000000),
    vec4(665,0.121200000000,0.044580000000,0.000000000000),
    vec4(670,0.087400000000,0.032000000000,0.000000000000),
    vec4(675,0.063600000000,0.023200000000,0.000000000000),
    vec4(680,0.046770000000,0.017000000000,0.000000000000),
    vec4(685,0.032900000000,0.011920000000,0.000000000000),
    vec4(690,0.022700000000,0.008210000000,0.000000000000),
    vec4(695,0.015840000000,0.005723000000,0.000000000000),
    vec4(700,0.011359160000,0.004102000000,0.000000000000),
    vec4(705,0.008110916000,0.002929000000,0.000000000000),
    vec4(710,0.005790346000,0.002091000000,0.000000000000),
    vec4(715,0.004109457000,0.001484000000,0.000000000000),
    vec4(720,0.002899327000,0.001047000000,0.000000000000),
    vec4(725,0.002049190000,0.000740000000,0.000000000000),
    vec4(730,0.001439971000,0.000520000000,0.000000000000),
    vec4(735,0.000999949300,0.000361100000,0.000000000000),
    vec4(740,0.000690078600,0.000249200000,0.000000000000),
    vec4(745,0.000476021300,0.000171900000,0.000000000000),
    vec4(750,0.000332301100,0.000120000000,0.000000000000),
    vec4(755,0.000234826100,0.000084800000,0.000000000000),
    vec4(760,0.000166150500,0.000060000000,0.000000000000),
    vec4(765,0.000117413000,0.000042400000,0.000000000000),
    vec4(770,0.000083075270,0.000030000000,0.000000000000),
    vec4(775,0.000058706520,0.000021200000,0.000000000000),
    vec4(780,0.000041509940,0.000014990000,0.000000000000)
//    vec4(785,0.000029353260,0.000010600000,0.000000000000),
//    vec4(790,0.000020673830,0.000007465700,0.000000000000),
//    vec4(795,0.000014559770,0.000005257800,0.000000000000),
//    vec4(800,0.000010253980,0.000003702900,0.000000000000),
//    vec4(805,0.000007221456,0.000002607800,0.000000000000),
//    vec4(810,0.000005085868,0.000001836600,0.000000000000),
//    vec4(815,0.000003581652,0.000001293400,0.000000000000),
//    vec4(820,0.000002522525,0.000000910930,0.000000000000),
//    vec4(825,0.000001776509,0.000000641530,0.000000000000),
//    vec4(830,0.000001251141,0.000000451810,0.000000000000)
);


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
    float s = max(0, 1 - 32 * abs(r.x + r.y)) +  max(0, 1 - 32 * abs(r.x - r.y));
    return s * s;
}

float make_star(in vec2 r)
{
    float d2 = dot(r, r);
    float d = sqrt(d2);

    float k = 1024;
    float psf = airy(k * d);
    psf *= abs(2 - d2);
    psf *= 1 + 32 * d * spike(r);
    psf *= 0.125 * pow(k / 8, 4);

    return psf * v_brightness;
}

float black_body(float temperature, float wlen)
{
    // Calculate the emittance of a black body at given
    // temperature (in kelvin) and wavelength (in metres).
    float c1 = 3.7417718e-16;
    float c2 = 1.43878e-2;
    return c1 * pow(wlen, -5.0) / (exp(c2 / (wlen * temperature)) - 1.0);
}

vec3 spectrum_to_xyz(float temperature, int nsteps)
{
    vec3 XYZ = vec3(0.0);
    float dwlen = 5 * nsteps * 1.0e-9;
    for (int i = 0; i < 81; i += nsteps) {
        vec4 cie = xyzbar31[i].yzwx;
        float wlen = cie.w * 1.0e-9;
        float I = black_body(temperature, wlen);
        XYZ += I * cie.xyz * dwlen;
    }
    return XYZ;
}

// Main
// ------------------------------------
void main()
{
    vec2 r = 2 * (gl_FragCoord.xy - v_pcenter) / v_psize;
    float psf = make_star(r);
    for(int i = 0; i < 8; i++) {
        psf += make_star(r + offsets[i]);
    }
    psf /= 1 + 8;
    vec3 c = psf * spectrum_to_xyz(v_temp, 8);
    gl_FragColor = vec4(c, 1);
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
    vec2 r = 2 * (gl_FragCoord.xy - v_pcenter) / v_psize;
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
    vec2 r = 2 * (gl_FragCoord.xy - v_pcenter) / v_psize;
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

        self.u_temp = 0.0
        self.u_psize = 4.0
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

        tex = gloo.Texture2D(self.size+(4,),
                             format='rgba',
                             internalformat='rgba32f')
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
#            blend_func=('one', 'one_minus_src_color', 'zero', 'one'),
#            blend_func=('src_alpha', 'one', 'zero', 'one'),
            blend_func=('one', 'one', 'zero', 'one'),
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
            self.u_temp -= 50
        elif event.text == 'T':
            self.u_temp += 50
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

            attributes = [
                ('a_temp', np.float32, 1),
                ('a_psize', np.float32, 1),
                ('a_magnitude', np.float32, 1),
                ('a_position', np.float32, 3),
            ]
            self.data[name] = np.zeros(n, attributes)

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

            F = sigma * T**4
            self.program[name]['u_Mlim'] = -Msum
            self.program[name]['u_bolometric_flux'] = F.mean()

            self.data[name]['a_temp'][...] = T
            self.data[name]['a_psize'][...] = R
            self.data[name]['a_magnitude'][...] = M
            self.data[name]['a_position'][...] = pos

            self.vdata[name] = gloo.VertexBuffer(self.data[name])
            self.program[name].bind(self.vdata[name])

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
