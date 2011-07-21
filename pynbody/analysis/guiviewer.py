#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""

"""

#from gi.repository import Gtk
import os
import sys
import pygtk
pygtk.require("2.0")
import gtk as Gtk
import gtk.gtkgl
from OpenGL import GL





################################################################################





class LowLevelGLArea(object):
    def __init__(self):
        self.glarea = Gtk.DrawingArea()
#        self.glarea.set_size_request(384, 320)
        self.glconfig = self._on_glconfig()

        print("is RGBA:",                 self.glconfig.is_rgba())
        print("is double-buffered:",      self.glconfig.is_double_buffered())
        print("is stereo:",               self.glconfig.is_stereo())
        print("has alpha:",               self.glconfig.has_alpha())
        print("has depth buffer:",        self.glconfig.has_depth_buffer())
        print("has stencil buffer:",      self.glconfig.has_stencil_buffer())
        print("has accumulation buffer:", self.glconfig.has_accum_buffer())

        Gtk.gtkgl.widget_set_gl_capability(self.glarea, self.glconfig)

        self.glarea.connect_after("realize", self._on_realize)
        self.glarea.connect("configure_event", self._on_configure_event)
        self.glarea.connect("expose_event", self._on_expose_event)
        self.glarea.connect("button-press-event", self._on_button_press)
        self.glarea.connect("button-release-event", self._on_button_release)
        self.glarea.connect("motion-notify-event", self._on_motion_notify)
        self.glarea.set_events(Gtk.gdk.SCROLL_MASK |
                               Gtk.gdk.KEY_PRESS_MASK |
                               Gtk.gdk.BUTTON_PRESS_MASK |
                               Gtk.gdk.POINTER_MOTION_MASK)

        self.rotx = 0
        self.roty = 0
        self.pointerx = 0
        self.pointery = 0
        self.frame_data = None


    def _on_glconfig(self):
        try:
            # try double-buffered
            glconfig = Gtk.gdkgl.Config(mode = Gtk.gdkgl.MODE_RGBA |
                                               Gtk.gdkgl.MODE_DOUBLE |
                                               Gtk.gdkgl.MODE_DEPTH)
        except Gtk.gdkgl.NoMatches:
            # try single-buffered
            glconfig = Gtk.gdkgl.Config(mode = Gtk.gdkgl.MODE_RGBA |
                                               Gtk.gdkgl.MODE_DEPTH)
        return glconfig


    def _on_realize(self, widget):
        # get GLContext and GLDrawable
        glcontext = Gtk.gtkgl.widget_get_gl_context(widget)
        gldrawable = Gtk.gtkgl.widget_get_gl_drawable(widget)

        # Begin GL calls
        if not gldrawable.gl_begin(glcontext): return

        GL.glEnable(GL.GL_DEPTH_TEST)
        GL.glClearColor(0.5, 0.5, 0.5, 1.0)
        GL.glColorMask(GL.GL_TRUE, GL.GL_TRUE, GL.GL_TRUE, GL.GL_TRUE)

        # End GL calls
        gldrawable.gl_end()


    def _on_configure_event(self, widget, event):
        # get GLContext and GLDrawable
        glcontext = Gtk.gtkgl.widget_get_gl_context(widget)
        gldrawable = Gtk.gtkgl.widget_get_gl_drawable(widget)

        # Begin GL calls
        if not gldrawable.gl_begin(glcontext): return

        x, y, width, height = widget.get_allocation()

        GL.glViewport(0, 0, width, height)
        GL.glMatrixMode(GL.GL_PROJECTION)
        GL.glLoadIdentity()
        if width > height:
            w = float(width) / float(height)
            GL.glFrustum(-w, w, -1.0, 1.0, 5.0, 60.0)
        else:
            h = float(height) / float(width)
            GL.glFrustum(-1.0, 1.0, -h, h, 5.0, 60.0)

        GL.glMatrixMode(GL.GL_MODELVIEW)

        # End GL calls
        gldrawable.gl_end()

        return True


    def _on_expose_event(self, widget, event):
        # get GLContext and GLDrawable
        glcontext = Gtk.gtkgl.widget_get_gl_context(widget)
        gldrawable = Gtk.gtkgl.widget_get_gl_drawable(widget)

        # Begin GL calls
        if not gldrawable.gl_begin(glcontext): return

        GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)

        GL.glLoadIdentity()

        GL.glTranslate(0, 0, -10)
        GL.glRotate(self.rotx, 1, 0, 0)
        GL.glRotate(self.roty, 0, 1, 0)

        GL.glColor(0.0, 1.0, 0.0, 0.0)
        Gtk.gdkgl.draw_teapot(False, 1.0)   # XXX: puts draw function right here.
        GL.glColor(0.0, 0.0, 1.0, 0.0)
        Gtk.gdkgl.draw_torus (False, 0.3, 0.6, 30, 30)

        if gldrawable.is_double_buffered():
            gldrawable.swap_buffers()
        else:
            GL.glFlush()

        # End GL calls
        gldrawable.gl_end()

        return True


    def _on_button_press(self, widget, event):
        self.pointerx = event.x
        self.pointery = event.y
        return True


    def _on_button_release(self, widget, event):
        pass


    def _on_motion_notify(self, widget, event):
        if event.state & Gtk.gdk.BUTTON1_MASK:
            x, y, width, height = widget.get_allocation()
            self.rotx += int(((event.y-self.pointery)/width)*360)
            self.roty += int(((event.x-self.pointerx)/height)*360)
        self.pointerx = event.x
        self.pointery = event.y
        return True


    def update_frame(self):
        self.glarea.window.invalidate_rect(self.glarea.allocation, False)
#        self.glarea.window.process_updates(False)




################################################################################





class GUIwindow(object):
    def __init__(self):
        self.window = Gtk.Window()
        self.window.set_title("PyNbody Viewer")
#        self.window.set_default_size(480, 400)
        self.window.set_size_request(480, 400)
        self.window.set_position(Gtk.WIN_POS_CENTER)
#        try:
#            path = os.path.dirname(__file__)
#            texture_path = os.path.join(path, "textures")
#            self.window.set_icon_from_file(os.path.join(texture_path, "glow.png"))
#        except Exception as e:
#            print e.message
#            sys.exit(1)
        self.window.connect("delete-event", self.hide_window)

        if sys.platform != "win32":
            self.window.set_resize_mode(Gtk.RESIZE_IMMEDIATE)
        self.window.set_reallocate_redraws(True)

        self.table = Gtk.Table(2, 2)
        self.table.set_border_width(1)
        self.table.set_col_spacings(1)
        self.table.set_row_spacings(1)
        self.window.add(self.table)
        self.table.show()

        self.widget = LowLevelGLArea()
        self.table.attach(self.widget.glarea, 0, 1, 0, 1)   # l, r, t, b
        self.widget.glarea.show()

        self.vadj = Gtk.Adjustment(0, -360, 360, 0.5, 1, 0)
        self.vadj.connect("value_changed", self.vchanged)
        vscale = Gtk.VScale(self.vadj)
        vscale.set_digits(0)
        vscale.set_size_request(32, -1)
        vscale.set_value_pos(Gtk.POS_BOTTOM)
        self.table.attach(vscale, 1, 2, 0, 1, xoptions=Gtk.FILL)   # l, r, t, b
        vscale.show()

        self.hadj = Gtk.Adjustment(0, -360, 360, 0.5, 1, 0)
        self.hadj.connect("value_changed", self.hchanged)
        hscale = Gtk.HScale(self.hadj)
        hscale.set_digits(0)
        hscale.set_size_request(-1, 32)
        hscale.set_value_pos(Gtk.POS_RIGHT)
        self.table.attach(hscale, 0, 1, 1, 2, yoptions=Gtk.FILL)   # l, r, t, b
        hscale.show()

        self.window.show_all()


    def hide_window(self, window, event):
        window.hide_all()
        if window.get_visible():
            return False
        else:
            return True


    def vchanged(self, vadj):
        self.widget.rotx = vadj.value


    def hchanged(self, hadj):
        self.widget.roty = hadj.value


    def update_window(self):
        if self.widget.rotx > self.vadj.upper:
            self.widget.rotx = self.vadj.upper
        if self.widget.rotx < self.vadj.lower:
            self.widget.rotx = self.vadj.lower
        if self.widget.roty > self.hadj.upper:
            self.widget.roty = self.hadj.upper
        if self.widget.roty < self.hadj.lower:
            self.widget.roty = self.hadj.lower
        self.vadj.value = self.widget.rotx
        self.hadj.value = self.widget.roty

        self.widget.update_frame()


    def set_frame_data(self, data):
        self.widget.frame_data = data





################################################################################





class GUIviewer(object):
    def __init__(self, app):
        self.app = app

    def update_frame_data(self, data):
        self.app.set_frame_data(data)

    def expose_event(self):
        while Gtk.events_pending():
            Gtk.main_iteration()
        if not Gtk.events_pending():
            self.app.update_window()

    def keep_exposure(self):
        while Gtk.events_pending():
            Gtk.main_iteration()
            self.app.update_window()








if __name__ == "__main__":
    gui = GUIviewer(GUIwindow())

    import time
    finished = False
    while not finished:
        gui.expose_event()
#        
        print "rotx:", gui.app.widget.rotx
        gui.app.widget.rotx += 2
        gui.app.widget.roty += 1
        if gui.app.widget.rotx > 359:
            finished = True
#        time.sleep(0.02)
#        

    gui.keep_exposure()

    print("hello...")


########## end of file ##########
