'''
(*)~----------------------------------------------------------------------------------
 Pupil - eye tracking platform
 Copyright (C) 2012-2016  Pupil Labs

 Distributed under the terms of the GNU Lesser General Public License (LGPL v3.0).
 License details are in the file license.txt, distributed as part of this software.
----------------------------------------------------------------------------------~(*)
'''

import os
import cv2
import numpy as np
from methods import normalize,denormalize, makeCalibDir
from gl_utils import adjust_gl_view,clear_gl_screen,basic_gl_setup
import OpenGL.GL as gl
from glfw import *
import calibrate
from file_methods import Persistent_Dict,load_object,save_object

from time import time
from circle_detector import find_concetric_circles
import audio

from pyglui import ui
from pyglui.cygl.utils import draw_points, draw_points_norm, draw_polyline, draw_polyline_norm, RGBA,draw_concentric_circles
from pyglui.pyfontstash import fontstash
from pyglui.ui import get_opensans_font_path
from plugin import Calibration_Plugin

from gaze_mappers import Simple_Gaze_Mapper, Binocular_Gaze_Mapper, Binocular_Glint_Gaze_Mapper, Glint_Gaze_Mapper, Bilateral_Glint_Gaze_Mapper
from finish_calibration import finish_calibration

#logging
import logging
logger = logging.getLogger(__name__)



# window calbacks
def on_resize(window,w,h):
    active_window = glfwGetCurrentContext()
    glfwMakeContextCurrent(window)
    adjust_gl_view(w,h)
    glfwMakeContextCurrent(active_window)

# easing functions for animation of the marker fade in/out
def easeInOutQuad(t, b, c, d):
    """Robert Penner easing function examples at: http://gizma.com/easing/
    t = current time in frames or whatever unit
    b = beginning/start value
    c = change in value
    d = duration

    """
    t /= d/2
    if t < 1:
        return c/2*t*t + b
    t-=1
    return -c/2 * (t*(t-2) - 1) + b

def interp_fn(t,b,c,d,start_sample=15.,stop_sample=55.):
    # ease in, sample, ease out
    if t < start_sample:
        return easeInOutQuad(t,b,c,start_sample)
    elif t > stop_sample:
        return 1-easeInOutQuad(t-stop_sample,b,c,d-stop_sample)
    else:
        return 1.0


class Screen_Marker_Calibration(Calibration_Plugin):
    """Calibrate using a marker on your screen
    We use a ring detector that moves across the screen to 9 sites
    Points are collected at sites - not between

    """
    def __init__(self, g_pool,fullscreen=True,marker_scale=1.0,sample_duration=45):
        super(Screen_Marker_Calibration, self).__init__(g_pool)
        self.active = False
        self.detected = False
        self.screen_marker_state = 0.
        self.sample_duration =  sample_duration # number of frames to sample per site
        self.lead_in = 25 #frames of marker shown before starting to sample
        self.lead_out = 5 #frames of markers shown after sampling is donw
        self.session_settings = Persistent_Dict(os.path.join(g_pool.user_dir,'user_settings_screen_calibration') )

        self.active_site = None
        self.sites = []
        self.display_pos = None
        self.on_position = False

        self.markers = []
        self.pos = None

        self.marker_scale = marker_scale

        self._window = None

        self.menu = None
        self.button = None

        self.fullscreen = fullscreen
        self.clicks_to_close = 5

        self.glfont = fontstash.Context()
        self.glfont.add_font('opensans',get_opensans_font_path())
        self.glfont.set_size(32)
        self.glfont.set_color_float((0.2,0.5,0.9,1.0))
        self.glfont.set_align_string(v_align='center')





    def init_gui(self):
        self.monitor_idx = self.session_settings.get('monitor', 0)
        self.monitor_names = [glfwGetMonitorName(m) for m in glfwGetMonitors()]

        #primary_monitor = glfwGetPrimaryMonitor()
        self.info = ui.Info_Text("Calibrate gaze parameters using a screen based animation.")
        self.g_pool.calibration_menu.append(self.info)

        self.menu = ui.Growing_Menu('Controls')
        self.g_pool.calibration_menu.append(self.menu)
        self.menu.append(ui.Selector('monitor_idx',self,selection = range(len(self.monitor_names)),labels=self.monitor_names,label='Monitor'))
        self.menu.append(ui.Switch('fullscreen',self,label='Use fullscreen'))
        self.menu.append(ui.Slider('marker_scale',self,step=0.1,min=0.5,max=2.0,label='Marker size'))
        self.menu.append(ui.Slider('sample_duration',self,step=1,min=10,max=100,label='Sample duration'))

        self.button = ui.Thumb('active',self,setter=self.toggle,label='Calibrate',hotkey='c')
        self.button.on_color[:] = (.3,.2,1.,.9)
        self.g_pool.quickbar.insert(0,self.button)


    def deinit_gui(self):
        if self.menu:
            self.g_pool.calibration_menu.remove(self.menu)
            self.g_pool.calibration_menu.remove(self.info)
            self.menu = None
        if self.button:
            self.g_pool.quickbar.remove(self.button)
            self.button = None


    def toggle(self,_=None):
        if self.active:
            self.stop()
        else:
            self.start()



    def start(self):
        # ##############
        # DEBUG
        #self.stop()

        logger.info("Starting Calibration")
        if self.g_pool.detection_mapping_mode == '3d':
            self.sites = [  (.5, .5),
                            (0.,1.),(1.,1.),
                            (1., 0.),(0.,0.)]

        else:
            self.sites = [  (.25, .5), (0,.5), (.25, .25), (.25, .75),
                        (0.,1.),(.5,1.),(1.,1.),
                        (1.,.5), (.75, .25), (.75, .75),
                        (1., 0.),(.5, .25),(0.,0.),(.5,.5), (.5, .75),
                        (.75,.5)]



        self.calGlint = self.g_pool.calGlint
        self.active_site = 0
        self.active_site = self.sites.pop(0)
        self.active = True
        self.ref_list = []
        self.pupil_list = []
        self.glint_list = []
        self.glint_pupil_list =[]
        self.clicks_to_close = 5
        self.open_window("Calibration")

    def open_window(self,title='new_window'):
        if not self._window:
            if self.fullscreen:
                monitor = glfwGetMonitors()[self.monitor_idx]
                width,height,redBits,blueBits,greenBits,refreshRate = glfwGetVideoMode(monitor)
            else:
                monitor = None
                width,height= 640,360

            self._window = glfwCreateWindow(width, height, title, monitor=monitor, share=glfwGetCurrentContext())
            if not self.fullscreen:
                glfwSetWindowPos(self._window,200,0)

            glfwSetInputMode(self._window,GLFW_CURSOR,GLFW_CURSOR_HIDDEN)

            #Register callbacks
            glfwSetFramebufferSizeCallback(self._window,on_resize)
            glfwSetKeyCallback(self._window,self.on_key)
            glfwSetMouseButtonCallback(self._window,self.on_button)
            on_resize(self._window,*glfwGetFramebufferSize(self._window))

            # gl_state settings
            active_window = glfwGetCurrentContext()
            glfwMakeContextCurrent(self._window)
            basic_gl_setup()
            # refresh speed settings
            glfwSwapInterval(0)

            glfwMakeContextCurrent(active_window)




    def on_key(self,window, key, scancode, action, mods):
        if action == GLFW_PRESS:
            if key == GLFW_KEY_ESCAPE:
                self.clicks_to_close = 0

    def on_button(self,window,button, action, mods):
        if action ==GLFW_PRESS:
            self.clicks_to_close -=1


    def stop(self):
        # TODO: redundancy between all gaze mappers -> might be moved to parent class
        logger.info('Stopping Calibration')
        self.screen_marker_state = 0
        self.active = False
        audio.say("Stopping Calibration")
        logger.info("Stopping Calibration")
        self.smooth_pos = 0,0
        self.counter = 0
        self.close_window()
        self.active = False
        self.button.status_text = ''

        ref_list_copy = list(self.ref_list)
        ref_list = list(self.ref_list)
        glint_pupil_list_copy = list(self.glint_pupil_list)
        try:
            cal_pt_cloud_glint = calibrate.preprocess_data(list(self.glint_pupil_list), list(self.ref_list),id_filter=(0,1), glints=True)
            cal_pt_cloud_glint_eye0 = calibrate.preprocess_data(list(self.glint_pupil_list),list(self.ref_list),id_filter=(0,), glints=True)
            cal_pt_cloud_glint_eye1 = calibrate.preprocess_data(list(self.glint_pupil_list),list(self.ref_list),id_filter=(1,), glints=True)
        except:
            cal_pt_cloud_glint = calibrate.preprocess_data_glint(self.glint_pupil_list, ref_list_copy)

        if self.calGlint and len(cal_pt_cloud_glint) < 20:
            self.calGlint = False
            logger.warning("Did not collect enough data on glint. Calibrating without glint.")


        cal_pt_cloud_glint = np.array(cal_pt_cloud_glint)
        np.save(os.path.join(self.g_pool.user_dir,'cal_pt_cloud_glint.npy'),cal_pt_cloud_glint)
        #replace current gaze mapper with new

        # if self.calGlint:
        #     map_fn_glint,params_glint = calibrate.get_map_from_cloud(cal_pt_cloud_glint,self.g_pool.capture.frame_size,return_params=True, binocular=self.g_pool.binocular, glint=True)
        #     if self.g_pool.binocular:
        #         cal_pt_cloud_eye0_glint= np.array(cal_pt_cloud_glint_eye0)
        #         cal_pt_cloud_eye1_glint = np.array(cal_pt_cloud_glint_eye1)
        #         _,params_eye0_glint = calibrate.get_map_from_cloud(cal_pt_cloud_eye0_glint,self.g_pool.capture.frame_size,return_params=True, glint=True)
        #         _,params_eye1_glint = calibrate.get_map_from_cloud(cal_pt_cloud_eye1_glint,self.g_pool.capture.frame_size,return_params=True, glint=True)
        #         self.g_pool.plugins.add(Bilateral_Glint_Gaze_Mapper, args={'params':params_glint, 'params_eye0':params_eye0_glint, 'params_eye1':params_eye1_glint})
        #     else:
        #         self.g_pool.plugins.add(Glint_Gaze_Mapper, args={'params': params_glint, 'interpolParams': params})

        base_dir = self.g_pool.user_dir.rsplit(os.path.sep,1)[0]
        dir = makeCalibDir(base_dir)

        finish_calibration(self.g_pool,self.pupil_list,self.ref_list, dir = dir)

        try:
            #np.save(os.path.join(dir,'cal_pt_cloud_eye0.npy'),cal_pt_cloud_eye0)
            #np.save(os.path.join(dir,'cal_pt_cloud_eye1.npy'),cal_pt_cloud_eye1)
            np.save(os.path.join(dir,'cal_pt_cloud_glint.npy'),cal_pt_cloud_glint)
            np.save(os.path.join(dir,'cal_ref_list.npy'),ref_list)
            np.save(os.path.join(dir,'cal_pt_cloud_eye0_glint.npy'),cal_pt_cloud_glint_eye0)
            np.save(os.path.join(dir,'cal_pt_cloud_eye1_glint.npy'),cal_pt_cloud_glint_eye1)
        except:
            #np.save(os.path.join(dir,'cal_pt_cloud.npy'),cal_pt_cloud)
            np.save(os.path.join(dir,'cal_pt_cloud_glint.npy'),cal_pt_cloud_glint)
            np.save(os.path.join(dir,'cal_ref_list.npy'),ref_list)


    def close_window(self):
        if self._window:
            # enable mouse display
            active_window = glfwGetCurrentContext();
            glfwSetInputMode(self._window,GLFW_CURSOR,GLFW_CURSOR_NORMAL)
            glfwDestroyWindow(self._window)
            self._window = None
            glfwMakeContextCurrent(active_window)


    def update(self,frame,events):
        if self.active:
            recent_pupil_positions = events['pupil_positions']
            recent_glint_positions = events['glint_positions']
            recent_glint_pupil_positions = events['glint_pupil_vectors']
            gray_img = frame.gray

            if self.clicks_to_close <=0:
                self.stop()
                return

            #detect the marker
            self.markers = find_concetric_circles(gray_img,min_ring_count=4)

            if len(self.markers) > 0:
                self.detected= True
                marker_pos = self.markers[0][0][0] # first marker, innermost ellipse,center
                self.pos = normalize(marker_pos,(frame.width,frame.height),flip_y=True)

            else:
                self.detected = False
                self.pos = None #indicate that no reference is detected

            #use np.arrays for per element wise math
            self.display_pos = np.array(self.active_site)
            p_window_size = glfwGetWindowSize(self._window)
            screen_pos = denormalize(self.display_pos,p_window_size,flip_y=True)
            #only save a valid ref position if within sample window of calibraiton routine
            on_position = self.lead_in < self.screen_marker_state < (self.lead_in+self.sample_duration)

            if on_position and self.detected:
                ref = {}
                ref["norm_pos"] = self.pos
                ref["screen_pos"] = marker_pos
                ref["timestamp"] = frame.timestamp
                ref["screenpos"] = self.actualScreenPos(screen_pos)
                self.ref_list.append(ref)

            #always save pupil positions
            for p_pt in recent_pupil_positions:
                if p_pt['confidence'] > self.g_pool.pupil_confidence_threshold:
                    self.pupil_list.append(p_pt)
            for g_pt in recent_glint_positions:
                if g_pt[0][3]:
                    self.glint_list.append(g_pt[0])
            for g_p_pt in recent_glint_pupil_positions:
                if g_p_pt['glint_found'] and g_p_pt['pupil_confidence'] > self.g_pool.pupil_confidence_threshold:
                    self.glint_pupil_list.append(g_p_pt)

            # Animate the screen marker
            if self.screen_marker_state < self.sample_duration+self.lead_in+self.lead_out:
                if self.detected or not on_position:
                    self.screen_marker_state += 1
            else:
                self.screen_marker_state = 0
                if not self.sites:
                    self.stop()
                    return
                self.active_site = self.sites.pop(0)
                logger.debug("Moving screen marker to site at %s %s"%tuple(self.active_site))



            #use np.arrays for per element wise math
            self.display_pos = np.array(self.active_site)
            self.on_position = on_position
            self.button.status_text = '%s / %s'%(self.active_site,9)




    def gl_display(self):
        """
        use gl calls to render
        at least:
            the published position of the reference
        better:
            show the detected postion even if not published
        """

        # debug mode within world will show green ellipses around detected ellipses
        if self.active and self.detected:
            for marker in self.markers:
                e = marker[-1] #outermost ellipse
                pts = cv2.ellipse2Poly( (int(e[0][0]),int(e[0][1])),
                                    (int(e[1][0]/2),int(e[1][1]/2)),
                                    int(e[-1]),0,360,15)
                draw_polyline(pts,1,RGBA(0.,1.,0.,1.))

        else:
            pass
        if self._window:
            self.gl_display_in_window()

    def draw_rect(self, x, y, width, height):
        gl.glBegin(gl.GL_QUADS)                               # start drawing a rectangle
        gl.glVertex2f(x, y)                                   # bottom left point
        gl.glVertex2f(x + width, y)                           # bottom right point
        gl.glVertex2f(x + width, y + height)                  # top right point
        gl.glVertex2f(x, y + height)                          # top left point
        gl.glEnd()


    def gl_display_in_window(self):
        active_window = glfwGetCurrentContext()
        if glfwWindowShouldClose(self._window):
            self.close_window()
            return

        glfwMakeContextCurrent(self._window)
        clear_gl_screen()
        gl.glColor3f(.85, .85, .85)
        self.draw_rect(0, 0, 2000, 2000)
        hdpi_factor = glfwGetFramebufferSize(self._window)[0]/glfwGetWindowSize(self._window)[0]
        r = 110*self.marker_scale * hdpi_factor
        gl.glMatrixMode(gl.GL_PROJECTION)
        gl.glLoadIdentity()
        p_window_size = glfwGetWindowSize(self._window)
        gl.glOrtho(0,p_window_size[0],p_window_size[1],0 ,-1,1)
        # Switch back to Model View Matrix
        gl.glMatrixMode(gl.GL_MODELVIEW)
        gl.glLoadIdentity()

        def map_value(value,in_range=(0,1),out_range=(0,1)):
            ratio = (out_range[1]-out_range[0])/(in_range[1]-in_range[0])
            return (value-in_range[0])*ratio+out_range[0]

        pad = 0.1*1920
        pad = .7*r

        screen_pos = map_value(self.display_pos[0],out_range=(pad,p_window_size[0]-pad)),map_value(self.display_pos[1],out_range=(p_window_size[1]-pad,pad))
        alpha = interp_fn(self.screen_marker_state,0.,1.,float(self.sample_duration+self.lead_in+self.lead_out),float(self.lead_in),float(self.sample_duration+self.lead_in))

        draw_concentric_circles(screen_pos,r,6,alpha)
        #some feedback on the detection state

        if self.detected and self.on_position:
            draw_points([screen_pos],size=10*self.marker_scale,color=RGBA(0.,.8,0.,alpha),sharpness=0.5)
        else:
            draw_points([screen_pos],size=10*self.marker_scale,color=RGBA(0.8,0.,0.,alpha),sharpness=0.5)

        if self.clicks_to_close <5:
            self.glfont.set_size(int(p_window_size[0]/30.))
            self.glfont.draw_text(p_window_size[0]/2.,p_window_size[1]/4.,'Touch %s more times to cancel calibration.'%self.clicks_to_close)

        glfwSwapBuffers(self._window)
        glfwMakeContextCurrent(active_window)

    def actualScreenPos(self, screen_pos):
        p_window_size = glfwGetWindowSize(self._window)
        hdpi_factor = glfwGetFramebufferSize(self._window)[0]/glfwGetWindowSize(self._window)[0]
        r = 110*self.marker_scale * hdpi_factor
        x = screen_pos[0] + r*.6
        y = screen_pos[1] +  r*.7
        x /= (p_window_size[0]+2*0.6*r)
        y /= (p_window_size[1]+2*0.7*r)
        x *= p_window_size[0]
        y *= p_window_size[1]
        return (int(x),int(y))

    def get_init_dict(self):
        d = {}
        d['fullscreen'] = self.fullscreen
        d['marker_scale'] = self.marker_scale
        return d

    def cleanup(self):
        """gets called when the plugin get terminated.
           either voluntarily or forced.
        """
        self.session_settings['monitor'] = self.monitor_idx
        self.session_settings.close()

        if self.active:
            self.stop()
        if self._window:
            self.close_window()
        self.deinit_gui()


