# python demo/webcam_demo_1.py configs/rtmdet/rtmdet-ins_x_8xb16-300e_coco.py checkpoints/rtmdet-ins_x_8xb16-300e_coco_20221124_111313-33d4595b.pth
# C:\Users\User\Desktop\mmdetection-main\mmdet\models\utils\misc.py 345行 改想要的label
# C:\Users\User\Desktop\mmdetection-main\mmdet\visualization\local_visualizer.py 改text, bbox 讓want_label以外變黑
import argparse
import numba

import cv2
# import torch
import math
import time
import numpy as np
import pyrealsense2 as rs
import os

ori_img_path='original_images'
depth_img_path='depth_images'

if not os.path.exists(ori_img_path):
    os.makedirs(ori_img_path)
if not os.path.exists(depth_img_path):
    os.makedirs(depth_img_path)

class AppState:

    def __init__(self, *args, **kwargs):
        self.WIN_NAME = 'RealSense'
        self.pitch, self.yaw = math.radians(-10), math.radians(-15)
        self.translation = np.array([0, 0, -1], dtype=np.float32)
        self.distance = 2
        self.prev_mouse = 0, 0
        self.mouse_btns = [False, False, False]
        self.paused = False
        self.decimate = 1
        self.scale = True
        self.color = True

    def reset(self):
        self.pitch, self.yaw, self.distance = 0, 0, 2
        self.translation[:] = 0, 0, -1

    @property
    def rotation(self):
        Rx, _ = cv2.Rodrigues((self.pitch, 0, 0))
        Ry, _ = cv2.Rodrigues((0, self.yaw, 0))
        return np.dot(Ry, Rx).astype(np.float32)

    @property
    def pivot(self):
        return self.translation + np.array((0, 0, self.distance), dtype=np.float32)


state = AppState()

# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()

pipeline_wrapper = rs.pipeline_wrapper(pipeline)
pipeline_profile = config.resolve(pipeline_wrapper)
device = pipeline_profile.get_device()

found_rgb = False
for s in device.sensors:
    if s.get_info(rs.camera_info.name) == 'RGB Camera':
        found_rgb = True
        break
if not found_rgb:
    print("The demo requires Depth camera with Color sensor")
    exit(0)

config.enable_stream(rs.stream.depth, rs.format.z16, 30)
config.enable_stream(rs.stream.color, rs.format.bgr8, 30)

# Start streaming
pipeline.start(config)

# Get stream profile and camera intrinsics
profile = pipeline.get_active_profile()
depth_profile = rs.video_stream_profile(profile.get_stream(rs.stream.depth))
depth_intrinsics = depth_profile.get_intrinsics()
w, h = depth_intrinsics.width, depth_intrinsics.height

# Processing blocks
pc = rs.pointcloud()
decimate = rs.decimation_filter()
decimate.set_option(rs.option.filter_magnitude, 2 ** state.decimate)
colorizer = rs.colorizer()
colorizer.set_option(rs.option.visual_preset, 1) # 0=Dynamic, 1=Fixed, 2=Near, 3=Far
value_min = 0
value_max = 0.5
colorizer.set_option(rs.option.min_distance, value_min)
colorizer.set_option(rs.option.max_distance, value_max)

def mouse_cb(event, x, y, flags, param):

    if event == cv2.EVENT_LBUTTONDOWN:
        state.mouse_btns[0] = True

    if event == cv2.EVENT_LBUTTONUP:
        state.mouse_btns[0] = False

    if event == cv2.EVENT_RBUTTONDOWN:
        state.mouse_btns[1] = True

    if event == cv2.EVENT_RBUTTONUP:
        state.mouse_btns[1] = False

    if event == cv2.EVENT_MBUTTONDOWN:
        state.mouse_btns[2] = True

    if event == cv2.EVENT_MBUTTONUP:
        state.mouse_btns[2] = False

    if event == cv2.EVENT_MOUSEMOVE:

        h, w = out.shape[:2]
        dx, dy = x - state.prev_mouse[0], y - state.prev_mouse[1]

        if state.mouse_btns[0]:
            state.yaw += float(dx) / w * 2
            state.pitch -= float(dy) / h * 2

        elif state.mouse_btns[1]:
            dp = np.array((dx / w, dy / h, 0), dtype=np.float32)
            state.translation -= np.dot(state.rotation, dp)

        elif state.mouse_btns[2]:
            dz = math.sqrt(dx**2 + dy**2) * math.copysign(0.01, -dy)
            state.translation[2] += dz
            state.distance -= dz

    if event == cv2.EVENT_MOUSEWHEEL:
        dz = math.copysign(0.1, flags)
        state.translation[2] += dz
        state.distance -= dz

    state.prev_mouse = (x, y)

cv2.namedWindow(state.WIN_NAME, cv2.WINDOW_AUTOSIZE)
cv2.resizeWindow(state.WIN_NAME, w, h)
cv2.setMouseCallback(state.WIN_NAME, mouse_cb)

def project(v):
    """project 3d vector array to 2d"""
    h, w = out.shape[:2]
    view_aspect = float(h)/w

    # ignore divide by zero for invalid depth
    with np.errstate(divide='ignore', invalid='ignore'):
        proj = v[:, :-1] / v[:, -1, np.newaxis] * \
            (w*view_aspect, h) + (w/2.0, h/2.0)

    # near clipping
    znear = 0.03
    proj[v[:, 2] < znear] = np.nan
    return proj


def view(v):
    """apply view transformation on vector array"""
    return np.dot(v - state.pivot, state.rotation) + state.pivot - state.translation


def line3d(out, pt1, pt2, color=(0x80, 0x80, 0x80), thickness=1):
    """draw a 3d line from pt1 to pt2"""
    p0 = project(pt1.reshape(-1, 3))[0]
    p1 = project(pt2.reshape(-1, 3))[0]
    if np.isnan(p0).any() or np.isnan(p1).any():
        return
    p0 = tuple(p0.astype(int))
    p1 = tuple(p1.astype(int))
    rect = (0, 0, out.shape[1], out.shape[0])
    inside, p0, p1 = cv2.clipLine(rect, p0, p1)
    if inside:
        cv2.line(out, p0, p1, color, thickness, cv2.LINE_AA)


def grid(out, pos, rotation=np.eye(3), size=1, n=10, color=(0x80, 0x80, 0x80)):
    """draw a grid on xz plane"""
    pos = np.array(pos)
    s = size / float(n)
    s2 = 0.5 * size
    for i in range(0, n+1):
        x = -s2 + i*s
        line3d(out, view(pos + np.dot((x, 0, -s2), rotation)),
               view(pos + np.dot((x, 0, s2), rotation)), color)
    for i in range(0, n+1):
        z = -s2 + i*s
        line3d(out, view(pos + np.dot((-s2, 0, z), rotation)),
               view(pos + np.dot((s2, 0, z), rotation)), color)


def axes(out, pos, rotation=np.eye(3), size=0.075, thickness=2):
    """draw 3d axes"""
    line3d(out, pos, pos +
           np.dot((0, 0, size), rotation), (0xff, 0, 0), thickness)
    line3d(out, pos, pos +
           np.dot((0, size, 0), rotation), (0, 0xff, 0), thickness)
    line3d(out, pos, pos +
           np.dot((size, 0, 0), rotation), (0, 0, 0xff), thickness)


def frustum(out, intrinsics, color=(0x40, 0x40, 0x40)):
    """draw camera's frustum"""
    orig = view([0, 0, 0])
    w, h = intrinsics.width, intrinsics.height

    for d in range(1, 6, 2):
        def get_point(x, y):
            p = rs.rs2_deproject_pixel_to_point(intrinsics, [x, y], d)
            line3d(out, orig, view(p), color)
            return p

        top_left = get_point(0, 0)
        top_right = get_point(w, 0)
        bottom_right = get_point(w, h)
        bottom_left = get_point(0, h)

        line3d(out, view(top_left), view(top_right), color)
        line3d(out, view(top_right), view(bottom_right), color)
        line3d(out, view(bottom_right), view(bottom_left), color)
        line3d(out, view(bottom_left), view(top_left), color)


def pointcloud(out, verts, texcoords, color, painter=True):
    """draw point cloud with optional painter's algorithm"""
    if painter:
        # Painter's algo, sort points from back to front

        # get reverse sorted indices by z (in view-space)
        # https://gist.github.com/stevenvo/e3dad127598842459b68
        v = view(verts)
        s = v[:, 2].argsort()[::-1]
        proj = project(v[s])
    else:
        proj = project(view(verts))

    if state.scale:
        proj *= 0.5**state.decimate

    h, w = out.shape[:2]

    # proj now contains 2d image coordinates
    j, i = proj.astype(np.uint32).T

    # create a mask to ignore out-of-bound indices
    im = (i >= 0) & (i < h)
    jm = (j >= 0) & (j < w)
    m = im & jm

    cw, ch = color.shape[:2][::-1]
    if painter:
        # sort texcoord with same indices as above
        # texcoords are [0..1] and relative to top-left pixel corner,
        # multiply by size and add 0.5 to center
        v, u = (texcoords[s] * (cw, ch) + 0.5).astype(np.uint32).T
    else:
        v, u = (texcoords * (cw, ch) + 0.5).astype(np.uint32).T
    # clip texcoords to image
    np.clip(u, 0, ch-1, out=u)
    np.clip(v, 0, cw-1, out=v)

    # perform uv-mapping
    out[i[m], j[m]] = color[u[m], v[m]]


out = np.empty((h, w, 3), dtype=np.uint8)

@numba.jit(nopython=True)
def image_matting(pred_img_data : np.ndarray, image : np.ndarray):
    for i in range(480):
        for j in range(640):
            for k in range(3):
                if pred_img_data[i, j, k] >= 20  and pred_img_data[i, j, k] <= 234:
                    break
                if k == 2:
                    image[i:i+3, j:j+3, k] = 0
                    image[i:i+3, j:j+3, k-1] = 0
                    image[i:i+3, j:j+3, k-2] = 0
    return image

num = 1

# counting the number of the images
while True:
    if (not os.path.exists(f'{ori_img_path}/original_{num}.jpg')) and (not os.path.exists(f'{depth_img_path}/depth_{num}.jpg')):
        if num>1:
            num=num-1
        break
    else:
        num=num+1

align_to = rs.stream.color
align = rs.align(align_to)

while True:
    if not state.paused:
        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()

        # depth_frame = frames.get_depth_frame()
        # color_frame = frames.get_color_frame()
        
        # # depth_frame.shape (720, 1280)----->(360, 640)
        # depth_frame = decimate.process(depth_frame)

        # # Grab new intrinsics (may be changed by decimation)
        # depth_intrinsics = rs.video_stream_profile(
        #     depth_frame.profile).get_intrinsics()
        # w, h = depth_intrinsics.width, depth_intrinsics.height

        # depth_image = np.asanyarray(depth_frame.get_data())

        # Align the depth frame to color frame
        aligned_frames = align.process(frames)

        # Get aligned frames
        aligned_depth_frame = aligned_frames.get_depth_frame() # aligned_depth_frame is a 640x480 depth image
        color_frame = aligned_frames.get_color_frame()

        # Validate that both frames are valid
        if not aligned_depth_frame or not color_frame:
            continue

        depth_image = np.asanyarray(aligned_depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())
        # color_image = cv2.resize(color_image, (480, 640))
        depth_colormap = np.asanyarray(colorizer.colorize(aligned_depth_frame).get_data())
        # depth_image = depth_image/16
        # depth_image = depth_image.astype(np.uint8)
        # ii = cv2.applyColorMap(depth_image, cv2.COLORMAP_JET)
        # depth_colormap = cv2.resize(depth_colormap, (480, 640))
        # color_image=cv2.resize(color_image,(480, 640))
        # color_image = cv2.resize(color_image, (480, 640))
        cv2.imwrite(f'application/original_{num}.jpg', color_image)
        cv2.imwrite(f'application/depth_{num}.jpg', depth_colormap)
        
        depth_colormap=cv2.resize(depth_colormap,(853,480))
        color_image=cv2.resize(color_image,(853,480))
        cv2.rectangle(depth_colormap,(75,19),(715,453),(0,0,255),3)
        cv2.rectangle(color_image,(75,19),(715,453),(0,0,255),3)

        # original_img = original_img[19:453, 75:715]
        # heat_img = heat_img[46:, :]
        cv2.imshow('depth_image', depth_colormap)
        cv2.imshow('RGB_image',color_image)

        color_image = cv2.resize(color_image, (1920, 1080))

        # depth_colormap = np.asanyarray(
        #     colorizer.colorize(aligned_depth_frame).get_data())

        if state.color:
            mapped_frame, color_source = color_frame, color_image
        else:
            mapped_frame, color_source = aligned_depth_frame, depth_colormap

        points = pc.calculate(aligned_depth_frame)
        pc.map_to(mapped_frame)

        # Pointcloud data to arrays
        v, t = points.get_vertices(), points.get_texture_coordinates()
        verts = np.asanyarray(v).view(np.float32).reshape(-1, 3)  # xyz
        texcoords = np.asanyarray(t).view(np.float32).reshape(-1, 2)  # uv
    # now = time.time()

    # out.fill(0)

    # grid(out, (0, 0.5, 1), size=1, n=10)
    # frustum(out, depth_intrinsics)
    # axes(out, view([0, 0, 0]), state.rotation, size=0.1, thickness=1)
    
    # if not state.scale or out.shape[:2] == (h, w):
    #     pointcloud(out, verts, texcoords, color_source)
    # else:
    #     tmp = np.zeros((h, w, 3), dtype=np.uint8)
    #     pointcloud(tmp, verts, texcoords, color_source)
    #     tmp = cv2.resize(
    #         tmp, out.shape[:2][::-1], interpolation=cv2.INTER_NEAREST)
    #     np.putmask(out, tmp > 0, tmp)

    # if any(state.mouse_btns):
    #     axes(out, view(state.pivot), state.rotation, thickness=4)

    # dt = time.time() - now

    # cv2.setWindowTitle(
    #     state.WIN_NAME, "RealSense (%dx%d) %dFPS (%.2fms) %s" %
    #     (w, h, 1.0/dt, dt*1000, "PAUSED" if state.paused else ""))
    
    # cv2.imshow(state.WIN_NAME, out)
    key = cv2.waitKey(1)

    if key == ord("r"):
        state.reset()

    if key == ord("p"):
        state.paused ^= True

    if key == ord("j"):
        num = num+1

    if key == ord("d"):
        state.decimate = (state.decimate + 1) % 3
        decimate.set_option(rs.option.filter_magnitude, 2 ** state.decimate)

    if key == ord("z"):
        state.scale ^= True

    if key == ord("c"):
        state.color ^= True

    if key == ord("s"):
        cv2.imwrite('./out.png', out)

    if key == ord("e"):
        points.export_to_ply('./out.ply', mapped_frame)

    if key in (27, ord("q")) or cv2.getWindowProperty(state.WIN_NAME, cv2.WND_PROP_AUTOSIZE) < 0:
        break

# Stop streaming
pipeline.stop()