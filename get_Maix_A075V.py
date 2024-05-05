# Frame encode and decode impl
# structure of one frame

# packet	    field	    size	type	description
# 
# frame	        frame_id    8B	    uint64	indicate which frame is it
#               frame_stamp	8B	    uint64	xxx ms passed
#               config	    12B	    config	info needed by decode
#               deep_size	4B	    int32	size of deepth data
#               rgb_size	4B	    int32	size of rgb data
#               payload	    adaptive	bytes	payload data, size is not fixed
# 
# config	    trigger_mode	1B	uint8	0:STOP 1:AUTO 2:SINGLE
#               deep_mode	1B	    uint8	0:16bit 1:8bit
#               deep_shift	1B	    uint8	shift xxx from 16bit, just working for 8bit mode
#               ir_mode	    1B	    uint8	0:16bit 1:8bit
#               status_mode	1B	    uint8	0:16bit 1:2bit 2:8bit 3:1bit
#               status_mask	1B	    uint8	just working for 1bit mode 1:1 2:2 4:3
#               rgb_mode	1B	    uint8	0:YUV 1:JPG 2:NULL
#               rgb_res	    1B	    uint8	0:800*600 1:1600*1200
#               expose_time	4B	    int32	expose time of this tof, 0 means AE(auto expose)


import struct
import numpy as np
import cv2, time


def frame_config_decode(frame_config):
    '''
        @frame_config bytes

        @return fields, tuple (trigger_mode, deep_mode, deep_shift, ir_mode, status_mode, status_mask, rgb_mode, rgb_res, expose_time)
    '''
    return struct.unpack("<BBBBBBBBi", frame_config)


def frame_config_encode(trigger_mode=1, deep_mode=1, deep_shift=255, ir_mode=1, status_mode=2, status_mask=7, rgb_mode=1, rgb_res=0, expose_time=0):
    '''
        @trigger_mode, deep_mode, deep_shift, ir_mode, status_mode, status_mask, rgb_mode, rgb_res, expose_time

        @return frame_config bytes
    '''
    return struct.pack("<BBBBBBBBi",
                       trigger_mode, deep_mode, deep_shift, ir_mode, status_mode, status_mask, rgb_mode, rgb_res, expose_time)


def frame_payload_decode(frame_data: bytes, with_config: tuple):
    '''
        @frame_data, bytes

        @with_config, tuple (trigger_mode, deep_mode, deep_shift, ir_mode, status_mode, status_mask, rgb_mode, rgb_res, expose_time)

        @return imgs, tuple (deepth_img, ir_img, status_img, rgb_img)
    '''

    deep_data_size, rgb_data_size = struct.unpack("<ii", frame_data[:8])
    frame_payload = frame_data[8:]
    # 0:16bit 1:8bit, resolution: 320*240
    deepth_size = (320*240*2) >> with_config[1]
    deepth_img = struct.unpack("<%us" % deepth_size, frame_payload[:deepth_size])[
        0] if 0 != deepth_size else None
    frame_payload = frame_payload[deepth_size:]

    # 0:16bit 1:8bit, resolution: 320*240
    ir_size = (320*240*2) >> with_config[3]
    ir_img = struct.unpack("<%us" % ir_size, frame_payload[:ir_size])[
        0] if 0 != ir_size else None
    frame_payload = frame_payload[ir_size:]

    status_size = (320*240//8) * (16 if 0 == with_config[4] else
                                  2 if 1 == with_config[4] else 8 if 2 == with_config[4] else 1)
    status_img = struct.unpack("<%us" % status_size, frame_payload[:status_size])[
        0] if 0 != status_size else None
    frame_payload = frame_payload[status_size:]

    assert(deep_data_size == deepth_size+ir_size+status_size)

    rgb_size = len(frame_payload)
    assert(rgb_data_size == rgb_size)
    rgb_img = struct.unpack("<%us" % rgb_size, frame_payload[:rgb_size])[
        0] if 0 != rgb_size else None

    if (not rgb_img is None):
        if (1 == with_config[6]):
            jpeg = cv2.imdecode(np.frombuffer(
                rgb_img, 'uint8', rgb_size), cv2.IMREAD_COLOR)
            if not jpeg is None:
                rgb = cv2.cvtColor(jpeg, cv2.COLOR_BGR2RGB)
                rgb_img = rgb.tobytes()
            else:
                rgb_img = None
        # elif 0 == with_config[6]:
        #     yuv = np.frombuffer(rgb_img, 'uint8', rgb_size)
        #     print(len(yuv))
        #     if not yuv is None:
        #         rgb = cv2.cvtColor(yuv, cv2.COLOR_YUV420P2RGB)
        #         rgb_img = rgb.tobytes()
        #     else:
        #         rgb_img = None

    return (deepth_img, ir_img, status_img, rgb_img)

# Frame capture from network impl
import numpy as np
import matplotlib.pyplot as plt
import cv2
import requests

HOST = '192.168.233.1'
PORT = 80

def post_encode_config(config=frame_config_encode(),host=HOST, port=PORT):
    r = requests.post('http://{}:{}/set_cfg'.format(host, port), config)
    if(r.status_code == requests.codes.ok):
        return True
    return False

def post_CameraParmsBytes(cameraParms:bytes,host=HOST, port=PORT):
    r = requests.post('http://{}:{}/calibration'.format(host, port), cameraParms)
    if(r.status_code == requests.codes.ok):
        print("ok")

def get_frame_from_http(host=HOST, port=PORT):
    r = requests.get('http://{}:{}/getdeep'.format(host, port))
    if(r.status_code == requests.codes.ok):
        # print('Get deep image')
        deepimg = r.content
        # print('Length={}'.format(len(deepimg)))

        # # Define the format strings for each field in the packet
        # frame_format = "<QQ12sii"  # Frame ID (uint64), Frame Stamp (uint64), Config (12 bytes), Deep Size (int32), RGB Size (int32)
        # config_format = "BBBBBBBBI"  # Trigger Mode (uint8), Deep Mode (uint8), Deep Shift (uint8), IR Mode (uint8), Status Mode (uint8), Status Mask (uint8), RGB Mode (uint8), RGB Resolution (uint8), Expose Time (int32)

        # # frame_id, frame_stamp, config_bytes, deep_size, rgb_size = struct.unpack(frame_format, frame_data[:36])
        # frame_id, frame_stamp = struct.unpack("<QQ", frame_data[:16])
        # print(frame_id, frame_stamp)
        (frameid, stamp_msec) = struct.unpack('<QQ', deepimg[0:8+8])
        # print(frameid, stamp_msec/1000)
        last_time = time.time()
        return deepimg

# Example of how to show one frame
# network example(2D)
# local file example(2D, 3D)
# 
# show frame impl
import numpy as np
import matplotlib.pyplot as plt
import cv2

idx = 0
def show_frame(frame_data: bytes):
    config = frame_config_decode(frame_data[16:16+12])
    frame_bytes = frame_payload_decode(frame_data[16+12:], config)

    depth = np.frombuffer(frame_bytes[0], 'uint16' if 0 == config[1] else 'uint8').reshape(
        240, 320) if frame_bytes[0] else None

    ir = np.frombuffer(frame_bytes[1], 'uint16' if 0 == config[3] else 'uint8').reshape(
        240, 320) if frame_bytes[1] else None

    status = np.frombuffer(frame_bytes[2], 'uint16' if 0 == config[4] else 'uint8').reshape(
        240, 320) if frame_bytes[2] else None

    rgb = np.frombuffer(frame_bytes[3], 'uint8').reshape(
        (480, 640, 3) if config[6] == 1 else (600, 800, 3)) if frame_bytes[3] else None

    # cv2.imshow("depth", depth/2600)
    # print(depth.shape[0], np.max(depth))

    depth_i = depth.copy()
    # depth_i[220:260, 300:340] = depth[220:260, 300:340]/2
    color = (0, 0, 0)
    cv2.rectangle(depth_i, (150, 110), (170, 130), color, thickness = 1)
    # cv2.putText(depth_i, 'O', (220, 300), cv2.FONT_HERSHEY_COMPLEX, .6, color, 1)
    depth_pcolor = cv2.applyColorMap(
        (depth_i/600*255).astype('uint8'), cv2.COLORMAP_JET)

    depth_pcolor = cv2.resize(depth_pcolor, (depth.shape[1]*2, depth.shape[0]*2))
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    img = np.hstack((depth_pcolor, bgr))
    cv2.imshow("depth_pcolor", img)
    cv2.rectangle(depth, (150, 110), (170, 130), color, thickness = 1)
    depth_ir = np.hstack((depth, ir))
    cv2.imshow("depth", (depth_ir/2600*255).astype('uint8'))
    
    print(idx, depth_pcolor.shape, np.sum(depth[150:170, 110:130])/400)#np.sum(depth[220:260, 300:340])/1600)

    # cv2.imshow("rgb", rgb/255)
    cv2.waitKey(1)

    return depth, ir, rgb
    
    figsize = (12, 12)
    fig = plt.figure(figsize=figsize)

    ax1 = fig.add_subplot(221)
    if not depth is None:
        ax1.imshow(depth)
        # print(depth)
        # print(np.max(rgb))
        # np.save("fg1.npy", depth)
        # np.savetxt("depth.csv", (depth/4).astype('uint16'), delimiter="," )
    ax2 = fig.add_subplot(222)
    if not ir is None:
        ax2.imshow(ir)
    ax3 = fig.add_subplot(223)
    if not status is None:
        ax3.imshow(status)
    ax4 = fig.add_subplot(224)
    if not rgb is None:
        ax4.imshow(rgb)

    return depth, ir, rgb

# network example
last_time = time.time()
while True:
    if post_encode_config(frame_config_encode(1,0,255,0,2,7,1,0,0)):
        p = get_frame_from_http()
        # print(time.time() - last_time)
        last_time = time.time()
        show_frame(p)
        idx += 1
        # with open("rgbd.raw", 'wb') as f:
        #     f.write(p)
        #     f.flush()

exit()

# local file example
with open("rgbd.raw", 'rb') as fp:
    file_data = fp.read()
    fp.close()
    show_frame(file_data)
with open("rgbd.raw", 'rb') as fp:
    file_data = fp.read()
    fp.close()
    show_frame(file_data)

# local file example (3D)
# need library open3d, your python version should be less than 3.9(included)
# 
# website: https://pypi.org/project/open3d/
# install: pip install open3d
import numpy as np
import open3d as o3d

points = o3d.io.read_point_cloud("rgbd.pcd")
o3d.visualization.draw_geometries([points])

# utils
with open("./sipeed/CameraParms.json", "rb") as f:
    post_CameraParmsBytes(f.read())
with open("./sipeed/CameraParms.json", "rb") as f:
    post_CameraParmsBytes(f.read())
