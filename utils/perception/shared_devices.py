from utils.vision.multicam import MultiCam
# realsense_serial_numbers = ['042222070680', '145422071576', '241222076578']
realsense_serial_numbers = ['042222070680'] #, '838212072814', '241222076578']
# realsense_serial_numbers = ['042222070680', '145422071576']
# realsense_serial_numbers = ['042222070680']
# realsense_serial_numbers = ['042222070680', '145422071576']
multi_cam = MultiCam(realsense_serial_numbers)

# import hid
# gamepad = hid.device()
# gamepad.open(0x1a86, 0xe026)
# gamepad.set_nonblocking(True)