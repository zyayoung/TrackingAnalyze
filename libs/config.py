def get_yolo_body(inputs, num_anchors, num_classes):
    from .yolo3.model import tiny_yolo_body, tiny_yolo_body_half, tiny_yolo_body_half_half, tiny_yolo_body_half_half_half, yolo_body
    return tiny_yolo_body(inputs, num_anchors, num_classes)

model_image_size = (224, 320)

classes = ["mouse"]
anchors = "24,24, 18,36, 36,18, 48,48, 36,72, 72,36"
model_path = "libs/model_data/yolo.h5"

strides = {0:32, 1:16, 2:8}
noise = True

data_arg = False
batch_size = 32
time_step = 3
show_dir = False

use_opti = False
