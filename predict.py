import os
import json

import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt

from model import PRLNet
from draw_utils import draw_keypoints
import transforms


def predict_all_person():
    # TODO
    pass

def connect_keypoints(image, keypoints, connections, color='r', linewidth=2):
    """
    连接关键点并绘制在图像上。
    
    参数:
    image -- 原始图像，用于背景
    keypoints -- 关键点坐标列表，形状为(N, 2)，N为关键点数量
    connections -- 关键点连接的索引对列表，如[(0, 1), (1, 2), ...]
    color -- 连线颜色，默认红色
    linewidth -- 线宽，默认2
    """
    plt.imshow(image)
    for connection in connections:
        start, end = connection
        plt.plot([keypoints[start, 0], keypoints[end, 0]],   # x坐标
             [keypoints[start, 1], keypoints[end, 1]],   # y坐标
             color=color, linewidth=linewidth)
    plt.axis('off')  # 隐藏坐标轴
    plt.show()



def predict_single_person():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"using device: {device}")

    flip_test = True
    resize_hw = (256, 192)
    img_path = "./person.png"
    weights_path = "./prlnet.pth"
    keypoint_json_path = "person_keypoints.json"
    assert os.path.exists(img_path), f"file: {img_path} does not exist."
    assert os.path.exists(weights_path), f"file: {weights_path} does not exist."
    assert os.path.exists(keypoint_json_path), f"file: {keypoint_json_path} does not exist."

    data_transform = transforms.Compose([
        transforms.AffineTransform(scale=(1.25, 1.25), fixed_size=resize_hw),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # read json file
    with open(keypoint_json_path, "r") as f:
        person_info = json.load(f)

    # read single-person image
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_tensor, target = data_transform(img, {"box": [0, 0, img.shape[1] - 1, img.shape[0] - 1]})
    img_tensor = torch.unsqueeze(img_tensor, dim=0)

    # create model

    model = PRLNet(base_channel=32)
    weights = torch.load(weights_path, map_location=device)
    weights = weights if "model" not in weights else weights["model"]
    model.load_state_dict(weights)
    model.to(device)
    model.eval()

    with torch.inference_mode():
        outputs = model(img_tensor.to(device))

        if flip_test:
            flip_tensor = transforms.flip_images(img_tensor)
            flip_outputs = torch.squeeze(
                transforms.flip_back(model(flip_tensor.to(device)), person_info["flip_pairs"]),
            )
            # feature is not aligned, shift flipped heatmap for higher accuracy
            flip_outputs[..., 1:] = flip_outputs.clone()[..., 0: -1]
            outputs = (outputs + flip_outputs) * 0.5

        keypoints, scores = transforms.get_final_preds(outputs, [target["reverse_trans"]], True)
        keypoints = np.squeeze(keypoints)
        scores = np.squeeze(scores)
        
        plot_img = draw_keypoints(img, keypoints, scores, thresh=0.2, r=3)
        connections_example = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (0, 9), (9, 10), (10, 11), (0, 12), (12, 13), (13, 14), (0, 15), (15, 16) ] 
        connect_keypoints(plot_img, keypoints,connections_example,color='g')  
        plt.imshow(connect_keypoints)
        plt.show()
        plot_img.save("test_result.jpg")

if __name__ == '__main__':
    predict_single_person()
