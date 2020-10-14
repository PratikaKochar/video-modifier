#!/usr/bin/python
# -*- encoding: utf-8 -*-

from logger import setup_logger
from model import BiSeNet

import torch

import os
import os.path as osp
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from skimage.filters import gaussian
import cv2
import sys

def sharpen(img):
    img = img * 1.0
    gauss_out = gaussian(img, sigma=5, multichannel=True)

    alpha = 1.5
    img_out = (img - gauss_out) * alpha + img

    img_out = img_out / 255.0

    mask_1 = img_out < 0
    mask_2 = img_out > 1

    img_out = img_out * (1 - mask_1)
    img_out = img_out * (1 - mask_2) + mask_2
    img_out = np.clip(img_out, 0, 1)
    img_out = img_out * 255
    return np.array(img_out, dtype=np.uint8)


def hair(image, parsing, part=17, color=[159, 29, 53]):
    b, g, r = color      #[10, 50, 250]       # [10, 250, 10]
    tar_color = np.zeros_like(image)
    tar_color[:, :, 0] = b
    tar_color[:, :, 1] = g
    tar_color[:, :, 2] = r

    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    tar_hsv = cv2.cvtColor(tar_color, cv2.COLOR_BGR2HSV)

    if part == 12 or part == 13:
        image_hsv[:, :, 0:2] = tar_hsv[:, :, 0:2]
    else:
        image_hsv[:, :, 0:1] = tar_hsv[:, :, 0:1]

    changed = cv2.cvtColor(image_hsv, cv2.COLOR_HSV2BGR)

    if part == 17:
        changed = sharpen(changed)

    changed[parsing != part] = image[parsing != part]
    #changed = cv2.resize(changed, (512, 512))
    return changed

def vis_parsing_maps(makeup,blurred,width, height,im, parsing_anno, stride, save_im=False, save_path='vis_results/parsing_map_on_im.jpg'):
    # Colors for all 20 parts
    part_colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0],
                   [255, 0, 85], [255, 0, 170],
                   [0, 255, 0], [85, 255, 0], [170, 255, 0],
                   [0, 255, 85], [0, 255, 170],
                   [0, 0, 255], [85, 0, 255], [170, 0, 255],
                   [0, 85, 255], [0, 170, 255],
                   [255, 255, 0], [255, 255, 85], [255, 255, 170],
                   [255, 0, 255], [255, 85, 255], [255, 170, 255],
                   [0, 255, 255], [85, 255, 255], [170, 255, 255]]

    im = np.array(im)

    vis_im = im.copy().astype(np.uint8)
    orig_im = vis_im
    vis_parsing_anno = parsing_anno.copy().astype(np.uint8)
    #print(vis_parsing_anno.shape,'vis anno')
    vis_parsing_anno = cv2.resize(vis_parsing_anno, None, fx=stride, fy=stride, interpolation=cv2.INTER_NEAREST)
    vis_parsing_anno_color = np.zeros((vis_parsing_anno.shape[0], vis_parsing_anno.shape[1], 3)) + 255
    #vis_parsing_anno_color_person = np.zeros((vis_parsing_anno.shape[0], vis_parsing_anno.shape[1], 3)) + 255
    #vis_parsing_anno_color = np.zeros((vis_parsing_anno.shape[0], vis_parsing_anno.shape[1]))

    num_of_class = np.max(vis_parsing_anno)
    #print('nmber of classes', num_of_class)
    for pi in range(1, num_of_class + 1):

        index = np.where(vis_parsing_anno == pi)
        vis_parsing_anno_color[index[0], index[1],:] = [1,1,1]#part_colors[pi]
        #vis_parsing_anno_color[index[0], index[1], :] = [1,1,1]

    mask2 = np.where((vis_parsing_anno == 0), 1, 0).astype('uint8')
    img = im * mask2[:, :, np.newaxis]
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    blur_background  = cv2.GaussianBlur(img, (15, 15), 0)


    vis_parsing_anno_color = vis_parsing_anno_color.astype(np.uint8)
    # print(vis_parsing_anno_color.shape, vis_im.shape)

    img1 = Image.open("./datavideo/background.jpg")
    #print(img1.size)
    #print('shown')
    im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
    # get masked foreground
    fg_masked = cv2.bitwise_and(im, im, mask=vis_parsing_anno)

    image1 = img1.resize((512, 512), Image.BILINEAR)
    #image1.show()
    im = np.array(image1)
    vis_im = im.copy().astype(np.uint8)

    bk_masked = cv2.bitwise_and(vis_im, vis_im, mask=mask2)
    final = cv2.bitwise_or(fg_masked, bk_masked )
    dst = cv2.add(fg_masked, bk_masked )
    rows, cols, channels = im.shape
    vis_im[0:rows, 0:cols] = dst
    blurred_image = cv2.bitwise_or(fg_masked,blur_background)
    #vis_im = cv2.cvtColor(vis_im, cv2.COLOR_RGB2BGR)
    #print(vis_parsing_anno_color.shape, vis_im.shape)
    #vis_im = cv2.addWeighted(vis_im , 0.4, vis_parsing_anno_color, 0.6, 0)
    #vis_im= Image.fromarray(vis_im, 'RGB')
    #vis_im.show()



    # Save result or not
    if save_im:
        cv2.imwrite(save_path[:-4] +'.png', vis_parsing_anno)
        cv2.imwrite(save_path, vis_im, [int(cv2.IMWRITE_JPEG_QUALITY), 100])

    dim = (width, height)

    vis_im = cv2.resize(vis_im, dim, interpolation = cv2.INTER_AREA)
    vis_parsing_anno = cv2.resize(vis_parsing_anno , dim, interpolation = cv2.INTER_AREA)
    blurred_image = cv2.resize(blurred_image, dim, interpolation = cv2.INTER_AREA)
    if blurred:
        return blurred_image,vis_parsing_anno
    if makeup:

        orig_im = cv2.resize(orig_im, dim, interpolation=cv2.INTER_AREA)
        orig_im = cv2.cvtColor(orig_im, cv2.COLOR_RGB2BGR)
        return orig_im,vis_parsing_anno
    return vis_im,vis_parsing_anno

def evaluate(makeup,blurred,respth='./res/test_res', dspth='./data', cp='model_final_diss.pth'):

    if not os.path.exists(respth):
        os.makedirs(respth)

    n_classes = 19
    net = BiSeNet(n_classes=n_classes)
    net.cuda()
    save_pth = osp.join('res/cp', cp)
    net.load_state_dict(torch.load(save_pth))
    net.eval()

    to_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    with torch.no_grad():
        #for image_path in os.listdir(dspth):
            #img = Image.open(osp.join(dspth, image_path))
            img = Image.open(dspth)
            width, height = img.size
            image = img.resize((512, 512), Image.BILINEAR)

            img = to_tensor(image)
            img = torch.unsqueeze(img, 0)
            img = img.cuda()
            out = net(img)[0]
            parsing = out.squeeze(0).cpu().numpy().argmax(0)
            # print(parsing)
            #print(np.unique(parsing))
            image_path=''
            res,vis_parsing_anno = vis_parsing_maps(makeup,blurred,width, height, image, parsing, stride=1, save_im=False, save_path=osp.join(respth, image_path))

    return res,vis_parsing_anno





if __name__ == "__main__":
    #res = evaluate(dspth='./data/new.jpg', cp='79999_iter.pth')


    import numpy as np
    import cv2


    import imageio

    operation_type = sys.argv[1]

    reader = imageio.get_reader('./datavideo/video.mp4')
    fps = reader.get_meta_data()['fps']

    writer = imageio.get_writer('./datavideo/video6.mp4', fps=fps)

    for im in reader:
        #im = evaluate(dspth=im, cp='79999_iter.pth')

        cv2.imwrite("./frames/testframe" + ".jpg", im)

        if operation_type == 'blur':
            print('Blur mode')
            frame,vis_parsing_anno = evaluate(makeup=False,blurred=True,dspth='./frames/testframe' + '.jpg', cp='79999_iter.pth')
        elif operation_type == 'change':
            print('Background change mode')
            frame, vis_parsing_anno = evaluate(makeup=False,blurred=False, dspth='./frames/testframe' + '.jpg', cp='79999_iter.pth')
        elif operation_type == 'makeup':
            print('Makeup mode')
            frame, vis_parsing_anno = evaluate(makeup=True,blurred=False, dspth='./frames/testframe' + '.jpg', cp='79999_iter.pth')
            table = {
                'hair': 17,
                'upper_lip': 12,
                'lower_lip': 13,
                'nose': 10,
                'face': 1
            }
            #image = cv2.imread(r"C:\Users\PRATIKA\Desktop\face-parsing.PyTorch\frames\testframe" + ".jpg")
            ori = frame.copy()
            #im = cv2.imread(parsing_path)

            #parsing = np.array(cv2.imread(parsing_path, 0))
            parsing = vis_parsing_anno

            #parsing = cv2.resize(parsing, (512, 512), interpolation=cv2.INTER_NEAREST)
            # parsing = cv2.resize(parsing, image.shape[0:2], interpolation=cv2.INTER_NEAREST)

            parts = [table['hair'],table['lower_lip'], table['upper_lip']]
            # colors = [[20, 20, 200], [100, 100, 230], [100, 100, 230]]
            colors = [[159, 29, 53], [159, 29, 53],[159, 29, 53]]
            for part, color in zip(parts, colors):
                frame = hair(frame, parsing, part, color)


        writer.append_data(frame)
    writer.close()