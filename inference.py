import argparse
import torch
import os
from glob import glob
from PIL import Image
from torchvision.transforms.functional import to_pil_image, to_tensor

from main import save_path, network

network.load_state_dict(torch.load(save_path))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def recursive_breakdown(network, img, img_name, args):
    w, h = img.size
    left, right = img.crop((0, 0, w // 2, h)), img.crop((w // 2, 0, w, h))
    left_img = left
    right_img = right

    left, right = to_tensor(left), to_tensor(right)

    preds = []
    with torch.no_grad():
        left_logit = network(left.to(device))
        right_logit = network(right.to(device))

        _, left_pred = left_logit.topk(1, 1, True, True)
        _, right_pred = right_logit.topk(1, 1, True, True)
        preds.append((left_img, left_pred.item(), img_name + '_L'))
        preds.append((right_img, right_pred.item(), img_name + '_R'))

    for pred_tuple in preds:
        cur_img, pred, name = pred_tuple
        if pred == 0:
            cur_img.save(os.path.join(args.out_hand, name))
        elif pred == 1:
            cur_img.save(os.path.join(args.out_print), name)
        else:
            recursive_breakdown(network, cur_img, name, args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--detected_path', default='./detected')
    parser.add_argument('--out_print', default='./print')
    parser.add_argument('--out_hand', default='./handwritten')
    args = parser.parse_args()

    network.to(device)

    img_list = glob(os.path.join(args.detected_path, '*'))

    os.makedirs(args.out_print)
    os.makedirs(args.out_hand)

    for img in img_list:
        img_name = os.path.basename(img)
        img = Image.open(img).convert('L')
        orig_img = img
        img = to_tensor(img)

        with torch.no_grad():
            logit = network(img.to(device))
            _, pred = logit.topk(1, 1, True, True)
            pred = pred.item()

        if pred == 0:  # hand
            orig_img.save(os.path.join(args.out_hand, img_name))
        elif pred == 1:  # print
            orig_img.save(os.path.join(args.out_print), img_name)
        else:  # mixed
            # split and re-predict
            recursive_breakdown(network, orig_img, img_name, args)




