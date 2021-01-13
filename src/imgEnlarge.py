import argparse
import os
import shutil
import numpy as np
from PIL import Image

parser = argparse.ArgumentParser()

parser.add_argument("--root_dir", required=True)
parser.add_argument("--path_prefix", default="enlarge")
#parser.add_argument("--retain_height", type=int, default=128)
#parser.add_argument("--retain_width", type=int, default=64)
parser.add_argument("--enlarge_factor", type=int, default=8)

args = parser.parse_args()


def path_check():
    root_dir = args.root_dir
    origin_index = os.path.join(root_dir, "index.html")
    origin_images = os.path.join(root_dir, "images")
    to_check = [root_dir, origin_index, origin_images]

    for path in to_check:
        if not os.path.exists(path):
            print("{0} not exist!".format(path))
            exit(0)

    enlarge_path = os.path.join(root_dir, args.path_prefix + "_" + str(args.enlarge_factor))
    enlarge_index = os.path.join(enlarge_path, "index.html")
    enlarge_images = os.path.join(enlarge_path, "images")

    if not os.path.exists(enlarge_path):
        os.makedirs(enlarge_path)
    shutil.copyfile(origin_index, enlarge_index)
    if not os.path.exists(enlarge_images):
        os.makedirs(enlarge_images)

    return origin_images, enlarge_images


def enlarge_image(img):
    factor = args.enlarge_factor
    #src_size = (args.retain_height, args.retain_width, 3)
    src_size = img.shape
    dst_size = (src_size[0] * factor, src_size[1] * factor, 3)

    enlarged = np.zeros(dst_size, dtype=np.uint8)
    for si in range(src_size[0]):
        for sj in range(src_size[1]):
            value = img[si, sj]
            for di in range(si * factor, si * factor + factor):
                for dj in range(sj * factor, sj * factor + factor):
                    enlarged[di, dj] = value

    return enlarged


def main():
    src_path, dst_path = path_check()

    filenames = os.listdir(src_path)
    count, total = 0, len(filenames)
    for filename in filenames:
        count += 1
        print("Enlarging image {0}/{1} ...".format(count, total))
        src_file = os.path.join(src_path, filename)
        dst_file = os.path.join(dst_path, filename)
        original = np.array(Image.open(src_file))
        enlarged = enlarge_image(original)
        Image.fromarray(enlarged, 'RGB').save(dst_file)

    return


if __name__ == '__main__':
    main()