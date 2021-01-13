import os
import time
import random
import math
import argparse
import numpy as np
from PIL import Image

parser = argparse.ArgumentParser()

parser.add_argument("--source_file", required=True, help="path of the source text feature file")
parser.add_argument("--dataset_name", required=True, help="name of dataset, text feature file should exist in input_dir")
parser.add_argument("--test_prop", type=float, default=0.2, help="proportion of test data")
parser.add_argument("--image_height", type=int, default=128, help="generated image height")
parser.add_argument("--image_width", type=int, default=64, help="generated image width")
parser.add_argument("--output_mode", required=True, help="top_only, left_only, both")
parser.add_argument("--prop_limit", type=float, default=0., help="lower bound of prop limit")


args = parser.parse_args()

feature_dict = {"alphaProp": 0, "numberProp": 1,
                "hasText": 2, "logLength": 3,
                "timeChars": 4, "percentChars": 5,
                "hasTopBorder": 6, "hasBottomBorder": 7, "hasLeftBorder": 8, "hasRightBorder": 9,
                "backColorNotWhite": 10, "fontColorNotBlack": 11,
                "isMerged": 12,
                "isNumberFormat": 13, "isDateFormat": 14, "isOtherFormat": 15,
                "isFontBold": 16, "fontSize": 17,
                "hasFormula": 18,
                "mergedWithTop": 19, "mergedWithBottom": 20, "mergedWithLeft": 21, "mergedWithRight": 22,
                }
feature_len = len(feature_dict)

# For the 1st Feature Version
"""
input_feature = ["logLength", "alphaProp", "numberProp"]
output_feature = ["hasTopBorder", "hasTopBorder", "hasTopBorder"]
input_scale = [255./3, 255., 255.]
output_scale = [255., 255., 255.]
"""

# Merge-related Feature Added
# input_feature = ["logLength", "alphaProp", "numberProp", "isMerged", "mergedWithTop", "mergedWithLeft"]
# input_scale = [255./3, 255., 255., 255., 255., 255.]

# all 4-dimention merged-related
input_feature = ["logLength", "alphaProp", "numberProp", "isMerged", "mergedWithTop", "mergedWithBottom", "mergedWithLeft", "mergedWithRight"]
input_scale = [255./3] + [255.] * 7

input_feature += ["isNumberFormat", "isDateFormat", "isOtherFormat"]
input_scale += [255.] * 3

# Both Top & Left
# output_feature = ["hasTopBorder", "hasTopBorder", "hasTopBorder", "hasLeftBorder", "hasLeftBorder", "hasLeftBorder"]
# output_scale = [255., 255., 255., 255., 255., 255.]
output_feature = ["hasTopBorder", "hasLeftBorder"]
output_scale = [255.] * 2


calculated_feature = ["hasTopBorder", "hasLeftBorder"]

if args.output_mode == "top_only":
    #output_feature = ["hasTopBorder", "hasTopBorder", "hasTopBorder"]
    #output_scale = [255., 255., 255.]
    output_feature = ["hasTopBorder"]
    output_scale = [255.]
    calculated_feature = ["hasTopBorder"]
elif args.output_mode == "left_only":
    #output_feature = ["hasLeftBorder", "hasLeftBorder", "hasLeftBorder"]
    #output_scale = [255., 255., 255.]
    output_feature = ["hasLeftBorder"]
    output_scale = [255.]
    calculated_feature = ["hasLeftBorder"]

input_len = len(input_feature)
output_len = len(output_feature)
calculated_len = len(calculated_feature)

assert(len(input_scale) == input_len)
assert(len(output_scale) == output_len)
for feature in input_feature + output_feature + calculated_feature:
    assert (feature in feature_dict)

a_size = (args.image_height, args.image_width * math.ceil(input_len / 3), 3)
b_size = (args.image_height, args.image_width * math.ceil(output_len / 3), 3)
f_size = (args.image_height, args.image_width * math.ceil(calculated_len / 3), 3)


def get_image_name(path_dict, count, name):
    usage_type = "train"
    r = random.random()
    if r < args.test_prop:
        usage_type = "test"
    count[usage_type] += 1
    file_A = os.path.join(path_dict[usage_type + "_A"], usage_type + "_" + str(count[usage_type]) + "_" + name + ".png")
    file_B = os.path.join(path_dict[usage_type + "_B"], usage_type + "_" + str(count[usage_type]) + "_" + name + ".png")
    file_feat = os.path.join(path_dict[usage_type + "_feat"], usage_type + "_" + str(count[usage_type]) + "_" + name + ".png")
    return file_A, file_B, file_feat


def get_features(line):
    parts = line.strip().split("|")
    srcFile = parts[0]
    sheetName = parts[1]
    height = int(parts[2])
    width = int(parts[3])
    parts = parts[4:4+height*width*feature_len]
    if height > args.image_height or width > args.image_width:
        print("Spreedsheet {1} of {0} too big: {2} x {3}".format(sheetName, srcFile, height, width))
        print("Skip...")
        return None, None

    name = srcFile.strip().split("\\")[-1].split(".")[0] + "_" + sheetName
    feats = np.zeros((height, width, feature_len))
    for i in range(len(parts)):
        value = float(parts[i])
        ax1 = i // (width * feature_len)
        i = i % (width * feature_len)
        ax2 = i // feature_len
        ax3 = i % feature_len
        feats[ax1, ax2, ax3] = value

    return name, feats


def count_prop(feats):
    calculated = []
    (height, width, _) = feats.shape
    total = height * width
    for feature in calculated_feature:
        f = feature_dict[feature]
        count = 0
        for i in range(height):
            for j in range(width):
                if feats[i, j, f] != 0:
                    count += 1
        calculated += [float(count) / total]
    return calculated


def gen_feature_image(feats, featList):
    image_A = np.zeros(a_size)
    image_B = np.zeros(b_size)
    image_f = np.zeros(f_size)

    (height, width, _) = feats.shape
    for i in range(height):
        for j in range(width):

            for k in range(input_len):
                index = feature_dict[input_feature[k]]
                jk = j + (k // 3 * args.image_width)
                kk = k % 3
                assert (feats[i, j, index] >= 0)
                image_A[i, jk, kk] = min(255., feats[i, j, index] * input_scale[k])

            for k in range(output_len):
                index = feature_dict[output_feature[k]]
                jk = j + (k // 3 * args.image_width)
                kk = k % 3
                assert (feats[i, j, index] >= 0)
                image_B[i, jk, kk] = min(255., feats[i, j, index] * output_scale[k])

            for k in range(calculated_len):
                value = featList[k] * 255.
                jk = j + (k // 3 * args.image_width)
                kk = k % 3
                image_f[i, jk, kk] = value

    return image_A, image_B, image_f


def main():
    root_path = os.path.join("./datasets/", args.dataset_name)
    paths = {'train_A': os.path.join(root_path, "train_A"),
             'train_B': os.path.join(root_path, "train_B"),
             'train_feat': os.path.join(root_path, "train_feat"),
             'test_A': os.path.join(root_path, "test_A"),
             'test_B': os.path.join(root_path, "test_B"),
             'test_feat': os.path.join(root_path, "test_feat"),}
    for path in paths.values():
        if not os.path.exists(path):
            os.makedirs(path)

    input_file = args.source_file
    if not os.path.exists(input_file):
        print("{0} not exist!".format(input_file))
        return

    random.seed(time.time())
    count = {"train": 0, "test": 0}

    for line in open(input_file, "r"):
        name, feats = get_features(line)
        if feats is None:
            continue
        featList = count_prop(feats)

        reach_limit = True
        for prop in featList:
            if not prop > args.prop_limit:
                reach_limit = False
        if not reach_limit:
            continue

        image_A, image_B, image_f = gen_feature_image(feats, featList)

        file_A, file_B, file_f = get_image_name(paths, count, name)
        Image.fromarray(image_A.astype(np.uint8)).save(file_A)
        Image.fromarray(image_B.astype(np.uint8)).save(file_B)
        Image.fromarray(image_f.astype(np.uint8)).save(file_f)

    print("train: {}".format(count["train"]))
    print("test: {}".format(count["test"]))
    return


if __name__ == '__main__':
    main()