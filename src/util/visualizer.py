### Copyright (C) 2017 NVIDIA Corporation. All rights reserved.
### Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
import numpy as np
import os
import ntpath
import time
from . import util
from . import html
import scipy.misc
import math
try:
    from StringIO import StringIO  # Python 2.7
except ImportError:
    from io import BytesIO         # Python 3.x

class Visualizer():
    def __init__(self, opt):
        # self.opt = opt
        self.tf_log = opt.tf_log
        self.use_html = opt.isTrain and not opt.no_html
        self.win_size = opt.display_winsize
        self.name = opt.name
        if self.tf_log:
            import tensorflow as tf
            self.tf = tf
            opt.tf_log_dir = os.path.join(opt.checkpoints_dir, opt.name, 'logs')
            if opt.phase == 'test':
                opt.tf_log_dir = os.path.join(opt.results_dir, opt.name, '%s_%s' % (opt.phase, opt.which_epoch), 'logs')
            self.log_dir = opt.tf_log_dir
            self.writer = tf.summary.FileWriter(self.log_dir)

            self.jet_map = np.loadtxt("./FeatureMapVisualization/JetColorMap_float.txt", dtype=np.float)

        if self.use_html:
            self.web_dir = os.path.join(opt.checkpoints_dir, opt.name, 'web')
            self.img_dir = os.path.join(self.web_dir, 'images')
            print('create web directory %s...' % self.web_dir)
            util.mkdirs([self.web_dir, self.img_dir])
        self.log_name = os.path.join(opt.checkpoints_dir, opt.name, 'loss_log.txt')
        with open(self.log_name, "a") as log_file:
            now = time.strftime("%c")
            log_file.write('================ Training Loss (%s) ================\n' % now)

    # |visuals|: dictionary of images to display or save
    def display_current_results(self, visuals, epoch, step):
        if self.tf_log: # show images in tensorboard output
            img_summaries = []
            for label, image_numpy in visuals.items():
                # Write the image to a string
                try:
                    s = StringIO()
                except:
                    s = BytesIO()
                scipy.misc.toimage(image_numpy).save(s, format="png")
                # scipy.misc.toimage(image_numpy).save(s, format="jpeg")
                # Create an Image object
                img_sum = self.tf.Summary.Image(encoded_image_string=s.getvalue(), height=image_numpy.shape[0], width=image_numpy.shape[1])
                # Create a Summary value
                img_summaries.append(self.tf.Summary.Value(tag="epoch_{0}/{1}".format(epoch, label), image=img_sum))

            # Create and write Summary
            summary = self.tf.Summary(value=img_summaries)
            self.writer.add_summary(summary, step)

        if self.use_html: # save images to a html file
            for label, image_numpy in visuals.items():
                if isinstance(image_numpy, list):
                    for i in range(len(image_numpy)):
                        img_path = os.path.join(self.img_dir, 'epoch%.3d_%s_%d.png' % (epoch, label, i))
                        #img_path = os.path.join(self.img_dir, 'epoch%.3d_%s_%d.jpg' % (epoch, label, i))
                        util.save_image(image_numpy[i], img_path)
                else:
                    img_path = os.path.join(self.img_dir, 'epoch%.3d_%s.png' % (epoch, label))
                    #img_path = os.path.join(self.img_dir, 'epoch%.3d_%s.jpg' % (epoch, label))
                    util.save_image(image_numpy, img_path)

            # update website
            webpage = html.HTML(self.web_dir, 'Experiment name = %s' % self.name, refresh=30)
            for n in range(epoch, 0, -1):
                webpage.add_header('epoch [%d]' % n)
                ims = []
                txts = []
                links = []

                for label, image_numpy in visuals.items():
                    if isinstance(image_numpy, list):
                        for i in range(len(image_numpy)):
                            img_path = 'epoch%.3d_%s_%d.png' % (n, label, i)
                            #img_path = 'epoch%.3d_%s_%d.jpg' % (n, label, i)
                            ims.append(img_path)
                            txts.append(label+str(i))
                            links.append(img_path)
                    else:
                        img_path = 'epoch%.3d_%s.png' % (n, label)
                        #img_path = 'epoch%.3d_%s.jpg' % (n, label)
                        ims.append(img_path)
                        txts.append(label)
                        links.append(img_path)
                if len(ims) < 10:
                    webpage.add_images(ims, txts, links, width=self.win_size)
                else:
                    num = int(round(len(ims)/2.0))
                    webpage.add_images(ims[:num], txts[:num], links[:num], width=self.win_size)
                    webpage.add_images(ims[num:], txts[num:], links[num:], width=self.win_size)
            webpage.save()

    def gray2color(self, gray):
        (height, width) = gray.shape
        color = np.zeros((height, width, 3), np.float)
        for i in range(height):
            for j in range(width):
                idx = np.int_(gray[i, j])
                color[i, j] = self.jet_map[idx]
        return color

    def display_feature_maps(self, feature_maps, filename, epoch, step):
        if self.tf_log:
            visuals = {}
            for f in range(len(feature_maps)):
                array = feature_maps[f].data[0].cpu().float().numpy()
                array = (np.transpose(array, (1, 2, 0)) + 1) / 2. * 255.
                array = np.clip(array, 0, 255)
                (h, w, c) = array.shape

                nh = math.floor(np.sqrt(c))
                nw = math.ceil(c/nh)
                gray = np.zeros((nh * h, nw * w))
                for i in range(h):
                    for j in range(w):
                        for k in range(c):
                            gi = k // nw * h + i
                            gj = k % nw * w + j
                            gray[gi, gj] = array[i, j, k]

                colorImg = self.gray2color(gray).astype(np.uint8)
                visuals["layer{0}_({1},{2},{3})".format(f, h, w, c)] = colorImg

            img_summaries = []
            for label, image in visuals.items():
                try:
                    s = StringIO()
                except:
                    s = BytesIO()
                scipy.misc.toimage(image).save(s, format="png")
                img_sum = self.tf.Summary.Image(encoded_image_string=s.getvalue(), height=image.shape[0], width=image.shape[1])
                img_summaries.append(self.tf.Summary.Value(tag="epoch_{0}_{1}/{2}".format(epoch, filename, label), image=img_sum))

            summary = self.tf.Summary(value=img_summaries)
            self.writer.add_summary(summary, step)

        return

    # errors: dictionary of error labels and values
    def plot_current_errors(self, errors, step):
        if self.tf_log:
            sum_list = []
            for tag, value in errors.items():
                scope = "Total_Loss"
                if "D_" in tag and tag is not "D_loss":
                    scope = "Discriminator_Loss"
                elif "G_" in tag and tag is not "G_loss":
                    scope = "Generator_Loss"
                elif "C_" in tag and tag is not "C_loss":
                    scope = "Continuous_Discriminator_Loss"
                sum_list.append(self.tf.Summary.Value(tag="{0}/{1}".format(scope, tag), simple_value=value))
            summary = self.tf.Summary(value=sum_list)
            self.writer.add_summary(summary, step)

    # errors: same format as |errors| of plotCurrentErrors
    def print_current_errors(self, epoch, i, errors, t):
        message = '(epoch: %d, iters: %d, time: %.3f) ' % (epoch, i, t)
        for k, v in errors.items():
            if v != 0:
                message += '%s: %.3f ' % (k, v)

        print(message)
        with open(self.log_name, "a") as log_file:
            log_file.write('%s\n' % message)

    # save image to the disk
    def save_images(self, webpage, visuals, image_path):
        image_dir = webpage.get_image_dir()
        short_path = ntpath.basename(image_path[0])
        name = os.path.splitext(short_path)[0]

        webpage.add_header(name)
        ims = []
        txts = []
        links = []

        for label, image_numpy in visuals.items():
            image_name = '%s_%s.png' % (name, label)
            #image_name = '%s_%s.jpg' % (name, label)
            save_path = os.path.join(image_dir, image_name)
            util.save_image(image_numpy, save_path)

            ims.append(image_name)
            txts.append(label)
            links.append(image_name)
        webpage.add_images(ims, txts, links, width=self.win_size)
