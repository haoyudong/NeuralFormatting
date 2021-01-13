#!/bin/bash
GPU_ID=3
with_feat=true
ngf=64
which_epoch=890

#Name=saus_4dim_png_top
#Dataset=saus_4dims_png_top
Name=sausDes_8Mean
Dataset=sausDes
input_nc=11
output_nc=5
feat_num=3

feats=""
if $with_feat
    then
    Name=${Name}"_feat"
    feats=" --label_feat --load_features --feat_num ${feat_num} "
else
    Name=${Name}"_nofeat"
fi

Name=${Name}"_${ngf}"


name=" --name $Name "
epoch=" --which_epoch ${which_epoch} "
data=" --dataroot ./datasets/${Dataset}/ "
ncs=" --label_nc=0 --input_nc ${input_nc} --output_nc ${output_nc} --no_instance "
nets=" --ngf ${ngf} "
trans=" --resize_or_crop=none --no_flip"
visuals=" --tf_log --show_ground_truth "
netG=" --netG local"
n_local_enhancers="--n_local_enhancers 1"
n_downsample_global=" --n_downsample_global 6"
n_blocks_global=" --n_blocks_global 6"
paras=" $n_local_enhancers $netG $name $data $epoch $ncs $feats $nets $trans $visuals $n_blocks_global $n_downsample_global"

CUDA_VISIBLE_DEVICES=$GPU_ID python test.py $paras


python imgEnlarge.py --root_dir ./results/$Name/test_$which_epoch
