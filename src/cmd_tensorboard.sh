#!/bin/bash
GPU_ID=7
port=8082
with_feat=true
ngf=64

Name=saus1190_11input_withSize
#Name=saus_11input_top
if $with_feat
    then
    Name=${Name}"_feat"
else
    Name=${Name}"_nofeat"
fi

Name=${Name}"_"${ngf}
Port=" --port ${port} "
dir=" --logdir ./checkpoints/${Name}/logs "

CUDA_VISIBLE_DEVICES=${GPU_ID} tensorboard ${Port} ${dir}

