#!/bin/bash
GPU_ID=2
with_feat=true
ngf=64

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

ndf=$ngf


name=" --name $Name "
data=" --dataroot ./datasets/$Dataset/ "
iters=" --niter 700 --niter_decay 100"
ncs=" --label_nc 0 --input_nc ${input_nc} --output_nc ${output_nc} --no_instance"
nets=" --ngf $ngf --ndf $ndf"
trans=" --resize_or_crop none --no_flip "
#visuals=" --tf_log "

netG=" --netG local"
n_local_enhancers="--n_local_enhancers 1"
n_downsample_global=" --n_downsample_global 6"
n_blocks_global=" --n_blocks_global 6"
num_D="--num_D 3"
n_layers_D="--n_layers_D 5"
lambda_feat="--lambda_feat 150"
lambda_localEnhancerIncontinuous="--lambda_localEnhancerIncontinuous 10"
lambda_localEnhancerMatching="--lambda_localEnhancerMatching 10"
no_vgg_loss="--no_vgg_loss"
continue_train="--continue_train"
#which_epoch="--which_epoch 800"
paras=" $continue_train $netG $n_local_enhancers $no_vgg_loss $lambda_feat $lambda_localEnhancerIncontinuous $lambda_localEnhancerMatching $name $data $iters $ncs $feats $nets $trans $visuals $n_blocks_global $n_downsample_global $numD $n_layers_D"
CUDA_VISIBLE_DEVICES=$GPU_ID python train.py $paras
# global+local:300 niter and 300 decay
# local: 100 niter and 100 decay
netG=" --netG local"
n_local_enhancers="--n_local_enhancers 1"
iters=" --niter 850 --niter_decay 50"
niter_fix_global="--niter_fix_global 3000"
continue_train="--continue_train"
#which_epoch="--which_epoch latest"
paras=" $continue_train  $niter_fix_global $netG $n_local_enhancers $no_vgg_loss $lambda_feat $lambda_localEnhancerIncontinuous $lambda_localEnhancerMatching $name $data $iters $ncs $feats $nets $trans $visuals $n_blocks_global $n_downsample_global $numD $n_layers_D"

CUDA_VISIBLE_DEVICES=$GPU_ID python train.py $paras

exit 0


