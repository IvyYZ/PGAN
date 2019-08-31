#CUDA_VISIBLE_DEVICES=0 python3  baseline.py -b 4 -j 32 -d market1501 -a resnet50 --combine-trainval \--lr 0.01 --epochs 60 --step-size 20 --eval-step 5 \--logs-dir ./model/ 

#CUDA_VISIBLE_DEVICES=0 python3 train.py --display-port 6006 --display-id 1 \
#	--stage 1 -d market1501 --name ./checkpoints/ \
#	--pose-aug gauss -b 2  -j 16 --niter 50 --niter-decay 50 --lr 0.001 --save-step 10 \
#	--lambda-recon 100.0 --lambda-veri 10.0 --lambda-ssim 100.0  --lambda-iou 10.0 --smooth-label \
#	--netE-pretrain ./model/model_best.pth.tar 

#CUDA_VISIBLE_DEVICES=0 python3 train.py --display-port 6006 --display-id 1 \
#	--stage 2 -d market1501 --name ./checkpoints/ \
#	--pose-aug gauss -b 8 -j 8 --niter 25 --niter-decay 25 --lr 0.0001 --save-step 10 --eval-step 5 \
#	--lambda-recon 100.0 --lambda-veri 10.0 --lambda-sp 10.0 --lambda-iou 10.0 --smooth-label \
#	--netE-pretrain ./bestmodel/market1501/best_net_E.pth --netG-pretrain ./bestmodel/market1501/best_net_G.pth \
#	--netDi-pretrain ./bestmodel/market1501/best_net_Di.pth --netDp-pretrain ./bestmodel/market1501/best_net_Dp.pth 


CUDA_VISIBLE_DEVICES=0 python3 train.py --display-port 6006 --eval --stage 2\
	--netE-pretrain ./bestmodel/market1501/best_net_E.pth --netG-pretrain ./bestmodel/market1501/best_net_G.pth \
	--netDi-pretrain ./bestmodel/market1501/best_net_Di.pth --netDp-pretrain ./bestmodel/market1501/best_net_Dp.pth 


