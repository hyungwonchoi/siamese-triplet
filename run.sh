#CUDA_VISIBLE_DEVICES=1 python3   main_cls.py --epoch 20 --lr 0.01 --triplet 0.0 --dataset fashionMnist --nfeat 32 --ntry 3
CUDA_VISIBLE_DEVICES=1 python3  main_cls.py --dataset omni --type "print" --dataroot "/data/shared/export_print_181016/output/images" --nchannel 3 --epoch 10 --lr 0.001 --triplet 0.0 --nfeat 128 --ntry 1
