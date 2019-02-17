# without triplet loss, cls only 
CUDA_VISIBLE_DEVICES=0 python3   main_cls.py --epoch 30 --lr 0.001 --triplet 0.0 --dataset Mnist --nfeat 32 --ntry 1
CUDA_VISIBLE_DEVICES=0 python3   main_cls.py --epoch 30 --lr 0.001 --triplet 0.0 --dataset Mnist --nfeat 32 --ntry 2
CUDA_VISIBLE_DEVICES=0 python3   main_cls.py --epoch 30 --lr 0.001 --triplet 0.0 --dataset Mnist --nfeat 32 --ntry 3
CUDA_VISIBLE_DEVICES=0 python3   main_cls.py --epoch 30 --lr 0.001 --triplet 0.0 --dataset Mnist --nfeat 32 --ntry 4
CUDA_VISIBLE_DEVICES=0 python3   main_cls.py --epoch 30 --lr 0.001 --triplet 0.0 --dataset Mnist --nfeat 32 --ntry 5

CUDA_VISIBLE_DEVICES=0 python3   main_cls.py --epoch 30 --lr 0.001 --triplet 0.0 --dataset fashionMnist --nfeat 32 --ntry 1
CUDA_VISIBLE_DEVICES=0 python3   main_cls.py --epoch 30 --lr 0.001 --triplet 0.0 --dataset fashionMnist --nfeat 32 --ntry 2
CUDA_VISIBLE_DEVICES=0 python3   main_cls.py --epoch 30 --lr 0.001 --triplet 0.0 --dataset fashionMnist --nfeat 32 --ntry 3
CUDA_VISIBLE_DEVICES=0 python3   main_cls.py --epoch 30 --lr 0.001 --triplet 0.0 --dataset fashionMnist --nfeat 32 --ntry 4
CUDA_VISIBLE_DEVICES=0 python3   main_cls.py --epoch 30 --lr 0.001 --triplet 0.0 --dataset fashionMnist --nfeat 32 --ntry 5

#CUDA_VISIBLE_DEVICES=0 python3   main_cls.py --epoch 30 --lr 0.01 --triplet 0.0 --dataset Mnist --nfeat 32 --ntry 1
#CUDA_VISIBLE_DEVICES=0 python3   main_cls.py --epoch 30 --lr 0.01 --triplet 0.0 --dataset Mnist --nfeat 32 --ntry 2
#CUDA_VISIBLE_DEVICES=0 python3   main_cls.py --epoch 30 --lr 0.01 --triplet 0.0 --dataset Mnist --nfeat 32 --ntry 3
#CUDA_VISIBLE_DEVICES=0 python3   main_cls.py --epoch 30 --lr 0.01 --triplet 0.0 --dataset Mnist --nfeat 32 --ntry 4
#CUDA_VISIBLE_DEVICES=0 python3   main_cls.py --epoch 30 --lr 0.01 --triplet 0.0 --dataset Mnist --nfeat 32 --ntry 5

#CUDA_VISIBLE_DEVICES=0 python3   main_cls.py --epoch 30 --lr 0.01 --triplet 0.0 --dataset fashionMnist --nfeat 32 --ntry 1
#CUDA_VISIBLE_DEVICES=0 python3   main_cls.py --epoch 30 --lr 0.01 --triplet 0.0 --dataset fashionMnist --nfeat 32 --ntry 2
#CUDA_VISIBLE_DEVICES=0 python3   main_cls.py --epoch 30 --lr 0.01 --triplet 0.0 --dataset fashionMnist --nfeat 32 --ntry 3
#CUDA_VISIBLE_DEVICES=0 python3   main_cls.py --epoch 30 --lr 0.01 --triplet 0.0 --dataset fashionMnist --nfeat 32 --ntry 4
#CUDA_VISIBLE_DEVICES=0 python3   main_cls.py --epoch 30 --lr 0.01 --triplet 0.0 --dataset fashionMnist --nfeat 32 --ntry 5

# triplet + margin 
#CUDA_VISIBLE_DEVICES=0 python3   main_cls_triplet.py --cls 1.0 --epoch 30 --lr 0.001 --triplet 1.0 --margin 1.0 --dataset Mnist --nfeat 32 --ntry 1
#CUDA_VISIBLE_DEVICES=0 python3   main_cls_triplet.py --cls 1.0 --epoch 30 --lr 0.001 --triplet 1.0 --margin 1.0 --dataset Mnist --nfeat 32 --ntry 2
#CUDA_VISIBLE_DEVICES=0 python3   main_cls_triplet.py --cls 1.0 --epoch 30 --lr 0.001 --triplet 1.0 --margin 1.0 --dataset Mnist --nfeat 32 --ntry 3
#CUDA_VISIBLE_DEVICES=0 python3   main_cls_triplet.py --cls 1.0 --epoch 30 --lr 0.001 --triplet 1.0 --margin 1.0 --dataset Mnist --nfeat 32 --ntry 4
#CUDA_VISIBLE_DEVICES=0 python3   main_cls_triplet.py --cls 1.0 --epoch 30 --lr 0.001 --triplet 1.0 --margin 1.0 --dataset Mnist --nfeat 32 --ntry 5

#CUDA_VISIBLE_DEVICES=0 python3   main_cls_triplet.py --cls 1.0 --epoch 30 --lr 0.001 --triplet 1.0 --margin 1.0 --dataset fashionMnist --nfeat 32 --ntry 1
#CUDA_VISIBLE_DEVICES=0 python3   main_cls_triplet.py --cls 1.0 --epoch 30 --lr 0.001 --triplet 1.0 --margin 1.0 --dataset fashionMnist --nfeat 32 --ntry 2
#CUDA_VISIBLE_DEVICES=0 python3   main_cls_triplet.py --cls 1.0 --epoch 30 --lr 0.001 --triplet 1.0 --margin 1.0 --dataset fashionMnist --nfeat 32 --ntry 3
#CUDA_VISIBLE_DEVICES=0 python3   main_cls_triplet.py --cls 1.0 --epoch 30 --lr 0.001 --triplet 1.0 --margin 1.0 --dataset fashionMnist --nfeat 32 --ntry 4
#CUDA_VISIBLE_DEVICES=0 python3   main_cls_triplet.py --cls 1.0 --epoch 30 --lr 0.001 --triplet 1.0 --margin 1.0 --dataset fashionMnist --nfeat 32 --ntry 5


