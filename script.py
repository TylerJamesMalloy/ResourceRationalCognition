import os 

# bvae, vae, cnn, sparse cnn 
# b = 4, 10, 100 
# with model resetting and without 
# 



os.system("python train_bvae.py vae --betaH-B 1 --is-eval-only --model-epochs 10")
os.system("python train_bvae.py vae --betaH-B 1 --is-eval-only --model-epochs 1")