import os 

# bvae, vae, cnn, sparse cnn 
# b = 4, 10, 100 
# with model resetting and without 
# 


os.system("python train_bvae.py vae  --is-eval-only --model-epochs 10")
os.system("python train_bvae.py vae  --is-eval-only --model-epochs 5")
os.system("python train_bvae.py vae  --is-eval-only --model-epochs 2")