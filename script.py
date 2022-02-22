import os 

# bvae, vae, cnn, sparse cnn 
# b = 4, 10, 100 
# with model resetting and without 
# 


# python train_bvae.py bvae   --betaH-B 2  --is-eval-only --model-epochs 2 -u 10000
# python train_bvae.py vae --betaH-B 2 --is-eval-only --model-epochs 5 -u 10
# python train_bvae.py bvae_u0 --u 0 --is-eval-only --model-epochs 10 

os.system("python train_bvae.py vae --betaH-B 0 --is-eval-only --model-epochs 5 -u 10")
os.system("python train_cnn.py sparse  --latent-dim 6 --is-eval-only --model-epochs 5 -u 10")
os.system("python train_cnn.py cnn_64  --latent-dim 64  --is-eval-only --model-epochs 5 -u 10")