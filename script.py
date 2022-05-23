import os 

# bvae, vae, cnn, sparse cnn 
# b = 4, 10, 100 
# with model resetting and without 
# 


# python train_bvae.py bvae   --betaH-B 2  --is-eval-only --model-epochs 2 -u 10000
# python train_bvae.py vae --betaH-B 2 --is-eval-only --model-epochs 5 -u 10
# python train_bvae.py bvae_u0 --u 0 --is-eval-only --model-epochs 10 

os.system("python train_bvae.py bvae_b0_u10_m2 --betaH-B 0 -u 10 --model-epochs 2")
os.system("python train_bvae.py bvae_b10_u10_m2 --betaH-B 10 -u 10 --model-epochs 2")
os.system("python train_bvae.py bvae_b100_u10_m2 --betaH-B 100 -u 10 --model-epochs 2")
os.system("python train_bvae.py bvae_b1000_u10_m2 --betaH-B 1000 -u 10 --model-epochs 2")

os.system("python train_bvae.py bvae_b0_u100_m2 --betaH-B 0 -u 100 --model-epochs 2")
os.system("python train_bvae.py bvae_b10_u100_m2 --betaH-B 10 -u 100 --model-epochs 2")
os.system("python train_bvae.py bvae_b100_u100_m2 --betaH-B 100 -u 100 --model-epochs 2")
os.system("python train_bvae.py bvae_b1000_u100_m2 --betaH-B 1000 -u 100 --model-epochs 2")