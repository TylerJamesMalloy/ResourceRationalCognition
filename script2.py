import os 

# bvae, vae, cnn, sparse cnn 
# b = 4, 10, 100 
# with model resetting and without 




os.system("python train_bvae.py bvae_b0_u1000_m2 --betaH-B 0 -u 100 --model-epochs 2")
os.system("python train_bvae.py bvae_b10_u1000_m2 --betaH-B 10 -u 100 --model-epochs 2")
os.system("python train_bvae.py bvae_b100_u1000_m2 --betaH-B 100 -u 100 --model-epochs 2")
os.system("python train_bvae.py bvae_b1000_u1000_m2 --betaH-B 1000 -u 100 --model-epochs 2")