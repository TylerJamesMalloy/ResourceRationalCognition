import os 

# bvae, vae, cnn, sparse cnn 
# b = 4, 10, 100 
# with model resetting and without 
# 

#os.system("python train_bvae.py bvae/n3_b4_5e --betaH-B 4 --model-epochs 5")
os.system("python train_bvae.py bvae/n3_b4_2e --betaH-B 4 --model-epochs 2")
os.system("python train_bvae.py bvae/n3_b4_1e --betaH-B 4 --model-epochs 1")

os.system("python train_bvae.py vae/n3_b1_5e --betaH-B 1 --model-epochs 5")
os.system("python train_bvae.py vae/n3_b1_2e --betaH-B 1 --model-epochs 2")
os.system("python train_bvae.py vae/n3_b1_1e --betaH-B 1 --model-epochs 1")

os.system("python train_bvae.py bvae/n3_b10_1e --betaH-B 10 --model-epochs 1")
os.system("python train_bvae.py bvae/n3_b100_1e --betaH-B 100 --model-epochs 1")

os.system("python train_bvae.py bvae/n3_b10_2e --betaH-B 10 --model-epochs 2")
os.system("python train_bvae.py bvae/n3_b100_2e --betaH-B 100 --model-epochs 2")

os.system("python train_bvae.py bvae/n3_b10_5e --betaH-B 10 --model-epochs 5")
os.system("python train_bvae.py bvae/n3_b100_5e --betaH-B 100 --model-epochs 5")

