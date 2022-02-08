
import os 

# bvae, vae, cnn, sparse cnn 
# b = 4, 10, 100 
# with model resetting and without 
# 

#os.system("python train_cnn.py cnn/10e_n128 --latent-dim 64 --model-epochs 10")
os.system("python train_cnn.py cnn/5e_n128 --latent-dim 64 --model-epochs 5")
os.system("python train_cnn.py cnn/2e_n128 --latent-dim 64 --model-epochs 2")
os.system("python train_cnn.py cnn/1e_n128 --latent-dim 64 --model-epochs 1")

os.system("python train_cnn.py sparse/10e_n6 --latent-dim 3 --model-epochs 10 --dropout_percent 80")
os.system("python train_cnn.py sparse/5e_n6 --latent-dim 3 --model-epochs 5 --dropout_percent  80")
os.system("python train_cnn.py sparse/2e_n6 --latent-dim 3 --model-epochs 2 --dropout_percent  80")
os.system("python train_cnn.py sparse/1e_n6 --latent-dim 3 --model-epochs 1 --dropout_percent  80")