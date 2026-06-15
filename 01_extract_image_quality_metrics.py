#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import numpy as np
import os
import pandas as pd
# from skimage.filters import laplace
from skimage.measure import shannon_entropy
from tqdm import tqdm


# In[4]:


def laplacian_variance(img):
    """Calculate Laplacian variance with NumPy 2.0 compatibility"""
    try:
        # Method 1: Standard approach
        return cv2.Laplacian(img, cv2.CV_64F).var()
    except AttributeError:
        try:
            # Method 2: Manual calculation for NumPy 2.0
            laplacian = cv2.Laplacian(img.astype(np.float64), cv2.CV_64F)
            return np.mean(laplacian**2) - (np.mean(laplacian))**2
        except:
            # Method 3: Simple variance
            laplacian = cv2.Laplacian(img, cv2.CV_64F)
            return np.var(laplacian)
            
def analyse_image(path):
    img = cv2.imread(path)
    if img is None:
        return None
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    lap_var = laplacian_variance(img_gray)
    entropy_score = shannon_entropy(img_gray)
    return lap_var, entropy_score

def batch_analyse(image_folder):
    results = []
    
    # Daftar subfolder yang akan dianalisis
    subfolders = ['train'] # , 'valid']
    
    for subfolder in subfolders:
        subfolder_path = os.path.join(image_folder, subfolder)
        
        if os.path.exists(subfolder_path):
            print(f"Processing {subfolder}...")
            
            # Loop melalui semua folder kelas
            for class_name in os.listdir(subfolder_path):
                class_path = os.path.join(subfolder_path, class_name)
                
                if os.path.isdir(class_path):
                    print(f"  Processing class: {class_name}")
                    
                    # Loop melalui semua file gambar dalam folder kelas
                    image_files = [f for f in os.listdir(class_path) 
                                 if f.lower().endswith(('jpg','png','jpeg', 'JPG'))]
                    
                    for fn in tqdm(image_files, desc=f"    {class_name}"):
                        path = os.path.join(class_path, fn)
                        analysis = analyse_image(path)
                        
                        if analysis is not None:
                            lap_var, entropy_score = analysis
                            results.append({
                                'subset': subfolder,
                                'class': class_name,
                                'filename': fn,
                                'full_path': path,
                                'laplacian_var': lap_var,
                                'entropy': entropy_score
                            })
    
    return pd.DataFrame(results)

def simple_batch_analyse(image_folder):
    results = []
    for fn in os.listdir(image_folder):
        if fn.lower().endswith(('jpg','png','jpeg', 'JPG')):
            path = os.path.join(image_folder, fn)
            analysis = analyse_image(path)
            if analysis is not None:
                lap_var, entropy_score = analysis
                results.append({'filename': fn,
                                'laplacian_var': lap_var,
                                'entropy': entropy_score})
    return pd.DataFrame(results)



# In[5]:


# Ganti path dengan folder dataset Anda
image_folder = '/Users/coder/Downloads/datasets/new_pv_local'
results_df = batch_analyse(image_folder)

# Threshold empiris (modifikasi sesuai data)
lap_var_blur_thresh = 100
entropy_low_thresh = 4.0

results_df['is_blur'] = results_df['laplacian_var'] < lap_var_blur_thresh
results_df['is_low_entropy'] = results_df['entropy'] < entropy_low_thresh

results_df.to_csv('dokumentasi_kualitas_citra_new.csv', index=False)

# Dokumentasi Otomatis
with open('dokumentasi_kualitas_citra_new.txt', 'w') as f:
    f.write('''# Dokumentasi Analisis Kualitas Citra Dataset

- Analisis otomatis dilakukan menggunakan laplacian variance (blur) dan shannon entropy (detail informasi) seluruh gambar dataset.
- Gambar dengan laplacian variance < 100 ditandai blur; entropy < 4.0 ditandai detail rendah.
- Semua hasil, skor dan penandaan citra buruk didokumentasikan pada dokumentasi_kualitas_citra.csv.
- Konfirmasi noise/label salah tetap butuh validasi manual/expert bila memungkinkan.
''')


# In[ ]:




