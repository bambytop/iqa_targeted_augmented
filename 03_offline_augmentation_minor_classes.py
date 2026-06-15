#!/usr/bin/env python
# coding: utf-8

# In[17]:


import os
import pandas as pd
import shutil
from PIL import Image, ImageEnhance, ImageFilter
import numpy as np


# In[18]:


def augment_image(img):
    aug_imgs = []
    # Rotasi acak
    aug_imgs.append(img.rotate(np.random.uniform(-25, 25)))
    # Flip horizontal
    aug_imgs.append(img.transpose(Image.FLIP_LEFT_RIGHT))
    # Brightness
    enhancer = ImageEnhance.Brightness(img)
    aug_imgs.append(enhancer.enhance(np.random.uniform(0.7, 1.3)))
    # Contrast
    enhancer = ImageEnhance.Contrast(img)
    aug_imgs.append(enhancer.enhance(np.random.uniform(0.7, 1.3)))
    # Gaussian Blur
    aug_imgs.append(img.filter(ImageFilter.GaussianBlur(radius=np.random.uniform(0.5, 1.5))))
    return aug_imgs


# In[19]:


# Daftar kelas minor (ganti sesuai notasi nama folder kelas Anda)
minor_classes = [
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
    'Corn_(maize)___Northern_Leaf_Blight',
    'Grape___Black_rot',
    'Potato___Late_blight',
    'Potato___healthy',
    'Tomato___Bacterial_spot',
    'Tomato___Early_blight',
    'Tomato___Late_blight',
    'Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot',
    'Tomato___Spider_mites Two-spotted_spider_mite',
    'Tomato___Target_Spot'
]

# Path ke CSV dan dataset
csv_path = 'image_quality_analysis_with_defect_flags_new.csv'
dataset_path = '/Users/coder/Downloads/datasets/new_pv_local/train'  # Path folder train, update sesuai kebutuhan Anda
augmented_path = '/Users/coder/Downloads/datasets/new_pv_local/train_augmented'  # Hasil augmentasi

# Baca file CSV
df = pd.read_csv(csv_path)

# Filter hanya file bagus
df = df[
    (df['is_blur'] == False) &
    (df['is_low_entropy'] == False) &
    (df['is_defect'] == False)
]

for cls in minor_classes:
    src_folder = os.path.join(dataset_path, cls)
    aug_folder = os.path.join(augmented_path, cls)
    os.makedirs(aug_folder, exist_ok=True)

    # Ambil file yang kualitasnya baik untuk kelas ini
    keep_files = set(df[df['class'] == cls]['filename'])  # Pastikan kolom class & filename sesuai CSV Anda

    for fname in os.listdir(src_folder):
        if fname in keep_files:
            fpath = os.path.join(src_folder, fname)
            img = Image.open(fpath).convert('RGB')
            # Simpan file asli
            img.save(os.path.join(aug_folder, fname))
            # Lakukan augmentasi dan simpan hasilnya
            for i, aug_img in enumerate(augment_image(img)):
                aug_name = fname.rsplit('.', 1)[0] + f'_aug{i}.jpg'
                aug_img.save(os.path.join(aug_folder, aug_name))

print("Augmentasi selesai. Semua gambar minor class yang lolos filter kualitas telah diaugmentasi secara offline dan disimpan di folder baru.")


# In[20]:


import matplotlib.pyplot as plt
import os
from PIL import Image

def plot_augmented_samples(aug_folder, class_names, n_samples=10):
    for cls in class_names:
        files = [f for f in os.listdir(os.path.join(aug_folder, cls)) if 'aug' in f]
        samples = files[:n_samples]
        fig, axs = plt.subplots(1, n_samples, figsize=(15,3))
        for i, fname in enumerate(samples):
            img_path = os.path.join(aug_folder, cls, fname)
            img = Image.open(img_path)
            axs[i].imshow(img)
            axs[i].axis('off')
        plt.suptitle(f'Augmented samples: {cls}', fontsize=12)
        plt.savefig(f'{aug_folder}/{cls}_aug_vis.png', bbox_inches='tight')
        plt.close()
    print("Visualisasi hasil augmentasi tersimpan!")

# Usage
plot_augmented_samples(augmented_path, minor_classes, n_samples=10)


# In[21]:


# Path dataset dan CSV quality control
dataset_train = dataset_path           # folder train asli
dataset_train_aug = augmented_path    # folder hasil augmentasi
qc_csv_path = 'image_quality_analysis_with_defect_flags_new.csv'           # CSV hasil quality control

# Daftar kelas minor
minor_classes = [
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
    'Corn_(maize)___Northern_Leaf_Blight',
    'Grape___Black_rot',
    'Potato___Late_blight',
    'Potato___healthy',
    'Tomato___Bacterial_spot',
    'Tomato___Early_blight',
    'Tomato___Late_blight',
    'Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot',
    'Tomato___Spider_mites Two-spotted_spider_mite',
    'Tomato___Target_Spot'
]

# Baca df quality control
df_qc = pd.read_csv(qc_csv_path)

rekap = []
for cls in minor_classes:
    cls_folder = os.path.join(dataset_train, cls)
    if not os.path.isdir(cls_folder):
        print(f'Folder tidak ditemukan: {cls_folder}')
        continue
    # Semua file gambar asli (ekstensi dapat disesuaikan)
    files_ori = [f for f in os.listdir(cls_folder) if f.lower().endswith(('.jpg','.jpeg','.png','JPG'))]
    sample_ori = len(files_ori)
    # File kualitas baik di kelas ini
    files_qc = set(df_qc[(df_qc['class'] == cls) & 
                         (~df_qc['is_blur']) &
                         (~df_qc['is_low_entropy']) &
                         (~df_qc['is_defect'])]['filename'])
    sample_qc = len(files_qc)
    # File hasil augmentasi di folder hasil augmentasi
    aug_folder = os.path.join(dataset_train_aug, cls)
    files_aug = []
    if os.path.isdir(aug_folder):
        files_aug = [f for f in os.listdir(aug_folder) if ('aug' in f) and f.lower().endswith(('.jpg','.jpeg','.png','JPG'))]
    sample_aug = len(files_aug)
    # Total = sample_qc (asli) + sample_aug (hasil augmentasi)
    sample_total = sample_qc + sample_aug
    
    rekap.append({
        'Class': cls,
        'Sample_Original': sample_ori,
        'Sample_QC': sample_qc,
        'Sample_Augmented': sample_aug,
        'Sample_Total': sample_total
    })

df_rekap = pd.DataFrame(rekap)
print(df_rekap)
df_rekap.to_csv('rekap_distribusi_new.csv', index=False)
print("File 'rekap_distribusi_new.csv' berhasil dibuat.")


# In[22]:


# Load data rekap distribusi
df = pd.read_csv('rekap_distribusi_new.csv')

print("Tabel distribusi sample per kelas:")
print(df)

# Bar chart total sample setelah augmentasi
plt.figure(figsize=(10, 5))
plt.bar(df['Class'], df['Sample_Total'])
plt.xticks(rotation=45, ha='right')
plt.title('Distribusi Sample Total per Kelas Setelah Augmentasi')
plt.xlabel('Kelas')
plt.ylabel('Jumlah Sample')
plt.tight_layout()
plt.savefig('distribusi_total_bar.png')
plt.close()

# Pie chart sample total
plt.figure(figsize=(8,8))
plt.pie(df['Sample_Total'], labels=df['Class'], autopct='%1.1f%%')
plt.title('Proporsi Sample Total per Kelas Setelah Augmentasi')
plt.tight_layout()
plt.savefig('distribusi_total_pie.png')
plt.close()

print("Bar chart dan pie chart distribusi sample berhasil dibuat otomatis dan tersimpan sebagai file PNG.")


# In[ ]:




