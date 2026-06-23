#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import shutil
from sklearn.model_selection import train_test_split

# Create project structure
base_path = '/Users/coder/Downloads'


# In[2]:


base_dir = '/Users/coder/Downloads/datasets/new_pv_local'
train_dir = os.path.join(base_dir, 'train')
test_dir = os.path.join(base_dir, 'test')

os.makedirs(test_dir, exist_ok=True)

# Ambil 15% dari train untuk test
test_ratio = 0.15

# Daftar subfolder (dari data Anda)
subfolders = [
    "Grape___Esca_(Black_Measles)", "Tomato___Early_blight", "Peach___Bacterial_spot",
    "Corn_(maize)___healthy", "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn_(maize)___Northern_Leaf_Blight", "Corn_(maize)___Common_rust_",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)", "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Peach___healthy", "Potato___Late_blight", "Strawberry___Leaf_scorch",
    "Raspberry___healthy", "Tomato___Late_blight", "Cherry_(including_sour)___Powdery_mildew",
    "Tomato___Spider_mites Two-spotted_spider_mite", "Strawberry___healthy",
    "Apple___healthy", "Orange___Haunglongbing_(Citrus_greening)",
    "Cherry_(including_sour)___healthy", "Grape___Black_rot", "Pepper,_bell___Bacterial_spot",
    "Soybean___healthy", "Grape___healthy", "Apple___Apple_scab",
    "Tomato___Septoria_leaf_spot", "Potato___healthy", "Squash___Powdery_mildew",
    "Apple___Cedar_apple_rust", "Potato___Early_blight", "Blueberry___healthy",
    "Apple___Black_rot", "Tomato___healthy", "Tomato___Leaf_Mold",
    "Tomato___Bacterial_spot", "Tomato___Tomato_mosaic_virus", "Tomato___Target_Spot",
    "Pepper,_bell___healthy"
]

random_seed = 1337

for subfolder in subfolders:
    src_path = os.path.join(train_dir, subfolder)
    dst_path = os.path.join(test_dir, subfolder)
    os.makedirs(dst_path, exist_ok=True)

    images = [f for f in os.listdir(src_path)
              if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]

    # Split: train tetap di train, sisanya jadi test
    train_images, test_images = train_test_split(
        images,
        test_size=test_ratio,
        random_state=random_seed,
        shuffle=True
    )

    # Pindahkan ke test
    for img in test_images:
        shutil.move(os.path.join(src_path, img), os.path.join(dst_path, img))

    # Hitung sisa
    remaining = len([f for f in os.listdir(src_path)
                    if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))])

    print(f"{subfolder}:")
    print(f"  Awal: {len(images)}")
    print(f"  Ke test: {len(test_images)} ({len(test_images)/len(images)*100:.1f}%)")
    print(f"  Sisa di train: {remaining} ({remaining/len(images)*100:.1f}%)")
    print("-" * 40)

print("\n✅ Selesai!")
print(f"Train: {train_dir}")
print(f"Valid: {os.path.join(base_dir, 'valid')} (TETAP UTUH)")
print(f"Test: {test_dir}")


# In[ ]:




