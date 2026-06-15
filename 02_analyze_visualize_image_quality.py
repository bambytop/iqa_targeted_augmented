#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib import rcParams


# In[2]:


# Set style untuk plot
plt.style.use('default')
rcParams['figure.figsize'] = (12, 8)

def analyze_image_quality(csv_path, lap_var_threshold=100, entropy_threshold=6):
    """
    Analisis kualitas gambar dari file CSV hasil analisis
    
    Parameters:
    - csv_path: path ke file CSV
    - lap_var_threshold: threshold untuk blur detection (Laplacian variance)
    - entropy_threshold: threshold untuk low entropy
    """
    
    # Baca data
    df = pd.read_csv(csv_path)
    print(f"Total images analyzed: {len(df)}")
    
    # 1. PERSENTASE CITRA BLUR / LOW ENTROPY PER SUBSET
    print("\n" + "="*50)
    print("1. PERSENTASE CITRA BLUR / LOW ENTROPY PER SUBSET")
    print("="*50)
    
    # Hitung blur dan low entropy
    df['is_blur'] = df['laplacian_var'] < lap_var_threshold
    df['is_low_entropy'] = df['entropy'] < entropy_threshold
    df['is_defect'] = df['is_blur'] | df['is_low_entropy']
    
    if 'subset' in df.columns:
        subset_analysis = df.groupby('subset').agg({
            'filename': 'count',
            'is_blur': 'sum',
            'is_low_entropy': 'sum',
            'is_defect': 'sum'
        }).rename(columns={'filename': 'total_images'})
        
        subset_analysis['blur_percentage'] = (subset_analysis['is_blur'] / subset_analysis['total_images'] * 100).round(2)
        subset_analysis['low_entropy_percentage'] = (subset_analysis['is_low_entropy'] / subset_analysis['total_images'] * 100).round(2)
        subset_analysis['defect_percentage'] = (subset_analysis['is_defect'] / subset_analysis['total_images'] * 100).round(2)
        
        print(subset_analysis[['total_images', 'blur_percentage', 'low_entropy_percentage', 'defect_percentage']])
        
        # Visualisasi persentase per subset
        if len(subset_analysis) > 1:
            fig, ax = plt.subplots(1, 3, figsize=(18, 6))
            
            # Blur percentage
            subset_analysis['blur_percentage'].plot(kind='bar', ax=ax[0], color='lightcoral')
            ax[0].set_title('Persentase Gambar Blur per Subset')
            ax[0].set_ylabel('Persentase (%)')
            ax[0].tick_params(axis='x', rotation=45)
            
            # Low entropy percentage
            subset_analysis['low_entropy_percentage'].plot(kind='bar', ax=ax[1], color='lightblue')
            ax[1].set_title('Persentase Low Entropy per Subset')
            ax[1].set_ylabel('Persentase (%)')
            ax[1].tick_params(axis='x', rotation=45)
            
            # Total defect percentage
            subset_analysis['defect_percentage'].plot(kind='bar', ax=ax[2], color='lightgreen')
            ax[2].set_title('Persentase Total Defect per Subset')
            ax[2].set_ylabel('Persentase (%)')
            ax[2].tick_params(axis='x', rotation=45)
            
            plt.tight_layout()
            plt.show()
            
    else:
        print("Kolom 'subset' tidak ditemukan dalam data")
    
    # 2. DISTRIBUSI LAPLACIAN_VAR DAN ENTROPY PER KELAS
    print("\n" + "="*50)
    print("2. DISTRIBUSI METRIK KUALITAS PER KELAS")
    print("="*50)
    
    if 'class' in df.columns:
        # Statistik per kelas
        class_stats = df.groupby('class').agg({
            'laplacian_var': ['mean', 'std', 'min', 'max'],
            'entropy': ['mean', 'std', 'min', 'max'],
            'is_defect': 'mean'
        }).round(3)
        
        class_stats.columns = ['lap_var_mean', 'lap_var_std', 'lap_var_min', 'lap_var_max',
                              'entropy_mean', 'entropy_std', 'entropy_min', 'entropy_max',
                              'defect_ratio']
        class_stats['defect_percentage'] = (class_stats['defect_ratio'] * 100).round(2)
        
        print("Statistik per Kelas (diurutkan berdasarkan % defect):")
        print(class_stats.sort_values('defect_percentage', ascending=False))
        
        # Visualisasi distribusi
        fig, axes = plt.subplots(2, 2, figsize=(20, 15))
        
        # Boxplot Laplacian variance per kelas
        df.boxplot(column='laplacian_var', by='class', ax=axes[0,0])
        axes[0,0].set_title('Distribusi Laplacian Variance per Kelas')
        axes[0,0].tick_params(axis='x', rotation=45)
        axes[0,0].set_ylabel('Laplacian Variance')
        
        # Boxplot Entropy per kelas
        df.boxplot(column='entropy', by='class', ax=axes[0,1])
        axes[0,1].set_title('Distribusi Entropy per Kelas')
        axes[0,1].tick_params(axis='x', rotation=45)
        axes[0,1].set_ylabel('Entropy')
        
        # Histogram Laplacian variance
        axes[1,0].hist(df['laplacian_var'], bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        axes[1,0].axvline(lap_var_threshold, color='red', linestyle='--', label=f'Threshold: {lap_var_threshold}')
        axes[1,0].set_title('Distribusi Laplacian Variance (Semua Gambar)')
        axes[1,0].set_xlabel('Laplacian Variance')
        axes[1,0].set_ylabel('Frekuensi')
        axes[1,0].legend()
        
        # Histogram Entropy
        axes[1,1].hist(df['entropy'], bins=50, alpha=0.7, color='lightgreen', edgecolor='black')
        axes[1,1].axvline(entropy_threshold, color='red', linestyle='--', label=f'Threshold: {entropy_threshold}')
        axes[1,1].set_title('Distribusi Entropy (Semua Gambar)')
        axes[1,1].set_xlabel('Entropy')
        axes[1,1].set_ylabel('Frekuensi')
        axes[1,1].legend()
        
        plt.suptitle('')  # Hapus automatic suptitle dari boxplot
        plt.tight_layout()
        plt.show()
        
    else:
        print("Kolom 'class' tidak ditemukan dalam data")
    
    # 3. DAFTAR KELAS BERDASARKAN RASIO CACAT
    print("\n" + "="*50)
    print("3. PERINGKAT KELAS BERDASARKAN RASIO DEFECT")
    print("="*50)
    
    if 'class' in df.columns:
        defect_by_class = df.groupby('class').agg({
            'filename': 'count',
            'is_defect': 'sum'
        }).rename(columns={'filename': 'total_images', 'is_defect': 'defect_count'})
        
        defect_by_class['defect_ratio'] = (defect_by_class['defect_count'] / defect_by_class['total_images']).round(3)
        defect_by_class['defect_percentage'] = (defect_by_class['defect_ratio'] * 100).round(2)
        
        defect_by_class = defect_by_class.sort_values('defect_percentage', ascending=False)
        
        print("Peringkat Kelas berdasarkan Rasio Defect:")
        print(defect_by_class)
        
        # Visualisasi rasio defect per kelas
        plt.figure(figsize=(12, 8))
        colors = ['red' if x > 20 else 'orange' if x > 10 else 'yellow' if x > 5 else 'green' for x in defect_by_class['defect_percentage']]
        bars = plt.bar(defect_by_class.index, defect_by_class['defect_percentage'], color=colors)
        
        plt.title('Persentase Gambar Defect per Kelas')
        plt.xlabel('Kelas')
        plt.ylabel('Persentase Defect (%)')
        plt.xticks(rotation=45)
        plt.grid(axis='y', alpha=0.3)
        
        # Tambah nilai di atas bar
        for bar, percentage in zip(bars, defect_by_class['defect_percentage']):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                    f'{percentage}%', ha='center', va='bottom', fontweight='bold')
        
        # Tambah garis threshold
        plt.axhline(y=10, color='red', linestyle='--', alpha=0.7, label='Threshold 10%')
        plt.axhline(y=5, color='orange', linestyle='--', alpha=0.7, label='Threshold 5%')
        plt.legend()
        
        plt.tight_layout()
        plt.show()
        
    else:
        print("Kolom 'class' tidak ditemukan dalam data")
    
    # 4. DETAIL FILE DEFECT
    print("\n" + "="*50)
    print("4. CONTOH FILE DENGAN KUALITAS RENDAH")
    print("="*50)
    
    defect_files = df[df['is_defect']].sort_values('laplacian_var').head(10)
    
    if len(defect_files) > 0:
        print("10 File dengan Laplacian Variance Terendah (Paling Blur):")
        for idx, row in defect_files.iterrows():
            issues = []
            if row['is_blur']:
                issues.append("BLUR")
            if row['is_low_entropy']:
                issues.append("LOW_ENTROPY")
            
            print(f"- {row['filename']} (Kelas: {row['class']})")
            print(f"  Laplacian: {row['laplacian_var']:.2f}, Entropy: {row['entropy']:.2f} - Issues: {', '.join(issues)}")
    else:
        print("Tidak ada file yang terdeteksi sebagai defect")
    
    # 5. SUMMARY
    print("\n" + "="*50)
    print("5. SUMMARY KESELURUHAN")
    print("="*50)
    total_defect = df['is_defect'].sum()
    total_images = len(df)
    overall_defect_percentage = (total_defect / total_images * 100).round(2)
    
    print(f"Total Gambar: {total_images}")
    print(f"Total Defect: {total_defect}")
    print(f"Persentase Defect Keseluruhan: {overall_defect_percentage}%")
    print(f"Threshold yang digunakan:")
    print(f"  - Laplacian Variance: {lap_var_threshold}")
    print(f"  - Entropy: {entropy_threshold}")
    
    return df

# 6. FUNGSI UNTUK MENCARI THRESHOLD OPTIMAL
def find_optimal_thresholds(csv_path):
    """
    Mencari threshold optimal berdasarkan distribusi data
    """
    df = pd.read_csv(csv_path)
    
    # Threshold berdasarkan percentil
    lap_var_threshold = np.percentile(df['laplacian_var'], 10)  # 10% terendah dianggap blur
    entropy_threshold = np.percentile(df['entropy'], 10)  # 10% terendah dianggap low entropy
    
    print("="*50)
    print("THRESHOLD OPTIMAL BERDASARKAN DISTRIBUSI DATA")
    print("="*50)
    print(f"Laplacian Variance Threshold (percentil 10): {lap_var_threshold:.2f}")
    print(f"Entropy Threshold (percentil 10): {entropy_threshold:.2f}")
    print(f"\nRekomendasi threshold:")
    print(f"- Laplacian Variance: {max(50, min(200, round(lap_var_threshold)))}")
    print(f"- Entropy: {max(4, min(7, round(entropy_threshold, 1)))}")
    
    return lap_var_threshold, entropy_threshold


# In[6]:


# PENGGUNAAN
if __name__ == "__main__":
    # Ganti dengan path file CSV Anda
    csv_file = 'dokumentasi_kualitas_citra_new.csv'  # atau path lengkap ke file Anda
    
    try:
        # Cari threshold optimal
        lap_thresh, entropy_thresh = find_optimal_thresholds(csv_file)
        
        # Analisis dengan threshold
        print("\n" + "="*70)
        print("ANALISIS LENGKAP KUALITAS GAMBAR")
        print("="*70)
        
        # Gunakan threshold yang ditemukan (diperbaiki syntaxnya)
        lap_thresh_rounded = max(50, min(150, round(lap_thresh)))
        entropy_thresh_rounded = max(5, min(7, round(entropy_thresh, 1)))
        
        analyzed_df = analyze_image_quality(
            csv_file, 
            lap_var_threshold=lap_thresh_rounded,
            entropy_threshold=entropy_thresh_rounded
        )
        
        # Simpan hasil analisis
        analyzed_df.to_csv('image_quality_analysis_with_defect_flags_new.csv', index=False)
        print(f"\nHasil analisis lengkap disimpan ke: image_quality_analysis_with_defect_flags.csv")
        
    except FileNotFoundError:
        print(f"Error: File {csv_file} tidak ditemukan!")
        print("Pastikan file CSV hasil analisis gambar sudah tersedia")
    except Exception as e:
        print(f"Error: {e}")
        print("Cek instalasi dependencies: pip install pandas matplotlib seaborn numpy")


# In[ ]:




