# ============================================================================
# STEP 2: FIXED - Generate RGB Composite CWT Representations
# ============================================================================

import os
import pickle
import numpy as np
import pywt
from scipy.ndimage import zoom
from tqdm import tqdm
import cv2

PROCESSED_PATH = '../santosh_lab/shared/KagoziA/wavelets/xresnet_baseline/'
SAMPLING_RATE = 100
IMAGE_SIZE = 224
BATCH_SIZE = 100

print("="*80)
print("STEP 2: GENERATE RGB COMPOSITE CWT REPRESENTATIONS (FIXED)")
print("="*80)

class CWTGeneratorFixed:
    """
    Generate RGB composite scalograms and phasograms
    THIS IS THE CORRECT APPROACH!
    """
    
    def __init__(self, sampling_rate=100, image_size=224):
        self.sampling_rate = sampling_rate
        self.image_size = image_size
        self.wavelet = 'cmor2.0-1.0'  # ✅ FIXED: Better wavelet
        
        # Optimized scales for ECG
        freq_min, freq_max = 0.5, 40.0
        n_scales = 128
        
        cf = pywt.central_frequency(self.wavelet)
        freqs = np.logspace(np.log10(freq_min), np.log10(freq_max), n_scales)
        self.scales = (cf * sampling_rate) / freqs
        
        print(f"\nFixed CWT Generator:")
        print(f"  Wavelet: {self.wavelet}")
        print(f"  Output: RGB composite (3, {image_size}, {image_size})")
    
    def compute_cwt_single_lead(self, signal_1d):
        """Compute CWT for a single lead"""
        try:
            coefficients, _ = pywt.cwt(
                signal_1d,
                self.scales,
                self.wavelet,
                sampling_period=1.0 / self.sampling_rate
            )
            return coefficients
        except Exception as e:
            print(f"Warning - CWT error: {e}")
            return None
    
    def generate_scalogram(self, coefficients):
        """Generate scalogram from CWT coefficients"""
        scalogram = np.abs(coefficients) ** 2
        scalogram = np.log1p(scalogram)  # ✅ FIXED: log1p instead of log10
        
        # Normalize to [0, 1]
        min_val, max_val = scalogram.min(), scalogram.max()
        if max_val - min_val > 1e-10:
            scalogram = (scalogram - min_val) / (max_val - min_val)
        else:
            scalogram = np.zeros_like(scalogram)
        
        return scalogram.astype(np.float32)
    
    def generate_phasogram(self, coefficients):
        """Generate phasogram from CWT coefficients"""
        phase = np.angle(coefficients)
        phasogram = (phase + np.pi) / (2 * np.pi)
        return phasogram.astype(np.float32)
    
    def resize_to_image(self, cwt_matrix):
        """Resize CWT matrix to target size"""
        return cv2.resize(cwt_matrix, (self.image_size, self.image_size))
    
    def process_12_lead_to_rgb(self, ecg_12_lead, use_scalogram=True):
        """
        ✅ FIXED: Generate 3-channel RGB composite
        
        Returns:
            rgb_composite: (3, H, W) array
        """
        # Ensure shape is (12, time)
        if ecg_12_lead.shape[0] != 12:
            ecg_12_lead = ecg_12_lead.T
        
        # Group leads into 3 channels
        channel_groups = [
            [0, 1, 2],              # Limb leads (I, II, III) → R
            [3, 4, 5],              # Augmented leads (aVR, aVL, aVF) → G  
            [6, 7, 8, 9, 10, 11]    # Chest leads (V1-V6) → B
        ]
        
        channels = []
        for group in channel_groups:
            # Average signals in group
            avg_signal = np.mean(ecg_12_lead[group], axis=0)
            
            # Compute CWT
            coeffs = self.compute_cwt_single_lead(avg_signal)
            
            if coeffs is None:
                channels.append(np.zeros((self.image_size, self.image_size), dtype=np.float32))
                continue
            
            # Generate scalogram or phasogram
            if use_scalogram:
                cwt_image = self.generate_scalogram(coeffs)
            else:
                cwt_image = self.generate_phasogram(coeffs)
            
            # Resize
            cwt_resized = self.resize_to_image(cwt_image)
            channels.append(cwt_resized)
        
        # Stack as RGB: (3, H, W)
        rgb_composite = np.stack(channels, axis=0)
        
        return rgb_composite
    
    def process_dataset_batched(self, X, output_scalo_path, output_phaso_path, batch_size=100):
        """Process entire dataset in batches"""
        n_samples = len(X)
        n_batches = (n_samples + batch_size - 1) // batch_size
        
        print(f"\nProcessing {n_samples} samples...")
        print(f"Output: ({n_samples}, 3, {self.image_size}, {self.image_size})")  # ✅ 3 channels!
        
        shape = (n_samples, 3, self.image_size, self.image_size)  # ✅ FIXED: 3 channels
        scalograms = np.memmap(output_scalo_path + '.tmp', dtype='float32', mode='w+', shape=shape)
        phasograms = np.memmap(output_phaso_path + '.tmp', dtype='float32', mode='w+', shape=shape)
        
        for batch_idx in tqdm(range(n_batches), desc="Processing"):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, n_samples)
            
            for i in range(start_idx, end_idx):
                scalo = self.process_12_lead_to_rgb(X[i], use_scalogram=True)
                phaso = self.process_12_lead_to_rgb(X[i], use_scalogram=False)
                
                scalograms[i] = scalo
                phasograms[i] = phaso
            
            scalograms.flush()
            phasograms.flush()
        
        del scalograms
        del phasograms
        
        # Convert to standard numpy
        print("Saving final arrays...")
        scalograms_final = np.memmap(output_scalo_path + '.tmp', dtype='float32', mode='r', shape=shape)
        phasograms_final = np.memmap(output_phaso_path + '.tmp', dtype='float32', mode='r', shape=shape)
        
        np.save(output_scalo_path, scalograms_final)
        np.save(output_phaso_path, phasograms_final)
        
        del scalograms_final
        del phasograms_final
        
        # Clean up temp files
        os.remove(output_scalo_path + '.tmp')
        os.remove(output_phaso_path + '.tmp')
        
        print(f"✓ Saved: {output_scalo_path}")
        print(f"✓ Saved: {output_phaso_path}")

def main():
    print("\n[1/4] Loading metadata...")
    with open(os.path.join(PROCESSED_PATH, 'metadata.pkl'), 'rb') as f:
        metadata = pickle.load(f)
    
    print(f"Classes: {metadata['classes']}")
    
    print("\n[2/4] Initializing FIXED CWT generator...")
    cwt_gen = CWTGeneratorFixed(sampling_rate=SAMPLING_RATE, image_size=IMAGE_SIZE)
    
    # Process training set
    print("\n[3/4] Processing TRAINING set...")
    X_train = np.load(os.path.join(PROCESSED_PATH, 'train_standardized.npy'), mmap_mode='r')
    cwt_gen.process_dataset_batched(
        X_train,
        output_scalo_path=os.path.join(PROCESSED_PATH, 'train_scalograms.npy'),
        output_phaso_path=os.path.join(PROCESSED_PATH, 'train_phasograms.npy'),
        batch_size=BATCH_SIZE
    )
    del X_train
    
    # Process validation set
    print("\n[3/4] Processing VALIDATION set...")
    X_val = np.load(os.path.join(PROCESSED_PATH, 'val_standardized.npy'), mmap_mode='r')
    cwt_gen.process_dataset_batched(
        X_val,
        output_scalo_path=os.path.join(PROCESSED_PATH, 'val_scalograms.npy'),
        output_phaso_path=os.path.join(PROCESSED_PATH, 'val_phasograms.npy'),
        batch_size=BATCH_SIZE
    )
    del X_val
    
    # Process test set
    print("\n[4/4] Processing TEST set...")
    X_test = np.load(os.path.join(PROCESSED_PATH, 'test_standardized.npy'), mmap_mode='r')
    cwt_gen.process_dataset_batched(
        X_test,
        output_scalo_path=os.path.join(PROCESSED_PATH, 'test_scalograms.npy'),
        output_phaso_path=os.path.join(PROCESSED_PATH, 'test_phasograms.npy'),
        batch_size=BATCH_SIZE
    )
    del X_test
    
    print("\n✓ STEP 2 COMPLETE - RGB COMPOSITES GENERATED!")

if __name__ == '__main__':
    main()