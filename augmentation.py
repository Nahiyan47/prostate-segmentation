#augmentation_1
import os
import numpy as np
import torch
import torch.nn.functional as F
import nibabel as nib
from torch.utils.data import Dataset, DataLoader  # Added Dataset import
from tqdm import tqdm
import random
from scipy import ndimage
from scipy.ndimage import rotate, gaussian_filter
import warnings
warnings.filterwarnings('ignore')

class OfflineAugmentor:
    def __init__(self, input_dir, output_dir, augmentation_factor=3):
        """
        Args:
            input_dir: Directory containing imagesTr and labelsTr folders
            output_dir: Directory to save augmented data
            augmentation_factor: Total multiplication factor (3 means 3x original data)
        """
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.augmentation_factor = augmentation_factor
        
        # Create output directories
        self.aug_images_dir = os.path.join(output_dir, 'imagesTr_augmented')
        self.aug_labels_dir = os.path.join(output_dir, 'labelsTr_augmented')
        
        os.makedirs(self.aug_images_dir, exist_ok=True)
        os.makedirs(self.aug_labels_dir, exist_ok=True)
        
    def apply_augmentations(self, image, label):
        """Apply various augmentation techniques"""
        augmented_pairs = []
        
        # Calculate how many augmented versions we need per image
        # If factor=3, we need 2 augmented versions + 1 original = 3 total
        num_augmentations = self.augmentation_factor - 1
        
        for i in range(num_augmentations):
            aug_image = image.copy()
            aug_label = label.copy()
            
            # Randomly select augmentation techniques to apply
            augmentation_applied = False
            
            # 1. Horizontal flip (50% chance)
            if random.random() > 0.5:
                aug_image = np.fliplr(aug_image)
                aug_label = np.fliplr(aug_label)
                augmentation_applied = True
            
            # 2. Vertical flip (30% chance - less common for medical images)
            if random.random() > 0.7:
                aug_image = np.flipud(aug_image)
                aug_label = np.flipud(aug_label)
                augmentation_applied = True
            
            # 3. Rotation (-15 to +15 degrees, 60% chance)
            if random.random() > 0.4:
                angle = random.uniform(-15, 15)
                aug_image = rotate(aug_image, angle, order=1, reshape=False, mode='constant')
                aug_label = rotate(aug_label, angle, order=0, reshape=False, mode='constant')
                augmentation_applied = True
            
            # 4. Add Gaussian noise (40% chance)
            if random.random() > 0.6:
                noise_std = 0.05 * (aug_image.max() - aug_image.min())
                noise = np.random.normal(0, noise_std, aug_image.shape)
                aug_image = aug_image + noise
                aug_image = np.clip(aug_image, aug_image.min(), aug_image.max())
                augmentation_applied = True
            
            # 5. Brightness adjustment (50% chance)
            if random.random() > 0.5:
                brightness_factor = random.uniform(0.8, 1.2)
                aug_image = aug_image * brightness_factor
                aug_image = np.clip(aug_image, 0, aug_image.max())
                augmentation_applied = True
            
            # 6. Contrast adjustment (50% chance)
            if random.random() > 0.5:
                contrast_factor = random.uniform(0.8, 1.2)
                mean_val = aug_image.mean()
                aug_image = (aug_image - mean_val) * contrast_factor + mean_val
                aug_image = np.clip(aug_image, 0, aug_image.max())
                augmentation_applied = True
            
            # If no augmentation was applied, force at least one
            if not augmentation_applied:
                if random.random() > 0.5:
                    # Apply horizontal flip
                    aug_image = np.fliplr(aug_image)
                    aug_label = np.fliplr(aug_label)
                else:
                    # Apply rotation
                    angle = random.uniform(-10, 10)
                    aug_image = rotate(aug_image, angle, order=1, reshape=False, mode='constant')
                    aug_label = rotate(aug_label, angle, order=0, reshape=False, mode='constant')
            
            augmented_pairs.append((aug_image, aug_label))
        
        return augmented_pairs
    
    def process_single_case(self, image_path, label_path, case_name):
        """Process a single image-label pair"""
        try:
            print(f"Processing case: {case_name}")
            print(f"Image path: {image_path}")
            print(f"Label path: {label_path}")
            
            # Check if files exist
            if not os.path.exists(image_path):
                print(f"ERROR: Image file does not exist: {image_path}")
                return False
            if not os.path.exists(label_path):
                print(f"ERROR: Label file does not exist: {label_path}")
                return False
            
            # Load original image and label
            img_nii = nib.load(image_path)
            label_nii = nib.load(label_path)
            
            image_data = img_nii.get_fdata()
            label_data = label_nii.get_fdata()
            
            print(f"Image shape: {image_data.shape}")
            print(f"Label shape: {label_data.shape}")
            
            # Process each slice (assuming 3D volumes)
            if len(image_data.shape) == 4:
                # Remove the 4th dimension if present
                image_data = image_data[:, :, :, 0] if image_data.shape[3] == 1 else image_data
                label_data = label_data[:, :, :, 0] if len(label_data.shape) == 4 and label_data.shape[3] == 1 else label_data
            
            num_slices = image_data.shape[2] if len(image_data.shape) >= 3 else 1
            print(f"Number of slices: {num_slices}")
            
            # Copy original files first (always save as .nii.gz)
            original_img_path = os.path.join(self.aug_images_dir, f"{case_name}_original.nii.gz")
            original_label_path = os.path.join(self.aug_labels_dir, f"{case_name}_original.nii.gz")
            
            print(f"Saving original to: {original_img_path}")
            nib.save(img_nii, original_img_path)
            nib.save(label_nii, original_label_path)
            
            # Generate augmented versions
            for aug_idx in range(self.augmentation_factor - 1):
                print(f"Creating augmentation {aug_idx + 1}")
                aug_image_volume = np.zeros_like(image_data)
                aug_label_volume = np.zeros_like(label_data)
                
                # Apply augmentations slice by slice for consistency
                random.seed(42 + aug_idx)  # Ensure reproducibility
                
                if len(image_data.shape) >= 3:
                    for slice_idx in range(num_slices):
                        image_slice = image_data[:, :, slice_idx]
                        label_slice = label_data[:, :, slice_idx]
                        
                        # Skip empty slices
                        if image_slice.max() == 0:
                            aug_image_volume[:, :, slice_idx] = image_slice
                            aug_label_volume[:, :, slice_idx] = label_slice
                            continue
                        
                        # Apply augmentations to this slice
                        augmented_pairs = self.apply_augmentations(image_slice, label_slice)
                        if augmented_pairs:
                            aug_image_slice, aug_label_slice = augmented_pairs[0]  # Take first augmentation
                            aug_image_volume[:, :, slice_idx] = aug_image_slice
                            aug_label_volume[:, :, slice_idx] = aug_label_slice
                        else:
                            aug_image_volume[:, :, slice_idx] = image_slice
                            aug_label_volume[:, :, slice_idx] = label_slice
                else:
                    # Handle 2D case
                    augmented_pairs = self.apply_augmentations(image_data, label_data)
                    if augmented_pairs:
                        aug_image_volume, aug_label_volume = augmented_pairs[0]
                    else:
                        aug_image_volume = image_data
                        aug_label_volume = label_data
                
                # Save augmented volume (always as .nii.gz)
                aug_img_nii = nib.Nifti1Image(aug_image_volume, img_nii.affine, img_nii.header)
                aug_label_nii = nib.Nifti1Image(aug_label_volume, label_nii.affine, label_nii.header)
                
                aug_img_path = os.path.join(self.aug_images_dir, f"{case_name}_aug_{aug_idx+1}.nii.gz")
                aug_label_path = os.path.join(self.aug_labels_dir, f"{case_name}_aug_{aug_idx+1}.nii.gz")
                
                print(f"Saving augmented to: {aug_img_path}")
                nib.save(aug_img_nii, aug_img_path)
                nib.save(aug_label_nii, aug_label_path)
            
            print(f"Successfully processed {case_name}")
            return True
            
        except Exception as e:
            print(f"Error processing {case_name}: {str(e)}")
            import traceback
            traceback.print_exc()
            return False

    def generate_augmented_dataset(self):
        """Generate the complete augmented dataset"""
        images_tr_dir = os.path.join(self.input_dir, 'imagesTr')
        labels_tr_dir = os.path.join(self.input_dir, 'labelsTr')
        
        # FIXED: Look for both .nii and .nii.gz files
        image_files = [f for f in os.listdir(images_tr_dir) 
                       if (f.endswith('.nii.gz') or f.endswith('.nii')) and not f.startswith('._')]
        image_files.sort()
        
        print(f"Found {len(image_files)} images to augment")
        print(f"Will generate {len(image_files) * self.augmentation_factor} total images")
        
        successful_cases = 0
        failed_cases = 0
        
        for image_file in tqdm(image_files, desc="Generating augmented dataset"):
            # FIXED: Handle both file extensions
            if image_file.endswith('.nii.gz'):
                case_name = image_file.replace('.nii.gz', '')
            else:
                case_name = image_file.replace('.nii', '')
            
            image_path = os.path.join(images_tr_dir, image_file)
            label_path = os.path.join(labels_tr_dir, image_file)
            
            if not os.path.exists(label_path):
                print(f"Warning: Label file not found for {image_file}")
                failed_cases += 1
                continue
            
            if self.process_single_case(image_path, label_path, case_name):
                successful_cases += 1
            else:
                failed_cases += 1
        
        print(f"\nAugmentation complete!")
        print(f"Successfully processed: {successful_cases} cases")
        print(f"Failed: {failed_cases} cases")
        print(f"Total augmented images: {successful_cases * self.augmentation_factor}")
        print(f"Augmented data saved in: {self.output_dir}")
    

def get_augmented_data_paths(augmented_data_dir):
    """Get paths for augmented dataset"""
    aug_images_dir = os.path.join(augmented_data_dir, 'imagesTr_augmented')
    aug_labels_dir = os.path.join(augmented_data_dir, 'labelsTr_augmented')
    
    # FIXED: Look for .nii.gz files (our augmented files are saved as .nii.gz)
    image_files = [f for f in os.listdir(aug_images_dir) if f.endswith('.nii.gz')]
    image_files.sort()
    
    image_paths = [os.path.join(aug_images_dir, f) for f in image_files]
    label_paths = [os.path.join(aug_labels_dir, f) for f in image_files]
    
    return image_paths, label_paths

def train_val_split_augmented(augmented_data_dir, val_ratio=0.2, random_seed=42):
    """
    Split augmented data ensuring original images go to validation
    and augmented versions stay in training
    """
    random.seed(random_seed)
    np.random.seed(random_seed)
    
    aug_images_dir = os.path.join(augmented_data_dir, 'imagesTr_augmented')
    aug_labels_dir = os.path.join(augmented_data_dir, 'labelsTr_augmented')
    
    # FIXED: Check if directories exist and look for .nii.gz files
    if not os.path.exists(aug_images_dir):
        print(f"Error: {aug_images_dir} does not exist!")
        return [], [], [], []
    
    # Get all files - our augmented files are saved as .nii.gz
    all_files = [f for f in os.listdir(aug_images_dir) if f.endswith('.nii.gz')]
    
    if len(all_files) == 0:
        print(f"No .nii.gz files found in {aug_images_dir}")
        print("Available files:")
        all_available = os.listdir(aug_images_dir)
        for f in all_available[:10]:  # Show first 10 files
            print(f"  {f}")
        return [], [], [], []
    
    # Separate original and augmented files
    original_files = [f for f in all_files if '_original.nii.gz' in f]
    augmented_files = [f for f in all_files if '_aug_' in f]
    
    print(f"Original files: {len(original_files)}")
    print(f"Augmented files: {len(augmented_files)}")
    
    if len(original_files) == 0:
        print("Warning: No original files found! Check file naming.")
        return [], [], [], []
    
    # Use 20% of original files for validation
    random.shuffle(original_files)
    val_size = int(len(original_files) * val_ratio)
    val_files = original_files[:val_size]
    train_original_files = original_files[val_size:]
    
    # All augmented files go to training
    train_files = train_original_files + augmented_files
    
    # Create full paths
    train_images = [os.path.join(aug_images_dir, f) for f in train_files]
    val_images = [os.path.join(aug_images_dir, f) for f in val_files]
    train_labels = [os.path.join(aug_labels_dir, f) for f in train_files]
    val_labels = [os.path.join(aug_labels_dir, f) for f in val_files]
    
    print(f"Training set: {len(train_images)} images")
    print(f"Validation set: {len(val_images)} images")
    
    return train_images, val_images, train_labels, val_labels

def debug_before_augmentation():
    INPUT_DIR = '/kaggle/input/my-dataset/Task05_Prostate'
    
    # Check exactly what the augmentor will see
    images_path = os.path.join(INPUT_DIR, 'imagesTr')
    labels_path = os.path.join(INPUT_DIR, 'labelsTr')
    
    print(f"Augmentor will look for:")
    print(f"Images: {images_path}")
    print(f"Labels: {labels_path}")
    
    # Check files
    if os.path.exists(images_path):
        image_files = [f for f in os.listdir(images_path) if not f.startswith('.') and f.endswith(('.nii', '.nii.gz'))]
        print(f"\nFound {len(image_files)} image files:")
        for f in image_files[:5]:
            print(f"  {f}")
    
    if os.path.exists(labels_path):
        label_files = [f for f in os.listdir(labels_path) if not f.startswith('.') and f.endswith(('.nii', '.nii.gz'))]
        print(f"\nFound {len(label_files)} label files:")
        for f in label_files[:5]:
            print(f"  {f}")

def debug_after_augmentation():
    """Debug function to check augmented files"""
    AUGMENTED_DIR = '/kaggle/working/Task05_Prostate'
    aug_images_dir = os.path.join(AUGMENTED_DIR, 'imagesTr_augmented')
    
    if os.path.exists(aug_images_dir):
        all_files = os.listdir(aug_images_dir)
        print(f"\nFiles in augmented directory ({len(all_files)} total):")
        for f in all_files[:10]:  # Show first 10
            print(f"  {f}")
        
        # Count different types
        nii_gz_files = [f for f in all_files if f.endswith('.nii.gz')]
        nii_files = [f for f in all_files if f.endswith('.nii') and not f.endswith('.nii.gz')]
        original_files = [f for f in nii_gz_files if '_original' in f]
        aug_files = [f for f in nii_gz_files if '_aug_' in f]
        
        print(f"\n.nii.gz files: {len(nii_gz_files)}")
        print(f".nii files: {len(nii_files)}")
        print(f"Original files: {len(original_files)}")
        print(f"Augmented files: {len(aug_files)}")
    else:
        print(f"Augmented directory does not exist: {aug_images_dir}")

# Example usage
def main_augmentation():
    # Set paths
    INPUT_DIR = '/kaggle/input/my-dataset/Task05_Prostate'
    OUTPUT_DIR = '/kaggle/working/Task05_Prostate'
    
    # Create augmentor and generate dataset
    augmentor = OfflineAugmentor(INPUT_DIR, OUTPUT_DIR, augmentation_factor=3)
    
    print("Starting offline data augmentation...")
    augmentor.generate_augmented_dataset()
    
    print("\nDebugging augmented files...")
    debug_after_augmentation()
    
    print("\nTesting train/val split...")
    train_images, val_images, train_labels, val_labels = train_val_split_augmented(OUTPUT_DIR)
    
    print(f"Final dataset summary:")
    print(f"Training: {len(train_images)} images")
    print(f"Validation: {len(val_images)} images")

# Modified ProstateDataset for offline augmented data
class ProstateDatasetOffline(Dataset):
    def __init__(self, image_paths, label_paths):
        self.image_paths = image_paths
        self.label_paths = label_paths
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_nii = nib.load(self.image_paths[idx])
        label_nii = nib.load(self.label_paths[idx])
        
        image = img_nii.get_fdata()
        label = label_nii.get_fdata()
        
        # Handle dimensions and extract middle slice
        if len(image.shape) == 4:
            mid_slice = image.shape[2] // 2
            image = image[:, :, mid_slice, 0]
            label = label[:, :, mid_slice] if len(label.shape) == 3 else label[:, :, mid_slice, 0]
        elif len(image.shape) == 3:
            mid_slice = image.shape[2] // 2
            image = image[:, :, mid_slice]
            label = label[:, :, mid_slice]
        
        if len(image.shape) > 2:
            image = np.squeeze(image)
        if len(label.shape) > 2:
            label = np.squeeze(label)
            
        # Enhanced normalization
        image = np.clip(image, np.percentile(image, 1), np.percentile(image, 99))
        image_min, image_max = image.min(), image.max()
        if image_max > image_min:
            image = (image - image_min) / (image_max - image_min)
        else:
            image = np.zeros_like(image)
        
        image = torch.tensor(image, dtype=torch.float32).unsqueeze(0)
        label = torch.tensor(label, dtype=torch.long)
        
        # Resize to 128x128 for better feature extraction
        image = F.interpolate(image.unsqueeze(0), size=(128, 128), mode='bilinear', align_corners=False).squeeze(0)
        label = F.interpolate(label.unsqueeze(0).unsqueeze(0).float(), size=(128, 128), mode='nearest').squeeze(0).squeeze(0).long()
        
        # No augmentation here - it's already done offline
        return image, label

# Updated main function for offline augmented training
def main_training_with_offline_augmentation():
    # Step 1: Generate augmented dataset (run this once)
    INPUT_DIR = '/kaggle/input/my-dataset/Task05_Prostate'
    AUGMENTED_DIR = '/kaggle/working/Task05_Prostate'
    os.makedirs(AUGMENTED_DIR, exist_ok=True)
    
    # Check if augmented data already exists and has files
    aug_images_dir = os.path.join(AUGMENTED_DIR, 'imagesTr_augmented')
    should_generate = True
    
    if os.path.exists(aug_images_dir):
        existing_files = [f for f in os.listdir(aug_images_dir) if f.endswith('.nii.gz')]
        if len(existing_files) > 0:
            print(f"Using existing augmented dataset with {len(existing_files)} files...")
            should_generate = False
        else:
            print("Augmented directory exists but is empty. Regenerating...")
    
    if should_generate:
        print("Generating offline augmented dataset...")
        augmentor = OfflineAugmentor(INPUT_DIR, AUGMENTED_DIR, augmentation_factor=3)
        augmentor.generate_augmented_dataset()
        debug_after_augmentation()  # Debug newly created files
    else:
        debug_after_augmentation()  # Debug existing files
    
    # Step 2: Load augmented data for training
    BATCH_SIZE = 2
    LEARNING_RATE = 1e-4
    NUM_EPOCHS = 200
    
    # Use the offline augmented data
    train_images, val_images, train_labels, val_labels = train_val_split_augmented(AUGMENTED_DIR)
    
    if len(train_images) == 0:
        print("ERROR: No training images found! Check augmentation process.")
        return
    
    # Use the offline dataset (no online augmentation needed)
    train_dataset = ProstateDatasetOffline(train_images, train_labels)
    val_dataset = ProstateDatasetOffline(val_images, val_labels)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    
    print(f"Training with {len(train_images)} images ({len(train_images)//3} original × 3 with augmentation)")
    print(f"Validation with {len(val_images)} original images")
    
    # Rest of training code would go here...
    # (You can use your existing EnhancedAttentionUNet model and training loop)

if __name__ == "__main__":
    # Choose what to run:
    # main_augmentation()  # Run this first to generate augmented data
    debug_before_augmentation()
    main_training_with_offline_augmentation()  # Run this to train with augmented data
