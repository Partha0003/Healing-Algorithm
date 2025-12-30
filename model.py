import os
import glob
import torch
import numpy as np
import nibabel as nib
from monai.networks.nets import SegResNet
from monai.inferers import sliding_window_inference
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Orientationd, 
    Spacingd, NormalizeIntensityd
)
from monai.data import Dataset, DataLoader


TEST_DIR = "test"
OUTPUT_DIR = "predictions"
MODEL_FILE = "best_metric_model.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs(OUTPUT_DIR, exist_ok=True)


test_transforms = Compose([
    LoadImaged(keys=["image"]),
    EnsureChannelFirstd(keys=["image"]),
    Orientationd(keys=["image"], axcodes="RAS"),
    Spacingd(keys=["image"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear")),
    NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
])


model = SegResNet(
    spatial_dims=3, 
    in_channels=4, 
    out_channels=3, 
    init_filters=16, 
    dropout_prob=0.2
).to(DEVICE)

if os.path.exists(MODEL_FILE):
    model.load_state_dict(torch.load(MODEL_FILE, map_location=DEVICE))
    print(f" Loaded weights from {MODEL_FILE}")
else:
    raise FileNotFoundError(f" Error: {MODEL_FILE} not found. Please put it in the same folder.")

model.eval()


patient_folders = sorted(glob.glob(os.path.join(TEST_DIR, "HelioGLI-*")))
print(f"🔍 Found {len(patient_folders)} patients in {TEST_DIR}")

for subject_path in patient_folders:
    patient_id = os.path.basename(subject_path)
    print(f"Processing {patient_id}...")
    
    # Locate the 4 input files
    def get_file(key):
        files = glob.glob(os.path.join(subject_path, f"*{key}.nii.gz"))
        return files[0] if files else None

    try:
        files = {
            "image": [
                get_file("t1n"),
                get_file("t1c"),
                get_file("t2w"),
                get_file("t2f")
            ]
        }
        # Check if any file is missing
        if None in files["image"]:
            raise ValueError("Missing one of the 4 MRI files.")
            
    except (IndexError, ValueError):
        print(f" Skipping {patient_id}: Missing MRI files.")
        continue

    ds = Dataset(data=[files], transform=test_transforms)
    loader = DataLoader(ds, batch_size=1, shuffle=False)
    
    original_affine = nib.load(files["image"][0]).affine

    with torch.no_grad():
        for batch_data in loader:
            inputs = batch_data["image"].to(DEVICE)
            
            val_outputs = sliding_window_inference(inputs, (128, 128, 128), 4, model, overlap=0.5)
            val_outputs = (val_outputs.sigmoid() > 0.5).float()
            
        
            result = val_outputs.cpu().numpy()[0]
            result = np.moveaxis(result, 0, -1) 
            
            output_filename = os.path.join(OUTPUT_DIR, f"{patient_id}.nii.gz")
            nib.save(nib.Nifti1Image(result, original_affine), output_filename)
            print(f"   Saved -> {output_filename}")
