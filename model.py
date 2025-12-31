import os
import glob
import torch
import numpy as np
import nibabel as nib
from monai.networks.nets import SegResNet
from monai.inferers import sliding_window_inference
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, 
    Orientationd, Spacingd, NormalizeIntensityd, Invertd
)
from monai.data import Dataset, DataLoader, decollate_batch


TEST_DIR = "test"
OUTPUT_DIR = "predictions"
MODEL_FILE = "best_metric_model.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs(OUTPUT_DIR, exist_ok=True)


test_transforms = Compose([
    LoadImaged(keys=["image"]),
    EnsureChannelFirstd(keys=["image"]),
    
    Orientationd(keys=["image"], axcodes="RAS"), 
    
    Spacingd(keys=["image"], pixdim=(1.0, 1.0, 1.0), mode="bilinear"), 
    NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
])


post_transforms = Compose([
    Invertd(
        keys="pred",            
        transform=test_transforms, 
        orig_keys="image",      
        meta_keys="pred_meta_dict",
        orig_meta_keys="image_meta_dict",
        meta_key_postfix="meta_dict",
        nearest_interp=False,   
        to_tensor=True,
    ),
])


model = SegResNet(
    spatial_dims=3, 
    in_channels=4, 
    out_channels=3, 
    init_filters=16, 
    dropout_prob=0.2
).to(DEVICE)

if os.path.exists(MODEL_FILE):
    try:
        model.load_state_dict(torch.load(MODEL_FILE, map_location=DEVICE))
    except:
        model.load_state_dict(torch.load(MODEL_FILE, map_location=DEVICE, weights_only=False))
    print(f" Loaded weights from {MODEL_FILE}")
else:
    raise FileNotFoundError(f" Error: {MODEL_FILE} not found.")

patient_folders = sorted(glob.glob(os.path.join(TEST_DIR, "HelioGLI-*")))
print(f" Found {len(patient_folders)} patients in {TEST_DIR}")

for subject_path in patient_folders:
    patient_id = os.path.basename(subject_path)
    print(f"Processing {patient_id}...")
    
    def get_file(key):
        files = glob.glob(os.path.join(subject_path, f"*{key}.nii.gz"))
        return files[0] if files else None

    try:
        files = {
            "image": [
                get_file("t1n"), get_file("t1c"),
                get_file("t2w"), get_file("t2f")
            ]
        }
        if None in files["image"]: continue
    except: continue

    ds = Dataset(data=[files], transform=test_transforms)
    loader = DataLoader(ds, batch_size=1, shuffle=False)
    
    with torch.no_grad():
        for batch_data in loader:
            inputs = batch_data["image"].to(DEVICE)
            
            val_outputs = sliding_window_inference(inputs, (128, 128, 128), 4, model, overlap=0.5)
            val_outputs = torch.sigmoid(val_outputs) 
            
            
            batch_data["pred"] = val_outputs
            
            batch_data_list = [post_transforms(i) for i in decollate_batch(batch_data)]
            final_pred = batch_data_list[0]["pred"]
            
            final_pred = (final_pred > 0.5).float()
            
            result = final_pred.cpu().numpy()
            if len(result.shape) == 4: result = np.moveaxis(result, 0, -1)
            
          
            original_nifti = nib.load(files["image"][0])
            output_filename = os.path.join(OUTPUT_DIR, f"{patient_id}.nii.gz")
            nib.save(nib.Nifti1Image(result, original_nifti.affine), output_filename)
            
            print(f"   Saved -> {output_filename}")