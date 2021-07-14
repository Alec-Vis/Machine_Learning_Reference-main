import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt

# path to nii file
# when running with ALT + SHIFT + E, path is '.\<data_path>' otherwise use '..\<data_path>'
t1_fn = r'..\data\Biomedical_images\1010_brain_mr_02.nii'

# read nii file image
sitk_t1 = sitk.ReadImage(t1_fn)

t1 = sitk.GetArrayFromImage(sitk_t1)

image = sitk.Image(sitk_t1)

z = 100
slice = t1[z,:,:]
plt.imshow(slice)
plt.show()

print(sitk.Version())

print("Width, Height, Depth")
print(image.GetSize())
