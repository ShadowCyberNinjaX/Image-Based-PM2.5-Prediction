import cv2
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
from image_dehazer import remove_haze

# -------------------------
# Read Image
# -------------------------
image = cv2.imread("18[R][0@0][0][2].jpg")

if image is None:
    raise Exception("Image not found")

# -------------------------
# Dehaze Image
# -------------------------
dehazed = remove_haze(image)

# Convert BGR → RGB
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
dehazed = cv2.cvtColor(dehazed, cv2.COLOR_BGR2RGB)

# -------------------------
# Select Near ROI
# -------------------------
print("Select NEAR ROI")
near = cv2.selectROI(
    "Near ROI",
    cv2.cvtColor(image, cv2.COLOR_RGB2BGR),
    False
)
cv2.destroyAllWindows()

# -------------------------
# Select Far ROI
# -------------------------
print("Select FAR ROI")
far = cv2.selectROI(
    "Far ROI",
    cv2.cvtColor(image, cv2.COLOR_RGB2BGR),
    False
)
cv2.destroyAllWindows()

# -------------------------
# Function
# -------------------------
def calculate_ssim(rect):

    x,y,w,h = rect

    hazy_roi = image[y:y+h,x:x+w]
    clear_roi = dehazed[y:y+h,x:x+w]

    hazy_gray = cv2.cvtColor(hazy_roi,cv2.COLOR_RGB2GRAY)
    clear_gray = cv2.cvtColor(clear_roi,cv2.COLOR_RGB2GRAY)

    score = ssim(hazy_gray,clear_gray)

    return score,hazy_roi,clear_roi

near_ssim,near_hazy,near_clear = calculate_ssim(near)
far_ssim,far_hazy,far_clear = calculate_ssim(far)

# -------------------------
# Results
# -------------------------
print("--------------------------------")
print("Near ROI SSIM :",near_ssim)
print("Far ROI SSIM  :",far_ssim)

# -------------------------
# Display
# -------------------------
fig,ax = plt.subplots(2,3,figsize=(15,8))

ax[0,0].imshow(image)
ax[0,0].set_title("Original")

ax[0,1].imshow(near_hazy)
ax[0,1].set_title("Near ROI")

ax[0,2].imshow(near_clear)
ax[0,2].set_title(f"Near Dehazed\nSSIM={near_ssim:.4f}")

ax[1,0].imshow(dehazed)
ax[1,0].set_title("Dehazed")

ax[1,1].imshow(far_hazy)
ax[1,1].set_title("Far ROI")

ax[1,2].imshow(far_clear)
ax[1,2].set_title(f"Far Dehazed\nSSIM={far_ssim:.4f}")

for a in ax.ravel():
    a.axis("off")

plt.tight_layout()
plt.show()
