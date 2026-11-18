import cv2
import numpy as np
#---------READ IMAGE----------------------
img = cv2.imread("image.jpg")
if img is None:
    raise FileNotFoundError("image.jpg not found")

#--------------------------------------------
img = img.astype(np.float32)#convert to float
#---------------SPLIT------------------------
b, g, r = cv2.split(img)

avg_b = np.mean(b)#average intencity for blue
avg_g = np.mean(g)#average intensity for green
avg_r = np.mean(r)#average intensity for red

gray = (avg_b + avg_g + avg_r) / 3
#----------------WHITNING---------------------
b = b * (gray / avg_b)
g = g * (gray / avg_g)
r = r * (gray / avg_r)
#---------------MERGE-------------------------
IWB = cv2.merge((b,g,r))

IWB = np.clip(IWB,0,255).astype(np.uint8)

cv2.imwrite("white_balance.jpg",IWB)

#-----------CONTRAST ENHANCEMENT---------------

# Convert White Balanced image to LAB
lab = cv2.cvtColor(IWB, cv2.COLOR_BGR2LAB)

# Split LAB channels
L, A, B = cv2.split(lab)

# Create CLAHE object
clahe = cv2.createCLAHE(
    clipLimit=3.0,
    tileGridSize=(8,8)
)

# Apply CLAHE only on L channel
L = clahe.apply(L)

# Merge channels
lab = cv2.merge((L, A, B))

# Convert back to BGR
ICE = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

cv2.imwrite("Contrast_Enhancement.jpg", ICE)

#--------------------DISPLAY-----------------------------

import matplotlib.pyplot as plt

plt.figure(figsize=(15,5))

plt.subplot(131)
plt.imshow(cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2RGB))
plt.title("Original")

plt.subplot(132)
plt.imshow(cv2.cvtColor(IWB, cv2.COLOR_BGR2RGB))
plt.title("White Balanced (IWB)")

plt.subplot(133)
plt.imshow(cv2.cvtColor(ICE, cv2.COLOR_BGR2RGB))
plt.title("Contrast Enhanced (ICE)")

plt.tight_layout()
plt.savefig("2.jpg",dpi=300, bbox_inches="tight")
plt.show()

#-------------------------------COLOURFULLNESS------------------

def colourfulness_map(img, patch_size=15):

    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

    L, A, B = cv2.split(lab)

    A = A.astype(np.float32)
    B = B.astype(np.float32)

    h, w = A.shape

    colour_map = np.zeros((h, w), dtype=np.float32)

    w1 = 0.5
    w2 = 0.5

    offset = patch_size // 2

    for y in range(offset, h-offset):

        for x in range(offset, w-offset):

            patch_a = A[
                y-offset:y+offset+1,
                x-offset:x+offset+1
            ]

            patch_b = B[
                y-offset:y+offset+1,
                x-offset:x+offset+1
            ]

            mu_a = np.mean(patch_a)
            mu_b = np.mean(patch_b)

            sigma_a = np.std(patch_a)
            sigma_b = np.std(patch_b)

            colour = (
                w1*np.sqrt(sigma_a**2 + sigma_b**2)
                +
                w2*np.sqrt(mu_a**2 + mu_b**2)
            )

            colour_map[y,x] = colour

    return colour_map
colour_map_IWB = colourfulness_map(IWB)
colour_map_ICE = colourfulness_map(ICE)

colour_map_IWB = cv2.normalize(
    colour_map_IWB,
    None,
    0,
    255,
    cv2.NORM_MINMAX
).astype(np.uint8)

colour_map_ICE = cv2.normalize(
    colour_map_ICE,
    None,
    0,
    255,
    cv2.NORM_MINMAX
).astype(np.uint8)

cv2.imwrite("Colourfulness_IWB.jpg", colour_map_IWB)
cv2.imwrite("Colourfulness_ICE.jpg", colour_map_ICE)
#------------------------------------------DISPLAY------------------------------
plt.figure(figsize=(12,5))

plt.subplot(121)
plt.imshow(colour_map_IWB, cmap="jet")
plt.title("Colourfulness Map - IWB")
plt.axis("off")

plt.subplot(122)
plt.imshow(colour_map_ICE, cmap="jet")
plt.title("Colourfulness Map - ICE")
plt.axis("off")

plt.tight_layout()
plt.savefig("Colourfulness_Map.jpg", dpi=300)
plt.show()

#----------------VISUAL SALIENCY------------------------
import cv2
import numpy as np

def saliency_map(img):

    # Create saliency detector
    saliency = cv2.saliency.StaticSaliencySpectralResidual_create()

    # Compute saliency
    success, saliencyMap = saliency.computeSaliency(img)

    if not success:
        raise Exception("Saliency computation failed.")

    # Convert from float (0-1) to uint8 (0-255)
    saliencyMap = (saliencyMap * 255).astype("uint8")

    return saliencyMap
saliency_IWB = saliency_map(IWB)
saliency_ICE = saliency_map(ICE)

cv2.imwrite("Saliency_IWB.jpg", saliency_IWB)
cv2.imwrite("Saliency_ICE.jpg", saliency_ICE)

plt.figure(figsize=(12,5))

plt.subplot(121)
plt.imshow(saliency_IWB, cmap="gray")
plt.title("Saliency Map - IWB")
plt.axis("off")

plt.subplot(122)
plt.imshow(saliency_ICE, cmap="gray")
plt.title("Saliency Map - ICE")
plt.axis("off")

plt.tight_layout()
plt.savefig("Saliency_Map.jpg", dpi=300)
plt.show()

#---------------------ENTROPY-----------------------------

def entropy_map(img, patch_size=15):

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    h, w = gray.shape

    entropy = np.zeros((h, w), dtype=np.float32)

    offset = patch_size // 2

    for y in range(offset, h-offset):

        for x in range(offset, w-offset):

            patch = gray[
                y-offset:y+offset+1,
                x-offset:x+offset+1
            ]

            hist = cv2.calcHist(
                [patch],
                [0],
                None,
                [256],
                [0,256]
            )

            hist = hist / hist.sum()

            hist = hist[hist > 0]

            entropy[y,x] = -np.sum(
                hist * np.log2(hist)
            )

    return entropy

entropy_IWB = entropy_map(IWB)
entropy_ICE = entropy_map(ICE)

entropy_IWB = cv2.normalize(
    entropy_IWB,
    None,
    0,
    255,
    cv2.NORM_MINMAX
).astype(np.uint8)

entropy_ICE = cv2.normalize(
    entropy_ICE,
    None,
    0,
    255,
    cv2.NORM_MINMAX
).astype(np.uint8)

cv2.imwrite("Entropy_IWB.jpg", entropy_IWB)
cv2.imwrite("Entropy_ICE.jpg", entropy_ICE)

plt.figure(figsize=(12,5))

plt.subplot(121)
plt.imshow(entropy_IWB, cmap="gray")
plt.title("Entropy Map - IWB")
plt.axis("off")

plt.subplot(122)
plt.imshow(entropy_ICE, cmap="gray")
plt.title("Entropy Map - ICE")
plt.axis("off")

plt.tight_layout()
plt.savefig("Entropy_Map.jpg", dpi=300)
plt.show()

#--------------------------------VISIBILITY----------------------------
def visibility_map(img):

    gray = cv2.cvtColor(
        img,
        cv2.COLOR_BGR2GRAY
    )

    grad_x = cv2.Sobel(
        gray,
        cv2.CV_64F,
        1,
        0,
        ksize=3
    )

    grad_y = cv2.Sobel(
        gray,
        cv2.CV_64F,
        0,
        1,
        ksize=3
    )

    magnitude = cv2.magnitude(
        grad_x,
        grad_y
    )

    visibility = cv2.normalize(
        magnitude,
        None,
        0,
        255,
        cv2.NORM_MINMAX
    )

    return visibility.astype(np.uint8)
visibility_IWB = visibility_map(IWB)

visibility_ICE = visibility_map(ICE)

cv2.imwrite(
    "Visibility_IWB.jpg",
    visibility_IWB
)

cv2.imwrite(
    "Visibility_ICE.jpg",
    visibility_ICE
)

plt.figure(figsize=(12,5))

plt.subplot(121)
plt.imshow(
    visibility_IWB,
    cmap="gray"
)
plt.title("Visibility Map - IWB")
plt.axis("off")

plt.subplot(122)
plt.imshow(
    visibility_ICE,
    cmap="gray"
)
plt.title("Visibility Map - ICE")
plt.axis("off")

plt.tight_layout()

plt.savefig(
    "Visibility_Map.jpg",
    dpi=300
)

plt.show()
#------------------------------CHROMATICITY-------------------------
def chromaticity_map(img):

    hsv = cv2.cvtColor(
        img,
        cv2.COLOR_BGR2HSV
    )

    H, S, V = cv2.split(hsv)

    chromaticity = cv2.normalize(
        S,
        None,
        0,
        255,
        cv2.NORM_MINMAX
    )

    return chromaticity.astype(np.uint8)

chromaticity_IWB = chromaticity_map(IWB)

chromaticity_ICE = chromaticity_map(ICE)

cv2.imwrite(
    "Chromaticity_IWB.jpg",
    chromaticity_IWB
)

cv2.imwrite(
    "Chromaticity_ICE.jpg",
    chromaticity_ICE
)

plt.figure(figsize=(12,5))

plt.subplot(121)
plt.imshow(
    chromaticity_IWB,
    cmap="gray"
)
plt.title("Chromaticity Map - IWB")
plt.axis("off")

plt.subplot(122)
plt.imshow(
    chromaticity_ICE,
    cmap="gray"
)
plt.title("Chromaticity Map - ICE")
plt.axis("off")

plt.tight_layout()

plt.savefig(
    "Chromaticity_Map.jpg",
    dpi=300
)

plt.show()

#-------------------------WEIGHT MAP----------------------------

def weight_map(
    colour,
    saliency,
    visibility,
    chromaticity
):

    weight = (

        colour.astype(np.float32)

        + saliency.astype(np.float32)

        + visibility.astype(np.float32)

        + chromaticity.astype(np.float32)

    ) / 4

    return weight


weight_IWB = weight_map(
    colour_map_IWB,
    saliency_IWB,
    visibility_IWB,
    chromaticity_IWB
)

weight_ICE = weight_map(
    colour_map_ICE,
    saliency_ICE,
    visibility_ICE,
    chromaticity_ICE
)

# Normalize each map
weight_IWB = cv2.normalize(weight_IWB, None, 0, 1, cv2.NORM_MINMAX)
weight_ICE = cv2.normalize(weight_ICE, None, 0, 1, cv2.NORM_MINMAX)

# Make both weights sum to 1
sum_weight = weight_IWB + weight_ICE + 1e-6

weight_IWB = weight_IWB / sum_weight
weight_ICE = weight_ICE / sum_weight

#-------------------------GAUSSIAN PYRAMID------------
def gaussian_pyramid(img, levels=5):

    pyramid = [img]

    current = img.copy()

    for i in range(levels - 1):

        current = cv2.pyrDown(current)

        pyramid.append(current)

    return pyramid
gaussian_IWB = gaussian_pyramid(IWB)

gaussian_ICE = gaussian_pyramid(ICE)

gaussian_weight_IWB = gaussian_pyramid(weight_IWB)

gaussian_weight_ICE = gaussian_pyramid(weight_ICE)

for i, level in enumerate(gaussian_IWB):

    print(f"Level {i}: {level.shape}")

#--------------------------LAPLACIAN PYRAMID--------------------
def laplacian_pyramid(gaussian):

    laplacian = []

    for i in range(len(gaussian)-1):

        size = (
            gaussian[i].shape[1],
            gaussian[i].shape[0]
        )

        expanded = cv2.pyrUp(
            gaussian[i+1],
            dstsize=size
        )

        lap = cv2.subtract(
            gaussian[i],
            expanded
        )

        laplacian.append(lap)

    laplacian.append(
        gaussian[-1]
    )

    return laplacian

lap_IWB = laplacian_pyramid(
    gaussian_IWB
)

lap_ICE = laplacian_pyramid(
    gaussian_ICE
)

for i, level in enumerate(lap_IWB):

    print(level.shape)

#---------------------------------PYRAMID FUSHION--------------

def pyramid_fusion(

    lap1,
    lap2,

    weight1,
    weight2

):

    fused = []

    for l1, l2, w1, w2 in zip(

        lap1,

        lap2,

        weight1,

        weight2

    ):
        l1 = l1.astype(np.float32)
        l2 = l2.astype(np.float32)

        if len(l1.shape) == 3:

            w1 = cv2.merge(
                [w1,w1,w1]
            )

            w2 = cv2.merge(
                [w2,w2,w2]
            )

        fused_level = (

            w1*l1

            +

            w2*l2

        )

        fused.append(

            fused_level.astype(np.float32)

        )

    return fused

fused = pyramid_fusion(

    lap_IWB,

    lap_ICE,

    gaussian_weight_IWB,

    gaussian_weight_ICE

)

def reconstruct(pyramid):

    image = pyramid[-1]

    for i in range(

        len(pyramid)-2,

        -1,

        -1

    ):

        size = (

            pyramid[i].shape[1],

            pyramid[i].shape[0]

        )

        image = cv2.pyrUp(

            image,

            dstsize=size

        )

        image = cv2.resize( image, ( pyramid[i].shape[1], pyramid[i].shape[0] ))

        image = cv2.add(

            image,

            pyramid[i]

        )

    return image

dehazed = reconstruct(fused)

dehazed = np.clip(

    dehazed,

    0,

    255

).astype(np.uint8)

cv2.imwrite(

    "Dehazed.jpg",

    dehazed

)

plt.figure(figsize=(10,5))

plt.subplot(121)

plt.imshow(

    cv2.cvtColor(IWB,cv2.COLOR_BGR2RGB)

)

plt.title("Input")

plt.axis("off")

plt.subplot(122)

plt.imshow(

    cv2.cvtColor(dehazed,cv2.COLOR_BGR2RGB)

)

plt.title("Dehazed")

plt.axis("off")

plt.tight_layout()

plt.savefig("Final_Result.jpg",dpi=300)

plt.show()
