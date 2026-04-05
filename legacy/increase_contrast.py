from PIL import Image, ImageEnhance

img = Image.open("hand_removed.png")

# increase contrast
enhancer = ImageEnhance.Contrast(img)
img = enhancer.enhance(1.5)

# slight sharpness
sharp = ImageEnhance.Sharpness(img)
img = sharp.enhance(1.3)

img.save("enhanced.png")