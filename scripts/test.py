import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Buat frame kosong
frame = np.zeros((300, 600, 3), dtype=np.uint8)

# Konversi ke format PIL
pil_frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
draw = ImageDraw.Draw(pil_frame)

# Jalur font
font_path = "fonts/Moderna.TTF"  # Pastikan file ada
font = ImageFont.truetype(font_path, 24)

# Tambahkan teks
draw.text((50, 50), "Testing Moderna Font", font=font, fill=(255, 255, 255))

# Konversi kembali ke OpenCV
frame = cv2.cvtColor(np.array(pil_frame), cv2.COLOR_RGB2BGR)

# Tampilkan frame
cv2.imshow("Test Font", frame)
cv2.waitKey(0)
cv2.destroyAllWindows()
