from PIL import Image
import os
import re 

# Folder containing your images
folder = "map_frames"

# Collect only valid image files
files = [f for f in os.listdir(folder) if f.endswith(('.png', '.jpg', '.jpeg'))]

# Sort by numeric index in the filename
files.sort(key=lambda f: int(re.search(r'\d+', f).group()))

# Load images
frames = [Image.open(os.path.join(folder, f)) for f in files]

# Save as GIF
output_path = "map_animation.gif"
frames[0].save(
    output_path,
    save_all=True,
    append_images=frames[1:],
    duration=50,   
    loop=0          # 0 = infinite loop
)

print(f"GIF saved as {output_path}")
