import matplotlib.animation as animation
import matplotlib.pyplot as plt
from matplotlib.image import imread
import os
import re

def create_gif_from_frames(frame_pattern='output_frame_{}.png', 
                          output_gif='output_animation.gif',
                          fps=10):
    """
    Create GIF from numbered frame files
    
    Args:
        frame_pattern: filename pattern with {} for frame number
        output_gif: output GIF filename
        fps: frames per second
    """
    # Find all matching frame files
    directory = '.'  # Current directory
    files = os.listdir(directory)
    
    # Extract frame numbers from filenames
    frame_files = []
    pattern = frame_pattern.replace('{}', r'(\d+)')
    
    for filename in files:
        match = re.match(pattern, filename)
        if match:
            frame_num = int(match.group(1))
            frame_files.append((frame_num, filename))
    
    # Sort by frame number
    frame_files.sort(key=lambda x: x[0])
    
    if not frame_files:
        print(f"No frames found matching pattern: {frame_pattern}")
        return
    
    print(f"Found {len(frame_files)} frames")
    print(f"Frame range: {frame_files[0][0]} to {frame_files[-1][0]}")
    
    # Load first frame to get dimensions
    first_frame = imread(frame_files[0][1])
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.axis('off')
    
    # Initialize with first frame
    im = ax.imshow(first_frame)
    
    def init():
        im.set_data(first_frame)
        return [im]
    
    def animate(frame_idx):
        _, filename = frame_files[frame_idx]
        img = imread(filename)
        im.set_data(img)
        return [im]
    
    # Create animation
    print(f"Creating animation...")
    anim = animation.FuncAnimation(
        fig, animate, init_func=init,
        frames=len(frame_files), 
        interval=1000/fps,  # Convert fps to milliseconds
        blit=True, 
        repeat=True
    )
    
    # Save as GIF
    print(f"Saving to {output_gif}...")
    anim.save(output_gif, writer='pillow', fps=fps)
    print(f"✓ Saved {output_gif} ({len(frame_files)} frames @ {fps} FPS)")
    
    plt.close()


def create_gif_from_frames_simple(frame_pattern='output_frame_{}.png',
                                   output_gif='output_animation.gif',
                                   fps=10,
                                   loop=0):
    """
    Create GIF using PIL directly (simpler, no matplotlib dependency)
    
    Args:
        frame_pattern: filename pattern with {} for frame number
        output_gif: output GIF filename
        fps: frames per second
        loop: 0 for infinite loop, N for N loops
    """
    from PIL import Image
    import os
    import re
    
    # Find all matching frame files
    directory = '.'
    files = os.listdir(directory)
    
    # Extract frame numbers
    frame_files = []
    pattern = frame_pattern.replace('{}', r'(\d+)')
    
    for filename in files:
        match = re.match(pattern, filename)
        if match:
            frame_num = int(match.group(1))
            frame_files.append((frame_num, filename))
    
    # Sort numerically
    frame_files.sort(key=lambda x: x[0])
    
    if not frame_files:
        print(f"No frames found matching pattern: {frame_pattern}")
        return
    
    print(f"Found {len(frame_files)} frames")
    print(f"Frame range: {frame_files[0][0]} to {frame_files[-1][0]}")
    
    # Load all frames
    print("Loading frames...")
    frames = []
    for frame_num, filename in frame_files:
        img = Image.open(filename)
        frames.append(img.copy())
        img.close()
    
    # Save as GIF
    print(f"Saving to {output_gif}...")
    duration = int(1000 / fps)  # milliseconds per frame
    
    frames[0].save(
        output_gif,
        save_all=True,
        append_images=frames[1:],
        duration=duration,
        loop=loop
    )
    
    print(f"✓ Saved {output_gif} ({len(frames)} frames @ {fps} FPS)")


# Method 2: Using PIL directly (simpler, often faster)
os.chdir("out")
create_gif_from_frames_simple(
    frame_pattern='output_frame_{}.png',
    output_gif='seed 33 output_animation.gif',
    fps=25,
    loop=0  # 0 = infinite loop
)
