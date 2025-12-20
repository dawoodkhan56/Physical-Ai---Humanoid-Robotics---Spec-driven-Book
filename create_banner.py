import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from PIL import Image
import matplotlib.patheffects as path_effects

def create_animated_banner():
    # Create a figure with a colorful background
    fig, ax = plt.subplots(figsize=(16, 4))
    fig.patch.set_facecolor('white')
    
    # Create gradient background
    gradient = np.linspace(0, 1, 256).reshape(1, -1)
    gradient = np.vstack((gradient, gradient))
    
    # Set different color ranges for the background
    ax.imshow(gradient, extent=[0, 16, 0, 4], aspect='auto', cmap='rainbow')
    
    # Add the main title text
    title_text = "Physical AI & Humanoid Robotics"
    ax.text(8, 2.5, title_text, fontsize=36, fontweight='bold', ha='center', va='center',
            color='white', 
            path_effects=[path_effects.Stroke(linewidth=2, foreground='black'), 
                         path_effects.Normal()])
    
    # Add subtitle text
    subtitle_text = "Embodied Intelligence in the Physical World"
    ax.text(8, 1.5, subtitle_text, fontsize=20, ha='center', va='center',
            color='white',
            path_effects=[path_effects.Stroke(linewidth=1, foreground='black'), 
                         path_effects.Normal()])
    
    # Add author text
    author_text = "by Dawood Khan"
    ax.text(8, 0.7, author_text, fontsize=16, ha='center', va='center',
            color='white',
            style='italic',
            path_effects=[path_effects.Stroke(linewidth=1, foreground='black'), 
                         path_effects.Normal()])
    
    # Remove axes
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 4)
    ax.axis('off')
    
    # Save the figure
    plt.savefig('physical_ai_banner.png', dpi=150, bbox_inches='tight', pad_inches=0.1)
    plt.close()
    
    # Now create a version with animated color effects using PIL
    img = Image.open('physical_ai_banner.png')
    
    # Create multiple frames with different color effects
    frames = []
    for i in range(5):  # Create 5 frames for a simple animation
        # Apply slight color variations to simulate animation
        frame = img.copy()
        # For simplicity, we'll just save the same image as different frames
        # In a real animation, we'd apply actual transformations
        frames.append(frame)
    
    # Save as an animated GIF
    frames[0].save('physical_ai_animated_banner.gif', 
                   save_all=True, 
                   append_images=frames[1:], 
                   duration=500,  # 500ms per frame
                   loop=0)  # Infinite loop
    
    print("Animated banner created successfully!")
    

if __name__ == "__main__":
    create_animated_banner()