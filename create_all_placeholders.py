import os
from PIL import Image, ImageDraw, ImageFont

def create_placeholder_image(filename, width=800, height=450, text=None):
    """Create a placeholder image with text indicating its purpose"""
    img = Image.new('RGB', (width, height), color=(77, 150, 255))  # Soft blue background
    draw = ImageDraw.Draw(img)
    
    # Draw a border
    draw.rectangle([10, 10, width-10, height-10], outline=(255, 255, 255), width=3)
    
    # Add the filename as text in the center
    if text is None:
        text = os.path.basename(filename).replace('.jpg', '').replace('.png', '').replace('_', ' ').title()
        text = text.replace('-', ' ')
    
    # Try to use a default font, fallback to default if not available
    try:
        font = ImageFont.truetype("arial.ttf", 36)
    except:
        try:
            font = ImageFont.truetype("DejaVuSans.ttf", 36)
        except:
            font = ImageFont.load_default()
    
    # Calculate text size and position
    try:
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
    except:
        # Fallback if textbbox isn't available
        text_width = len(text) * 20
        text_height = 40
    
    text_x = (width - text_width) // 2
    text_y = (height - text_height) // 2
    
    # Draw the text
    draw.text((text_x, text_y), text, fill=(255, 255, 255), font=font)
    
    # Save the image - only for image formats that Pillow supports
    if filename.endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff')):
        img.save(filename)
        print(f"Created placeholder image: {filename}")
    else:
        print(f"Skipped unsupported format: {filename}")

def create_all_placeholders():
    """Create all required placeholder images"""
    # Create the directory if it doesn't exist
    os.makedirs("static/img", exist_ok=True)
    
    # List of all required images
    images = [
        "physical-ai-concept.jpg",
        "embodied-intelligence-architecture.jpg",
        "ros2-architecture.jpg",
        "humanoid-urdf-structure.jpg",
        "gazebo-simulation-humanoid.jpg",
        "unity-humanoid-visualization.jpg",
        "isaac-sim-humanoid.jpg",
        "isaac-ros-vslam.jpg",
        "vla-humanoid-system.jpg",
        "conversational-robotics.jpg",
        "autonomous-humanoid-capstone.jpg",
        "docusaurus-social-card.jpg",
    ]
    
    for img in images:
        filepath = f"static/img/{img}"
        create_placeholder_image(filepath)
    
    # Create special files that aren't images
    # For logo.svg, create a simple text file that describes the logo concept
    with open("static/img/logo.svg", "w") as f:
        f.write("""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100">
  <rect width="100" height="100" fill="#4D96FF"/>
  <circle cx="50" cy="40" r="15" fill="white"/>
  <rect x="35" y="60" width="30" height="20" rx="5" fill="white"/>
  <text x="50" y="90" text-anchor="middle" fill="white" font-family="Arial" font-size="12">
    PAI Logo
  </text>
</svg>""")
        print("Created placeholder SVG: static/img/logo.svg")
    
    # For favicon.ico, we'll create a simple version as PNG (conceptual)
    favicon_img = Image.new('RGB', (32, 32), color=(77, 150, 255))
    favicon_draw = ImageDraw.Draw(favicon_img)
    favicon_draw.rectangle([2, 2, 30, 30], outline=(255, 255, 255), width=2)
    favicon_draw.text((8, 12), "PAI", fill=(255, 255, 255))
    favicon_img.save("static/img/favicon.png")  # Save as PNG since PIL can't handle ICO
    print("Created placeholder favicon: static/img/favicon.png")

# Run the function to create all placeholders
create_all_placeholders()