import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch
import numpy as np

def create_physical_ai_architecture_diagram():
    fig, ax = plt.subplots(figsize=(16, 9))
    fig.patch.set_facecolor('white')
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 9)
    ax.axis('off')

    # Color scheme
    soft_blue = '#4D96FF'
    soft_purple = '#8A4FFF'
    light_gray = '#F0F0F0'
    medium_gray = '#D0D0D0'
    dark_gray = '#A0A0A0'
    accent_green = '#6BCB77'
    accent_orange = '#FFA07A'

    # Draw layers
    layer_height = 1.2
    layer_spacing = 0.4
    
    # Perception Layer
    perception_y = 8.2
    ax.add_patch(FancyBboxPatch((0.5, perception_y), 15, layer_height, 
                               boxstyle="round,pad=0.05", 
                               facecolor=soft_blue, alpha=0.2, 
                               edgecolor=soft_blue, linewidth=2))
    ax.text(8, perception_y + layer_height/2, 'Perception Layer', 
            fontsize=16, fontweight='bold', ha='center', va='center', color=soft_blue)

    # Perception components
    sensors = ['Cameras', 'LiDAR', 'Microphones', 'Tactile Sensors']
    sensor_colors = [accent_green, accent_orange, '#9B59B6', '#3498DB']
    for i, (sensor, color) in enumerate(zip(sensors, sensor_colors)):
        x_pos = 2 + i * 3.5
        ax.add_patch(FancyBboxPatch((x_pos, perception_y + 0.2), 2.5, 0.8, 
                                   boxstyle="round,pad=0.02", 
                                   facecolor=color, alpha=0.7, 
                                   edgecolor='white', linewidth=1))
        ax.text(x_pos + 1.25, perception_y + 0.6, sensor, 
                fontsize=12, ha='center', va='center', color='white', fontweight='bold')

    # Sensor Fusion Layer
    fusion_y = 6.6
    ax.add_patch(FancyBboxPatch((3, fusion_y), 10, layer_height, 
                               boxstyle="round,pad=0.05", 
                               facecolor=medium_gray, alpha=0.2, 
                               edgecolor=dark_gray, linewidth=2))
    ax.text(8, fusion_y + layer_height/2, 'Sensor Fusion & State Estimation', 
            fontsize=16, fontweight='bold', ha='center', va='center', color=dark_gray)

    fusion_items = ['Multi-sensor fusion', 'World model', 'Localization']
    for i, item in enumerate(fusion_items):
        x_pos = 4.5 + i * 3
        ax.add_patch(FancyBboxPatch((x_pos, fusion_y + 0.2), 2.5, 0.8, 
                                   boxstyle="round,pad=0.02", 
                                   facecolor=dark_gray, alpha=0.7, 
                                   edgecolor='white', linewidth=1))
        ax.text(x_pos + 1.25, fusion_y + 0.6, item, 
                fontsize=12, ha='center', va='center', color='white', fontweight='bold')

    # Arrow from perception to fusion
    for i in range(4):
        x_start = 2.5 + i * 3.5
        arrow = patches.FancyArrowPatch((x_start+1.25, perception_y), 
                                       (7.5+i*0.3, fusion_y + layer_height), 
                                       arrowstyle='->', mutation_scale=20, 
                                       color=dark_gray, alpha=0.7, lw=1.5)
        ax.add_patch(arrow)

    # AI Cognition Layer
    cognition_y = 5.0
    ax.add_patch(FancyBboxPatch((0.5, cognition_y), 15, layer_height, 
                               boxstyle="round,pad=0.05", 
                               facecolor=soft_purple, alpha=0.2, 
                               edgecolor=soft_purple, linewidth=2))
    ax.text(8, cognition_y + layer_height/2, 'AI Cognition Layer', 
            fontsize=16, fontweight='bold', ha='center', va='center', color=soft_purple)

    cognition_items = ['Foundation Models', 'Large Language Models (LLMs)', 'Vision-Language Models', 'Reasoning & Planning']
    cognition_colors = ['#9B59B6', '#8E44AD', '#9B59B6', '#8E44AD']
    for i, (item, color) in enumerate(zip(cognition_items, cognition_colors)):
        x_pos = 1 + i * 4.5
        width = 3.5 if i == 1 else 3  # Make LLMs slightly wider
        ax.add_patch(FancyBboxPatch((x_pos, cognition_y + 0.2), width, 0.8, 
                                   boxstyle="round,pad=0.02", 
                                   facecolor=color, alpha=0.7, 
                                   edgecolor='white', linewidth=1))
        ax.text(x_pos + width/2, cognition_y + 0.6, item, 
                fontsize=11, ha='center', va='center', color='white', fontweight='bold', wrap=True)

    # Arrow from fusion to cognition
    arrow = patches.FancyArrowPatch((8, fusion_y), 
                                   (8, cognition_y + layer_height), 
                                   arrowstyle='->', mutation_scale=25, 
                                   color=soft_purple, alpha=0.8, lw=2)
    ax.add_patch(arrow)

    # Decision & Control Layer
    control_y = 3.4
    ax.add_patch(FancyBboxPatch((2, control_y), 12, layer_height, 
                               boxstyle="round,pad=0.05", 
                               facecolor=accent_green, alpha=0.2, 
                               edgecolor=accent_green, linewidth=2))
    ax.text(8, control_y + layer_height/2, 'Decision & Control Layer', 
            fontsize=16, fontweight='bold', ha='center', va='center', color=accent_green)

    control_items = ['Task Planning', 'Motion Planning', 'Reinforcement Learning']
    control_colors = ['#27AE60', '#219653', '#2ECC71']
    for i, (item, color) in enumerate(zip(control_items, control_colors)):
        x_pos = 3.5 + i * 4
        ax.add_patch(FancyBboxPatch((x_pos, control_y + 0.2), 3, 0.8, 
                                   boxstyle="round,pad=0.02", 
                                   facecolor=color, alpha=0.8, 
                                   edgecolor='white', linewidth=1))
        ax.text(x_pos + 1.5, control_y + 0.6, item, 
                fontsize=12, ha='center', va='center', color='white', fontweight='bold')

    # Arrow from cognition to control
    arrow = patches.FancyArrowPatch((8, cognition_y), 
                                   (8, control_y + layer_height), 
                                   arrowstyle='->', mutation_scale=25, 
                                   color=accent_green, alpha=0.8, lw=2)
    ax.add_patch(arrow)

    # Actuation Layer
    actuation_y = 1.8
    ax.add_patch(FancyBboxPatch((0.5, actuation_y), 15, layer_height, 
                               boxstyle="round,pad=0.05", 
                               facecolor=accent_orange, alpha=0.2, 
                               edgecolor=accent_orange, linewidth=2))
    ax.text(8, actuation_y + layer_height/2, 'Actuation Layer', 
            fontsize=16, fontweight='bold', ha='center', va='center', color=accent_orange)

    actuation_items = ['Motors', 'Servos', 'Robotic Arms', 'Humanoid Legs']
    actuation_colors = ['#F39C12', '#E67E22', '#F1C40F', '#E74C3C']
    for i, (item, color) in enumerate(zip(actuation_items, actuation_colors)):
        x_pos = 2 + i * 3.5
        ax.add_patch(FancyBboxPatch((x_pos, actuation_y + 0.2), 2.5, 0.8, 
                                   boxstyle="round,pad=0.02", 
                                   facecolor=color, alpha=0.8, 
                                   edgecolor='white', linewidth=1))
        ax.text(x_pos + 1.25, actuation_y + 0.6, item, 
                fontsize=12, ha='center', va='center', color='white', fontweight='bold')

    # Arrow from control to actuation
    arrow = patches.FancyArrowPatch((8, control_y), 
                                   (8, actuation_y + layer_height), 
                                   arrowstyle='->', mutation_scale=25, 
                                   color=accent_orange, alpha=0.8, lw=2)
    ax.add_patch(arrow)

    # Feedback loop arrow
    feedback_arrow = patches.FancyArrowPatch((14, actuation_y + layer_height/2), 
                                            (14, perception_y + layer_height/2), 
                                            arrowstyle='->', mutation_scale=20, 
                                            color='gray', alpha=0.7, lw=2,
                                            connectionstyle="arc3,rad=-0.3")
    ax.add_patch(feedback_arrow)
    ax.text(14.5, 5.0, 'Feedback Loop\n(Actions → Environment → Perception)', 
            fontsize=10, ha='left', va='center', color='gray', style='italic')

    # Title
    ax.text(8, 0.3, 'Physical AI & Humanoid Robotics Architecture', 
            fontsize=20, fontweight='bold', ha='center', va='center', 
            color='#2C3E50')

    # Add legend
    legend_elements = [
        patches.Patch(color=soft_blue, alpha=0.3, label='Perception'),
        patches.Patch(color=medium_gray, alpha=0.3, label='Fusion'),
        patches.Patch(color=soft_purple, alpha=0.3, label='Cognition'),
        patches.Patch(color=accent_green, alpha=0.3, label='Control'),
        patches.Patch(color=accent_orange, alpha=0.3, label='Actuation')
    ]
    ax.legend(handles=legend_elements, loc='lower right', bbox_to_anchor=(0.98, 0.02))

    plt.tight_layout()
    plt.savefig('physical-ai-architecture.jpg', dpi=150, bbox_inches='tight', facecolor='white')
    plt.show()

# Run the function to create the diagram
create_physical_ai_architecture_diagram()