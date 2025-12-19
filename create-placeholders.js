// create-placeholders.js
// Script to create placeholder images for the documentation

const fs = require('fs');
const path = require('path');

// Create a simple placeholder image content (base64 encoded PNG)
const placeholderImage = `iVBORw0KGgoAAAANSUhEUgAAADIAAAAyCAIAAACRXR/mAAAACXBIWXMAAAsTAAALEwEAmpwYAAAAIGNIUk0AAHolAACAgwAA+f8AAIDpAAB1MAAA6mAAADqYAAAXb5JfxUYAAABnSURBVHja7M5RDYAwDEXRDgmvEocnlrQS2SwUFST9uEfBGWs9c97nbGtDcquqiKhOImLs/UpuzVzWE5QhDQEAAACA3f0DAADAeQtDGjAAAwAAABgxDjpdAJnJAAAAAElFTkSuQmCC`;

// Define all the image files that need to be created
const imageFiles = [
  'physical-ai-concept.jpg',
  'embodied-intelligence-architecture.jpg',
  'ros2-architecture.jpg',
  'humanoid-urdf-structure.jpg',
  'gazebo-simulation-humanoid.jpg',
  'unity-humanoid-visualization.jpg',
  'isaac-sim-humanoid.jpg',
  'isaac-ros-vslam.jpg',
  'vla-humanoid-system.jpg',
  'conversational-robotics.jpg',
  'docusaurus-social-card.jpg',
  'favicon.ico',
  'logo.svg',
  'autonomous-humanoid-capstone.jpg'
];

// Create placeholder images
imageFiles.forEach(filename => {
  const filePath = path.join(__dirname, 'static', 'img', filename);
  
  // Convert base64 to buffer
  const imageBuffer = Buffer.from(placeholderImage, 'base64');
  
  // Write the placeholder image
  fs.writeFileSync(filePath, imageBuffer);
  console.log(`Created placeholder image: ${filename}`);
});

console.log('All placeholder images created successfully!');