# Quickstart Guide: Physical AI & Humanoid Robotics

This guide helps you get started with the Physical AI & Humanoid Robotics educational content.

## Prerequisites

Before beginning this curriculum, you should have:
- Basic understanding of programming concepts (Python preferred)
- Familiarity with command-line interfaces
- Understanding of fundamental mathematics (linear algebra, calculus)
- Basic knowledge of robotics concepts (optional but helpful)

## Environment Setup

### 1. Install Node.js and npm
You need Node.js 18+ and npm to run the Docusaurus documentation site:

```bash
# Check your Node.js version (should be 18+)
node --version

# If Node.js is not installed or version is too low:
# Download from https://nodejs.org/
```

### 2. Install Dependencies
```bash
# Navigate to the project root
cd path/to/physical-ai-humanoid-book

# Install project dependencies
npm install
```

### 3. Start Local Development Server
```bash
# Run the development server
npm start

# This command starts a local development server and opens the documentation in your browser
# Most changes are reflected live without restarting the server
```

## Curriculum Structure

The curriculum is organized into 5 core modules across 13 weeks:

### Module 1: The Robotic Nervous System (Weeks 3-4)
- ROS 2 fundamentals and architecture
- Communication patterns and middleware
- Practical exercises with ROS 2 nodes and messages

### Module 2: The Digital Twin (Weeks 5-6)
- Gazebo simulation environment
- Unity simulation environment
- Simulation workflows and testing

### Module 3: The AI-Robot Brain (Weeks 7-9)
- NVIDIA Isaac platform overview
- Perception systems
- Decision-making algorithms

### Module 4: Vision-Language Action (Weeks 10-11)
- VLA architecture
- Multimodal learning
- Perception-action loops

### Module 5: Capstone - The Autonomous Humanoid (Weeks 12-13)
- Integration challenges
- Complete system design
- Final project

## Navigation Tips

- Use the sidebar to navigate between modules and lessons
- Each module includes practical exercises and knowledge checks
- Cross-links connect related concepts throughout the curriculum
- Use the search function to find specific topics quickly

## Practical Exercises

Each module includes hands-on exercises that reinforce the concepts covered. These exercises often involve:
- Simulating robotic behaviors in Gazebo or Unity
- Implementing algorithms using ROS 2 and NVIDIA Isaac
- Analyzing system performance and debugging issues

## Getting Help

- Check the troubleshooting section for common issues
- Review the glossary for definitions of technical terms
- Consult the further reading resources for additional depth
- Report issues with the content through the project's issue tracker

## Deployment

The documentation can be deployed to various platforms:

### GitHub Pages
```bash
# Build the static site
npm run build

# Deploy to GitHub Pages
npm run deploy
```

### Vercel or Netlify
The `build` directory can be deployed directly to Vercel or Netlify following their standard procedures.