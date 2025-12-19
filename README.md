# Physical AI & Humanoid Robotics: Embodied Intelligence in the Physical World

This repository contains the complete documentation for "Physical AI & Humanoid Robotics: Embodied Intelligence in the Physical World", a comprehensive guide to embodied artificial intelligence and humanoid robotics.

## Project Structure

```
├── docs/                   # All book chapters in Markdown format
│   ├── introduction.md
│   ├── physical-ai-foundations.md
│   ├── embodied-intelligence.md
│   ├── ros2-fundamentals.md
│   ├── urdf-humanoids.md
│   ├── gazebo-simulation.md
│   ├── unity-visualization.md
│   ├── nvidia-isaac-sim.md
│   ├── isaac-ros-vslam.md
│   ├── vision-language-action.md
│   ├── conversational-robotics.md
│   └── capstone-autonomous-humanoid.md
├── src/                    # Docusaurus source files
│   └── css/
│       └── custom.css
├── static/                 # Static assets
│   └── img/
├── package.json           # Dependencies and scripts
├── docusaurus.config.js   # Docusaurus configuration
├── sidebars.js           # Sidebar navigation
└── README.md             # This file
```

## Getting Started

1. Make sure you have Node.js 18+ installed
2. Install dependencies: `npm install`
3. Start development server: `npm start`
4. Open your browser to http://localhost:3000

## Building for Production

To build the static site:
```bash
npm run build
```

The resulting static files will be in the `build/` directory and can be deployed to any static hosting service.

## Contributing

This book was created using the Docusaurus framework to ensure accessibility and maintainability. To contribute:

1. Fork the repository
2. Create your feature branch
3. Make your changes
4. Submit a pull request

## License

This book is released under a Creative Commons license. See the individual chapters for specific licensing information.