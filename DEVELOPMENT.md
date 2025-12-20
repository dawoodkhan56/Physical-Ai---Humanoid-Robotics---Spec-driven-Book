# Development Setup

This document explains how to set up and work with the Physical AI & Humanoid Robotics documentation project using Node.js and TypeScript.

## Prerequisites

- Node.js 18.17.0 or higher (recommend using nvm to manage Node versions)
- npm or yarn package manager
- Git for version control

## Setup Instructions

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Physical-AI-Humanoid/physical-ai-humanoid-book.git
   cd physical-ai-humanoid-book
   ```

2. **Install dependencies:**
   ```bash
   npm install
   ```
   
   Or if using Yarn:
   ```bash
   yarn install
   ```

3. **Verify Node.js version:**
   Make sure you're using the correct Node.js version (specified in `.nvmrc`):
   ```bash
   # If using nvm:
   nvm use
   node --version
   ```

## Available Scripts

Once you've installed the dependencies, you can run the following scripts:

- `npm start` - Starts the development server at http://localhost:3000
- `npm run build` - Builds the static site for production
- `npm run serve` - Serves the built site locally for testing
- `npm run clear` - Clears the Docusaurus cache
- `npm run deploy` - Deploys to GitHub Pages (if configured)
- `npm run write-translations` - Extracts translation messages
- `npm run write-heading-ids` - Generates heading IDs for anchor links

## Development Workflow

1. **Start development server:**
   ```bash
   npm start
   ```
   This will start a hot-reloading development server that watches for changes to your files.

2. **Edit documentation files:**
   Edit the Markdown files in the `docs/` directory to make changes to the content.

3. **Preview changes:**
   Changes will automatically be reflected in your browser as you edit.

4. **Build for production:**
   ```bash
   npm run build
   ```
   This creates an optimized build in the `build/` directory.

## TypeScript Configuration

This project uses TypeScript for type safety. The configuration is in `tsconfig.json`:
- Target: ES2017 for broad compatibility
- JSX: Preserved for Docusaurus compatibility
- Strict mode: Disabled to maintain compatibility with Docusaurus patterns
- Modules: ESNext for modern JavaScript support

## Testing

Run the test suite:
```bash
npm test
```

To run tests in watch mode:
```bash
npm test -- --watch
```

## Deployment

The site is configured for deployment to multiple platforms:

### GitHub Pages
1. Update the `organizationName` and `projectName` in `docusaurus.config.js`
2. Run: `npm run deploy`

### Netlify
1. Connect your GitHub repository to Netlify
2. Set build command to: `npm run build`
3. Set publish directory to: `build`

### Vercel
1. Connect your GitHub repository to Vercel
2. Set build command to: `npm run build`
3. Set output directory to: `build`

## Troubleshooting

- **npm install fails**: Clear npm cache with `npm cache clean --force` and try again
- **Port 3000 is in use**: Use `npm start -- --port 3001` to use a different port
- **Build fails**: Run `npm run clear` to clear cache and try building again

## Updating Dependencies

To update Docusaurus to the latest version:
```bash
npm update @docusaurus/core @docusaurus/preset-classic
```

Then check the [Docusaurus migration guide](https://docusaurus.io/docs/migration) and run:
```bash
npm run swizzle @docusaurus/theme-classic -- --danger
```

## Directory Structure

```
├── docs/                   # Documentation Markdown files
├── src/                    # Custom React components and CSS
│   ├── css/
│   │   └── custom.css
│   └── tests/
│       └── setupTests.js
├── static/                 # Static assets (images, PDFs, etc.)
│   └── img/
├── package.json           # Dependencies and scripts
├── docusaurus.config.js   # Website configuration
├── sidebars.js            # Sidebar navigation configuration
├── tsconfig.json          # TypeScript configuration
├── jest.config.js         # Testing configuration
└── README.md             # Project overview
```