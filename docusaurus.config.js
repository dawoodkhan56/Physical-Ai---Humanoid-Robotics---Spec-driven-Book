// @ts-check
// `@type` JSDoc annotations allow IDEs and type-checking tools to autocomplete
/** @type {import('@docusaurus/types').Config} */
const config = {
  title: 'Physical AI & Humanoid Robotics: Embodied Intelligence in the Physical World',
  tagline: 'A comprehensive guide to embodied artificial intelligence and humanoid robotics',
  favicon: 'img/favicon.ico',

  // Set the production url of your site here
  url: 'https://physical-ai-humanoid-guide.netlify.app/', // Placeholder URL
  // Set the /<baseUrl>/ pathname under which your site is served
  // For GitHub pages deployment, it is often '/<projectName>/'
  baseUrl: '/',

  // GitHub pages deployment config.
  organizationName: 'Physical-AI-Humanoid', // Usually your GitHub org/user name.
  projectName: 'physical-ai-humanoid-book', // Usually your repo name.

  onBrokenLinks: 'throw',
  onBrokenMarkdownLinks: 'warn',

  // Even if you don't use internationalization, you can use this field to set
  // useful metadata like html lang. For example, if your site is Chinese, you
  // may want to replace "en" with "zh-Hans".
  i18n: {
    defaultLocale: 'en',
    locales: ['en'],
  },

  presets: [
    [
      'classic',
      /** @type {import('@docusaurus/preset-classic').Options} */
      ({
        docs: {
          sidebarPath: './sidebars.js',
          // Please change this to your repo.
          // Remove this to remove the "edit this page" links.
          editUrl:
            'https://github.com/Physical-AI-Humanoid/physical-ai-humanoid-book/tree/main/',
        },
        blog: {
          showReadingTime: true,
          // Please change this to your repo.
          // Remove this to remove the "edit this page" links.
          editUrl:
            'https://github.com/Physical-AI-Humanoid/physical-ai-humanoid-book/tree/main/',
        },
        theme: {
          customCss: './src/css/custom.css',
        },
      }),
    ],
  ],

  customFields: {
    onBrokenMarkdownImages: 'warn',  // Handle broken images
  },
  themeConfig:
    /** @type {import('@docusaurus/preset-classic').ThemeConfig} */
    ({
      // Replace with your project's social card
      image: 'img/docusaurus-social-card.jpg',
      prism: {
        theme: require('prism-react-renderer').themes.github,
        darkTheme: require('prism-react-renderer').themes.dracula,
        additionalLanguages: ['python', 'bash', 'docker', 'json', 'yaml'],
      },
      navbar: {
        title: 'Physical AI & Humanoid Robotics',
        logo: {
          alt: 'Physical AI & Humanoid Robotics Logo',
          src: 'img/logo.png',  // Use our colorful banner as logo
        },
        items: [
          {
            type: 'docSidebar',
            sidebarId: 'tutorialSidebar',
            position: 'left',
            label: 'Book Chapters',
          },
          {
            href: 'https://github.com/Physical-AI-Humanoid/physical-ai-humanoid-book',
            label: 'GitHub',
            position: 'right',
          },
        ],
      },
      footer: {
        style: 'dark',
        links: [
          {
            title: 'Chapters',
            items: [
              {
                label: 'Introduction',
                to: '/docs/introduction',
              },
              {
                label: 'Physical AI Foundations',
                to: '/docs/physical-ai-foundations',
              },
              {
                label: 'Embodied Intelligence',
                to: '/docs/embodied-intelligence',
              },
              {
                label: 'Capstone: Autonomous Humanoid',
                to: '/docs/capstone-autonomous-humanoid',
              },
            ],
          },
          {
            title: 'Community',
            items: [
              {
                label: 'Discussions',
                href: 'https://github.com/Physical-AI-Humanoid/physical-ai-humanoid-book/discussions',
              },
              {
                label: 'Issues',
                href: 'https://github.com/Physical-AI-Humanoid/physical-ai-humanoid-book/issues',
              },
            ],
          },
          {
            title: 'More',
            items: [
              {
                label: 'GitHub',
                href: 'https://github.com/Physical-AI-Humanoid/physical-ai-humanoid-book',
              },
            ],
          },
        ],
        copyright: `Copyright © ${new Date().getFullYear()} Physical AI & Humanoid Robotics. Built with Docusaurus.`,
      },
      prism: {
        theme: require('prism-react-renderer').themes.github,
        darkTheme: require('prism-react-renderer').themes.dracula,
        additionalLanguages: ['python', 'bash', 'docker', 'json', 'yaml'],
      },
    }),
};

module.exports = config;