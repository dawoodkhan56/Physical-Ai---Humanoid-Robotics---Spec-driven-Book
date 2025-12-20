module.exports = {
  preset: '@docusaurus/core/jest',
  testEnvironment: 'jsdom',
  transformIgnorePatterns: [
    '/node_modules/(?!(rehype-katex|remark-math|@mapbox/rehype-prism)/)'
  ],
  setupFilesAfterEnv: ['<rootDir>/src/tests/setupTests.js'],
  moduleNameMapping: {
    '^@/(.*)$': '<rootDir>/src/$1',
  },
  testPathIgnorePatterns: [
    '<rootDir>/node_modules/',
    '<rootDir>/build/'
  ]
};