import type { StorybookConfig } from '@storybook/react-vite';
import { mergeConfig } from 'vite';
import path from 'node:path';

const config: StorybookConfig = {
  stories: ['../src/**/*.stories.@(ts|tsx)'],
  addons: ['@storybook/addon-essentials', '@storybook/addon-interactions'],
  framework: {
    name: '@storybook/react-vite',
    options: {}
  },
  viteFinal: async (cfg) =>
    mergeConfig(cfg, {
      resolve: {
        alias: {
          '@': path.resolve(__dirname, '../src')
        }
      }
    })
};

export default config;
