import type { Meta, StoryObj } from '@storybook/react';
import React from 'react';
import { Typography } from '@mui/material';
import { WidgetCard } from '@/components/WidgetCard';

const meta: Meta<typeof WidgetCard> = {
  title: 'Design System/WidgetCard',
  component: WidgetCard
};

export default meta;

type Story = StoryObj<typeof WidgetCard>;

export const Default: Story = {
  args: {
    title: 'Widget title',
    subtitle: 'Helpful context'
  },
  render: (args) => (
    <WidgetCard {...args}>
      <Typography variant="body2" color="text.secondary">
        Widget content goes here.
      </Typography>
    </WidgetCard>
  )
};
