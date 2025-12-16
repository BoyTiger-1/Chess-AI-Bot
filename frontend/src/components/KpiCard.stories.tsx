import type { Meta, StoryObj } from '@storybook/react';
import React from 'react';
import { KpiCard } from '@/components/KpiCard';

const meta: Meta<typeof KpiCard> = {
  title: 'Design System/KpiCard',
  component: KpiCard
};

export default meta;

type Story = StoryObj<typeof KpiCard>;

export const Positive: Story = {
  args: {
    title: 'MRR',
    value: '$1.2M',
    delta: { label: '+4.5% WoW', tone: 'positive' },
    footnote: 'Updated a few seconds ago'
  }
};
