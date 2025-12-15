import React from 'react';
import { Box } from '@mui/material';
import { Responsive, WidthProvider, type Layouts, type Layout } from 'react-grid-layout';
import { readJson, writeJson } from '@/utils/storage';

const ResponsiveGridLayout = WidthProvider(Responsive);

export interface DashboardWidget {
  id: string;
  minW?: number;
  minH?: number;
  default: { w: number; h: number; x: number; y: number };
  render: () => React.ReactNode;
}

function defaultLayouts(widgets: DashboardWidget[]): Layouts {
  const lg: Layout[] = widgets.map((w, idx) => ({
    i: w.id,
    x: w.default.x,
    y: w.default.y ?? idx,
    w: w.default.w,
    h: w.default.h,
    minW: w.minW,
    minH: w.minH
  }));

  return { lg };
}

export function DashboardGrid({
  widgets,
  storageKey
}: {
  widgets: DashboardWidget[];
  storageKey: string;
}) {
  const [layouts, setLayouts] = React.useState<Layouts>(() => {
    const fallback = defaultLayouts(widgets);
    return readJson<Layouts>(storageKey, fallback);
  });

  const widgetKey = widgets.map((w) => w.id).join('|');

  React.useEffect(() => {
    const fallback = defaultLayouts(widgets);
    setLayouts((prev) => {
      const next = { ...fallback, ...prev };
      writeJson(storageKey, next);
      return next;
    });
  }, [storageKey, widgetKey]);

  return (
    <Box
      sx={{
        '.react-grid-item': {
          transition: 'transform 180ms ease, width 180ms ease, height 180ms ease'
        },
        '.react-grid-item.react-grid-placeholder': {
          background: 'rgba(110,231,255,0.18)',
          borderRadius: 2
        }
      }}
    >
      <ResponsiveGridLayout
        className="layout"
        layouts={layouts}
        onLayoutChange={(_, all) => {
          setLayouts(all);
          writeJson(storageKey, all);
        }}
        draggableHandle=".drag-handle"
        breakpoints={{ lg: 1200, md: 900, sm: 600, xs: 0 }}
        cols={{ lg: 12, md: 12, sm: 6, xs: 2 }}
        rowHeight={88}
        margin={[16, 16]}
        containerPadding={[0, 0]}
        compactType="vertical"
        isBounded
      >
        {widgets.map((w) => (
          <Box key={w.id} data-grid={{ i: w.id }}>
            {w.render()}
          </Box>
        ))}
      </ResponsiveGridLayout>
    </Box>
  );
}
