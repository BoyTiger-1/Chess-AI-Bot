import React from 'react';

export function useSimulatedStream<T>(
  initial: T,
  reducer: (prev: T) => T,
  opts?: { intervalMs?: number; enabled?: boolean }
) {
  const { intervalMs = 2000, enabled = true } = opts ?? {};
  const [value, setValue] = React.useState<T>(initial);

  React.useEffect(() => {
    if (!enabled) return;
    const id = window.setInterval(() => setValue((v) => reducer(v)), intervalMs);
    return () => window.clearInterval(id);
  }, [enabled, intervalMs, reducer]);

  return value;
}
