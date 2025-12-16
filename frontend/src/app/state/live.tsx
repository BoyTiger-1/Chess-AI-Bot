import React, { createContext, PropsWithChildren, useContext, useEffect, useMemo, useRef, useState } from 'react';

export type LiveConnectionStatus = 'connecting' | 'connected' | 'disconnected';

export interface LiveMessage {
  type: string;
  payload: unknown;
  ts: string;
}

interface LiveContextValue {
  status: LiveConnectionStatus;
  lastMessage: LiveMessage | null;
}

const LiveContext = createContext<LiveContextValue | null>(null);

function getWsUrl() {
  const fromEnv = import.meta.env.VITE_WS_URL as string | undefined;
  return fromEnv || 'ws://localhost:8000/ws';
}

export function LiveProvider({ children }: PropsWithChildren) {
  const [status, setStatus] = useState<LiveConnectionStatus>('connecting');
  const [lastMessage, setLastMessage] = useState<LiveMessage | null>(null);
  const wsRef = useRef<WebSocket | null>(null);

  useEffect(() => {
    const url = getWsUrl();

    try {
      const ws = new WebSocket(url);
      wsRef.current = ws;

      setStatus('connecting');

      ws.addEventListener('open', () => setStatus('connected'));
      ws.addEventListener('close', () => setStatus('disconnected'));
      ws.addEventListener('error', () => setStatus('disconnected'));
      ws.addEventListener('message', (evt) => {
        const ts = new Date().toISOString();
        try {
          const parsed = JSON.parse(String(evt.data)) as { type?: string; payload?: unknown };
          setLastMessage({ type: parsed.type ?? 'message', payload: parsed.payload ?? parsed, ts });
        } catch {
          setLastMessage({ type: 'message', payload: String(evt.data), ts });
        }
      });

      return () => {
        ws.close();
      };
    } catch {
      setStatus('disconnected');
    }
  }, []);

  const value = useMemo<LiveContextValue>(() => ({ status, lastMessage }), [status, lastMessage]);

  return <LiveContext.Provider value={value}>{children}</LiveContext.Provider>;
}

export function useLive() {
  const ctx = useContext(LiveContext);
  if (!ctx) throw new Error('useLive must be used within LiveProvider');
  return ctx;
}
