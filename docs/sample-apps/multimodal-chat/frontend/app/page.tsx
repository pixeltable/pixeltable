// app/page.tsx
import { Providers } from './providers';
import ChatInterface from '@/components/ChatInterface';

export default function Home() {
  return (
    <Providers>
      <main className="min-h-screen">
        <ChatInterface />
      </main>
    </Providers>
  );
}