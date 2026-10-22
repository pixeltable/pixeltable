import { createRoot } from 'react-dom/client';
import { EmbeddedDashboard } from './EmbeddedDashboard';
import '@/index.css';

const container = document.getElementById('root');
if (!container) throw new Error('Dashboard root is missing');
createRoot(container).render(<EmbeddedDashboard />);
