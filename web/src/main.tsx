import { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';
import App from './App';

const container = document.getElementById('root');
if (!container) {
  throw new Error('No #root element; index.html and main.tsx disagree.');
}

createRoot(container).render(
  <StrictMode>
    <App />
  </StrictMode>,
);
