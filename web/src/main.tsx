import '@fontsource-variable/literata';
import '@fontsource-variable/hanken-grotesk';
import { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';
import App from './App';
import './styles.css';

const container = document.getElementById('root');
if (!container) {
  throw new Error('No #root element; index.html and main.tsx disagree.');
}

createRoot(container).render(
  <StrictMode>
    <App />
  </StrictMode>,
);
