# DeepForensics — Frontend White Screen Diagnosis & Audit Report

## 1. Current Frontend Architecture
- **Framework**: React 18 + Vite + TypeScript (SPA)
- **Styling**: Custom CSS Design System (`src/index.css`)
- **State & Data Fetching**: Async fetch wrappers (`src/api.ts`)
- **Routing**: `react-router-dom` (v7)

## 2. Entry Point
- `frontend/index.html` → `<script type="module" src="/src/main.tsx"></script>`
- Root DOM element: `<div id="root"></div>`

## 3. Root Component
- `src/main.tsx` mounts `<App />` into `createRoot(document.getElementById('root')!)`.

## 4. Routing System
- `src/App.tsx` using `BrowserRouter` with routes:
  - `/` -> Landing page
  - `/image` -> Image Analysis dashboard
  - `/video` -> Video Analysis dashboard
  - `/history` -> Detection History
  - `/models` -> Active Model Registry
  - `/evaluation` -> Performance & Metrics Dashboard
  - `/settings` -> Configuration & Settings
  - `/methodology` -> System Architecture & Methodology

## 5. API Layer
- `src/api.ts` connecting to `http://localhost:8000/api`

## 6. Environment Variables
- `VITE_API_BASE_URL` (defaults safely to `http://localhost:8000/api`)

## 7. Dependencies Audit & Root Causes of Blank Screen
1. **Missing `"jsx": "react-jsx"` in `tsconfig.json`**:
   - TypeScript compiler was missing the `--jsx` flag, causing `npm run build` (`tsc`) to fail with 890+ JSX syntax errors.
2. **Missing `vite.config.ts` & `@vitejs/plugin-react`**:
   - Without `@vitejs/plugin-react` configured in `vite.config.ts`, Vite could not properly compile JSX/TSX components in browser dev mode, resulting in un-transformed JS that threw runtime syntax errors resulting in a blank white screen.
3. **Missing `react` and `react-dom` in `frontend/package.json`**:
   - `package.json` only declared `axios`, `lucide-react`, `react-router-dom`, `recharts`.
4. **Missing Global React Error Boundary**:
   - Component rendering exceptions or network crashes had no safety net, causing the React component tree to unmount and leave a blank white canvas.
5. **CORS / Port Mismatch Handling**:
   - Backend CORS allowed `http://localhost:5173`, but if Vite launched on `5174` or `5175`, API calls failed without fallback UI.

## 8. Required Fixes
1. Add `react`, `react-dom`, `@types/react`, `@types/react-dom`, `@vitejs/plugin-react` to `frontend/package.json`.
2. Configure `"jsx": "react-jsx"` and `"jsxImportSource": "react"` in `frontend/tsconfig.json`.
3. Create `frontend/vite.config.ts` with React plugin and API proxy to `http://localhost:8000`.
4. Create a robust Global React Error Boundary (`src/components/ErrorBoundary.tsx`) wrapping `<App />` in `src/main.tsx`.
5. Update FastAPI backend CORS to support `http://localhost:5174`, `http://localhost:5175`, `http://localhost:3000`, `http://127.0.0.1:5173`, `http://127.0.0.1:5174`.
