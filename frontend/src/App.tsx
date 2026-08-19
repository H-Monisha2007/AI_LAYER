import { BrowserRouter, Routes, Route } from 'react-router-dom'
import { Sidebar } from './components/Sidebar'
import { Landing } from './pages/Landing'
import { ImageAnalysis } from './pages/ImageAnalysis'
import { VideoAnalysis } from './pages/VideoAnalysis'
import { History } from './pages/History'
import { Models } from './pages/Models'
import { Evaluation } from './pages/Evaluation'
import { Settings } from './pages/Settings'
import { Methodology } from './pages/Methodology'

function AppLayout({ children }: { children: React.ReactNode }) {
    return (
        <div className="app-layout">
            <Sidebar />
            {children}
        </div>
    )
}

export default function App() {
    return (
        <BrowserRouter>
            <Routes>
                {/* Landing is full-page (no sidebar) */}
                <Route path="/" element={<Landing />} />

                {/* Sidebar layout for all dashboard pages */}
                <Route path="/image" element={<AppLayout><ImageAnalysis /></AppLayout>} />
                <Route path="/video" element={<AppLayout><VideoAnalysis /></AppLayout>} />
                <Route path="/history" element={<AppLayout><History /></AppLayout>} />
                <Route path="/models" element={<AppLayout><Models /></AppLayout>} />
                <Route path="/evaluation" element={<AppLayout><Evaluation /></AppLayout>} />
                <Route path="/settings" element={<AppLayout><Settings /></AppLayout>} />
                <Route path="/methodology" element={<AppLayout><Methodology /></AppLayout>} />
            </Routes>
        </BrowserRouter>
    )
}
