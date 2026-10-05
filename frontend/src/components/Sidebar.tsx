import { NavLink } from 'react-router-dom'
import { useLiveHealth } from '../api'

const navItems = [
    { icon: '🏠', label: 'Home', to: '/' },
    { icon: '🖼️', label: 'Image Analysis', to: '/image' },
    { icon: '🎬', label: 'Video Analysis', to: '/video' },
    { icon: '📋', label: 'History', to: '/history' },
    { icon: '🤖', label: 'Models', to: '/models' },
    { icon: '📊', label: 'Evaluation', to: '/evaluation' },
    { icon: '⚙️', label: 'Settings', to: '/settings' },
    { icon: '📖', label: 'Methodology', to: '/methodology' },
]

export function Sidebar() {
    const { health, error } = useLiveHealth()

    return (
        <aside className="sidebar">
            <div className="sidebar-logo">
                <div className="logo-mark">
                    <div className="logo-icon">🔬</div>
                    <div>
                        <div className="logo-text">Trust-AI</div>
                        <div className="logo-sub">AI Media Analysis</div>
                    </div>
                </div>
            </div>

            <nav className="sidebar-nav">
                <div className="nav-section-title">Navigation</div>
                {navItems.map(item => (
                    <NavLink
                        key={item.to}
                        to={item.to}
                        end={item.to === '/'}
                        className={({ isActive }) => `nav-item${isActive ? ' active' : ''}`}
                    >
                        <span className="nav-icon">{item.icon}</span>
                        {item.label}
                    </NavLink>
                ))}
            </nav>

            <div className="sidebar-footer">
                <div className="status-pill">
                    <div className={`status-dot${error ? ' offline' : ''}`} />
                    <span>
                        {error ? 'Backend offline' : health ? `API · ${health.device.toUpperCase()}` : 'Connecting…'}
                    </span>
                </div>
            </div>
        </aside>
    )
}
