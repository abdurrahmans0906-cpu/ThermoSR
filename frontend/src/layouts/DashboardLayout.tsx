import React, { useState, useEffect } from 'react';
import { NavLink, Outlet, useLocation } from 'react-router-dom';
import {
  LayoutDashboard,
  Cpu,
  Layers,
  FlaskConical,
  BarChart3,
  Archive,
  Settings,
  Menu,
  X,
  Flame,
  CheckCircle2,
} from 'lucide-react';
import { api } from '../services/api';
import { HealthResponse, DeviceInfo } from '../types';

export const DashboardLayout: React.FC = () => {
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [device, setDevice] = useState<DeviceInfo | null>(null);
  const location = useLocation();

  useEffect(() => {
    const fetchStatus = async () => {
      try {
        const [hData, dData] = await Promise.all([
          api.getHealth(),
          api.getDeviceInfo(),
        ]);
        setHealth(hData);
        setDevice(dData);
      } catch (err) {
        console.error('Failed to query backend health:', err);
      }
    };
    fetchStatus();
    const interval = setInterval(fetchStatus, 15000);
    return () => clearInterval(interval);
  }, []);

  const navItems = [
    { name: 'Dashboard', path: '/', icon: LayoutDashboard },
    { name: 'Reconstruct', path: '/reconstruct', icon: Flame },
    { name: 'Dataset', path: '/dataset', icon: Layers },
    { name: 'Training Lab', path: '/training', icon: FlaskConical },
    { name: 'Experiments', path: '/experiments', icon: BarChart3 },
    { name: 'Model Registry', path: '/models', icon: Archive },
    { name: 'Settings', path: '/settings', icon: Settings },
  ];

  const getPageTitle = () => {
    const active = navItems.find((item) => item.path === location.pathname);
    return active ? active.name : 'ThermoSR';
  };

  return (
    <div className="flex h-screen bg-slate-50 text-slate-800 overflow-hidden font-sans">
      {/* Sidebar Desktop */}
      <aside className="hidden md:flex flex-col w-60 border-r border-slate-200 bg-white flex-shrink-0">
        {/* Brand */}
        <div className="flex items-center gap-3 px-5 h-16 border-b border-slate-200">
          <div className="h-8 w-8 rounded bg-blue-600 flex items-center justify-center text-white font-bold text-sm">
            T
          </div>
          <div>
            <div className="font-bold text-slate-900 tracking-tight text-sm">
              ThermoSR
            </div>
            <p className="text-[11px] text-slate-500">Thermal SR Research Lab</p>
          </div>
        </div>

        {/* Navigation Links */}
        <nav className="flex-1 px-3 py-4 space-y-0.5 overflow-y-auto">
          {navItems.map((item) => {
            const Icon = item.icon;
            return (
              <NavLink
                key={item.path}
                to={item.path}
                className={({ isActive }) =>
                  `flex items-center gap-3 px-3 py-2 rounded-md text-xs font-medium transition-colors ${
                    isActive
                      ? 'bg-blue-50 text-blue-700 font-semibold'
                      : 'text-slate-600 hover:text-slate-900 hover:bg-slate-100'
                  }`
                }
              >
                <Icon className="h-4 w-4" />
                <span>{item.name}</span>
              </NavLink>
            );
          })}
        </nav>

        {/* Project Academic Info */}
        <div className="p-3.5 m-3 rounded-lg border border-slate-200 bg-slate-50 text-xs">
          <div className="font-semibold text-slate-800 text-[11px] mb-1">
            Research Contribution
          </div>
          <p className="text-[11px] text-slate-500 leading-relaxed">
            Reliability-Aware RGB Guidance for Thermal Infrared Super-Resolution (×8).
          </p>
        </div>
      </aside>

      {/* Main Content Area */}
      <div className="flex-1 flex flex-col min-w-0 overflow-hidden">
        {/* Top Header */}
        <header className="h-16 border-b border-slate-200 bg-white px-6 flex items-center justify-between z-10">
          <div className="flex items-center gap-4">
            <button
              onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
              className="md:hidden p-2 rounded text-slate-600 hover:bg-slate-100"
            >
              {mobileMenuOpen ? <X className="h-5 w-5" /> : <Menu className="h-5 w-5" />}
            </button>
            <h1 className="text-base font-semibold text-slate-900">{getPageTitle()}</h1>
          </div>

          {/* Right Status Indicators */}
          <div className="flex items-center gap-2.5">
            {/* Compute Device */}
            <div className="flex items-center gap-1.5 px-3 py-1 rounded border border-slate-200 bg-slate-50 text-xs text-slate-600">
              <Cpu className="h-3.5 w-3.5 text-slate-500" />
              <span>
                {device?.cuda_available ? (
                  <span className="font-medium text-slate-800">{device.device_name}</span>
                ) : (
                  <span>CPU Mode</span>
                )}
              </span>
            </div>

            {/* API Status */}
            <div className="flex items-center gap-1.5 px-3 py-1 rounded border border-slate-200 bg-slate-50 text-xs">
              <div
                className={`h-2 w-2 rounded-full ${
                  health?.status === 'healthy' ? 'bg-emerald-500' : 'bg-amber-500'
                }`}
              />
              <span className="text-slate-600 font-medium text-[11px]">
                {health?.status === 'healthy' ? 'API Online' : 'Connecting...'}
              </span>
            </div>
          </div>
        </header>

        {/* Mobile Navigation Drawer */}
        {mobileMenuOpen && (
          <div className="md:hidden border-b border-slate-200 bg-white px-4 py-3 space-y-1">
            {navItems.map((item) => (
              <NavLink
                key={item.path}
                to={item.path}
                onClick={() => setMobileMenuOpen(false)}
                className={({ isActive }) =>
                  `flex items-center gap-3 px-3 py-2 rounded text-xs font-medium ${
                    isActive ? 'bg-blue-50 text-blue-700' : 'text-slate-600 hover:text-slate-900'
                  }`
                }
              >
                <item.icon className="h-4 w-4" />
                <span>{item.name}</span>
              </NavLink>
            ))}
          </div>
        )}

        {/* Main Routed Page Content */}
        <main className="flex-1 overflow-y-auto p-6 md:p-8">
          <div className="max-w-6xl mx-auto">
            <Outlet />
          </div>
        </main>
      </div>
    </div>
  );
};
