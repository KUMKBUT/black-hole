import type { Metadata } from 'next';
import './globals.css';

export const metadata: Metadata = {
  title: 'Black Hole — Cinematic 3D Simulation',
  description:
    'An interactive, cinematic 3D black hole simulation with gravitational lensing, accretion disk, and photon sphere effects built with Three.js and Next.js.',
};

export default function RootLayout({ children }: LayoutProps<"/">) {
  return (
    <html lang="en">
      <body className="m-0 p-0 w-full h-screen overflow-hidden bg-black">
        {children}
      </body>
    </html>
  );
}
