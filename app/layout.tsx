import type { Metadata } from 'next';
import { Geist, Geist_Mono } from 'next/font/google';
import './globals.css';

const geistSans = Geist({
  variable: '--font-geist-sans',
  subsets: ['latin'],
});

const geistMono = Geist_Mono({
  variable: '--font-geist-mono',
  subsets: ['latin'],
});

export const metadata: Metadata = {
  metadataBase: new URL(
    process.env.NEXT_PUBLIC_SITE_ORIGIN ?? 'http://localhost:3000',
  ),
  title: 'VoiceMerge9000',
  description:
    'Map the speakers in an audio scene, cast any compatible character voice models, and merge the converted performances back on the original timeline.',
  openGraph: {
    title: 'VoiceMerge9000',
    description: 'Scene in. Characters out.',
    images: [{ url: '/og-voicemerge9000.png', width: 1728, height: 908, alt: 'VoiceMerge9000 character voice casting console' }],
  },
  twitter: {
    card: 'summary_large_image',
    title: 'VoiceMerge9000',
    description: 'Scene in. Characters out.',
    images: ['/og-voicemerge9000.png'],
  },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body
        className={`${geistSans.variable} ${geistMono.variable} antialiased`}
      >
        {children}
      </body>
    </html>
  );
}
