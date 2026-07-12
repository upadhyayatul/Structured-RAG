import type { Metadata } from "next";
import { Courier_Prime, Special_Elite } from "next/font/google";
import "./globals.css";

// Typewriter pairing for the government-dossier theme:
// Courier Prime for body copy, Special Elite for headings/labels/stamps.
const courierPrime = Courier_Prime({
  variable: "--font-courier-prime",
  weight: ["400", "700"],
  subsets: ["latin"],
});

const specialElite = Special_Elite({
  variable: "--font-special-elite",
  weight: "400",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "UPSC Polity RAG",
  description: "Grounded Q&A over M. Laxmikanth's Indian Polity",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html
      lang="en"
      suppressHydrationWarning
      className={`${courierPrime.variable} ${specialElite.variable} dark h-full antialiased`}
    >
      <body className="h-full">{children}</body>
    </html>
  );
}
