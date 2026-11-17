import type { Metadata } from "next";
import type { ReactNode } from "react";
import { Source_Sans_3, STIX_Two_Text } from "next/font/google";
import "./globals.css";

const sans = Source_Sans_3({ subsets: ["latin"], variable: "--font-sans" });
const serif = STIX_Two_Text({ subsets: ["latin"], variable: "--font-serif" });

export const metadata: Metadata = {
  title: "WikiLens | Search Simple Wikipedia",
  description: "Search Simple English Wikipedia with keyword, semantic, and reranked retrieval.",
};

export default function Layout({ children }: { children: ReactNode }) {
  return <html lang="en"><body className={sans.variable + " " + serif.variable}>{children}</body></html>;
}
