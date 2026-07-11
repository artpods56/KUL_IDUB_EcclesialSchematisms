import type { Metadata } from "next";
import "./globals.css";
import { Providers } from "@/components/providers";

export const metadata: Metadata = {
  title: "Notarius Workbench",
  description:
    "A node-first workbench for testing typed artifact graphs and nested-field projections.",
};

const themeScript = `
(function () {
  try {
    var theme = localStorage.getItem("ns-theme");
    if (theme === "light" || theme === "dark") {
      document.documentElement.style.colorScheme = theme;
    }
  } catch (e) {}
})();
`;

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" suppressHydrationWarning>
      <body>
        <script dangerouslySetInnerHTML={{ __html: themeScript }} />
        <Providers>{children}</Providers>
      </body>
    </html>
  );
}
