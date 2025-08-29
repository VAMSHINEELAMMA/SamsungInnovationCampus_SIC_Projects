"use client";

import { usePathname } from "next/navigation";
import { AppLayout } from "./app-layout";
import { AuthLayout } from "./auth-layout";
import { Toaster } from "@/components/ui/toaster";

const appRoutes = [
  "/dashboard",
  "/assessment",
  "/projects",
  "/feedback",
  "/calculator",
  "/games",
  "/connect",
];
const authRoutes = ["/login", "/signup"];

export function LayoutProvider({ children }: { children: React.ReactNode }) {
  const pathname = usePathname();

  if (appRoutes.includes(pathname)) {
    return (
      <AppLayout>
        {children}
        <Toaster />
      </AppLayout>
    );
  }

  if (authRoutes.includes(pathname)) {
    return (
      <AuthLayout>
        {children}
        <Toaster />
      </AuthLayout>
    );
  }

  return <>
    {children}
    <Toaster />
  </>;
}
