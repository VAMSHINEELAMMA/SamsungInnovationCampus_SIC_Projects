
"use client";

import { usePathname, useRouter } from "next/navigation";
import { AppLayout } from "./app-layout";
import { AuthLayout } from "./auth-layout";
import { Toaster } from "@/components/ui/toaster";
import { AuthProvider, useAuth } from "@/hooks/use-auth";
import { useEffect } from "react";

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

function InnerLayout({ children }: { children: React.ReactNode }) {
  const pathname = usePathname();
  const router = useRouter();
  const { isAuthenticated, isLoading } = useAuth();

  useEffect(() => {
    if (!isLoading && !isAuthenticated && appRoutes.includes(pathname)) {
      router.push("/login");
    }
    if (!isLoading && isAuthenticated && authRoutes.includes(pathname)) {
      router.push("/dashboard");
    }
  }, [isAuthenticated, pathname, router, isLoading]);

  if (isLoading) {
    return (
      <div className="flex h-screen w-screen items-center justify-center">
        {/* You can replace this with a nice spinner component */}
        <p>Loading...</p>
      </div>
    );
  }

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


export function LayoutProvider({ children }: { children: React.ReactNode }) {
  return (
    <AuthProvider>
      <InnerLayout>{children}</InnerLayout>
    </AuthProvider>
  );
}
