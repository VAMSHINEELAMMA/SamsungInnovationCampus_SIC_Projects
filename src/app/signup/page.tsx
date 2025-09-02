
"use client";

import Link from "next/link";
import { useRouter } from "next/navigation";
import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle, CardFooter } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { signup } from "@/lib/auth";
import { useToast } from "@/hooks/use-toast";

export default function SignupPage() {
  const [fullName, setFullName] = useState("");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const router = useRouter();
  const { toast } = useToast();


  const handleSignup = () => {
    if (!fullName || !email || !password) {
      toast({
        variant: "destructive",
        title: "Error",
        description: "Please fill in all fields.",
      });
      return;
    }
    
    try {
      signup({ fullName, email, password });
      toast({
        title: "Account Created",
        description: "Welcome! You have been logged in.",
      });
      router.push("/dashboard");
    } catch (error: any) {
        toast({
            variant: "destructive",
            title: "Signup Failed",
            description: error.message,
        });
    }
  };

  return (
    <Card className="w-full max-w-sm shadow-xl">
      <CardHeader>
        <CardTitle className="text-2xl font-headline">Create an Account</CardTitle>
        <CardDescription>
          Enter your information to create a new account.
        </CardDescription>
      </CardHeader>
      <CardContent className="grid gap-4">
        <div className="grid gap-2">
          <Label htmlFor="full-name">Full Name</Label>
          <Input 
            id="full-name" 
            placeholder="John Doe" 
            required 
            value={fullName}
            onChange={(e) => setFullName(e.target.value)}
          />
        </div>
        <div className="grid gap-2">
          <Label htmlFor="email">Email</Label>
          <Input 
            id="email" 
            type="email" 
            placeholder="student@example.com" 
            required 
            value={email}
            onChange={(e) => setEmail(e.target.value)}
          />
        </div>
        <div className="grid gap-2">
          <Label htmlFor="password">Password</Label>
          <Input 
            id="password" 
            type="password" 
            required 
            value={password}
            onChange={(e) => setPassword(e.target.value)}
          />
        </div>
      </CardContent>
      <CardFooter className="flex flex-col gap-4">
        <Button className="w-full" onClick={handleSignup}>
          Create Account
        </Button>
         <div className="text-sm text-center text-muted-foreground">
          Already have an account?{" "}
          <Link href="/login" className="underline font-medium text-primary">
            Login
          </Link>
        </div>
      </CardFooter>
    </Card>
  );
}
