import Link from 'next/link';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { GraduationCap, BarChart2, FolderKanban, MessageSquareHeart, Cpu, Puzzle, Users, Rocket } from 'lucide-react';
import Image from 'next/image';

const features = [
  {
    icon: <BarChart2 className="h-8 w-8 text-primary" />,
    title: 'Student Dashboard',
    description: 'Track your academic journey, view strengths, and identify areas for improvement in real-time.',
    dataAiHint: 'student dashboard',
  },
  {
    icon: <FolderKanban className="h-8 w-8 text-primary" />,
    title: 'Project Portfolio',
    description: 'Showcase your best work by uploading projects and building a comprehensive portfolio.',
    dataAiHint: 'project portfolio',
  },
  {
    icon: <Cpu className="h-8 w-8 text-primary" />,
    title: 'AI Performance Calculator',
    description: 'Get AI-driven predictions on your performance based on scores and efficiency.',
    dataAiHint: 'artificial intelligence',
  },
  {
    icon: <Puzzle className="h-8 w-8 text-primary" />,
    title: 'Mind-Refreshing Games',
    description: 'Take a break and sharpen your mind with a selection of engaging mini-games.',
    dataAiHint: 'puzzle game',
  },
];

export default function LandingPage() {
  return (
    <div className="flex flex-col min-h-screen bg-background">
      <header className="container mx-auto px-4 sm:px-6 lg:px-8 h-20 flex items-center justify-between">
        <Link href="/" className="flex items-center gap-2 font-bold text-xl">
          <GraduationCap className="h-7 w-7 text-primary" />
          <span className="font-headline">EduPerform</span>
        </Link>
        <nav className="flex items-center gap-4">
          <Button variant="ghost" asChild>
            <Link href="/login">Login</Link>
          </Button>
          <Button asChild>
            <Link href="/signup">Get Started</Link>
          </Button>
        </nav>
      </header>

      <main className="flex-grow">
        <section className="container mx-auto px-4 sm:px-6 lg:px-8 py-20 md:py-32 text-center">
          <div className="max-w-4xl mx-auto">
            <h1 className="text-4xl md:text-6xl font-headline font-bold tracking-tighter mb-4">
              Unlock Your Full Academic Potential
            </h1>
            <p className="text-lg md:text-xl text-muted-foreground mb-8">
              EduPerform uses AI to provide personalized insights, helping you excel in your studies and beyond.
            </p>
            <Button size="lg" asChild>
              <Link href="/signup">
                Sign Up for Free <Rocket className="ml-2 h-5 w-5" />
              </Link>
            </Button>
          </div>
        </section>

        <section id="features" className="py-20 md:py-24 bg-secondary">
          <div className="container mx-auto px-4 sm:px-6 lg:px-8">
            <div className="text-center max-w-3xl mx-auto mb-12">
              <h2 className="text-3xl md:text-4xl font-headline font-bold">A Smarter Way to Learn</h2>
              <p className="text-muted-foreground mt-4 text-lg">
                Everything you need to stay on top of your academic performance, all in one platform.
              </p>
            </div>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-8">
              {features.map((feature) => (
                <Card key={feature.title} className="text-center shadow-lg hover:shadow-xl transition-shadow duration-300">
                  <CardHeader>
                    <div className="mx-auto bg-primary/10 p-4 rounded-full w-fit">
                      {feature.icon}
                    </div>
                    <CardTitle className="mt-4 font-headline">{feature.title}</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <p className="text-muted-foreground">{feature.description}</p>
                  </CardContent>
                </Card>
              ))}
            </div>
          </div>
        </section>

        <section className="container mx-auto px-4 sm:px-6 lg:px-8 py-20 md:py-24">
            <div className="grid md:grid-cols-2 gap-12 items-center">
                <div>
                    <h2 className="text-3xl md:text-4xl font-headline font-bold">Connect and Grow</h2>
                    <p className="text-muted-foreground mt-4 text-lg">
                        Our knowledge exchange platform allows you to connect with faculty and peers, fostering a collaborative learning environment where you can ask questions, share insights, and grow together.
                    </p>
                    <Button asChild className="mt-6">
                        <Link href="/connect">
                            Explore Connections <Users className="ml-2 h-5 w-5" />
                        </Link>
                    </Button>
                </div>
                <div className="relative h-80 rounded-lg overflow-hidden shadow-2xl">
                    <Image src="https://picsum.photos/600/400" alt="Collaboration" layout="fill" objectFit="cover" data-ai-hint="collaboration teamwork"/>
                </div>
            </div>
        </section>

      </main>

      <footer className="bg-secondary py-8">
        <div className="container mx-auto px-4 sm:px-6 lg:px-8 text-center text-muted-foreground">
          <p>&copy; {new Date().getFullYear()} EduPerform. All rights reserved.</p>
        </div>
      </footer>
    </div>
  );
}
