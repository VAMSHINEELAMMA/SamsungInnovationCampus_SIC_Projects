import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card";
import { Upload } from "lucide-react";
import Image from "next/image";

const projects = [
  {
    title: "E-commerce Website",
    description: "A full-stack e-commerce platform built with Next.js, Stripe, and PostgreSQL.",
    imageUrl: "https://picsum.photos/600/400?random=1",
    dataAiHint: 'ecommerce website'
  },
  {
    title: "Data Visualization Dashboard",
    description: "An analytics dashboard for visualizing sales data using D3.js and React.",
    imageUrl: "https://picsum.photos/600/400?random=2",
    dataAiHint: 'data dashboard'
  },
  {
    title: "Mobile Fitness App",
    description: "A cross-platform mobile app developed with React Native to track workouts and nutrition.",
    imageUrl: "https://picsum.photos/600/400?random=3",
    dataAiHint: 'fitness app'
  },
  {
    title: "Machine Learning Model",
    description: "A Python-based model to predict stock market trends using historical data.",
    imageUrl: "https://picsum.photos/600/400?random=4",
    dataAiHint: 'machine learning'
  },
];

export default function ProjectsPage() {
  return (
    <div className="space-y-6">
      <div className="flex justify-end">
        <Button>
          <Upload className="mr-2 h-4 w-4" />
          Upload Project
        </Button>
      </div>
      <div className="grid gap-6 md:grid-cols-2 lg:grid-cols-3">
        {projects.map((project) => (
          <Card key={project.title} className="shadow-md hover:shadow-lg transition-shadow overflow-hidden">
            <CardHeader className="p-0">
              <div className="relative h-48 w-full">
                <Image
                  src={project.imageUrl}
                  alt={project.title}
                  layout="fill"
                  objectFit="cover"
                  data-ai-hint={project.dataAiHint}
                />
              </div>
            </CardHeader>
            <div className="p-6">
              <CardTitle>{project.title}</CardTitle>
              <CardDescription className="mt-2">{project.description}</CardDescription>
            </div>
            <CardFooter>
              <Button variant="outline" className="w-full">View Details</Button>
            </CardFooter>
          </Card>
        ))}
      </div>
    </div>
  );
}
