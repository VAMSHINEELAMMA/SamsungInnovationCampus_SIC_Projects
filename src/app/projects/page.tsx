
"use client";

import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card";
import { Dialog, DialogContent, DialogDescription, DialogFooter, DialogHeader, DialogTitle, DialogTrigger } from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
import { Upload, Linkedin } from "lucide-react";
import Image from "next/image";
import Link from "next/link";


const initialProjects = [
  {
    title: "E-commerce Website",
    description: "A full-stack e-commerce platform built with Next.js, Stripe, and PostgreSQL.",
    imageUrl: "https://picsum.photos/600/400?random=1",
    linkedinUrl: "https://linkedin.com/in/user",
    projectFile: null as File | null,
    dataAiHint: 'ecommerce website'
  },
  {
    title: "Data Visualization Dashboard",
    description: "An analytics dashboard for visualizing sales data using D3.js and React.",
    imageUrl: "https://picsum.photos/600/400?random=2",
    linkedinUrl: "https://linkedin.com/in/user",
    projectFile: null as File | null,
    dataAiHint: 'data dashboard'
  },
  {
    title: "Mobile Fitness App",
    description: "A cross-platform mobile app developed with React Native to track workouts and nutrition.",
    imageUrl: "https://picsum.photos/600/400?random=3",
    linkedinUrl: "https://linkedin.com/in/user",
    projectFile: null as File | null,
    dataAiHint: 'fitness app'
  },
  {
    title: "Machine Learning Model",
    description: "A Python-based model to predict stock market trends using historical data.",
    imageUrl: "https://picsum.photos/600/400?random=4",
    linkedinUrl: "https://linkedin.com/in/user",
    projectFile: null as File | null,
    dataAiHint: 'machine learning'
  },
];

type Project = typeof initialProjects[0];

export default function ProjectsPage() {
  const [projects, setProjects] = useState<Project[]>(initialProjects);
  const [isDialogOpen, setIsDialogOpen] = useState(false);
  const [newProject, setNewProject] = useState({
    title: "",
    description: "",
    imageUrl: "",
    linkedinUrl: "",
    projectFile: null as File | null
  });

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement | HTMLTextAreaElement>) => {
    const { id, value } = e.target;
    setNewProject(prev => ({ ...prev, [id]: value }));
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files) {
      setNewProject(prev => ({ ...prev, projectFile: e.target.files?.[0] || null }));
    }
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!newProject.title || !newProject.description || !newProject.imageUrl) {
        alert("Please fill in all required fields.");
        return;
    }
    const projectToAdd: Project = { ...newProject, dataAiHint: 'custom project' };
    setProjects(prev => [projectToAdd, ...prev]);
    setIsDialogOpen(false);
    setNewProject({
        title: "",
        description: "",
        imageUrl: "",
        linkedinUrl: "",
        projectFile: null
    });
  };


  return (
    <div className="space-y-6">
      <div className="flex justify-end">
        <Dialog open={isDialogOpen} onOpenChange={setIsDialogOpen}>
          <DialogTrigger asChild>
            <Button>
              <Upload className="mr-2 h-4 w-4" />
              Upload Project
            </Button>
          </DialogTrigger>
          <DialogContent className="sm:max-w-[480px]">
            <DialogHeader>
              <DialogTitle>Upload New Project</DialogTitle>
              <DialogDescription>
                Showcase your work by adding a new project to your portfolio.
              </DialogDescription>
            </DialogHeader>
            <form onSubmit={handleSubmit}>
              <div className="grid gap-4 py-4">
                <div className="grid grid-cols-4 items-center gap-4">
                  <Label htmlFor="title" className="text-right">Title</Label>
                  <Input id="title" value={newProject.title} onChange={handleInputChange} className="col-span-3" required />
                </div>
                <div className="grid grid-cols-4 items-center gap-4">
                  <Label htmlFor="description" className="text-right">Description</Label>
                  <Textarea id="description" value={newProject.description} onChange={handleInputChange} className="col-span-3" required/>
                </div>
                <div className="grid grid-cols-4 items-center gap-4">
                  <Label htmlFor="imageUrl" className="text-right">Image URL</Label>
                  <Input id="imageUrl" value={newProject.imageUrl} onChange={handleInputChange} className="col-span-3" placeholder="https://picsum.photos/600/400" required />
                </div>
                 <div className="grid grid-cols-4 items-center gap-4">
                  <Label htmlFor="linkedinUrl" className="text-right">LinkedIn URL</Label>
                  <Input id="linkedinUrl" value={newProject.linkedinUrl} onChange={handleInputChange} className="col-span-3" placeholder="https://linkedin.com/in/your-profile" />
                </div>
                <div className="grid grid-cols-4 items-center gap-4">
                  <Label htmlFor="projectFile" className="text-right">Project File</Label>
                  <Input id="projectFile" type="file" accept=".zip" onChange={handleFileChange} className="col-span-3" />
                </div>
              </div>
              <DialogFooter>
                <Button type="submit">Save Project</Button>
              </DialogFooter>
            </form>
          </DialogContent>
        </Dialog>
      </div>
      <div className="grid gap-6 md:grid-cols-2 lg:grid-cols-3">
        {projects.map((project, index) => (
          <Card key={`${project.title}-${index}`} className="shadow-md hover:shadow-lg transition-shadow overflow-hidden flex flex-col">
            <CardHeader className="p-0">
              <div className="relative h-48 w-full">
                <Image
                  src={project.imageUrl}
                  alt={project.title}
                  width={600}
                  height={400}
                  className="object-cover w-full h-full"
                  data-ai-hint={project.dataAiHint}
                />
              </div>
            </CardHeader>
            <div className="p-6 flex flex-col flex-grow">
              <CardTitle>{project.title}</CardTitle>
              <CardDescription className="mt-2 flex-grow">{project.description}</CardDescription>
            </div>
            <CardFooter className="flex flex-col sm:flex-row gap-2">
              <Button variant="outline" className="w-full">View Details</Button>
               {project.linkedinUrl && (
                <Button asChild variant="secondary" className="w-full">
                    <Link href={project.linkedinUrl} target="_blank">
                        <Linkedin className="mr-2 h-4 w-4" />
                        LinkedIn
                    </Link>
                </Button>
              )}
            </CardFooter>
          </Card>
        ))}
      </div>
    </div>
  );
}
