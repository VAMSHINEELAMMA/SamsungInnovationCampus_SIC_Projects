
"use client";

import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { FileText, Upload, Clock, CheckCircle, FileUp, Download } from "lucide-react";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Dialog, DialogContent, DialogDescription, DialogFooter, DialogHeader, DialogTitle, DialogTrigger } from "@/components/ui/dialog";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";


const initialAssessments = [
  { title: "Calculus Midterm", subject: "Mathematics", dueDate: "2024-08-15", status: "Not Submitted", file: null as File | null },
  { title: "World War II Essay", subject: "History", dueDate: "2024-08-10", status: "Submitted", file: new File([], "history_essay.pdf") },
  { title: "React Components Lab", subject: "Computer Science", dueDate: "2024-08-12", status: "Not Submitted", file: null as File | null },
];

const submissions = [
    { student: "Alice Johnson", assessment: "World War II Essay", date: "2024-08-09" },
    { student: "Bob Williams", assessment: "Calculus Midterm", date: "2024-08-14" },
    { student: "Charlie Brown", assessment: "React Components Lab", date: "2024-08-11" },
];


export default function AssessmentPage() {
  const [assessments, setAssessments] = useState(initialAssessments);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);

  const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    if (event.target.files) {
      setSelectedFile(event.target.files[0]);
    }
  };

  const handleSubmit = (assessmentTitle: string) => {
    if (!selectedFile) return;

    setAssessments(prev =>
      prev.map(assessment =>
        assessment.title === assessmentTitle
          ? { ...assessment, status: "Submitted", file: selectedFile }
          : assessment
      )
    );
    setSelectedFile(null);
    // Here you would typically handle the file upload to a server
  };

  return (
    <Tabs defaultValue="student" className="w-full">
      <TabsList className="grid w-full grid-cols-2 max-w-md mx-auto">
        <TabsTrigger value="student">Student View</TabsTrigger>
        <TabsTrigger value="faculty">Faculty View</TabsTrigger>
      </TabsList>
      <TabsContent value="student">
        <div className="grid gap-6 mt-6 md:grid-cols-2 lg:grid-cols-3">
          {assessments.map((assessment) => (
            <Card key={assessment.title} className="shadow-md hover:shadow-lg transition-shadow">
              <CardHeader>
                <CardTitle className="flex items-center gap-2"><FileText className="text-primary"/>{assessment.title}</CardTitle>
                <CardDescription>{assessment.subject}</CardDescription>
              </CardHeader>
              <CardContent className="space-y-2">
                <div className="flex items-center gap-2 text-sm text-muted-foreground">
                  <Clock className="h-4 w-4" />
                  <span>Due: {assessment.dueDate}</span>
                </div>
                <div className={`flex items-center gap-2 text-sm ${assessment.status === 'Submitted' ? 'text-green-500' : 'text-yellow-500'}`}>
                  {assessment.status === 'Submitted' ? <CheckCircle className="h-4 w-4" /> : <Clock className="h-4 w-4" />}
                  <span>{assessment.status}</span>
                </div>
              </CardContent>
              <CardFooter>
                 <Dialog>
                   <DialogTrigger asChild>
                      <Button className="w-full" disabled={assessment.status === 'Submitted'}>
                        <Upload className="mr-2 h-4 w-4" />
                        {assessment.status === 'Submitted' ? 'Submitted' : 'Submit Now'}
                      </Button>
                   </DialogTrigger>
                   <DialogContent className="sm:max-w-[425px]">
                     <DialogHeader>
                       <DialogTitle>Upload Submission</DialogTitle>
                       <DialogDescription>
                         Select the file for your submission. Click submit when you're done.
                       </DialogDescription>
                     </DialogHeader>
                     <div className="grid gap-4 py-4">
                       <div className="grid grid-cols-4 items-center gap-4">
                         <Label htmlFor="submission-file" className="text-right">
                           File
                         </Label>
                         <Input id="submission-file" type="file" className="col-span-3" onChange={handleFileChange} />
                       </div>
                     </div>
                     <DialogFooter>
                       <Button type="submit" onClick={() => handleSubmit(assessment.title)} disabled={!selectedFile}>
                         Submit
                       </Button>
                     </DialogFooter>
                   </DialogContent>
                 </Dialog>
              </CardFooter>
            </Card>
          ))}
        </div>
      </TabsContent>
      <TabsContent value="faculty">
        <div className="grid gap-8 mt-6 lg:grid-cols-2">
            <Card className="shadow-lg">
                <CardHeader>
                  <CardTitle>Upload New Assessment</CardTitle>
                  <CardDescription>Create and distribute a new assessment for your students.</CardDescription>
                </CardHeader>
                <CardContent className="space-y-4">
                   <div className="grid gap-2">
                     <Label htmlFor="assessment-title">Title</Label>
                     <Input id="assessment-title" placeholder="e.g., Final Exam"/>
                   </div>
                   <div className="grid gap-2">
                     <Label htmlFor="assessment-subject">Subject</Label>
                     <Input id="assessment-subject" placeholder="e.g., Computer Science"/>
                   </div>
                   <div className="grid gap-2">
                     <Label htmlFor="assessment-file">Assessment File</Label>
                     <Input id="assessment-file" type="file" />
                   </div>
                </CardContent>
                <CardFooter>
                  <Button className="w-full">
                    <FileUp className="mr-2 h-4 w-4"/>
                    Upload Assessment
                  </Button>
                </CardFooter>
              </Card>

            <Card className="shadow-lg">
                <CardHeader>
                    <CardTitle>Student Submissions</CardTitle>
                    <CardDescription>Review and download assignments submitted by students.</CardDescription>
                </CardHeader>
                <CardContent>
                   <Table>
                      <TableHeader>
                        <TableRow>
                          <TableHead>Student</TableHead>
                          <TableHead>Assessment</TableHead>
                          <TableHead className="text-right">Action</TableHead>
                        </TableRow>
                      </TableHeader>
                      <TableBody>
                        {submissions.map((submission) => (
                          <TableRow key={`${submission.student}-${submission.assessment}`}>
                            <TableCell className="font-medium">{submission.student}</TableCell>
                            <TableCell>{submission.assessment}</TableCell>
                            <TableCell className="text-right">
                              <Button variant="outline" size="sm">
                                <Download className="mr-2 h-4 w-4" />
                                View
                              </Button>
                            </TableCell>
                          </TableRow>
                        ))}
                      </TableBody>
                    </Table>
                </CardContent>
            </Card>
        </div>
      </TabsContent>
    </Tabs>
  );
}
