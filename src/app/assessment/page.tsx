import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { FileText, Upload, Clock, CheckCircle, FileUp } from "lucide-react";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";

const assessments = [
  { title: "Calculus Midterm", subject: "Mathematics", dueDate: "2024-08-15", status: "Not Submitted" },
  { title: "World War II Essay", subject: "History", dueDate: "2024-08-10", status: "Submitted" },
  { title: "React Components Lab", subject: "Computer Science", dueDate: "2024-08-12", status: "Not Submitted" },
];

export default function AssessmentPage() {
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
                <Button className="w-full" disabled={assessment.status === 'Submitted'}>
                  <Upload className="mr-2 h-4 w-4" />
                  {assessment.status === 'Submitted' ? 'Submitted' : 'Submit Now'}
                </Button>
              </CardFooter>
            </Card>
          ))}
        </div>
      </TabsContent>
      <TabsContent value="faculty">
        <div className="mt-6 max-w-2xl mx-auto">
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
        </div>
      </TabsContent>
    </Tabs>
  );
}
