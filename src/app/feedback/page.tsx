
"use client";

import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Textarea } from "@/components/ui/textarea";
import { Label } from "@/components/ui/label";
import { ThumbsUp, ThumbsDown, Send } from "lucide-react";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { useToast } from "@/hooks/use-toast";

type Feedback = {
  subject: string;
  experience: "Like" | "Dislike" | null;
  comments: string;
};

export default function FeedbackPage() {
  const [feedbacks, setFeedbacks] = useState<Feedback[]>([]);
  const [subject, setSubject] = useState("");
  const [experience, setExperience] = useState<"Like" | "Dislike" | null>(null);
  const [comments, setComments] = useState("");
  const { toast } = useToast();

  const handleSubmit = () => {
    if (!subject || !experience || !comments) {
      toast({
        variant: "destructive",
        title: "Incomplete Feedback",
        description: "Please fill out all fields before submitting.",
      });
      return;
    }

    const newFeedback: Feedback = { subject, experience, comments };
    setFeedbacks(prev => [...prev, newFeedback]);
    
    toast({
        title: "Feedback Submitted",
        description: "Thank you for your feedback!",
    });

    // Reset form
    setSubject("");
    setExperience(null);
    setComments("");
  };


  return (
    <Tabs defaultValue="student" className="w-full max-w-4xl mx-auto">
      <TabsList className="grid w-full grid-cols-2 max-w-md mx-auto">
        <TabsTrigger value="student">Student View</TabsTrigger>
        <TabsTrigger value="faculty">Faculty View</TabsTrigger>
      </TabsList>
      <TabsContent value="student">
        <div className="flex justify-center items-start pt-10">
          <Card className="w-full max-w-2xl shadow-xl">
            <CardHeader>
              <CardTitle className="text-2xl font-headline">Submit Your Feedback</CardTitle>
              <CardDescription>
                We value your opinion. Let us know what you think about your courses or assessments.
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-6">
              <div className="grid gap-2">
                <Label htmlFor="feedback-subject">Feedback Subject</Label>
                <Select value={subject} onValueChange={setSubject}>
                  <SelectTrigger id="feedback-subject">
                    <SelectValue placeholder="Select a course or assessment" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="Calculus Midterm">Calculus Midterm</SelectItem>
                    <SelectItem value="World War II Essay">World War II Essay</SelectItem>
                    <SelectItem value="React Components Lab">React Components Lab</SelectItem>
                    <SelectItem value="General Platform Feedback">General Platform Feedback</SelectItem>
                  </SelectContent>
                </Select>
              </div>
              
              <div className="grid gap-2">
                <Label>Overall Experience</Label>
                <div className="flex gap-4">
                  <Button 
                    variant={experience === 'Like' ? 'default' : 'outline'} 
                    className="flex-1 group"
                    onClick={() => setExperience('Like')}
                  >
                    <ThumbsUp className="h-5 w-5 mr-2 text-green-500 transition-transform group-hover:scale-110" />
                    Like
                  </Button>
                  <Button 
                    variant={experience === 'Dislike' ? 'destructive' : 'outline'} 
                    className="flex-1 group"
                    onClick={() => setExperience('Dislike')}
                  >
                    <ThumbsDown className="h-5 w-5 mr-2 text-red-500 transition-transform group-hover:scale-110" />
                    Dislike
                  </Button>
                </div>
              </div>

              <div className="grid gap-2">
                <Label htmlFor="comments">Comments</Label>
                <Textarea 
                    id="comments" 
                    placeholder="Tell us more about your experience..." 
                    rows={5} 
                    value={comments}
                    onChange={(e) => setComments(e.target.value)}
                />
              </div>

              <Button className="w-full" onClick={handleSubmit}>
                <Send className="mr-2 h-4 w-4" />
                Submit Feedback
              </Button>
            </CardContent>
          </Card>
        </div>
      </TabsContent>
      <TabsContent value="faculty">
         <Card className="mt-6 shadow-lg">
            <CardHeader>
                <CardTitle>Student Feedback Submissions</CardTitle>
                <CardDescription>Review feedback submitted by students for courses and assessments.</CardDescription>
            </CardHeader>
            <CardContent>
                {feedbacks.length > 0 ? (
                    <Table>
                        <TableHeader>
                            <TableRow>
                            <TableHead>Subject</TableHead>
                            <TableHead>Rating</TableHead>
                            <TableHead>Comment</TableHead>
                            </TableRow>
                        </TableHeader>
                        <TableBody>
                            {feedbacks.map((feedback, index) => (
                            <TableRow key={index}>
                                <TableCell className="font-medium">{feedback.subject}</TableCell>
                                <TableCell>
                                    <span className={`flex items-center gap-2 ${feedback.experience === 'Like' ? 'text-green-500' : 'text-red-500'}`}>
                                        {feedback.experience === 'Like' ? <ThumbsUp className="h-4 w-4"/> : <ThumbsDown className="h-4 w-4"/>}
                                        {feedback.experience}
                                    </span>
                                </TableCell>
                                <TableCell>{feedback.comments}</TableCell>
                            </TableRow>
                            ))}
                        </TableBody>
                    </Table>
                ) : (
                    <div className="text-center py-12 text-muted-foreground">
                        <p>No feedback has been submitted yet.</p>
                    </div>
                )}
            </CardContent>
        </Card>
      </TabsContent>
    </Tabs>
  );
}
