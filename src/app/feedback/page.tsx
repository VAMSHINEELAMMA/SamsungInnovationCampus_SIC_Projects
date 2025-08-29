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
} from "@/components/ui/select"

export default function FeedbackPage() {
  return (
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
            <Select>
              <SelectTrigger id="feedback-subject">
                <SelectValue placeholder="Select a course or assessment" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="math-calculus">Calculus Midterm</SelectItem>
                <SelectItem value="history-ww2">World War II Essay</SelectItem>
                <SelectItem value="cs-react-lab">React Components Lab</SelectItem>
                <SelectItem value="general">General Platform Feedback</SelectItem>
              </SelectContent>
            </Select>
          </div>
          
          <div className="grid gap-2">
            <Label>Overall Experience</Label>
            <div className="flex gap-4">
              <Button variant="outline" className="flex-1 group">
                <ThumbsUp className="h-5 w-5 mr-2 text-green-500 transition-transform group-hover:scale-110" />
                Like
              </Button>
              <Button variant="outline" className="flex-1 group">
                <ThumbsDown className="h-5 w-5 mr-2 text-red-500 transition-transform group-hover:scale-110" />
                Dislike
              </Button>
            </div>
          </div>

          <div className="grid gap-2">
            <Label htmlFor="comments">Comments</Label>
            <Textarea id="comments" placeholder="Tell us more about your experience..." rows={5} />
          </div>

          <Button className="w-full">
            <Send className="mr-2 h-4 w-4" />
            Submit Feedback
          </Button>
        </CardContent>
      </Card>
    </div>
  );
}
