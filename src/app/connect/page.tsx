import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card";
import { MessageSquare } from "lucide-react";
import { Input } from "@/components/ui/input";

const faculty = [
  { name: "Dr. Evelyn Reed", subject: "Physics & Astronomy", avatar: "https://picsum.photos/100?random=1", dataAiHint: "female professor" },
  { name: "Prof. Samuel Chen", subject: "Computer Science", avatar: "https://picsum.photos/100?random=2", dataAiHint: "male professor" },
  { name: "Dr. Isabella Rossi", subject: "History & Arts", avatar: "https://picsum.photos/100?random=3", dataAiHint: "female academic" },
  { name: "Prof. Ben Carter", subject: "Mathematics", avatar: "https://picsum.photos/100?random=4", dataAiHint: "male academic" },
  { name: "Dr. Anya Sharma", subject: "Biology & Chemistry", avatar: "https://picsum.photos/100?random=5", dataAiHint: "scientist woman" },
  { name: "Prof. Omar Badawi", subject: "Economics", avatar: "https://picsum.photos/100?random=6", dataAiHint: "professor man" },
];

export default function ConnectPage() {
  return (
    <div className="space-y-8">
       <div className="max-w-2xl mx-auto">
         <Input
           type="search"
           placeholder="Search for faculty by name or subject..."
           className="w-full"
         />
       </div>

      <div className="grid gap-6 md:grid-cols-2 lg:grid-cols-3">
        {faculty.map((member) => (
          <Card key={member.name} className="shadow-md hover:shadow-lg transition-shadow text-center">
            <CardHeader className="items-center">
                <Avatar className="h-24 w-24 border-4 border-primary/20">
                    <AvatarImage src={member.avatar} alt={member.name} data-ai-hint={member.dataAiHint} />
                    <AvatarFallback>{member.name.charAt(0)}</AvatarFallback>
                </Avatar>
            </CardHeader>
            <CardContent>
                <CardTitle>{member.name}</CardTitle>
                <CardDescription className="mt-1 text-primary">{member.subject}</CardDescription>
            </CardContent>
            <CardFooter>
              <Button className="w-full">
                <MessageSquare className="mr-2 h-4 w-4" />
                Message
              </Button>
            </CardFooter>
          </Card>
        ))}
      </div>
    </div>
  );
}
