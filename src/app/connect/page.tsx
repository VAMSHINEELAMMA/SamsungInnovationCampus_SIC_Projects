
"use client";

import { useState } from "react";
import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card";
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle, DialogTrigger } from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { MessageSquare, Send } from "lucide-react";
import { ScrollArea } from "@/components/ui/scroll-area";
import { useAuth } from "@/hooks/use-auth";

const faculty = [
  { name: "Dr. Evelyn Reed", subject: "Physics & Astronomy", avatar: "https://picsum.photos/100?random=1", dataAiHint: "female professor" },
  { name: "Prof. Samuel Chen", subject: "Computer Science", avatar: "https://picsum.photos/100?random=2", dataAiHint: "male professor" },
  { name: "Dr. Isabella Rossi", subject: "History & Arts", avatar: "https://picsum.photos/100?random=3", dataAiHint: "female academic" },
  { name: "Prof. Ben Carter", subject: "Mathematics", avatar: "https://picsum.photos/100?random=4", dataAiHint: "male academic" },
  { name: "Dr. Anya Sharma", subject: "Biology & Chemistry", avatar: "https://picsum.photos/100?random=5", dataAiHint: "scientist woman" },
  { name: "Prof. Omar Badawi", subject: "Economics", avatar: "https://picsum.photos/100?random=6", dataAiHint: "professor man" },
];

type Message = {
  text: string;
  sender: 'user' | 'faculty';
  timestamp: string;
};

export default function ConnectPage() {
  const [searchTerm, setSearchTerm] = useState("");
  const [chattingWith, setChattingWith] = useState<typeof faculty[0] | null>(null);
  const [messages, setMessages] = useState<Record<string, Message[]>>({});
  const [currentMessage, setCurrentMessage] = useState("");
  const { user } = useAuth();

  const filteredFaculty = faculty.filter(
    (f) =>
      f.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
      f.subject.toLowerCase().includes(searchTerm.toLowerCase())
  );

  const handleSendMessage = () => {
    if (!currentMessage.trim() || !chattingWith) return;

    const newMessage: Message = {
      text: currentMessage,
      sender: 'user',
      timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
    };

    const facultyName = chattingWith.name;
    const existingMessages = messages[facultyName] || [];
    const updatedMessages = [...existingMessages, newMessage];
    
    setMessages(prev => ({ ...prev, [facultyName]: updatedMessages }));
    setCurrentMessage("");

    // Simulate faculty reply
    setTimeout(() => {
      const reply: Message = {
        text: `Hello! Thanks for reaching out. I'll get back to you about "${currentMessage}" shortly.`,
        sender: 'faculty',
        timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
      };
      setMessages(prev => ({ ...prev, [facultyName]: [...updatedMessages, reply] }));
    }, 1000);
  };
  
  const handleKeyPress = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === 'Enter') {
      handleSendMessage();
    }
  }


  return (
    <div className="space-y-8">
      <div className="max-w-2xl mx-auto">
        <Input
          type="search"
          placeholder="Search for faculty by name or subject..."
          className="w-full"
          value={searchTerm}
          onChange={(e) => setSearchTerm(e.target.value)}
        />
      </div>

      <div className="grid gap-6 md:grid-cols-2 lg:grid-cols-3">
        {filteredFaculty.map((member) => (
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
              <Button className="w-full" onClick={() => setChattingWith(member)}>
                <MessageSquare className="mr-2 h-4 w-4" />
                Message
              </Button>
            </CardFooter>
          </Card>
        ))}
      </div>

      <Dialog open={!!chattingWith} onOpenChange={(open) => { if (!open) setChattingWith(null); }}>
        <DialogContent className="sm:max-w-[425px] flex flex-col h-[600px] p-0">
          <DialogHeader className="p-6 pb-2">
            <DialogTitle>Chat with {chattingWith?.name}</DialogTitle>
            <DialogDescription>
              Ask any questions you have about {chattingWith?.subject}.
            </DialogDescription>
          </DialogHeader>
          <ScrollArea className="flex-grow my-0 px-6">
             <div className="space-y-4 py-4">
                {(messages[chattingWith?.name || ''] || []).map((msg, index) => (
                   <div key={index} className={`flex items-end gap-2 ${msg.sender === 'user' ? 'justify-end' : 'justify-start'}`}>
                      {msg.sender === 'faculty' && <Avatar className="h-8 w-8"><AvatarImage src={chattingWith?.avatar} /><AvatarFallback>{chattingWith?.name.charAt(0)}</AvatarFallback></Avatar>}
                      <div className={`rounded-lg px-3 py-2 max-w-[80%] ${msg.sender === 'user' ? 'bg-primary text-primary-foreground' : 'bg-muted'}`}>
                          <p className="text-sm">{msg.text}</p>
                          <p className="text-xs text-right opacity-70 mt-1">{msg.timestamp}</p>
                      </div>
                      {msg.sender === 'user' && <Avatar className="h-8 w-8"><AvatarFallback>{user?.fullName?.[0]?.toUpperCase() || 'U'}</AvatarFallback></Avatar>}
                   </div>
                ))}
             </div>
          </ScrollArea>
          <div className="flex items-center space-x-2 p-6 pt-2 border-t">
            <Input
              placeholder="Type your message..."
              value={currentMessage}
              onChange={(e) => setCurrentMessage(e.target.value)}
              onKeyPress={handleKeyPress}
            />
            <Button onClick={handleSendMessage} disabled={!currentMessage.trim()}>
              <Send className="h-4 w-4" />
            </Button>
          </div>
        </DialogContent>
      </Dialog>
    </div>
  );
}
