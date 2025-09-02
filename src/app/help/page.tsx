
import {
  Accordion,
  AccordionContent,
  AccordionItem,
  AccordionTrigger,
} from "@/components/ui/accordion"
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card"
import { LifeBuoy } from "lucide-react"

const faqs = [
  {
    question: "How do I check my overall performance?",
    answer: "You can view your overall performance, including detailed breakdowns and strengths, on the Dashboard page. The main KPIs are displayed at the top, and a performance chart gives you a visual breakdown.",
  },
  {
    question: "How do I submit an assessment?",
    answer: "Navigate to the Assessments page. Find the assessment you want to submit and click the 'Submit Now' button. A dialog will appear allowing you to upload your file.",
  },
  {
    question: "Where can I see feedback on my submissions?",
    answer: "Scores for your submitted assessments will appear on the assessment card on the Assessments page once they have been graded. For more detailed written feedback, check the Feedback page.",
  },
  {
    question: "Can I add my personal projects to the portfolio?",
    answer: "Yes! Go to the Projects page and click the 'Upload Project' button. You can add details, an image, and links to your GitHub or LinkedIn.",
  },
  {
    question: "What is the AI Performance Calculator?",
    answer: "The Calculator on the 'Calculator' page uses AI to predict your future performance based on your current scores and efficiency. It also provides personalized suggestions for improvement.",
  },
  {
    question: "How do I contact a faculty member?",
    answer: "The Connect page allows you to search for faculty members by name or subject. Click the 'Message' button on their card to start a real-time chat.",
  },
];


export default function HelpPage() {
  return (
    <div className="max-w-4xl mx-auto">
        <Card className="shadow-lg">
            <CardHeader>
                <div className="flex items-center gap-4">
                    <LifeBuoy className="h-8 w-8 text-primary" />
                    <div>
                        <CardTitle className="text-2xl font-headline">Help & Frequently Asked Questions</CardTitle>
                        <CardDescription>
                            Find answers to common questions about using the EduPerform platform.
                        </CardDescription>
                    </div>
                </div>
            </CardHeader>
            <CardContent>
                <Accordion type="single" collapsible className="w-full">
                    {faqs.map((faq, index) => (
                        <AccordionItem value={`item-${index}`} key={index}>
                            <AccordionTrigger className="text-left font-semibold">{faq.question}</AccordionTrigger>
                            <AccordionContent className="text-muted-foreground">
                                {faq.answer}
                            </AccordionContent>
                        </AccordionItem>
                    ))}
                </Accordion>
            </CardContent>
        </Card>
    </div>
  )
}
