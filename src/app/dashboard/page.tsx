import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { BarChart, CheckCircle2, AlertTriangle, Target, Activity } from "lucide-react";
import { PerformanceChart } from './performance-chart';

const kpiData = [
  { title: "Overall Performance", value: "88%", icon: <BarChart className="h-6 w-6 text-primary" />, change: "+5% this month" },
  { title: "Assessment Score", value: "92%", icon: <CheckCircle2 className="h-6 w-6 text-green-500" />, change: "Avg. Score" },
  { title: "Project Score", value: "85%", icon: <CheckCircle2 className="h-6 w-6 text-green-500" />, change: "Avg. Score" },
  { title: "Efficiency", value: "95%", icon: <Activity className="h-6 w-6 text-primary" />, change: "Tasks on time" },
];

export default function DashboardPage() {
  return (
    <div className="space-y-6">
      <div className="grid gap-6 md:grid-cols-2 lg:grid-cols-4">
        {kpiData.map((kpi) => (
          <Card key={kpi.title} className="shadow-md hover:shadow-lg transition-shadow">
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium">{kpi.title}</CardTitle>
              {kpi.icon}
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">{kpi.value}</div>
              <p className="text-xs text-muted-foreground">{kpi.change}</p>
            </CardContent>
          </Card>
        ))}
      </div>

      <div className="grid gap-6 lg:grid-cols-2">
        <Card className="shadow-md">
          <CardHeader>
            <CardTitle className="flex items-center gap-2"><Target className="text-green-500"/> Strengths</CardTitle>
          </CardHeader>
          <CardContent>
            <ul className="list-disc list-inside space-y-2 text-muted-foreground">
                <li>Excellent problem-solving skills demonstrated in project work.</li>
                <li>Consistently high scores in programming-related assessments.</li>
                <li>Proactive participation in class discussions and feedback sessions.</li>
            </ul>
          </CardContent>
        </Card>
        <Card className="shadow-md">
          <CardHeader>
            <CardTitle className="flex items-center gap-2"><AlertTriangle className="text-yellow-500" /> Areas for Improvement</CardTitle>
          </CardHeader>
          <CardContent>
             <ul className="list-disc list-inside space-y-2 text-muted-foreground">
                <li>Focus on time management for larger assessment deadlines.</li>
                <li>Improve documentation clarity in project submissions.</li>
                <li>Engage more with theoretical concepts during lectures.</li>
            </ul>
          </CardContent>
        </Card>
      </div>

       <Card className="shadow-md col-span-1 lg:col-span-2">
          <CardHeader>
            <CardTitle>Performance Breakdown</CardTitle>
          </CardHeader>
          <CardContent>
            <PerformanceChart />
          </CardContent>
        </Card>
    </div>
  );
}
