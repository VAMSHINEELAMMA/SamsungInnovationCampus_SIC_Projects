'use server';

/**
 * @fileOverview This file defines a Genkit flow for analyzing student feedback and generating suggestions for improvement.
 *
 * - analyzeStudentFeedback - A function that takes student feedback and returns improvement suggestions.
 * - AnalyzeFeedbackInput - The input type for the analyzeStudentFeedback function.
 * - AnalyzeFeedbackOutput - The return type for the analyzeStudentFeedback function.
 */

import {ai} from '@/ai/genkit';
import {z} from 'genkit';

const AnalyzeFeedbackInputSchema = z.object({
  subject: z.string().describe('The course or assessment the feedback is for.'),
  rating: z.enum(['Like', 'Dislike']).describe('The student rating.'),
  comments: z.string().describe('The detailed comments from the student.'),
});
export type AnalyzeFeedbackInput = z.infer<typeof AnalyzeFeedbackInputSchema>;

const AnalyzeFeedbackOutputSchema = z.object({
  suggestions: z
    .string()
    .describe(
      'Actionable suggestions for the faculty on how to improve the course or assessment.'
    ),
});
export type AnalyzeFeedbackOutput = z.infer<
  typeof AnalyzeFeedbackOutputSchema
>;

export async function analyzeStudentFeedback(
  input: AnalyzeFeedbackInput
): Promise<AnalyzeFeedbackOutput> {
  return analyzeFeedbackFlow(input);
}

const prompt = ai.definePrompt({
  name: 'analyzeFeedbackPrompt',
  input: {schema: AnalyzeFeedbackInputSchema},
  output: {schema: AnalyzeFeedbackOutputSchema},
  prompt: `You are an expert educational consultant. A student has provided feedback on "{{subject}}".

  Student Rating: {{rating}}
  Student Comments:
  "{{comments}}"

  Based on this feedback, provide concise, actionable suggestions for the faculty to improve this course/assessment.
  Focus on constructive advice. Address the student's concerns directly if possible.

  Output the suggestions in the specified JSON format.`,
});

const analyzeFeedbackFlow = ai.defineFlow(
  {
    name: 'analyzeFeedbackFlow',
    inputSchema: AnalyzeFeedbackInputSchema,
    outputSchema: AnalyzeFeedbackOutputSchema,
  },
  async input => {
    const {output} = await prompt(input);
    return output!;
  }
);
