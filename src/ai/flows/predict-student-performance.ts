'use server';

/**
 * @fileOverview This file defines a Genkit flow for predicting student performance based on various input features.
 *
 * - predictStudentPerformance - A function that takes student performance data as input and returns a predicted performance score.
 * - PredictStudentPerformanceInput - The input type for the predictStudentPerformance function.
 * - PredictStudentPerformanceOutput - The return type for the predictStudentPerformance function.
 */

import {ai} from '@/ai/genkit';
import {z} from 'genkit';

const PredictStudentPerformanceInputSchema = z.object({
  assessmentScore: z
    .number()
    .describe('The student score on assessments (0-100).'),
  projectScore: z
    .number()
    .describe('The student score on projects (0-100).'),
  feedbackScore: z
    .number()
    .describe('The student score based on feedback (0-100).'),
  efficiency: z
    .number()
    .describe('A measure of student efficiency in completing tasks (0-100).'),
});
export type PredictStudentPerformanceInput = z.infer<
  typeof PredictStudentPerformanceInputSchema
>;

const PredictStudentPerformanceOutputSchema = z.object({
  predictedPerformance: z
    .number()
    .describe(
      'The predicted overall performance score of the student (0-100).'
    ),
  areasForImprovement: z
    .string()
    .describe(
      'A textual description of areas where the student can improve.'
    ),
});
export type PredictStudentPerformanceOutput = z.infer<
  typeof PredictStudentPerformanceOutputSchema
>;

export async function predictStudentPerformance(
  input: PredictStudentPerformanceInput
): Promise<PredictStudentPerformanceOutput> {
  return predictStudentPerformanceFlow(input);
}

const prompt = ai.definePrompt({
  name: 'predictStudentPerformancePrompt',
  input: {schema: PredictStudentPerformanceInputSchema},
  output: {schema: PredictStudentPerformanceOutputSchema},
  prompt: `You are an AI assistant designed to predict student performance.

  Based on the student's assessment score ({{assessmentScore}}), project score ({{projectScore}}), feedback score ({{feedbackScore}}), and efficiency ({{efficiency}}), predict the student's overall performance score (out of 100).

  Also, identify and describe areas where the student can improve, considering their scores and efficiency.
  Be encouraging and specific in your recommendations.

  Output the predicted performance and areas for improvement in the specified JSON format.`,
});

const predictStudentPerformanceFlow = ai.defineFlow(
  {
    name: 'predictStudentPerformanceFlow',
    inputSchema: PredictStudentPerformanceInputSchema,
    outputSchema: PredictStudentPerformanceOutputSchema,
  },
  async input => {
    const {output} = await prompt(input);
    return output!;
  }
);
