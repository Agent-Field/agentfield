import { AgentRouter } from '@agentfield/sdk';
import { z } from 'zod';

// Group related reasoners with a router
export const reasonersRouter = new AgentRouter({ prefix: 'demo', tags: ['example'] });

reasonersRouter.reasoner<{ message: string }, { original: string; echoed: string; length: number }>(
  'echo',
  async (ctx) => {
    /**
     * Simple echo reasoner - works without AI configured.
     *
     * Example usage:
     * curl -X POST http://localhost:8080/api/v1/execute/init-example.demo_echo \
     *   -H "Content-Type: application/json" \
     *   -d '{"input": {"message": "Hello World"}}'
     */
    const message = ctx.input.message ?? '';
    return {
      original: message,
      echoed: message,
      length: message.length
    };
  }
);

const sentimentSchema = z.object({
  sentiment: z.enum(['positive', 'negative', 'neutral']),
  confidence: z.number().min(0).max(1),
  reasoning: z.string()
});
type SentimentResult = z.infer<typeof sentimentSchema>;

const analyzeSentimentOutputSchema = sentimentSchema.extend({
  text: z.string()
});

reasonersRouter.reasoner<{ text: string }, SentimentResult & { text: string }>(
  'analyzeSentiment',
  async (ctx) => {
    /**
     * AI-powered sentiment analysis with structured output.
     *
     * Example usage:
     * curl -X POST http://localhost:8080/api/v1/execute/init-example.demo_analyzeSentiment \
     *   -H "Content-Type: application/json" \
     *   -d '{"input": {"text": "I love this product!"}}'
     */
    // Add a note at the start of processing
    ctx.note('Starting sentiment analysis', ['debug', 'processing']);

    const raw = await ctx.ai(
      `You are a sentiment analysis expert. Analyze the sentiment of this text and respond ONLY as strict JSON, no code fences or prose.`,
      {
        schema: sentimentSchema,
        temperature: 0.2
      }
    );

    // Handle either structured object (preferred) or string fallback
    const parsed =
      typeof raw === 'string'
        ? JSON.parse((raw as string).replace(/```(json)?/gi, '').trim())
        : raw;
    const sentiment = sentimentSchema.parse(parsed);

    // Add a note with the analysis result
    ctx.note(`Sentiment analysis completed: ${sentiment.sentiment} (confidence: ${sentiment.confidence.toFixed(2)})`, [
      'info',
      'sentiment'
    ]);

    // Add a note for observability
    await ctx.workflow.progress(100, {
      status: 'succeeded',
      result: { sentiment: sentiment.sentiment, confidence: sentiment.confidence }
    });

    return { ...sentiment, text: ctx.input.text };
  },
  { outputSchema: analyzeSentimentOutputSchema }
);

const processWithNotesOutputSchema = z.object({
  processed: z.number(),
  notes: z.number()
});

reasonersRouter.reasoner<{ items: string[] }, { processed: number; notes: number }>(
  'processWithNotes',
  async (ctx) => {
    /**
     * Example reasoner demonstrating the note() method for fire-and-forget execution logging.
     *
     * Example usage:
     * curl -X POST http://localhost:8080/api/v1/execute/init-example.demo_processWithNotes \
     *   -H "Content-Type: application/json" \
     *   -d '{"input": {"items": ["item1", "item2", "item3"]}}'
     */
    const items = ctx.input.items ?? [];
    let notesSent = 0;

    ctx.note(`Processing ${items.length} items`, ['debug', 'start']);
    notesSent++;

    const processed: string[] = [];
    for (let i = 0; i < items.length; i++) {
      const item = items[i];
      ctx.note(`Processing item ${i + 1}/${items.length}: ${item}`, ['debug', 'item-processing']);
      notesSent++;
      await new Promise((resolve) => setTimeout(resolve, 10));
      processed.push(item);
    }

    ctx.note(`Successfully processed ${processed.length} items`, ['info', 'completion']);
    notesSent++;

    return {
      processed: processed.length,
      notes: notesSent
    };
  },
  { outputSchema: processWithNotesOutputSchema }
);