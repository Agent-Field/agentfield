import { AgentRouter } from '@agentfield/sdk';
import { z } from 'zod';

/**
 * Verifiable Credentials Reasoners
 *
 * Each reasoner demonstrates different VC generation patterns:
 * 1. Basic VC generation with explicit call
 * 2. AI-powered analysis with VC audit trail
 * 3. Data transformation with integrity proof
 * 4. Multi-step workflow with chained VCs
 */

export const reasonersRouter = new AgentRouter({
  prefix: 'vc',
  tags: ['verifiable-credentials', 'demo'],
});

// ============================================================================
// Reasoner 1: Basic Processing with VC
// ============================================================================

interface ProcessInput {
  text: string;
  metadata?: Record<string, any>;
}

interface ProcessOutput {
  processed: string;
  wordCount: number;
  timestamp: string;
  vcGenerated: boolean;
  vcId?: string;
}

reasonersRouter.reasoner<ProcessInput, ProcessOutput>(
  'process',
  async (ctx) => {
    /**
     * Basic text processing with explicit VC generation.
     *
     * This demonstrates the fundamental VC flow:
     * 1. Process the input
     * 2. Generate output
     * 3. Create a VC that cryptographically attests to the execution
     *
     * Example:
     *   curl -X POST http://localhost:8080/api/v1/execute/vc-demo.vc_process \
     *     -H "Content-Type: application/json" \
     *     -d '{"input": {"text": "Hello, Verifiable World!"}}'
     */
    const startTime = Date.now();
    const text = ctx.input.text ?? '';

    // Process the input
    const processed = text.toUpperCase();
    const wordCount = text.split(/\s+/).filter(Boolean).length;
    const timestamp = new Date().toISOString();

    const result: ProcessOutput = {
      processed,
      wordCount,
      timestamp,
      vcGenerated: false,
    };

    // Generate Verifiable Credential for this execution
    try {
      const credential = await ctx.did.generateCredential({
        inputData: ctx.input,
        outputData: result,
        status: 'succeeded',
        durationMs: Date.now() - startTime,
      });

      result.vcGenerated = true;
      result.vcId = credential.vcId;

      console.log(`[VC] Generated credential: ${credential.vcId}`);
      console.log(`[VC] Status: ${credential.status}`);
      console.log(`[VC] Input Hash: ${credential.inputHash}`);
      console.log(`[VC] Output Hash: ${credential.outputHash}`);
    } catch (error) {
      console.error('[VC] Failed to generate credential:', error);
      // Continue without VC - the execution still succeeds
    }

    return result;
  },
  {
    description: 'Basic text processing with Verifiable Credential generation',
    inputSchema: {
      type: 'object',
      properties: {
        text: { type: 'string', description: 'Text to process' },
        metadata: { type: 'object', description: 'Optional metadata' },
      },
      required: ['text'],
    },
    outputSchema: {
      type: 'object',
      properties: {
        processed: { type: 'string' },
        wordCount: { type: 'number' },
        timestamp: { type: 'string' },
        vcGenerated: { type: 'boolean' },
        vcId: { type: 'string' },
      },
    },
    tags: ['vc', 'processing'],
  }
);

// ============================================================================
// Reasoner 2: AI Analysis with VC Audit Trail
// ============================================================================

const analysisSchema = z.object({
  sentiment: z.enum(['positive', 'negative', 'neutral', 'mixed']),
  confidence: z.number().min(0).max(1),
  topics: z.array(z.string()),
  summary: z.string(),
});

type AnalysisResult = z.infer<typeof analysisSchema>;

interface AnalyzeInput {
  text: string;
  analyzeTopics?: boolean;
}

interface AnalyzeOutput extends AnalysisResult {
  originalText: string;
  vcGenerated: boolean;
  vcId?: string;
  analysisTimestamp: string;
}

reasonersRouter.reasoner<AnalyzeInput, AnalyzeOutput>(
  'analyze',
  async (ctx) => {
    /**
     * AI-powered text analysis with VC audit trail.
     *
     * This demonstrates how VCs provide accountability for AI decisions:
     * - The VC records the exact input that was analyzed
     * - The VC records the AI's output (hashed for privacy)
     * - External auditors can verify the execution occurred
     *
     * Example:
     *   curl -X POST http://localhost:8080/api/v1/execute/vc-demo.vc_analyze \
     *     -H "Content-Type: application/json" \
     *     -d '{"input": {"text": "I love this new feature! It makes everything so much easier.", "analyzeTopics": true}}'
     */
    const startTime = Date.now();
    const text = ctx.input.text ?? '';

    // Report progress
    await ctx.workflow.progress(10, { status: 'Starting analysis' });

    let analysis: AnalysisResult;

    try {
      // Use AI to analyze the text
      const raw = await ctx.ai(
        `Analyze the following text and provide sentiment, confidence (0-1), key topics, and a brief summary.

Text: "${text}"

Respond as JSON only, no markdown.`,
        {
          schema: analysisSchema,
          temperature: 0.3,
        }
      );

      analysis = raw;
    } catch (aiError) {
      // Fallback for when AI is not configured
      console.warn('[AI] AI analysis failed, using fallback:', aiError);
      analysis = {
        sentiment: 'neutral',
        confidence: 0.5,
        topics: ['text', 'content'],
        summary: `Analyzed ${text.split(/\s+/).length} words of content.`,
      };
    }

    await ctx.workflow.progress(80, { status: 'Analysis complete, generating VC' });

    const result: AnalyzeOutput = {
      ...analysis,
      originalText: text,
      vcGenerated: false,
      analysisTimestamp: new Date().toISOString(),
    };

    // Generate VC for the AI analysis
    try {
      const credential = await ctx.did.generateCredential({
        inputData: { text, analyzeTopics: ctx.input.analyzeTopics },
        outputData: {
          sentiment: analysis.sentiment,
          confidence: analysis.confidence,
          topicCount: analysis.topics.length,
        },
        status: 'succeeded',
        durationMs: Date.now() - startTime,
      });

      result.vcGenerated = true;
      result.vcId = credential.vcId;

      console.log(`[VC] AI Analysis VC: ${credential.vcId}`);
    } catch (error) {
      console.error('[VC] Failed to generate analysis VC:', error);
    }

    await ctx.workflow.progress(100, {
      status: 'succeeded',
      result: { sentiment: analysis.sentiment, vcId: result.vcId },
    });

    return result;
  },
  {
    description: 'AI-powered text analysis with Verifiable Credential audit trail',
    tags: ['vc', 'ai', 'analysis'],
  }
);

// ============================================================================
// Reasoner 3: Data Transformation with Integrity Proof
// ============================================================================

interface TransformInput {
  data: Record<string, any>;
  operations: Array<'uppercase' | 'lowercase' | 'trim' | 'sort'>;
}

interface TransformOutput {
  original: Record<string, any>;
  transformed: Record<string, any>;
  operationsApplied: string[];
  vcGenerated: boolean;
  vcId?: string;
  integrityProof: {
    inputHash?: string;
    outputHash?: string;
    verifiable: boolean;
  };
}

reasonersRouter.reasoner<TransformInput, TransformOutput>(
  'transform',
  async (ctx) => {
    /**
     * Data transformation with integrity proof via VC.
     *
     * This demonstrates using VCs for data integrity:
     * - Proves the transformation was applied correctly
     * - Input and output hashes allow verification
     * - Useful for compliance and data lineage tracking
     *
     * Example:
     *   curl -X POST http://localhost:8080/api/v1/execute/vc-demo.vc_transform \
     *     -H "Content-Type: application/json" \
     *     -d '{"input": {"data": {"name": "  John Doe  ", "items": ["banana", "apple"]}, "operations": ["trim", "sort"]}}'
     */
    const startTime = Date.now();
    const { data, operations } = ctx.input;

    // Deep clone the data
    let transformed = JSON.parse(JSON.stringify(data));
    const appliedOps: string[] = [];

    // Apply transformations
    const transform = (obj: any): any => {
      if (typeof obj === 'string') {
        let result = obj;
        if (operations.includes('uppercase')) result = result.toUpperCase();
        if (operations.includes('lowercase')) result = result.toLowerCase();
        if (operations.includes('trim')) result = result.trim();
        return result;
      }
      if (Array.isArray(obj)) {
        let arr = obj.map(transform);
        if (operations.includes('sort')) arr = arr.sort();
        return arr;
      }
      if (typeof obj === 'object' && obj !== null) {
        const result: Record<string, any> = {};
        for (const [key, value] of Object.entries(obj)) {
          result[key] = transform(value);
        }
        return result;
      }
      return obj;
    };

    transformed = transform(transformed);
    operations.forEach((op) => appliedOps.push(op));

    const result: TransformOutput = {
      original: data,
      transformed,
      operationsApplied: appliedOps,
      vcGenerated: false,
      integrityProof: {
        verifiable: false,
      },
    };

    // Generate VC with integrity proof
    try {
      const credential = await ctx.did.generateCredential({
        inputData: { data, operations },
        outputData: transformed,
        status: 'succeeded',
        durationMs: Date.now() - startTime,
      });

      result.vcGenerated = true;
      result.vcId = credential.vcId;
      result.integrityProof = {
        inputHash: credential.inputHash,
        outputHash: credential.outputHash,
        verifiable: true,
      };

      console.log(`[VC] Transform VC: ${credential.vcId}`);
      console.log(`[VC] Input Hash: ${credential.inputHash}`);
      console.log(`[VC] Output Hash: ${credential.outputHash}`);
    } catch (error) {
      console.error('[VC] Failed to generate transform VC:', error);
    }

    return result;
  },
  {
    description: 'Data transformation with Verifiable Credential integrity proof',
    tags: ['vc', 'transform', 'integrity'],
  }
);

// ============================================================================
// Reasoner 4: Multi-Step Workflow with Chained VCs
// ============================================================================

interface ChainInput {
  text: string;
  steps: Array<'validate' | 'process' | 'enrich' | 'finalize'>;
}

interface StepResult {
  step: string;
  success: boolean;
  vcId?: string;
  output: any;
}

interface ChainOutput {
  input: string;
  stepsExecuted: StepResult[];
  finalResult: any;
  vcChain: string[];
  workflowVcGenerated: boolean;
}

reasonersRouter.reasoner<ChainInput, ChainOutput>(
  'chain',
  async (ctx) => {
    /**
     * Multi-step workflow with chained VCs.
     *
     * This demonstrates how VCs create an audit trail across workflow steps:
     * - Each step generates its own VC
     * - VCs are chained via workflow/execution IDs
     * - The final workflow VC aggregates all steps
     *
     * Example:
     *   curl -X POST http://localhost:8080/api/v1/execute/vc-demo.vc_chain \
     *     -H "Content-Type: application/json" \
     *     -d '{"input": {"text": "Process this through multiple steps", "steps": ["validate", "process", "enrich", "finalize"]}}'
     */
    const startTime = Date.now();
    const { text, steps } = ctx.input;

    const stepsExecuted: StepResult[] = [];
    const vcChain: string[] = [];
    let currentData: any = { text, validated: false };

    // Execute each step
    for (let i = 0; i < steps.length; i++) {
      const step = steps[i];
      const stepStart = Date.now();

      await ctx.workflow.progress(Math.round(((i + 1) / steps.length) * 80), {
        status: `Executing step: ${step}`,
      });

      // Simulate step processing
      let stepOutput: any;
      switch (step) {
        case 'validate':
          stepOutput = {
            ...currentData,
            validated: true,
            validatedAt: new Date().toISOString(),
          };
          break;
        case 'process':
          stepOutput = {
            ...currentData,
            processed: currentData.text?.toUpperCase() ?? '',
            wordCount: (currentData.text ?? '').split(/\s+/).filter(Boolean).length,
          };
          break;
        case 'enrich':
          stepOutput = {
            ...currentData,
            enriched: true,
            metadata: {
              source: 'vc-chain-demo',
              version: '1.0',
              stepIndex: i,
            },
          };
          break;
        case 'finalize':
          stepOutput = {
            ...currentData,
            finalized: true,
            completedAt: new Date().toISOString(),
            totalSteps: steps.length,
          };
          break;
        default:
          stepOutput = currentData;
      }

      const stepResult: StepResult = {
        step,
        success: true,
        output: stepOutput,
      };

      // Generate VC for this step
      try {
        const credential = await ctx.did.generateCredential({
          inputData: { step, input: currentData },
          outputData: stepOutput,
          status: 'succeeded',
          durationMs: Date.now() - stepStart,
        });

        stepResult.vcId = credential.vcId;
        vcChain.push(credential.vcId);

        console.log(`[VC] Step "${step}" VC: ${credential.vcId}`);
      } catch (error) {
        console.error(`[VC] Failed to generate VC for step "${step}":`, error);
      }

      currentData = stepOutput;
      stepsExecuted.push(stepResult);
    }

    await ctx.workflow.progress(90, {
      status: 'Generating final workflow VC',
    });

    const result: ChainOutput = {
      input: text,
      stepsExecuted,
      finalResult: currentData,
      vcChain,
      workflowVcGenerated: false,
    };

    // Generate final workflow-level VC
    try {
      const workflowCredential = await ctx.did.generateCredential({
        inputData: { text, steps, totalSteps: steps.length },
        outputData: {
          stepsCompleted: stepsExecuted.length,
          allSuccessful: stepsExecuted.every((s) => s.success),
          vcChainLength: vcChain.length,
        },
        status: 'succeeded',
        durationMs: Date.now() - startTime,
      });

      vcChain.push(workflowCredential.vcId);
      result.workflowVcGenerated = true;

      console.log(`[VC] Workflow VC: ${workflowCredential.vcId}`);
      console.log(`[VC] Total VCs in chain: ${vcChain.length}`);
    } catch (error) {
      console.error('[VC] Failed to generate workflow VC:', error);
    }

    await ctx.workflow.progress(100, {
      status: 'succeeded',
      result: { stepsCompleted: stepsExecuted.length, vcCount: vcChain.length },
    });

    return result;
  },
  {
    description: 'Multi-step workflow with chained Verifiable Credentials',
    tags: ['vc', 'workflow', 'chain'],
  }
);
