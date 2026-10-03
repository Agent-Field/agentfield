/**
 * Configurable templates for the text and tool-message framing the SDK injects
 * into LLM calls (issue #229).
 *
 * Field names line up with the Python SDK's `PromptTemplates` and the Go SDK's
 * `PromptConfig` (`toolCallLimitReached`, `toolErrorFormatter`,
 * `toolResultFormatter`). Defaults reproduce the SDK's prior output exactly, so
 * a reasoner that never sets `promptTemplates` sees no change.
 *
 * The TypeScript SDK uses the Vercel AI SDK's native structured output
 * (`generateObject`) and injects no schema instruction, so, unlike Python,
 * there is no `schemaInstruction` field here.
 */

/** Formats a failed tool call before it is returned to the model. */
export type ToolErrorFormatter = (toolName: string, error: string) => unknown;

/** Formats a successful tool call before it is returned to the model. */
export type ToolResultFormatter = (toolName: string, result: unknown) => unknown;

export interface PromptTemplates {
  /**
   * Optional system prompt describing how to use tools, appended after the
   * caller's system prompt. Undefined means nothing is injected (the default),
   * so existing `tools=` calls are unchanged; set it to opt in.
   */
  toolSystemPrompt?: string;
  /** Message returned to the model when the tool-call limit is reached. */
  toolCallLimitReached?: string;
  /**
   * Formats a failed tool call `(toolName, error)` before it is sent back to
   * the model. Return a string to send verbatim, or any value to be returned
   * as-is (the AI SDK serializes it).
   */
  toolErrorFormatter?: ToolErrorFormatter;
  /**
   * Formats a successful tool call `(toolName, result)` before it is sent back
   * to the model. Return a string to send verbatim, or any value to be
   * returned as-is.
   */
  toolResultFormatter?: ToolResultFormatter;
}

export const DEFAULT_TOOL_CALL_LIMIT_REACHED =
  'Tool call limit reached. Please provide a final response.';

/** Default tool-error framing: the prior `{ error, tool }` object. */
export const defaultToolErrorFormatter: ToolErrorFormatter = (toolName, error) => ({
  error,
  tool: toolName,
});

/** Default tool-result framing: the raw result, unframed. */
export const defaultToolResultFormatter: ToolResultFormatter = (_toolName, result) =>
  result;

/** A fully-populated set of templates with every default filled in. */
export interface ResolvedPromptTemplates {
  toolSystemPrompt?: string;
  toolCallLimitReached: string;
  toolErrorFormatter: ToolErrorFormatter;
  toolResultFormatter: ToolResultFormatter;
}

/**
 * Merges a partial override over the built-in defaults. An undefined field
 * (or an undefined `overrides`) keeps the default; `toolSystemPrompt` stays
 * undefined unless explicitly set.
 */
export function resolvePromptTemplates(
  overrides?: Partial<PromptTemplates>
): ResolvedPromptTemplates {
  return {
    toolSystemPrompt: overrides?.toolSystemPrompt,
    toolCallLimitReached:
      overrides?.toolCallLimitReached ?? DEFAULT_TOOL_CALL_LIMIT_REACHED,
    toolErrorFormatter: overrides?.toolErrorFormatter ?? defaultToolErrorFormatter,
    toolResultFormatter:
      overrides?.toolResultFormatter ?? defaultToolResultFormatter,
  };
}

/**
 * Cross-SDK message-source taxonomy, shared with the Python and Go SDKs so a
 * trace tagged in one SDK reads the same in another (issue #229).
 * `sdk.schema_instruction` exists only in the Python SDK (which injects it;
 * TypeScript uses native structured output).
 */
export type MessageSource =
  | 'user'
  | 'assistant'
  | 'sdk.tool_system_prompt'
  | 'sdk.tool_result'
  | 'sdk.tool_error'
  | 'sdk.tool_limit';

export const TRACE_SOURCE_USER = 'user';
export const TRACE_SOURCE_ASSISTANT = 'assistant';
export const TRACE_SOURCE_TOOL_SYSTEM_PROMPT = 'sdk.tool_system_prompt';
export const TRACE_SOURCE_TOOL_RESULT = 'sdk.tool_result';
export const TRACE_SOURCE_TOOL_ERROR = 'sdk.tool_error';
export const TRACE_SOURCE_TOOL_LIMIT = 'sdk.tool_limit';

/**
 * Tags a message the loop sent with where it came from. `message` references
 * the wire content (not a reshaped copy); `source` uses the taxonomy above.
 */
export interface TracedMessage {
  message: unknown;
  source: MessageSource;
}
