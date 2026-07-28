import { describe, it, expect } from 'vitest';

import {
  MODEL_VARIANT_SEP,
  resolveModelAndVariant,
  splitModelVariant,
} from '../src/harness/modelVariant.js';

describe('splitModelVariant', () => {
  it('parses a #variant suffix into model and variant', () => {
    expect(splitModelVariant('openrouter/z-ai/glm-5.2#high')).toEqual({
      model: 'openrouter/z-ai/glm-5.2',
      variant: 'high',
    });
  });

  it('passes a bare model through with no variant', () => {
    expect(splitModelVariant('deepseek/deepseek-v4-flash')).toEqual({
      model: 'deepseek/deepseek-v4-flash',
      variant: undefined,
    });
  });

  it('handles missing, empty, or non-string model', () => {
    expect(splitModelVariant(undefined)).toEqual({});
    expect(splitModelVariant(null)).toEqual({});
    expect(splitModelVariant('')).toEqual({});
    expect(splitModelVariant('  ')).toEqual({});
    expect(splitModelVariant(42)).toEqual({});
  });

  it('treats empty halves around the separator as absent', () => {
    expect(splitModelVariant('model#')).toEqual({ model: 'model', variant: undefined });
    expect(splitModelVariant('#high')).toEqual({ model: undefined, variant: 'high' });
  });

  it('splits on the first separator only', () => {
    expect(splitModelVariant('a#b#c')).toEqual({ model: 'a', variant: 'b#c' });
  });

  it('trims whitespace around both halves', () => {
    expect(splitModelVariant(' openai/gpt-5 # high ')).toEqual({
      model: 'openai/gpt-5',
      variant: 'high',
    });
  });

  it('uses # as the separator', () => {
    expect(MODEL_VARIANT_SEP).toBe('#');
  });
});

describe('resolveModelAndVariant', () => {
  it('lets an explicit variant option win over the suffix', () => {
    expect(resolveModelAndVariant({ model: 'openai/gpt-5#low', variant: 'max' })).toEqual({
      model: 'openai/gpt-5',
      variant: 'max',
    });
  });

  it('uses the suffix when no explicit option is set', () => {
    expect(resolveModelAndVariant({ model: 'openai/gpt-5#minimal' })).toEqual({
      model: 'openai/gpt-5',
      variant: 'minimal',
    });
  });

  it('ignores a whitespace-only explicit variant', () => {
    expect(resolveModelAndVariant({ model: 'openai/gpt-5#low', variant: '  ' })).toEqual({
      model: 'openai/gpt-5',
      variant: 'low',
    });
  });

  it('returns nothing for empty options', () => {
    expect(resolveModelAndVariant({})).toEqual({ model: undefined, variant: undefined });
  });
});
