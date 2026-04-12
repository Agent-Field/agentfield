import { afterEach, describe, expect, it, vi } from "vitest";
import { ZodError } from "zod";

import type { JsonSchema } from "@/types/execution";
import * as jsonSchemaToZodModule from "./jsonSchemaToZod";
import {
  generateExampleData,
  schemaToFormFields,
  validateFormData,
  validateValueAgainstSchema,
} from "./schemaUtils";

describe("schemaToFormFields", () => {
  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("returns fields with inferred types, placeholders, metadata, and variant titles", () => {
    const schema: JsonSchema = {
      properties: {
        userName: {
          type: ["null", "string"],
          description: "Primary username",
          default: "alice",
          example: "ally",
        },
        favorite_color: {
          enum: ["red", "blue"],
          description: "Preferred color",
        },
        tags: {
          type: "array",
          items: { type: "string" },
          minItems: 1,
          maxItems: 3,
        },
        tupleValue: {
          type: "array",
          items: [{ type: "string" }, { type: "integer" }],
          additionalItems: false,
        },
        settings: {
          properties: {
            mode: { const: "safe" },
          },
        },
        endpoint: {
          format: "uri",
          examples: ["https://api.example.com"],
        },
        variant: {
          oneOf: [
            { title: "Named Variant", type: "string" },
            { description: "Described Variant", type: "number" },
            { type: "boolean" },
          ],
        },
      },
      required: ["userName", "tags"],
    };

    const fields = schemaToFormFields(schema);

    expect(fields).toHaveLength(7);

    expect(fields[0]).toMatchObject({
      name: "userName",
      label: "User Name",
      type: "string",
      required: true,
      description: "Primary username",
      placeholder: "alice",
      defaultValue: "alice",
      examples: ["ally"],
    });

    expect(fields[1]).toMatchObject({
      name: "favorite_color",
      label: "Favorite Color",
      type: "select",
      options: ["red", "blue"],
      enumValues: ["red", "blue"],
      placeholder: "red",
    });

    expect(fields[2]).toMatchObject({
      name: "tags",
      type: "array",
      required: true,
      itemSchema: { type: "string" },
      tupleSchemas: undefined,
      minItems: 1,
      maxItems: 3,
      placeholder: "Add items...",
    });

    expect(fields[3]).toMatchObject({
      name: "tupleValue",
      type: "array",
      tupleSchemas: [{ type: "string" }, { type: "integer" }],
      itemSchema: null,
    });

    expect(fields[4]).toMatchObject({
      name: "settings",
      type: "object",
      placeholder: "Configure object...",
    });

    expect(fields[5]).toMatchObject({
      name: "endpoint",
      type: "string",
      placeholder: "https://api.example.com",
      examples: ["https://api.example.com"],
      format: "uri",
    });

    expect(fields[6]).toMatchObject({
      name: "variant",
      type: "string",
      combinator: "oneOf",
      variantTitles: ["Named Variant", "Described Variant", "Variant 3"],
      placeholder: "Enter value...",
    });
  });

  it("supports parent paths and infers top-level object schemas without an explicit type", () => {
    const fields = schemaToFormFields(
      {
        properties: {
          childField: { type: "number" },
        },
      },
      "parent",
    );

    expect(fields).toEqual([
      expect.objectContaining({
        name: "parent.childField",
        label: "Child Field",
        type: "number",
        placeholder: "Enter number...",
      }),
    ]);
  });

  it("returns an empty list for invalid or non-object schemas and warns for invalid properties", () => {
    const warnSpy = vi.spyOn(console, "warn").mockImplementation(() => {});

    expect(schemaToFormFields(null as unknown as JsonSchema)).toEqual([]);
    expect(schemaToFormFields({ type: "string" })).toEqual([]);
    expect(
      schemaToFormFields({
        type: "object",
        properties: {
          valid: { type: "boolean" },
          broken: null as unknown as JsonSchema,
        },
      }),
    ).toEqual([
      expect.objectContaining({
        name: "valid",
        type: "boolean",
        placeholder: "",
      }),
    ]);

    expect(warnSpy).toHaveBeenCalledTimes(2);
  });
});

describe("validateFormData", () => {
  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("passes valid data through zod-backed validation", () => {
    const result = validateFormData(
      { userName: "alice", count: 3, enabled: true },
      {
        type: "object",
        properties: {
          userName: { type: "string" },
          count: { type: "integer" },
          enabled: { type: "boolean" },
        },
        required: ["userName", "count"],
      },
    );

    expect(result).toEqual({ isValid: true, errors: [] });
  });

  it("formats zod validation issues into human-readable paths", () => {
    const result = validateFormData(
      { userName: 123, profile: { emailAddress: 9 } },
      {
        type: "object",
        properties: {
          userName: { type: "string" },
          profile: {
            type: "object",
            properties: {
              emailAddress: { type: "string" },
            },
            required: ["emailAddress"],
          },
        },
        required: ["userName", "profile"],
      },
    );

    expect(result.isValid).toBe(false);
    expect(result.errors).toContain("User Name: Invalid input: expected string, received number");
    expect(result.errors).toContain(
      "Profile › Email Address: Invalid input: expected string, received number",
    );
  });

  it("returns permissive success for invalid schemas and unexpected validator failures", () => {
    const warnSpy = vi.spyOn(console, "warn").mockImplementation(() => {});
    const validatorSpy = vi
      .spyOn(jsonSchemaToZodModule, "jsonSchemaToZodObject")
      .mockImplementationOnce(() => {
        throw new Error("unexpected");
      });

    expect(validateFormData({ any: "thing" }, null as unknown as JsonSchema)).toEqual({
      isValid: true,
      errors: [],
    });
    expect(validateFormData({ any: "thing" }, { type: "object", properties: {} })).toEqual({
      isValid: true,
      errors: [],
    });

    expect(validatorSpy).toHaveBeenCalledTimes(1);
    expect(warnSpy).toHaveBeenCalledTimes(1);
  });

  it("formats issues when a ZodError is thrown directly", () => {
    vi.spyOn(jsonSchemaToZodModule, "jsonSchemaToZodObject").mockImplementationOnce(() => {
      throw new ZodError([
        {
          code: "custom",
          path: ["request_id"],
          message: "is invalid",
        },
      ]);
    });

    expect(validateFormData({}, { type: "object", properties: {} })).toEqual({
      isValid: false,
      errors: ["Request Id: is invalid"],
    });
  });
});

describe("validateValueAgainstSchema", () => {
  it("validates strings, numbers, booleans, arrays, objects, enums, and const values", () => {
    expect(
      validateValueAgainstSchema("ab", {
        type: "string",
        minLength: 3,
        maxLength: 5,
        pattern: "^a+$",
      }),
    ).toEqual([
      "Value must be at least 3 characters",
      "Value format is invalid",
    ]);

    expect(
      validateValueAgainstSchema("abcdef", {
        type: "string",
        maxLength: 5,
      }),
    ).toEqual(["Value must be no more than 5 characters"]);

    expect(
      validateValueAgainstSchema("2.5", {
        type: "integer",
        minimum: 3,
        maximum: 10,
      }),
    ).toEqual([
      "Value must be at least 3",
      "Value must be an integer",
    ]);

    expect(
      validateValueAgainstSchema("not-a-number", {
        type: "number",
      }),
    ).toEqual(["Value must be a number"]);

    expect(validateValueAgainstSchema("yes", { type: "boolean" })).toEqual([
      "Value must be true or false",
    ]);

    expect(
      validateValueAgainstSchema(["ok"], {
        type: "array",
        items: [{ type: "string" }, { type: "integer" }],
      }),
    ).toEqual(["Value[1] is required"]);

    expect(
      validateValueAgainstSchema(["ok", 2, "extra"], {
        type: "array",
        items: [{ type: "string" }, { type: "integer" }],
        additionalItems: false,
      }),
    ).toEqual(["Value has too many items"]);

    expect(
      validateValueAgainstSchema([1, "two"], {
        type: "array",
        minItems: 3,
        maxItems: 1,
        items: { type: "integer" },
      }),
    ).toEqual([
      "Value must contain at least 3 items",
      "Value must contain no more than 1 items",
      "Value[1] must be a number",
    ]);

    expect(
      validateValueAgainstSchema(
        { firstName: "", age: "old", role: "guest" },
        {
          type: "object",
          required: ["firstName"],
          properties: {
            firstName: { type: "string", minLength: 1 },
            age: { type: "integer" },
            role: { enum: ["admin", "member"] },
          },
        },
      ),
    ).toEqual([
      "First Name is required",
      "Value.Age must be a number",
      "Value.Role must be one of: admin, member",
    ]);

    expect(validateValueAgainstSchema("value", { const: "fixed" })).toEqual([
      "Value must be exactly fixed",
    ]);

    expect(validateValueAgainstSchema("green", { enum: ["red", "blue"] })).toEqual([
      "Value must be one of: red, blue",
    ]);
  });

  it("short-circuits empty values and invalid schemas", () => {
    expect(validateValueAgainstSchema(undefined, { type: "string" })).toEqual([]);
    expect(validateValueAgainstSchema("", { type: "string", minLength: 10 })).toEqual([]);
    expect(validateValueAgainstSchema("value", null as unknown as JsonSchema)).toEqual([]);
  });

  it("validates combinators with anyOf, oneOf, and allOf semantics", () => {
    expect(
      validateValueAgainstSchema("alpha", {
        anyOf: [{ type: "string", minLength: 3 }, { type: "number" }],
      }),
    ).toEqual([]);

    expect(
      validateValueAgainstSchema({}, {
        anyOf: [{ type: "string" }, { type: "number" }],
      }),
    ).toEqual(["Value must be a string"]);

    expect(
      validateValueAgainstSchema(5, {
        oneOf: [{ type: "number" }, { type: "integer" }],
      }),
    ).toEqual(["Value matches multiple variants. Please choose one."]);

    expect(
      validateValueAgainstSchema({}, {
        oneOf: [{ type: "string" }, { type: "number" }],
      }),
    ).toEqual(["Value must be a string"]);

    expect(
      validateValueAgainstSchema("ab", {
        allOf: [{ type: "string", minLength: 3 }, { type: "string", pattern: "^z+$" }],
      }),
    ).toEqual([
      "Value must be at least 3 characters",
      "Value format is invalid",
    ]);
  });

  it("rejects values that do not match array or object types", () => {
    expect(validateValueAgainstSchema("text", { type: "array" })).toEqual([
      "Value must be an array",
    ]);
    expect(validateValueAgainstSchema([], { type: "object" })).toEqual([
      "Value must be an object",
    ]);
  });
});

describe("generateExampleData", () => {
  it("prefers schema defaults, examples, const values, and enums", () => {
    expect(generateExampleData({ default: { enabled: true } })).toEqual({ enabled: true });
    expect(generateExampleData({ examples: ["sample"] })).toBe("sample");
    expect(generateExampleData({ example: 12 })).toBe(12);
    expect(generateExampleData({ const: "fixed" })).toBe("fixed");
    expect(generateExampleData({ enum: ["first", "second"] })).toBe("first");
  });

  it("generates examples for formatted strings, numbers, arrays, objects, and combinators", () => {
    expect(generateExampleData({ type: "string", format: "email" })).toBe("user@example.com");
    expect(generateExampleData({ type: "string", format: "url" })).toBe("https://example.com");
    expect(generateExampleData({ type: "string", format: "uuid" })).toBe(
      "123e4567-e89b-12d3-a456-426614174000",
    );
    expect(generateExampleData({ type: "string" })).toBe("example");

    expect(generateExampleData({ type: "number", minimum: 2 })).toBe(2);
    expect(generateExampleData({ type: "number", maximum: 7 })).toBe(7);
    expect(generateExampleData({ type: "integer" })).toBe(1);
    expect(generateExampleData({ type: "number" })).toBe(1);
    expect(generateExampleData({ type: "boolean" })).toBe(true);

    expect(
      generateExampleData({
        type: "array",
        items: [{ type: "string" }, { type: "integer" }],
      }),
    ).toEqual(["example", 1]);
    expect(generateExampleData({ type: "array", items: { type: "boolean" } })).toEqual([true]);
    expect(generateExampleData({ type: "array" })).toEqual([]);

    expect(
      generateExampleData({
        type: "object",
        properties: {
          name: { type: "string" },
          active: { type: "boolean" },
        },
      }),
    ).toEqual({
      name: "example",
      active: true,
    });

    expect(
      generateExampleData({
        oneOf: [{ type: "number", minimum: 9 }, { type: "string" }],
      }),
    ).toBe(9);
  });

  it("returns null for invalid schemas and unknown schema shapes", () => {
    expect(generateExampleData(null as unknown as JsonSchema)).toBeNull();
    expect(generateExampleData({})).toBeNull();
  });
});
