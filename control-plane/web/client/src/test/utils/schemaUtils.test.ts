import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import type { JsonSchema } from "@/types/execution";
import {
  generateExampleData,
  schemaToFormFields,
  validateFormData,
  validateValueAgainstSchema,
} from "@/utils/schemaUtils";

describe("schemaUtils", () => {
  beforeEach(() => {
    vi.spyOn(console, "warn").mockImplementation(() => {});
    vi.spyOn(console, "error").mockImplementation(() => {});
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("builds form fields with labels, defaults, arrays, and combinators", () => {
    const schema: JsonSchema = {
      type: "object",
      required: ["displayName", "variant"],
      properties: {
        displayName: {
          type: "string",
          description: "Shown in the UI",
          default: "Ada",
        },
        role: {
          enum: ["admin", "viewer"],
          example: "viewer",
        },
        emails: {
          type: "array",
          items: { type: "string", format: "email" },
          minItems: 1,
          maxItems: 3,
        },
        tupleValue: {
          type: "array",
          items: [{ type: "string" }, { type: "integer" }],
          additionalItems: false,
        },
        variant: {
          oneOf: [
            { type: "string", title: "Text variant" },
            { type: "number", description: "Numeric variant" },
          ],
        },
        nullableOptions: {
          type: ["null", "object"],
          properties: {
            enabled: { type: "boolean" },
          },
          required: ["enabled"],
        },
      },
    };

    const fields = schemaToFormFields(schema);
    const displayName = fields.find((field) => field.name === "displayName");
    const role = fields.find((field) => field.name === "role");
    const emails = fields.find((field) => field.name === "emails");
    const tupleValue = fields.find((field) => field.name === "tupleValue");
    const variant = fields.find((field) => field.name === "variant");
    const nullableOptions = fields.find((field) => field.name === "nullableOptions");

    expect(fields).toHaveLength(6);
    expect(displayName).toMatchObject({
      label: "Display Name",
      type: "string",
      required: true,
      defaultValue: "Ada",
      placeholder: "Ada",
    });
    expect(role).toMatchObject({
      type: "select",
      options: ["admin", "viewer"],
      enumValues: ["admin", "viewer"],
      examples: ["viewer"],
      placeholder: "viewer",
    });
    expect(emails).toMatchObject({
      type: "array",
      minItems: 1,
      maxItems: 3,
      itemSchema: { type: "string", format: "email" },
      placeholder: "Add items...",
    });
    expect(tupleValue?.tupleSchemas).toHaveLength(2);
    expect(variant).toMatchObject({
      combinator: "oneOf",
      variantTitles: ["Text variant", "Numeric variant"],
      required: true,
    });
    expect(nullableOptions).toMatchObject({
      type: "object",
      placeholder: "Configure object...",
    });
  });

  it("returns an empty field list for invalid or non-object schemas", () => {
    expect(schemaToFormFields("bad-schema" as unknown as JsonSchema)).toEqual([]);
    expect(schemaToFormFields({ type: "string" })).toEqual([]);
    expect(console.warn).toHaveBeenCalledWith(
      "schemaToFormFields received invalid schema:",
      "bad-schema"
    );
  });

  it("validates form data through jsonSchemaToZodObject", () => {
    const schema: JsonSchema = {
      type: "object",
      required: ["name", "count"],
      properties: {
        name: { type: "string" },
        count: { type: "number" },
        enabled: { type: "boolean" },
      },
    };

    expect(
      validateFormData({ name: "AgentField", count: 2, enabled: true }, schema)
    ).toEqual({ isValid: true, errors: [] });

    const invalid = validateFormData({ count: "many" }, schema);
    expect(invalid.isValid).toBe(false);
    expect(invalid.errors.some((error) => error.includes("Name"))).toBe(true);
    expect(invalid.errors.some((error) => error.includes("Count"))).toBe(true);
  });

  it("validates strings, numbers, booleans, arrays, objects, enums, const, and combinators", () => {
    expect(
      validateValueAgainstSchema("a", {
        type: "string",
        minLength: 2,
        maxLength: 3,
        pattern: "^[A-Z]+$",
      })
    ).toEqual([
      "Value must be at least 2 characters",
      "Value format is invalid",
    ]);

    expect(
      validateValueAgainstSchema(3.2, {
        type: "integer",
        minimum: 4,
        maximum: 6,
      })
    ).toEqual([
      "Value must be at least 4",
      "Value must be an integer",
    ]);

    expect(validateValueAgainstSchema("yes", { type: "boolean" })).toEqual([
      "Value must be true or false",
    ]);

    expect(
      validateValueAgainstSchema(["ok", 2, "extra"], {
        type: "array",
        items: [{ type: "string" }, { type: "integer" }],
        additionalItems: false,
      })
    ).toEqual(["Value has too many items"]);

    expect(
      validateValueAgainstSchema(
        { title: "", details: { done: "no" } },
        {
          type: "object",
          required: ["title", "details"],
          properties: {
            title: { type: "string", minLength: 1 },
            details: {
              type: "object",
              required: ["done"],
              properties: {
                done: { type: "boolean" },
              },
            },
          },
        }
      )
    ).toEqual([
      "Title is required",
      "Value.Details.Done must be true or false",
    ]);

    expect(validateValueAgainstSchema("beta", { enum: ["alpha", "gamma"] })).toEqual([
      "Value must be one of: alpha, gamma",
    ]);

    expect(validateValueAgainstSchema("prod", { const: "dev" })).toEqual([
      "Value must be exactly dev",
    ]);

    expect(
      validateValueAgainstSchema("shared", {
        oneOf: [{ type: "string" }, { enum: ["shared"] }],
      })
    ).toEqual(["Value matches multiple variants. Please choose one."]);

    expect(
      validateValueAgainstSchema(5, {
        anyOf: [{ type: "string" }, { type: "number", minimum: 1 }],
      })
    ).toEqual([]);

    expect(
      validateValueAgainstSchema("oops", {
        allOf: [
          { type: "string", minLength: 5 },
          { type: "string", pattern: "^OK" },
        ],
      })
    ).toEqual([
      "Value must be at least 5 characters",
      "Value format is invalid",
    ]);
  });

  it("generates example data for defaults, combinators, arrays, objects, formats, and enums", () => {
    expect(generateExampleData({ default: { nested: true } })).toEqual({ nested: true });
    expect(
      generateExampleData({
        oneOf: [{ type: "string", format: "email" }, { type: "number" }],
      })
    ).toBe("user@example.com");
    expect(generateExampleData({ type: "array", items: { type: "integer", minimum: 4 } })).toEqual([
      4,
    ]);
    expect(
      generateExampleData({
        type: "object",
        properties: {
          enabled: { type: "boolean" },
          url: { type: "string", format: "url" },
        },
      })
    ).toEqual({
      enabled: true,
      url: "https://example.com",
    });
    expect(generateExampleData({ enum: ["primary", "secondary"] })).toBe("primary");
    expect(generateExampleData("bad-schema" as unknown as JsonSchema)).toBeNull();
  });
});
