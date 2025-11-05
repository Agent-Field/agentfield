import React, { useState, useEffect } from 'react';
import { Button } from '../ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '../ui/card';
import { Alert, AlertDescription } from '../ui/alert';
import { Loader2, Save, AlertCircle } from '@/components/ui/icon-bridge';
import { ConfigField } from './ConfigField';
import type { ConfigurationSchema, AgentConfiguration, ConfigField as ConfigFieldType } from '../../types/haxen';

interface ConfigurationFormProps {
  schema: ConfigurationSchema;
  initialValues?: AgentConfiguration;
  onSubmit: (configuration: AgentConfiguration) => Promise<void>;
  loading?: boolean;
  title?: string;
  description?: string;
}

interface ValidationErrors {
  [fieldName: string]: string;
}

export const ConfigurationForm: React.FC<ConfigurationFormProps> = ({
  schema,
  initialValues = {},
  onSubmit,
  loading = false,
  title = "Configuration",
  description = "Configure the agent settings below"
}) => {
  const fields = schema.fields ?? [];
  const [values, setValues] = useState<AgentConfiguration>(initialValues);
  const [errors, setErrors] = useState<ValidationErrors>({});
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [submitError, setSubmitError] = useState<string | null>(null);

  // Initialize form values with defaults
  useEffect(() => {
    const defaultValues: AgentConfiguration = {};
    fields.forEach((field) => {
      if (field.default !== undefined) {
        defaultValues[field.name] = field.default;
      }
    });
    setValues({ ...defaultValues, ...initialValues });
  }, [fields, initialValues]);

  const validateField = (field: ConfigFieldType, value: any): string | null => {
    // Required field validation
    if (field.required && (value === undefined || value === null || value === '')) {
      return `${field.name} is required`;
    }

    // Type-specific validation
    if (value !== undefined && value !== null && value !== '') {
      switch (field.type) {
        case 'number':
          const numValue = Number(value);
          if (isNaN(numValue)) {
            return `${field.name} must be a valid number`;
          }
          if (field.validation?.min !== undefined && numValue < field.validation.min) {
            return `${field.name} must be at least ${field.validation.min}`;
          }
          if (field.validation?.max !== undefined && numValue > field.validation.max) {
            return `${field.name} must be at most ${field.validation.max}`;
          }
          break;

        case 'text':
        case 'secret':
          if (field.validation?.pattern) {
            const regex = new RegExp(field.validation.pattern);
            if (!regex.test(String(value))) {
              return `${field.name} format is invalid`;
            }
          }
          break;
      }
    }

    return null;
  };

  const validateForm = (): boolean => {
    const newErrors: ValidationErrors = {};
    let isValid = true;

    fields.forEach((field) => {
      const error = validateField(field, values[field.name]);
      if (error) {
        newErrors[field.name] = error;
        isValid = false;
      }
    });

    setErrors(newErrors);
    return isValid;
  };

  const handleFieldChange = (fieldName: string, value: any) => {
    setValues(prev => ({ ...prev, [fieldName]: value }));
    
    // Clear field error when user starts typing
    if (errors[fieldName]) {
      setErrors(prev => ({ ...prev, [fieldName]: '' }));
    }
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    
    if (!validateForm()) {
      return;
    }

    setIsSubmitting(true);
    setSubmitError(null);

    try {
      await onSubmit(values);
    } catch (error) {
      setSubmitError(error instanceof Error ? error.message : 'An error occurred while saving configuration');
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <Card className="w-full max-w-2xl mx-auto">
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          {title}
          {loading && <Loader2 className="h-4 w-4 animate-spin" />}
        </CardTitle>
        {description && (
          <CardDescription>{description}</CardDescription>
        )}
      </CardHeader>
      <CardContent>
        {submitError && (
          <Alert variant="destructive" className="mb-6">
            <AlertCircle className="h-4 w-4" />
            <AlertDescription>{submitError}</AlertDescription>
          </Alert>
        )}

        <form onSubmit={handleSubmit} className="space-y-6">
          {fields.map((field) => (
            <ConfigField
              key={field.name}
              field={field}
              value={values[field.name]}
              onChange={(value) => handleFieldChange(field.name, value)}
              error={errors[field.name]}
            />
          ))}

          <div className="flex justify-end pt-4">
            <Button
              type="submit"
              disabled={isSubmitting || loading}
              className="min-w-[120px]"
            >
              {isSubmitting ? (
                <>
                  <Loader2 className="h-4 w-4 animate-spin mr-2" />
                  Saving...
                </>
              ) : (
                <>
                  <Save className="h-4 w-4 mr-2" />
                  Save Configuration
                </>
              )}
            </Button>
          </div>
        </form>
      </CardContent>
    </Card>
  );
};
