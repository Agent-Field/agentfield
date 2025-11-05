import React, { useState, useEffect } from 'react';
import { Button } from '../ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '../ui/card';
import { Alert, AlertDescription } from '../ui/alert';
import { Skeleton } from '../ui/skeleton';
import { Input } from '../ui/input';
import { Label } from '../ui/label';
import { Loader2, Save, AlertCircle, Eye, EyeOff, RefreshCw } from '@/components/ui/icon-bridge';
import { 
  getAgentEnvironmentVariables, 
  updateAgentEnvironmentVariables, 
  getAgentConfigurationSchema 
} from '../../services/api';
import { 
  useSuccessNotification,
  useErrorNotification
} from '../ui/notification';
import type { 
  ConfigField, 
  ConfigSchemaResponse 
} from '../../types/haxen';

interface EnvironmentVariableFormProps {
  agentId: string;
  packageId: string;
  onConfigurationChange?: () => void;
}

interface EnvFieldState {
  value: string;
  showSecret: boolean;
  error?: string;
}

interface ValidationErrors {
  [fieldName: string]: string;
}

export const EnvironmentVariableForm: React.FC<EnvironmentVariableFormProps> = ({
  agentId,
  packageId,
  onConfigurationChange
}) => {
  // Notification hooks
  const showSuccess = useSuccessNotification();
  const showError = useErrorNotification();

  // State management
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [schema, setSchema] = useState<ConfigSchemaResponse | null>(null);
  const [fieldStates, setFieldStates] = useState<Record<string, EnvFieldState>>({});
  const [errors, setErrors] = useState<ValidationErrors>({});
  const [hasChanges, setHasChanges] = useState(false);

  // Load initial data
  useEffect(() => {
    loadData();
  }, [agentId, packageId]);

  const loadData = async () => {
    setLoading(true);
    try {
      const [schemaResponse, envResponse] = await Promise.all([
        getAgentConfigurationSchema(agentId, packageId),
        getAgentEnvironmentVariables(agentId, packageId)
      ]);

      setSchema(schemaResponse);

      // Initialize field states
      const initialStates: Record<string, EnvFieldState> = {};
      
      // Process required and optional fields from schema
      const allFields = [
        ...(schemaResponse.schema.user_environment?.required || []),
        ...(schemaResponse.schema.user_environment?.optional || [])
      ];

      allFields.forEach(field => {
        const currentValue = envResponse.variables[field.name] || field.default || '';
        
        initialStates[field.name] = {
          value: currentValue,
          showSecret: false
        };
      });

      setFieldStates(initialStates);
      setHasChanges(false);
    } catch (error) {
      showError(`Failed to load configuration: ${error instanceof Error ? error.message : 'Unknown error'}`);
    } finally {
      setLoading(false);
    }
  };

  const validateField = (field: ConfigField, value: string): string | null => {
    // Required field validation
    if (field.required && (!value || value.trim() === '')) {
      return `${field.name} is required`;
    }

    // Type-specific validation
    if (value && value.trim() !== '') {
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
            if (!regex.test(value)) {
              return `${field.name} format is invalid`;
            }
          }
          break;
      }
    }

    return null;
  };

  const validateForm = (): boolean => {
    if (!schema) return false;

    const newErrors: ValidationErrors = {};
    let isValid = true;

    const allFields = [
      ...(schema.schema.user_environment?.required || []),
      ...(schema.schema.user_environment?.optional || [])
    ];

    allFields.forEach((field) => {
      const value = fieldStates[field.name]?.value || '';
      const error = validateField(field, value);
      if (error) {
        newErrors[field.name] = error;
        isValid = false;
      }
    });

    setErrors(newErrors);
    return isValid;
  };

  const handleFieldChange = (fieldName: string, value: string) => {
    setFieldStates(prev => ({
      ...prev,
      [fieldName]: {
        ...prev[fieldName],
        value
      }
    }));

    // Clear field error when user starts typing
    if (errors[fieldName]) {
      setErrors(prev => ({ ...prev, [fieldName]: '' }));
    }

    setHasChanges(true);
  };

  const toggleSecretVisibility = (fieldName: string) => {
    setFieldStates(prev => ({
      ...prev,
      [fieldName]: {
        ...prev[fieldName],
        showSecret: !prev[fieldName]?.showSecret
      }
    }));
  };

  const handleSave = async () => {
    if (!validateForm()) {
      showError('Please fix the validation errors before saving');
      return;
    }

    setSaving(true);
    try {
      // Prepare variables for submission
      const variables: Record<string, string> = {};
      Object.entries(fieldStates).forEach(([key, state]) => {
        if (state.value.trim() !== '') {
          variables[key] = state.value;
        }
      });

      await updateAgentEnvironmentVariables(agentId, packageId, variables);
      
      showSuccess('Environment variables saved successfully');
      setHasChanges(false);
      
      // Notify parent component about configuration change
      if (onConfigurationChange) {
        onConfigurationChange();
      }

      // Reload data to get updated state
      await loadData();
    } catch (error) {
      showError(`Failed to save environment variables: ${error instanceof Error ? error.message : 'Unknown error'}`);
    } finally {
      setSaving(false);
    }
  };

  const renderField = (field: ConfigField) => {
    const fieldState = fieldStates[field.name] || { value: '', showSecret: false };
    const error = errors[field.name];
    const isSecret = field.type === 'secret';

    return (
      <div key={field.name} className="space-y-2">
        <Label htmlFor={field.name} className="text-sm font-medium">
          {field.name}
          {field.required && <span className="text-red-500 ml-1">*</span>}
        </Label>
        {field.description && (
          <p className="text-body-small">{field.description}</p>
        )}
        
        {isSecret ? (
          <div className="relative">
            <Input
              id={field.name}
              type={fieldState.showSecret ? 'text' : 'password'}
              value={fieldState.value}
              onChange={(e) => handleFieldChange(field.name, e.target.value)}
              placeholder={field.description || `Enter ${field.name}`}
              className={`pr-10 ${error ? 'border-red-500' : ''}`}
            />
            <Button
              type="button"
              variant="ghost"
              size="sm"
              className="absolute right-0 top-0 h-full px-3 py-2 hover:bg-transparent"
              onClick={() => toggleSecretVisibility(field.name)}
            >
              {fieldState.showSecret ? (
                <EyeOff className="h-4 w-4 text-gray-400" />
              ) : (
                <Eye className="h-4 w-4 text-gray-400" />
              )}
            </Button>
          </div>
        ) : (
          <Input
            id={field.name}
            type={field.type === 'number' ? 'number' : 'text'}
            value={fieldState.value}
            onChange={(e) => handleFieldChange(field.name, e.target.value)}
            placeholder={field.description || `Enter ${field.name}`}
            className={error ? 'border-red-500' : ''}
            min={field.validation?.min}
            max={field.validation?.max}
          />
        )}
        
        {error && (
          <p className="text-xs text-red-500">{error}</p>
        )}
      </div>
    );
  };

  if (loading) {
    return (
      <Card className="w-full">
        <CardHeader>
          <Skeleton className="h-6 w-48" />
          <Skeleton className="h-4 w-96" />
        </CardHeader>
        <CardContent className="space-y-6">
          {[1, 2, 3].map(i => (
            <div key={i} className="space-y-2">
              <Skeleton className="h-4 w-32" />
              <Skeleton className="h-10 w-full" />
            </div>
          ))}
        </CardContent>
      </Card>
    );
  }

  if (!schema) {
    return (
      <Alert variant="destructive">
        <AlertCircle className="h-4 w-4" />
        <AlertDescription>
          Failed to load configuration schema. Please try refreshing the page.
        </AlertDescription>
      </Alert>
    );
  }

  const requiredFields = schema.schema.user_environment?.required || [];
  const optionalFields = schema.schema.user_environment?.optional || [];
  const packageName = schema.metadata?.package_name ?? 'this package';
  const packageVersion = schema.metadata?.package_version ?? 'unknown';

  return (
    <Card className="w-full">
      <CardHeader>
        <CardTitle className="flex items-center justify-between">
          <span>Environment Variables</span>
          <Button
            variant="outline"
            size="sm"
            onClick={loadData}
            disabled={loading}
          >
            <RefreshCw className={`h-4 w-4 mr-2 ${loading ? 'animate-spin' : ''}`} />
            Refresh
          </Button>
        </CardTitle>
        <CardDescription>
          Configure environment variables for {packageName} (v{packageVersion})
        </CardDescription>
      </CardHeader>
      
      <CardContent className="space-y-8">
        {/* Required Fields Section */}
        {requiredFields.length > 0 && (
          <div className="space-y-4">
            <div className="border-b pb-2">
              <h3 className="text-sm font-semibold text-foreground">Required Variables</h3>
              <p className="text-body-small">These variables must be configured</p>
            </div>
            <div className="space-y-4">
              {requiredFields.map(renderField)}
            </div>
          </div>
        )}

        {/* Optional Fields Section */}
        {optionalFields.length > 0 && (
          <div className="space-y-4">
            <div className="border-b pb-2">
              <h3 className="text-sm font-semibold text-foreground">Optional Variables</h3>
              <p className="text-body-small">These variables are optional but may enhance functionality</p>
            </div>
            <div className="space-y-4">
              {optionalFields.map(renderField)}
            </div>
          </div>
        )}

        {/* No fields message */}
        {requiredFields.length === 0 && optionalFields.length === 0 && (
          <div className="text-center py-8">
            <p className="text-muted-foreground">No environment variables are configured for this package.</p>
          </div>
        )}

        {/* Save Button */}
        {(requiredFields.length > 0 || optionalFields.length > 0) && (
          <div className="flex justify-end pt-4 border-t">
            <Button
              onClick={handleSave}
              disabled={saving || !hasChanges}
              className="min-w-[120px]"
            >
              {saving ? (
                <>
                  <Loader2 className="h-4 w-4 animate-spin mr-2" />
                  Saving...
                </>
              ) : (
                <>
                  <Save className="h-4 w-4 mr-2" />
                  Save Changes
                </>
              )}
            </Button>
          </div>
        )}
      </CardContent>
    </Card>
  );
};
