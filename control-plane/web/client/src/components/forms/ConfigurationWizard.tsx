import React, { useState } from 'react';
import { Button } from '../ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '../ui/card';
import { Badge } from '../ui/badge';
import { ChevronLeft, ChevronRight, Check, Settings, Package, Info } from '@/components/ui/icon-bridge';
import { ConfigurationForm } from './ConfigurationForm';
import type { ConfigurationSchema, AgentConfiguration, AgentPackage } from '../../types/agentfield';

interface ConfigurationWizardProps {
  package: AgentPackage;
  schema: ConfigurationSchema;
  initialValues?: AgentConfiguration;
  onComplete: (configuration: AgentConfiguration) => Promise<void>;
  onCancel?: () => void;
}

export const ConfigurationWizard: React.FC<ConfigurationWizardProps> = ({
  package: pkg,
  schema,
  initialValues,
  onComplete,
  onCancel
}) => {
  const fields = schema.fields ?? [];
  const [currentStep, setCurrentStep] = useState(0);
  const [isCompleting, setIsCompleting] = useState(false);
  const [configuration, setConfiguration] = useState<AgentConfiguration>(initialValues || {});

  const steps = [
    {
      title: 'Package Overview',
      description: 'Review the agent package details',
      icon: Package
    },
    {
      title: 'Configuration',
      description: 'Set up the agent configuration',
      icon: Settings
    },
    {
      title: 'Review & Complete',
      description: 'Review your settings and complete setup',
      icon: Check
    }
  ];

  const handleNext = () => {
    if (currentStep < steps.length - 1) {
      setCurrentStep(currentStep + 1);
    }
  };

  const handlePrevious = () => {
    if (currentStep > 0) {
      setCurrentStep(currentStep - 1);
    }
  };

  const handleConfigurationSubmit = async (config: AgentConfiguration) => {
    setConfiguration(config);
    handleNext();
  };

  const handleComplete = async () => {
    setIsCompleting(true);
    try {
      await onComplete(configuration);
    } catch (error) {
      // Error handling is done in the parent component
      console.error('Configuration completion failed:', error);
    } finally {
      setIsCompleting(false);
    }
  };

  const renderStepContent = () => {
    switch (currentStep) {
      case 0:
        return (
          <div className="space-y-6">
            <div className="flex items-start gap-4">
              <div className="p-3 bg-blue-100 rounded-lg">
                <Package className="h-6 w-6 text-blue-600" />
              </div>
              <div className="flex-1">
                <h3 className="text-heading-3">{pkg.name}</h3>
                <p className="text-gray-600 mt-1">{pkg.description}</p>
                <div className="flex items-center gap-4 mt-3 text-sm text-gray-500">
                  <span>Version: {pkg.version}</span>
                  <span>Author: {pkg.author}</span>
                </div>
              </div>
            </div>

            {pkg.tags && pkg.tags.length > 0 && (
              <div>
                <h4 className="text-sm font-medium mb-2">Tags</h4>
                <div className="flex flex-wrap gap-2">
                  {pkg.tags.map((tag) => (
                    <Badge key={tag} variant="secondary">
                      {tag}
                    </Badge>
                  ))}
                </div>
              </div>
            )}

            <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
              <div className="flex items-start gap-3">
                <Info className="h-5 w-5 text-blue-600 mt-0.5" />
                <div>
                  <h4 className="text-sm font-medium text-blue-900">Configuration Required</h4>
                  <p className="text-sm text-blue-700 mt-1">
                    This agent requires {fields.length} configuration field{fields.length !== 1 ? 's' : ''} to be set up before it can run.
                  </p>
                </div>
              </div>
            </div>
          </div>
        );

      case 1:
        return (
          <ConfigurationForm
            schema={{ ...schema, fields }}
            initialValues={configuration}
            onSubmit={handleConfigurationSubmit}
            title=""
            description=""
          />
        );

      case 2:
        return (
          <div className="space-y-6">
            <div className="bg-green-50 border border-green-200 rounded-lg p-4">
              <div className="flex items-start gap-3">
                <Check className="h-5 w-5 text-green-600 mt-0.5" />
                <div>
                  <h4 className="text-sm font-medium text-green-900">Configuration Complete</h4>
                  <p className="text-sm text-green-700 mt-1">
                    Your agent is configured and ready to start.
                  </p>
                </div>
              </div>
            </div>

            <div>
              <h4 className="text-sm font-medium mb-3">Configuration Summary</h4>
              <div className="space-y-2">
                {fields.map((field) => (
                  <div key={field.name} className="flex justify-between items-center py-2 border-b border-gray-100">
                    <span className="text-sm font-medium">{field.name}</span>
                    <span className="text-sm text-gray-600">
                      {field.type === 'secret' 
                        ? '••••••••' 
                        : field.type === 'boolean'
                        ? (configuration[field.name] ? 'Enabled' : 'Disabled')
                        : configuration[field.name] || 'Not set'
                      }
                    </span>
                  </div>
                ))}
              </div>
            </div>
          </div>
        );

      default:
        return null;
    }
  };

  return (
    <div className="w-full max-w-4xl mx-auto">
      {/* Step Indicator */}
      <div className="mb-8">
        <div className="flex items-center justify-between">
          {steps.map((step, index) => {
            const Icon = step.icon;
            const isActive = index === currentStep;
            const isCompleted = index < currentStep;
            
            return (
              <div key={index} className="flex items-center">
                <div className={`flex items-center justify-center w-10 h-10 rounded-full border-2 ${
                  isCompleted 
                    ? 'bg-green-500 border-green-500 text-white' 
                    : isActive 
                    ? 'bg-blue-500 border-blue-500 text-white' 
                    : 'bg-gray-100 border-gray-300 text-gray-400'
                }`}>
                  {isCompleted ? (
                    <Check className="h-5 w-5" />
                  ) : (
                    <Icon className="h-5 w-5" />
                  )}
                </div>
                <div className="ml-3">
                  <p className={`text-sm font-medium ${isActive ? 'text-blue-600' : isCompleted ? 'text-green-600' : 'text-gray-500'}`}>
                    {step.title}
                  </p>
                  <p className="text-xs text-gray-500">{step.description}</p>
                </div>
                {index < steps.length - 1 && (
                  <div className={`flex-1 h-0.5 mx-4 ${isCompleted ? 'bg-green-500' : 'bg-gray-200'}`} />
                )}
              </div>
            );
          })}
        </div>
      </div>

      {/* Step Content */}
      <Card>
        <CardHeader>
          <CardTitle>{steps[currentStep].title}</CardTitle>
          <CardDescription>{steps[currentStep].description}</CardDescription>
        </CardHeader>
        <CardContent>
          {renderStepContent()}
        </CardContent>
      </Card>

      {/* Navigation */}
      {currentStep !== 1 && (
        <div className="flex justify-between mt-6">
          <Button
            variant="outline"
            onClick={currentStep === 0 ? onCancel : handlePrevious}
            disabled={isCompleting}
          >
            <ChevronLeft className="h-4 w-4 mr-2" />
            {currentStep === 0 ? 'Cancel' : 'Previous'}
          </Button>

          <Button
            onClick={currentStep === steps.length - 1 ? handleComplete : handleNext}
            disabled={isCompleting}
          >
            {currentStep === steps.length - 1 ? (
              isCompleting ? (
                'Completing...'
              ) : (
                'Complete Setup'
              )
            ) : (
              <>
                Next
                <ChevronRight className="h-4 w-4 ml-2" />
              </>
            )}
          </Button>
        </div>
      )}
    </div>
  );
};
