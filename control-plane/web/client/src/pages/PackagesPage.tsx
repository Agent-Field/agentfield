import React, { useState } from "react";
import { ConfigurationWizard } from "../components/forms";
import { AgentPackageList } from "../components/packages";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from "../components/ui/dialog";
import {
  NotificationProvider,
  useSuccessNotification,
  useErrorNotification,
} from "../components/ui/notification";
import {
  ConfigurationApiError,
  getAgentConfiguration,
  getConfigurationSchema,
  setAgentConfiguration,
  startAgent,
  stopAgent,
} from "../services/configurationApi";
import type {
  AgentConfiguration,
  AgentPackage,
  ConfigurationSchema,
} from "../types/agentfield";

const PackagesPageContent: React.FC = () => {
  const [selectedPackage, setSelectedPackage] = useState<AgentPackage | null>(
    null
  );
  const [configSchema, setConfigSchema] = useState<ConfigurationSchema | null>(
    null
  );
  const [currentConfig, setCurrentConfig] = useState<AgentConfiguration>({});
  const [isConfigDialogOpen, setIsConfigDialogOpen] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Notification hooks
  const showSuccess = useSuccessNotification();
  const showError = useErrorNotification();

  const handleConfigure = async (pkg: AgentPackage) => {
    setError(null);
    setSelectedPackage(pkg);

    try {
      // Load configuration schema and current configuration
      const [schema, config] = await Promise.all([
        getConfigurationSchema(pkg.id),
        getAgentConfiguration(pkg.id).catch(() => ({})), // Default to empty if no config exists
      ]);

      setConfigSchema(schema);
      setCurrentConfig(config);
      setIsConfigDialogOpen(true);
    } catch (err) {
      const errorMessage =
        err instanceof ConfigurationApiError
          ? err.message
          : "Failed to load configuration";
      setError(errorMessage);
      showError("Configuration Error", errorMessage);
      console.error("Configuration load error:", err);
    }
  };

  const handleConfigurationComplete = async (
    configuration: AgentConfiguration
  ) => {
    if (!selectedPackage) return;

    setError(null);

    try {
      await setAgentConfiguration(selectedPackage.id, configuration);
      setIsConfigDialogOpen(false);
      setSelectedPackage(null);
      setConfigSchema(null);
      setCurrentConfig({});

      showSuccess(
        "Configuration Saved",
        `${selectedPackage.name} has been configured successfully`
      );
    } catch (err) {
      const errorMessage =
        err instanceof ConfigurationApiError
          ? err.message
          : "Failed to save configuration";
      setError(errorMessage);
      showError("Configuration Error", errorMessage);
      console.error("Configuration save error:", err);
    }
  };

  const handleStart = async (pkg: AgentPackage) => {
    setError(null);

    try {
      await startAgent(pkg.id);
      showSuccess(
        "Agent Started",
        `${pkg.name} is now starting up`,
        {
          label: "View Logs",
          onClick: () => {
            // TODO: Navigate to logs or node detail page
            console.log(`Navigate to logs for ${pkg.id}`);
          }
        }
      );
    } catch (err) {
      const errorMessage =
        err instanceof ConfigurationApiError
          ? err.message
          : "Failed to start agent";
      setError(errorMessage);
      showError(
        "Start Failed",
        `Could not start ${pkg.name}: ${errorMessage}`
      );
      console.error("Agent start error:", err);
    }
  };

  const handleStop = async (pkg: AgentPackage) => {
    setError(null);

    try {
      await stopAgent(pkg.id);
      showSuccess(
        "Agent Stopped",
        `${pkg.name} has been stopped successfully`
      );
    } catch (err) {
      const errorMessage =
        err instanceof ConfigurationApiError
          ? err.message
          : "Failed to stop agent";
      setError(errorMessage);
      showError(
        "Stop Failed",
        `Could not stop ${pkg.name}: ${errorMessage}`
      );
      console.error("Agent stop error:", err);
    }
  };

  const handleCloseDialog = () => {
    setIsConfigDialogOpen(false);
    setSelectedPackage(null);
    setConfigSchema(null);
    setCurrentConfig({});
    setError(null);
  };

  return (
    <div className="container mx-auto py-6">
      <AgentPackageList
        onConfigure={handleConfigure}
        onStart={handleStart}
        onStop={handleStop}
      />

      {/* Configuration Dialog */}
      <Dialog open={isConfigDialogOpen} onOpenChange={handleCloseDialog}>
        <DialogContent className="max-w-4xl max-h-[90vh] overflow-y-auto">
          <DialogHeader>
            <DialogTitle>Configure Agent Package</DialogTitle>
            <DialogDescription>
              {selectedPackage &&
                `Configure ${selectedPackage.name} to customize its behavior and settings.`}
            </DialogDescription>
          </DialogHeader>

          {selectedPackage && configSchema && (
            <ConfigurationWizard
              package={selectedPackage}
              schema={configSchema}
              initialValues={currentConfig}
              onComplete={handleConfigurationComplete}
              onCancel={handleCloseDialog}
            />
          )}

          {error && (
            <div className="mt-4 p-4 bg-red-50 border border-red-200 rounded-lg">
              <p className="text-sm text-red-600">{error}</p>
            </div>
          )}
        </DialogContent>
      </Dialog>
    </div>
  );
};

export const PackagesPage: React.FC = () => {
  return (
    <NotificationProvider>
      <PackagesPageContent />
    </NotificationProvider>
  );
};
