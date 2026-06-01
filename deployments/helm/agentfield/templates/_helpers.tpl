{{- define "agentfield.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" -}}
{{- end -}}

{{- define "agentfield.fullname" -}}
{{- if .Values.fullnameOverride -}}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" -}}
{{- else -}}
{{- $name := include "agentfield.name" . -}}
{{- if contains $name .Release.Name -}}
{{- .Release.Name | trunc 63 | trimSuffix "-" -}}
{{- else -}}
{{- printf "%s-%s" .Release.Name $name | trunc 63 | trimSuffix "-" -}}
{{- end -}}
{{- end -}}
{{- end -}}

{{- define "agentfield.labels" -}}
app.kubernetes.io/name: {{ include "agentfield.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
helm.sh/chart: {{ printf "%s-%s" .Chart.Name .Chart.Version | quote }}
{{- end -}}

{{- define "agentfield.selectorLabels" -}}
app.kubernetes.io/name: {{ include "agentfield.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end -}}

{{- define "agentfield.controlPlane.fullname" -}}
{{- printf "%s-control-plane" (include "agentfield.fullname" .) | trunc 63 | trimSuffix "-" -}}
{{- end -}}

{{- define "agentfield.postgres.fullname" -}}
{{- printf "%s-postgres" (include "agentfield.fullname" .) | trunc 63 | trimSuffix "-" -}}
{{- end -}}

{{- define "agentfield.demoAgent.fullname" -}}
{{- printf "%s-demo-agent" (include "agentfield.fullname" .) | trunc 63 | trimSuffix "-" -}}
{{- end -}}

{{- define "agentfield.controlPlane.grpcPort" -}}
{{- $grpcPort := int (default 0 .Values.controlPlane.service.grpcPort) -}}
{{- if eq $grpcPort 0 -}}
{{- add (int .Values.controlPlane.service.port) 100 -}}
{{- else -}}
{{- $grpcPort -}}
{{- end -}}
{{- end -}}

{{- define "agentfield.controlPlane.postgresUrl" -}}
{{- $url := default "" .Values.controlPlane.storage.postgresUrl -}}
{{- if $url -}}
{{- $url -}}
{{- else if and .Values.postgres.enabled (not .Values.postgres.auth.existingSecret) -}}
{{- printf "postgres://%s:%s@%s:5432/%s?sslmode=disable" .Values.postgres.auth.username .Values.postgres.auth.password (include "agentfield.postgres.fullname" .) .Values.postgres.auth.database -}}
{{- else -}}
{{- "" -}}
{{- end -}}
{{- end -}}

{{- define "agentfield.apiAuth.secretName" -}}
{{- if .Values.apiAuth.existingSecret -}}
{{- .Values.apiAuth.existingSecret -}}
{{- else -}}
{{- printf "%s-api-auth" (include "agentfield.fullname" .) | trunc 63 | trimSuffix "-" -}}
{{- end -}}
{{- end -}}

{{- define "agentfield.postgres.secretName" -}}
{{- if .Values.postgres.auth.existingSecret -}}
{{- .Values.postgres.auth.existingSecret -}}
{{- else -}}
{{- printf "%s-postgres-auth" (include "agentfield.fullname" .) | trunc 63 | trimSuffix "-" -}}
{{- end -}}
{{- end -}}
