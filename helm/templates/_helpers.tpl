{{/*
Expand the name of the chart.
*/}}
{{- define "insightforge.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Create a default fully qualified app name.
*/}}
{{- define "insightforge.fullname" -}}
{{- if .Values.fullnameOverride }}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- $name := default .Chart.Name .Values.nameOverride }}
{{- if contains $name .Release.Name }}
{{- .Release.Name | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- printf "%s-%s" .Release.Name $name | trunc 63 | trimSuffix "-" }}
{{- end }}
{{- end }}
{{- end }}

{{/*
Common labels
*/}}
{{- define "insightforge.labels" -}}
helm.sh/chart: {{ include "insightforge.name" . }}
app.kubernetes.io/name: {{ include "insightforge.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}

{{/*
Selector labels
*/}}
{{- define "insightforge.selectorLabels" -}}
app.kubernetes.io/name: {{ include "insightforge.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}
