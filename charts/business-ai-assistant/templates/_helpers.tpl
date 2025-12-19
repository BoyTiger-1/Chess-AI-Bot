{{- define "baa.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" -}}
{{- end -}}

{{- define "baa.fullname" -}}
{{- if .Values.fullnameOverride -}}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" -}}
{{- else -}}
{{- $name := default .Chart.Name .Values.nameOverride -}}
{{- printf "%s-%s" .Release.Name $name | trunc 63 | trimSuffix "-" -}}
{{- end -}}
{{- end -}}

{{- define "baa.labels" -}}
app.kubernetes.io/name: {{ include "baa.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
helm.sh/chart: {{ printf "%s-%s" .Chart.Name .Chart.Version | quote }}
{{- end -}}

{{- define "baa.selectorLabels" -}}
app.kubernetes.io/name: {{ include "baa.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end -}}

{{- define "baa.componentLabels" -}}
{{ include "baa.selectorLabels" . }}
app.kubernetes.io/component: {{ .component | quote }}
{{- end -}}

{{- define "baa.serviceName" -}}
{{- printf "%s-%s" (include "baa.fullname" .root) .component | trunc 63 | trimSuffix "-" -}}
{{- end -}}
