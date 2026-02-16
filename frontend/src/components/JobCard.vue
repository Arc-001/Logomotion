<template>
  <div class="job-card">
    <div class="job-header">
      <span class="job-id">{{ job.job_id }}</span>
      <span class="status-pill" :class="'status-' + job.status">
        {{ job.status }}
      </span>
    </div>

    <div v-if="job.status === 'running'" class="progress-bar">
      <div class="progress-bar-inner"></div>
    </div>

    <div v-if="job.status === 'completed'" class="job-actions">
      <a :href="jobDownloadUrl" class="btn btn-primary btn-sm" download>
        Download Video
      </a>
      <button
        v-if="job.code"
        class="code-toggle"
        @click="showCode = !showCode"
      >
        {{ showCode ? 'Hide code' : 'View code' }}
      </button>
    </div>

    <div v-if="job.error" class="job-error">{{ job.error }}</div>

    <div v-if="showCode && job.code" class="code-block">{{ job.code }}</div>
  </div>
</template>

<script setup>
import { ref, computed } from 'vue'
import { downloadUrl } from '../api.js'

const props = defineProps({
  job: { type: Object, required: true },
})

const showCode = ref(false)
const jobDownloadUrl = computed(() => downloadUrl(props.job.job_id))
</script>

<style scoped>
.job-card {
  background: var(--bg-secondary);
  border: 1px solid var(--border-light);
  border-radius: var(--radius-md);
  padding: 18px 20px;
  box-shadow: var(--shadow-sm);
  transition: box-shadow var(--transition);
}

.job-card:hover {
  box-shadow: var(--shadow-md);
}

.job-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 8px;
}

.job-id {
  font-size: 13px;
  font-weight: 600;
  font-family: 'SF Mono', 'Fira Code', monospace;
  color: var(--text-primary);
}

.status-pill {
  display: inline-flex;
  align-items: center;
  gap: 5px;
  font-size: 11px;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.5px;
  padding: 3px 10px;
  border-radius: 12px;
}

.status-pending  { background: var(--warning-bg); color: var(--warning); }
.status-running  { background: var(--accent-subtle); color: var(--accent); }
.status-completed { background: var(--success-bg); color: var(--success); }
.status-failed   { background: var(--error-bg); color: var(--error); }

.job-actions {
  display: flex;
  gap: 8px;
  align-items: center;
  margin-top: 10px;
}

.job-error {
  font-size: 12px;
  color: var(--error);
  background: var(--error-bg);
  padding: 8px 12px;
  border-radius: var(--radius-sm);
  margin-top: 10px;
  line-height: 1.5;
}

.code-toggle {
  font-size: 12px;
  font-weight: 500;
  color: var(--accent);
  background: none;
  border: none;
  cursor: pointer;
  padding: 0;
  text-decoration: underline;
  text-underline-offset: 2px;
}

.code-block {
  margin-top: 10px;
  background: var(--bg-code);
  border: 1px solid var(--border-light);
  border-radius: var(--radius-sm);
  padding: 14px 16px;
  font-family: 'SF Mono', 'Fira Code', monospace;
  font-size: 12px;
  line-height: 1.7;
  overflow-x: auto;
  white-space: pre;
  color: var(--text-primary);
  max-height: 320px;
  overflow-y: auto;
}

.progress-bar {
  height: 3px;
  background: var(--border-light);
  border-radius: 2px;
  overflow: hidden;
  margin-top: 10px;
}

.progress-bar-inner {
  height: 100%;
  background: var(--accent);
  border-radius: 2px;
  animation: progress-indeterminate 1.8s ease-in-out infinite;
}

@keyframes progress-indeterminate {
  0%   { width: 0%;   margin-left: 0%; }
  50%  { width: 40%;  margin-left: 30%; }
  100% { width: 0%;   margin-left: 100%; }
}
</style>
