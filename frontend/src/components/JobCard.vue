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

    <!-- Web research sources (shown when web_search was used) -->
    <div v-if="job.web_sources && job.web_sources.length" class="sources-section">
      <button class="sources-toggle" @click="showSources = !showSources">
        <svg class="icon-globe" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <circle cx="12" cy="12" r="10"/><line x1="2" y1="12" x2="22" y2="12"/>
          <path d="M12 2a15.3 15.3 0 0 1 4 10 15.3 15.3 0 0 1-4 10 15.3 15.3 0 0 1-4-10 15.3 15.3 0 0 1 4-10z"/>
        </svg>
        {{ showSources ? 'Hide' : 'Show' }} {{ job.web_sources.length }} web source{{ job.web_sources.length !== 1 ? 's' : '' }}
      </button>
      <ul v-if="showSources" class="sources-list">
        <li v-for="(src, i) in job.web_sources" :key="i" class="source-item">
          <span class="source-badge" :class="'badge-' + src.source_type">{{ src.source_type }}</span>
          <a :href="src.url" target="_blank" rel="noopener" class="source-link">{{ src.title || src.url }}</a>
          <span v-if="src.snippet" class="source-snippet">{{ src.snippet.slice(0, 120) }}…</span>
        </li>
      </ul>
    </div>
  </div>
</template>

<script setup>
import { ref, computed } from 'vue'
import { downloadUrl } from '../api.js'

const props = defineProps({
  job: { type: Object, required: true },
})

const showCode = ref(false)
const showSources = ref(false)
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

/* Web sources */
.sources-section {
  margin-top: 10px;
}

.sources-toggle {
  display: flex;
  align-items: center;
  gap: 6px;
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

.icon-globe {
  width: 13px;
  height: 13px;
  flex-shrink: 0;
}

.sources-list {
  list-style: none;
  margin: 8px 0 0 0;
  padding: 0;
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.source-item {
  display: flex;
  flex-direction: column;
  gap: 2px;
  padding: 8px 12px;
  background: var(--bg-primary, var(--bg-secondary));
  border: 1px solid var(--border-light);
  border-radius: var(--radius-sm);
}

.source-badge {
  font-size: 10px;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 0.5px;
  padding: 1px 6px;
  border-radius: 8px;
  width: fit-content;
}

.badge-wikipedia { background: var(--accent-subtle); color: var(--accent); }
.badge-web       { background: var(--success-bg, rgba(34,197,94,0.1)); color: var(--success, #16a34a); }

.source-link {
  font-size: 12px;
  font-weight: 500;
  color: var(--text-primary);
  text-decoration: underline;
  text-underline-offset: 2px;
  word-break: break-all;
}

.source-snippet {
  font-size: 11px;
  color: var(--text-secondary);
  line-height: 1.5;
}
</style>
