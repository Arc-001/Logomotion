<template>
  <div class="app-shell">
    <AppHeader />

    <div class="tab-bar">
      <button
        class="tab-btn"
        :class="{ active: activeTab === 'generate' }"
        @click="activeTab = 'generate'"
      >Generate</button>
      <button
        class="tab-btn"
        :class="{ active: activeTab === 'search' }"
        @click="activeTab = 'search'"
      >Search</button>
    </div>

    <div v-show="activeTab === 'generate'">
      <GenerateForm @submitted="onJobSubmitted" />
      <div style="margin-top: 24px">
        <div v-if="jobs.length" class="card-title" style="margin-bottom: 12px">Jobs</div>
        <JobList :jobs="jobs" />
      </div>
    </div>

    <div v-show="activeTab === 'search'">
      <SearchPanel />
    </div>
  </div>
</template>

<script setup>
import { ref, reactive, onUnmounted } from 'vue'
import { apiJSON } from './api.js'
import AppHeader from './components/AppHeader.vue'
import GenerateForm from './components/GenerateForm.vue'
import JobList from './components/JobList.vue'
import SearchPanel from './components/SearchPanel.vue'

const activeTab = ref('generate')
const jobs = reactive([])
const pollingIntervals = {}

function onJobSubmitted(data) {
  const job = reactive({
    job_id: data.job_id,
    status: data.status,
    video_path: null,
    code: null,
    error: null,
  })
  jobs.unshift(job)
  startPolling(job)
}

function startPolling(job) {
  if (pollingIntervals[job.job_id]) return

  const poll = async () => {
    try {
      const data = await apiJSON(`/jobs/${job.job_id}`)
      job.status = data.status
      job.video_path = data.video_path
      job.code = data.code
      job.error = data.error
      job.warnings = data.warnings
      job.web_sources = data.web_sources

      if (data.status === 'completed' || data.status === 'failed') {
        clearInterval(pollingIntervals[job.job_id])
        delete pollingIntervals[job.job_id]
      }
    } catch {
      // keep polling
    }
  }

  pollingIntervals[job.job_id] = setInterval(poll, 3000)
  poll()
}

onUnmounted(() => {
  Object.values(pollingIntervals).forEach(clearInterval)
})
</script>

<style scoped>
.app-shell {
  max-width: 860px;
  margin: 0 auto;
  padding: 32px 24px 64px;
}

.tab-bar {
  display: flex;
  gap: 4px;
  margin-bottom: 28px;
  border-bottom: 1px solid var(--border-light);
}

.tab-btn {
  padding: 10px 20px;
  font-size: 13px;
  font-weight: 500;
  color: var(--text-tertiary);
  background: none;
  border: none;
  border-bottom: 2px solid transparent;
  cursor: pointer;
  transition: var(--transition);
  margin-bottom: -1px;
}

.tab-btn:hover {
  color: var(--text-secondary);
}

.tab-btn.active {
  color: var(--accent);
  border-bottom-color: var(--accent);
}
</style>
