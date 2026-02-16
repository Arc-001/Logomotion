<template>
  <div class="job-list">
    <div v-if="jobs.length === 0" class="empty-state">
      No jobs yet. Submit a prompt to get started.
    </div>
    <JobCard v-for="job in jobs" :key="job.job_id" :job="job" />
  </div>
</template>

<script setup>
import JobCard from './JobCard.vue'

defineProps({
  jobs: { type: Array, required: true },
})
</script>

<style scoped>
.job-list {
  display: flex;
  flex-direction: column;
  gap: 12px;
}
</style>
