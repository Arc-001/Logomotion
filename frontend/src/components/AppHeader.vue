<template>
  <header class="app-header">
    <div>
      <h1>Logomotion</h1>
      <div class="subtitle">AI-powered mathematical animation generator</div>
    </div>
    <span class="health-badge" :class="{ offline: !online }">
      <span class="health-dot"></span>
      {{ online ? 'Connected' : 'Offline' }}
    </span>
  </header>
</template>

<script setup>
import { ref, onMounted, onUnmounted } from 'vue'
import { apiJSON } from '../api.js'

const online = ref(false)
let interval = null

async function checkHealth() {
  try {
    await apiJSON('/health')
    online.value = true
  } catch {
    online.value = false
  }
}

onMounted(() => {
  checkHealth()
  interval = setInterval(checkHealth, 15000)
})

onUnmounted(() => clearInterval(interval))
</script>

<style scoped>
.app-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 36px;
  padding-bottom: 20px;
  border-bottom: 1px solid var(--border-light);
}

.app-header h1 {
  font-size: 22px;
  font-weight: 700;
  letter-spacing: -0.3px;
  color: var(--text-primary);
}

.subtitle {
  font-size: 13px;
  font-weight: 400;
  color: var(--text-tertiary);
  margin-top: 2px;
}

.health-badge {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  font-size: 12px;
  font-weight: 500;
  padding: 5px 12px;
  border-radius: 20px;
  background: var(--success-bg);
  color: var(--success);
  transition: var(--transition);
}

.health-badge.offline {
  background: var(--error-bg);
  color: var(--error);
}

.health-dot {
  width: 7px;
  height: 7px;
  border-radius: 50%;
  background: currentColor;
}
</style>
