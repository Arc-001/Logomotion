<template>
  <div>
    <form @submit.prevent="search" class="search-input-row">
      <input
        id="search-query"
        v-model="query"
        type="text"
        placeholder="Search Manim examples..."
      />
      <button type="submit" class="btn btn-primary btn-sm" :disabled="loading || !query.trim()">
        {{ loading ? 'Searching...' : 'Search' }}
      </button>
    </form>

    <div v-if="searched && results.length === 0 && !loading" class="empty-state">
      No results found.
    </div>

    <div class="search-results">
      <div v-for="r in results" :key="r.id" class="search-result">
        <div class="search-result-header">
          <span class="search-result-id">{{ r.id }}</span>
          <span class="search-result-score">{{ (r.score * 100).toFixed(0) }}%</span>
        </div>
        <div class="search-result-prompt">{{ r.prompt }}</div>
        <div class="tag-list">
          <span v-for="cls in (r.classes || [])" :key="cls" class="tag">{{ cls }}</span>
          <span v-for="anim in (r.animations || [])" :key="anim" class="tag">{{ anim }}</span>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref } from 'vue'
import { apiJSON } from '../api.js'

const query = ref('')
const results = ref([])
const loading = ref(false)
const searched = ref(false)

async function search() {
  const q = query.value.trim()
  if (!q) return
  loading.value = true
  searched.value = true
  try {
    const data = await apiJSON(`/search?query=${encodeURIComponent(q)}&limit=8`)
    results.value = data.results || []
  } catch {
    results.value = []
  } finally {
    loading.value = false
  }
}
</script>

<style scoped>
.search-input-row {
  display: flex;
  gap: 10px;
  margin-bottom: 20px;
}

.search-input-row input {
  flex: 1;
  font-family: inherit;
  font-size: 14px;
  padding: 10px 14px;
  border: 1px solid var(--border);
  border-radius: var(--radius-sm);
  background: var(--bg-secondary);
  color: var(--text-primary);
  outline: none;
  transition: border-color var(--transition), box-shadow var(--transition);
}

.search-input-row input:focus {
  border-color: var(--accent);
  box-shadow: 0 0 0 3px var(--accent-subtle);
}

.search-results {
  display: flex;
  flex-direction: column;
  gap: 10px;
}

.search-result {
  background: var(--bg-secondary);
  border: 1px solid var(--border-light);
  border-radius: var(--radius-md);
  padding: 14px 18px;
  box-shadow: var(--shadow-sm);
}

.search-result-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 6px;
}

.search-result-id {
  font-size: 13px;
  font-weight: 600;
  color: var(--text-primary);
}

.search-result-score {
  font-size: 11px;
  font-weight: 600;
  color: var(--accent);
  background: var(--accent-subtle);
  padding: 2px 8px;
  border-radius: 10px;
}

.search-result-prompt {
  font-size: 13px;
  color: var(--text-secondary);
  margin-bottom: 8px;
  line-height: 1.5;
}
</style>
