<template>
  <div class="card">
    <div class="card-title">Generate Video</div>
    <form @submit.prevent="submit" class="form-grid">
      <div class="form-group">
        <label for="gen-prompt">Prompt</label>
        <textarea
          id="gen-prompt"
          v-model="form.prompt"
          placeholder="Describe the mathematical animation you want to create..."
          rows="3"
        ></textarea>
      </div>
      <div class="form-group">
        <label for="gen-title">Title (optional)</label>
        <input
          id="gen-title"
          v-model="form.title"
          type="text"
          placeholder="Scene title"
        />
      </div>
      <div class="form-row">
        <div class="form-group">
          <label for="gen-length">Length</label>
          <div class="range-wrapper">
            <input
              id="gen-length"
              v-model.number="form.length"
              type="range"
              min="0.1"
              max="10"
              step="0.1"
            />
            <span class="range-value">{{ form.length.toFixed(1) }}m</span>
          </div>
        </div>
        <div class="form-group">
          <label for="gen-depth">Depth</label>
          <select id="gen-depth" v-model="form.depth">
            <option value="basic">Basic</option>
            <option value="detailed">Detailed</option>
            <option value="comprehensive">Comprehensive</option>
          </select>
        </div>
        <div class="form-group">
          <label for="gen-orientation">Orientation</label>
          <select id="gen-orientation" v-model="form.orientation">
            <option value="landscape">Landscape</option>
            <option value="portrait">Portrait</option>
          </select>
        </div>
      </div>
      <div v-if="error" class="form-error">{{ error }}</div>
      <div class="form-actions">
        <button type="submit" class="btn btn-primary" :disabled="submitting || !form.prompt.trim()">
          <span v-if="submitting" class="spinner"></span>
          {{ submitting ? 'Submitting...' : 'Generate' }}
        </button>
      </div>
    </form>
  </div>
</template>

<script setup>
import { reactive, ref } from 'vue'
import { apiJSON } from '../api.js'

const emit = defineEmits(['submitted'])

const form = reactive({
  prompt: '',
  title: '',
  length: 1.0,
  depth: 'detailed',
  orientation: 'landscape',
})
const submitting = ref(false)
const error = ref('')

async function submit() {
  if (!form.prompt.trim()) return
  submitting.value = true
  error.value = ''
  try {
    const data = await apiJSON('/generate', {
      method: 'POST',
      body: JSON.stringify({
        prompt: form.prompt.trim(),
        title: form.title.trim() || undefined,
        length: form.length,
        depth: form.depth,
        orientation: form.orientation,
      }),
    })
    emit('submitted', data)
    form.prompt = ''
    form.title = ''
  } catch (e) {
    error.value = e.message
  } finally {
    submitting.value = false
  }
}
</script>

<style scoped>
.form-grid {
  display: grid;
  gap: 16px;
}

.form-group {
  display: flex;
  flex-direction: column;
  gap: 5px;
}

.form-group label {
  font-size: 12px;
  font-weight: 600;
  color: var(--text-secondary);
  text-transform: uppercase;
  letter-spacing: 0.5px;
}

.form-group input,
.form-group select,
.form-group textarea {
  font-family: inherit;
  font-size: 14px;
  padding: 10px 14px;
  border: 1px solid var(--border);
  border-radius: var(--radius-sm);
  background: var(--bg-secondary);
  color: var(--text-primary);
  transition: border-color var(--transition), box-shadow var(--transition);
  outline: none;
}

.form-group input:focus,
.form-group select:focus,
.form-group textarea:focus {
  border-color: var(--accent);
  box-shadow: 0 0 0 3px var(--accent-subtle);
}

.form-group textarea {
  resize: vertical;
  min-height: 80px;
}

.form-row {
  display: grid;
  grid-template-columns: 1fr 1fr 1fr;
  gap: 16px;
}

@media (max-width: 600px) {
  .form-row {
    grid-template-columns: 1fr;
  }
}

.range-wrapper {
  display: flex;
  align-items: center;
  gap: 12px;
}

.range-wrapper input[type="range"] {
  flex: 1;
  padding: 0;
  height: 4px;
  -webkit-appearance: none;
  appearance: none;
  background: var(--border);
  border: none;
  border-radius: 2px;
  outline: none;
}

.range-wrapper input[type="range"]::-webkit-slider-thumb {
  -webkit-appearance: none;
  appearance: none;
  width: 16px;
  height: 16px;
  border-radius: 50%;
  background: var(--accent);
  cursor: pointer;
  border: 2px solid var(--bg-secondary);
  box-shadow: var(--shadow-sm);
}

.range-value {
  font-size: 13px;
  font-weight: 600;
  color: var(--accent);
  min-width: 48px;
  text-align: right;
}

.form-error {
  font-size: 12px;
  color: var(--error);
  background: var(--error-bg);
  padding: 8px 12px;
  border-radius: var(--radius-sm);
  line-height: 1.5;
}

.form-actions {
  display: flex;
  justify-content: flex-end;
  margin-top: 8px;
}
</style>
