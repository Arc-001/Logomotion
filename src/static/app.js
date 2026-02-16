/* ===================================================================
   Logomotion — Vue 3 Application (CDN, Composition API)
   =================================================================== */

const { createApp, ref, reactive, computed, onMounted, onUnmounted, watch, nextTick } = Vue;

// ---------------------------------------------------------------------------
// API helpers
// ---------------------------------------------------------------------------

const API = 'http://localhost:8000';

async function api(path, opts = {}) {
    const res = await fetch(`${API}${path}`, {
        headers: { 'Content-Type': 'application/json', ...opts.headers },
        ...opts,
    });
    if (!res.ok) {
        const err = await res.json().catch(() => ({ detail: res.statusText }));
        throw new Error(err.detail || 'Request failed');
    }
    return res;
}

async function apiJSON(path, opts) {
    return (await api(path, opts)).json();
}

// ---------------------------------------------------------------------------
// AppHeader component
// ---------------------------------------------------------------------------

const AppHeader = {
    setup() {
        const online = ref(false);
        let interval = null;

        async function checkHealth() {
            try {
                await apiJSON('/health');
                online.value = true;
            } catch {
                online.value = false;
            }
        }

        onMounted(() => {
            checkHealth();
            interval = setInterval(checkHealth, 15000);
        });

        onUnmounted(() => clearInterval(interval));

        return { online };
    },
    template: `
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
  `,
};

// ---------------------------------------------------------------------------
// GenerateForm component
// ---------------------------------------------------------------------------

const GenerateForm = {
    emits: ['submitted'],
    setup(_, { emit }) {
        const form = reactive({
            prompt: '',
            title: '',
            length: 1.0,
            depth: 'detailed',
            orientation: 'landscape',
        });
        const submitting = ref(false);
        const error = ref('');

        async function submit() {
            if (!form.prompt.trim()) return;
            submitting.value = true;
            error.value = '';
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
                });
                emit('submitted', data);
                form.prompt = '';
                form.title = '';
            } catch (e) {
                error.value = e.message;
            } finally {
                submitting.value = false;
            }
        }

        return { form, submitting, error, submit };
    },
    template: `
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
        <div v-if="error" class="job-error">{{ error }}</div>
        <div class="form-actions">
          <button type="submit" class="btn btn-primary" :disabled="submitting || !form.prompt.trim()">
            <span v-if="submitting" class="spinner"></span>
            {{ submitting ? 'Submitting...' : 'Generate' }}
          </button>
        </div>
      </form>
    </div>
  `,
};

// ---------------------------------------------------------------------------
// JobCard component
// ---------------------------------------------------------------------------

const JobCard = {
    props: ['job'],
    setup(props) {
        const showCode = ref(false);

        function downloadUrl() {
            return `${API}/jobs/${props.job.job_id}/download`;
        }

        return { showCode, downloadUrl };
    },
    template: `
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

      <div v-if="job.status === 'completed'" class="job-actions" style="margin-top: 10px;">
        <a :href="downloadUrl()" class="btn btn-primary btn-sm" download>
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
  `,
};

// ---------------------------------------------------------------------------
// JobList component
// ---------------------------------------------------------------------------

const JobList = {
    props: ['jobs'],
    components: { JobCard },
    template: `
    <div class="job-list">
      <div v-if="jobs.length === 0" class="empty-state">
        No jobs yet. Submit a prompt to get started.
      </div>
      <job-card
        v-for="job in jobs"
        :key="job.job_id"
        :job="job"
      />
    </div>
  `,
};

// ---------------------------------------------------------------------------
// SearchPanel component
// ---------------------------------------------------------------------------

const SearchPanel = {
    setup() {
        const query = ref('');
        const results = ref([]);
        const loading = ref(false);
        const searched = ref(false);

        async function search() {
            const q = query.value.trim();
            if (!q) return;
            loading.value = true;
            searched.value = true;
            try {
                const data = await apiJSON(`/search?query=${encodeURIComponent(q)}&limit=8`);
                results.value = data.results || [];
            } catch {
                results.value = [];
            } finally {
                loading.value = false;
            }
        }

        return { query, results, loading, searched, search };
    },
    template: `
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
  `,
};

// ---------------------------------------------------------------------------
// AppShell — root component
// ---------------------------------------------------------------------------

const AppShell = {
    components: { AppHeader, GenerateForm, JobList, SearchPanel },
    setup() {
        const activeTab = ref('generate');
        const jobs = reactive([]);
        const pollingIntervals = {};

        function onJobSubmitted(data) {
            const job = reactive({
                job_id: data.job_id,
                status: data.status,
                video_path: null,
                code: null,
                error: null,
            });
            jobs.unshift(job);
            startPolling(job);
        }

        function startPolling(job) {
            if (pollingIntervals[job.job_id]) return;

            const poll = async () => {
                try {
                    const data = await apiJSON(`/jobs/${job.job_id}`);
                    job.status = data.status;
                    job.video_path = data.video_path;
                    job.code = data.code;
                    job.error = data.error;

                    if (data.status === 'completed' || data.status === 'failed') {
                        clearInterval(pollingIntervals[job.job_id]);
                        delete pollingIntervals[job.job_id];
                    }
                } catch {
                    // keep polling
                }
            };

            pollingIntervals[job.job_id] = setInterval(poll, 3000);
            poll();
        }

        onUnmounted(() => {
            Object.values(pollingIntervals).forEach(clearInterval);
        });

        return { activeTab, jobs, onJobSubmitted };
    },
    template: `
    <div class="app-shell">
      <app-header />

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
        <generate-form @submitted="onJobSubmitted" />
        <div style="margin-top: 24px;">
          <div class="card-title" style="margin-bottom: 12px;" v-if="jobs.length">Jobs</div>
          <job-list :jobs="jobs" />
        </div>
      </div>

      <div v-show="activeTab === 'search'">
        <search-panel />
      </div>
    </div>
  `,
};

// ---------------------------------------------------------------------------
// Mount
// ---------------------------------------------------------------------------

createApp({
    components: { AppShell },
    template: '<app-shell />',
}).mount('#app');
