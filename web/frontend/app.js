function renderList(containerId, items, renderer) {
  const container = document.getElementById(containerId);
  container.innerHTML = '';
  items.forEach((item) => {
    container.appendChild(renderer(item));
  });
}

function createTextNode(tagName, text) {
  const el = document.createElement(tagName);
  el.textContent = text;
  return el;
}

async function loadDashboard() {
  const healthPill = document.getElementById('health-pill');
  const healthText = document.getElementById('health-text');
  const summarySource = document.getElementById('summary-source');

  try {
    const [healthRes, summaryRes, guideRes] = await Promise.all([
      fetch('/api/health'),
      fetch('/api/summary'),
      fetch('/api/guide'),
    ]);

    if (!healthRes.ok || !summaryRes.ok || !guideRes.ok) {
      throw new Error('Някой от API endpoint-ите върна грешка.');
    }

    const health = await healthRes.json();
    const summary = await summaryRes.json();
    const guide = await guideRes.json();

    healthPill.textContent = `Backend: ${health.status}`;
    healthPill.classList.add('ok');
    healthText.textContent = `${health.service} работи нормално.`;

    document.getElementById('fleet-size').textContent = summary.fleet_size;
    document.getElementById('windows-per-day').textContent = summary.windows_per_day;
    document.getElementById('failure-modes').textContent = summary.failure_modes;
    document.getElementById('feature-count').textContent = summary.feature_count;
    document.getElementById('rows-total').textContent = summary.rows_total ?? '—';

    const sourceText = summary.source === 'live'
      ? `Източник: LIVE parquet (${summary.data_files} файла). ${summary.message}`
      : `Източник: DEMO стойности. ${summary.message}`;
    summarySource.textContent = sourceText;

    renderList('mode-list', summary.operating_modes, (mode) => {
      const chip = document.createElement('span');
      chip.textContent = mode;
      return chip;
    });

    renderList('pipeline-steps', guide.pipeline_steps, (step) => createTextNode('li', step));
    renderList('deliverables', guide.final_deliverables, (item) => createTextNode('li', item));

    renderList('commands', guide.run_commands, (command) => {
      const wrapper = document.createElement('article');
      wrapper.className = 'command';

      const label = document.createElement('div');
      label.className = 'command__label';
      label.textContent = command.title;

      const code = document.createElement('code');
      code.textContent = command.command;

      wrapper.appendChild(label);
      wrapper.appendChild(code);
      return wrapper;
    });
  } catch (error) {
    healthPill.textContent = 'Backend: unavailable';
    healthPill.classList.add('error');
    healthText.textContent = `Грешка при връзка с API: ${error.message}`;
    summarySource.textContent = '';
  }
}

loadDashboard();
