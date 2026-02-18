async function loadDashboard() {
  const healthPill = document.getElementById('health-pill');
  const healthText = document.getElementById('health-text');

  try {
    const healthRes = await fetch('/api/health');
    const health = await healthRes.json();

    healthPill.textContent = `Backend: ${health.status}`;
    healthPill.classList.add('ok');
    healthText.textContent = `${health.service} работи нормално.`;

    const summaryRes = await fetch('/api/summary');
    const summary = await summaryRes.json();

    document.getElementById('fleet-size').textContent = summary.fleet_size;
    document.getElementById('windows-per-day').textContent = summary.windows_per_day;
    document.getElementById('failure-modes').textContent = summary.failure_modes;
    document.getElementById('feature-count').textContent = summary.feature_count;

    const modeList = document.getElementById('mode-list');
    modeList.innerHTML = '';
    summary.operating_modes.forEach((mode) => {
      const chip = document.createElement('span');
      chip.textContent = mode;
      modeList.appendChild(chip);
    });
  } catch (error) {
    healthPill.textContent = 'Backend: unavailable';
    healthText.textContent = `Грешка при връзка с API: ${error}`;
  }
}

loadDashboard();
