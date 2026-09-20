/**
 * CocoHealth AI - Mobile Application Client Logic
 * Handles camera capture, canvas bounding box rendering, API communication,
 * single-palm analysis, and multi-palm plantation aggregation.
 */

// State Management
const state = {
  currentFile: null,
  currentImageElement: null,
  currentRecord: null,
  scoutingSession: [],
  activeMode: 'single', // 'single' or 'scouting'
  apiReady: false,
  classColors: {
    'bud root dropping': '#e7298a',
    'bud rot': '#e41a1c',
    'gray leaf spot': '#7570b3',
    'leaf rot': '#d95f02',
    'stembleeding': '#a6761d'
  }
};

// DOM Elements
const elements = {
  // Screens
  screenHome: document.getElementById('screen-home'),
  screenPreview: document.getElementById('screen-preview'),
  screenResults: document.getElementById('screen-results'),
  screenPlantation: document.getElementById('screen-plantation'),

  // Header status
  apiStatusText: document.getElementById('api-status-text'),
  apiStatusBadge: document.getElementById('api-status-badge'),

  // Inputs
  cameraInput: document.getElementById('camera-input'),
  galleryInput: document.getElementById('gallery-input'),
  inputPlantationId: document.getElementById('input-plantation-id'),

  // Mode buttons
  btnModeSingle: document.getElementById('btn-mode-single'),
  btnModeScouting: document.getElementById('btn-mode-scouting'),
  sessionCountBadge: document.getElementById('session-count-badge'),

  // Preview elements
  imagePreview: document.getElementById('image-preview-element'),
  previewFileName: document.getElementById('preview-file-name'),
  previewFileSize: document.getElementById('preview-file-size'),
  btnRunAnalysis: document.getElementById('btn-run-analysis'),
  btnRetakePhoto: document.getElementById('btn-retake-photo'),
  btnBackToHome: document.getElementById('btn-back-to-home'),
  loadingOverlay: document.getElementById('loading-overlay'),
  loadingStepText: document.getElementById('loading-step-text'),

  // Results elements
  detectionCanvas: document.getElementById('detection-canvas'),
  resultLatencyText: document.getElementById('result-latency-text'),
  resultStatusPill: document.getElementById('result-status-pill'),
  resultPhiScore: document.getElementById('result-phi-score'),
  resultTierName: document.getElementById('result-tier-name'),
  resultRecommendation: document.getElementById('result-recommendation'),
  resultConfirmedCount: document.getElementById('result-confirmed-count'),
  resultCandidateCount: document.getElementById('result-candidate-count'),
  resultAreaProxy: document.getElementById('result-area-proxy'),
  detectionsContainer: document.getElementById('detections-container'),
  btnAddToSession: document.getElementById('btn-add-to-session'),
  btnScanAnother: document.getElementById('btn-scan-another'),
  btnViewPlantationReport: document.getElementById('btn-view-plantation-report'),
  btnBackToPreview: document.getElementById('btn-back-to-preview'),

  // Plantation report elements
  reportPlantationId: document.getElementById('report-plantation-id'),
  reportSessionDate: document.getElementById('report-session-date'),
  reportPhiVal: document.getElementById('report-phi-val'),
  reportPhiTier: document.getElementById('report-phi-tier'),
  reportTotalPalms: document.getElementById('report-total-palms'),
  reportPosRate: document.getElementById('report-pos-rate'),
  reportTotalLesions: document.getElementById('report-total-lesions'),
  pathologyTableBody: document.getElementById('pathology-table-body'),
  btnBackFromPlantation: document.getElementById('btn-back-from-plantation'),
  btnExportJson: document.getElementById('btn-export-json'),
  btnResetSession: document.getElementById('btn-reset-session'),

  // Toast
  toastBanner: document.getElementById('toast-banner'),
  toastMessage: document.getElementById('toast-message'),

  // Specs
  specModelName: document.getElementById('spec-model-name'),
  specConfCutoff: document.getElementById('spec-conf-cutoff')
};

// Initialize Application
document.addEventListener('DOMContentLoaded', () => {
  checkApiHealth();
  setupEventListeners();
});

// 1. API Health Check
async function checkApiHealth() {
  try {
    const res = await fetch('/api/v1/health');
    if (res.ok) {
      const data = await res.json();
      state.apiReady = true;
      elements.apiStatusText.textContent = 'API Online';
      elements.apiStatusBadge.style.backgroundColor = 'var(--primary-tint)';
      elements.apiStatusBadge.style.color = 'var(--primary)';
      if (elements.specModelName && data.model_id) {
        elements.specModelName.textContent = `${data.model_id} (${data.input_resolution}x${data.input_resolution})`;
      }
      if (elements.specConfCutoff && data.operational_confidence_threshold) {
        elements.specConfCutoff.textContent = `τ = ${data.operational_confidence_threshold}`;
      }
    } else {
      throw new Error(`API returned HTTP ${res.status}`);
    }
  } catch (err) {
    state.apiReady = false;
    elements.apiStatusText.textContent = 'Offline';
    elements.apiStatusBadge.style.backgroundColor = '#fee2e2';
    elements.apiStatusBadge.style.color = '#b91c1c';
    showToast('Warning: Backend inference API is currently unreachable.');
  }
}

// 2. Event Listeners
function setupEventListeners() {
  // File inputs
  elements.cameraInput.addEventListener('change', handleFileSelection);
  elements.galleryInput.addEventListener('change', handleFileSelection);

  // Mode toggles
  elements.btnModeSingle.addEventListener('click', () => setMode('single'));
  elements.btnModeScouting.addEventListener('click', () => setMode('scouting'));

  // Preview navigation
  elements.btnBackToHome.addEventListener('click', () => showScreen('home'));
  elements.btnRetakePhoto.addEventListener('click', () => {
    elements.cameraInput.value = '';
    elements.galleryInput.value = '';
    showScreen('home');
  });
  elements.btnRunAnalysis.addEventListener('click', executeAnalysis);

  // Results navigation
  elements.btnBackToPreview.addEventListener('click', () => showScreen('preview'));
  elements.btnScanAnother.addEventListener('click', () => {
    elements.cameraInput.value = '';
    elements.galleryInput.value = '';
    showScreen('home');
  });
  elements.btnAddToSession.addEventListener('click', addCurrentToSession);
  elements.btnViewPlantationReport.addEventListener('click', generatePlantationReport);

  // Plantation report navigation
  elements.btnBackFromPlantation.addEventListener('click', () => showScreen('results'));
  elements.btnExportJson.addEventListener('click', exportPlantationJson);
  elements.btnResetSession.addEventListener('click', resetSession);
}

// Mode Management
function setMode(mode) {
  state.activeMode = mode;
  if (mode === 'single') {
    elements.btnModeSingle.classList.add('active');
    elements.btnModeScouting.classList.remove('active');
  } else {
    elements.btnModeSingle.classList.remove('active');
    elements.btnModeScouting.classList.add('active');
  }
}

// Screen Navigation
function showScreen(screenId) {
  const screens = {
    home: elements.screenHome,
    preview: elements.screenPreview,
    results: elements.screenResults,
    plantation: elements.screenPlantation
  };

  Object.values(screens).forEach(screen => screen.classList.remove('active'));
  if (screens[screenId]) {
    screens[screenId].classList.add('active');
    window.scrollTo({ top: 0, behavior: 'smooth' });
  }
}

// Handle Image Selection
function handleFileSelection(e) {
  const file = e.target.files[0];
  if (!file) return;

  // Basic validation
  if (!file.type.match(/^image\/(jpeg|png|webp|jpg)$/i)) {
    showToast('Invalid file format. Please choose a JPEG, PNG, or WebP image.');
    return;
  }

  state.currentFile = file;
  elements.previewFileName.textContent = file.name;
  elements.previewFileSize.textContent = formatBytes(file.size);

  const reader = new FileReader();
  reader.onload = (event) => {
    elements.imagePreview.src = event.target.result;
    const img = new Image();
    img.onload = () => {
      state.currentImageElement = img;
      showScreen('preview');
    };
    img.src = event.target.result;
  };
  reader.readAsDataURL(file);
}

// 3. Execute Analysis
async function executeAnalysis() {
  if (!state.currentFile) {
    showToast('No palm image selected.');
    return;
  }

  // Show loading overlay
  elements.loadingOverlay.classList.add('active');
  elements.loadingStepText.textContent = 'Preprocessing image buffer...';

  const formData = new FormData();
  formData.append('file', state.currentFile);
  formData.append('plantation_id', elements.inputPlantationId.value.trim() || 'COCO_BLOCK_01');

  // Attempt to attach device GPS if available (non-blocking)
  if (navigator.geolocation) {
    navigator.geolocation.getCurrentPosition(
      (pos) => {
        formData.append('gps_lat', pos.coords.latitude);
        formData.append('gps_lng', pos.coords.longitude);
      },
      () => { /* Geolocation optional */ },
      { timeout: 1500 }
    );
  }

  try {
    setTimeout(() => {
      elements.loadingStepText.textContent = 'Running YOLOv8 localized inference...';
    }, 400);

    setTimeout(() => {
      elements.loadingStepText.textContent = 'Computing 2D sweep-line area proxy & health index...';
    }, 800);

    const res = await fetch('/api/v1/analyze', {
      method: 'POST',
      body: formData
    });

    if (!res.ok) {
      const errData = await res.json().catch(() => ({ detail: `HTTP ${res.status}` }));
      throw new Error(errData.detail || 'Inference failed');
    }

    const record = await res.json();
    state.currentRecord = record;

    // Render results
    renderResults(record);
    showScreen('results');

    // If in scouting mode, automatically prompt or allow adding
    if (state.activeMode === 'scouting') {
      showToast('Scan complete. Tap "Save to Walk" to include in plantation report.');
    }
  } catch (err) {
    console.error('Analysis error:', err);
    showToast(`Error: ${err.message}`);
  } finally {
    elements.loadingOverlay.classList.remove('active');
  }
}

// 4. Render Detection Results & Canvas Overlay
function renderResults(record) {
  // Latency
  elements.resultLatencyText.textContent = `${record.inference_latency_ms || 74.5} ms`;

  // Status Pill
  const status = record.observation_status;
  elements.resultStatusPill.textContent = status.replace(/_/g, ' ');
  elements.resultStatusPill.className = 'status-pill-large';
  if (status === 'DISEASE_DETECTED') {
    elements.resultStatusPill.classList.add('detected');
  } else if (status === 'NO_DISEASE_DETECTED') {
    elements.resultStatusPill.classList.add('healthy');
  } else {
    elements.resultStatusPill.classList.add('candidate');
  }

  // Health Assessment
  const health = record.health_assessment || {};
  elements.resultPhiScore.textContent = health.phi_score !== undefined ? health.phi_score.toFixed(1) : '100.0';
  elements.resultTierName.textContent = health.health_tier || 'EXCELLENT';
  elements.resultRecommendation.textContent = health.recommended_action || 'Routine monitoring.';

  // Metrics
  elements.resultConfirmedCount.textContent = record.accepted_detection_count || 0;
  elements.resultCandidateCount.textContent = record.candidate_detection_count || 0;
  const areaProxyPct = ((record.relative_affected_area_proxy || 0) * 100).toFixed(1);
  elements.resultAreaProxy.textContent = `${areaProxyPct}%`;

  // Draw Bounding Boxes on Canvas
  drawDetectionsOnCanvas(record);

  // Populate Detections List
  populateDetectionsList(record);
}

// 5. Draw Bounding Boxes on HTML5 Canvas
function drawDetectionsOnCanvas(record) {
  const canvas = elements.detectionCanvas;
  const ctx = canvas.getContext('2d');
  const img = state.currentImageElement;

  if (!img) return;

  // Set native canvas dimensions matching source image
  canvas.width = img.naturalWidth || img.width;
  canvas.height = img.naturalHeight || img.height;

  // Draw base image
  ctx.drawImage(img, 0, 0, canvas.width, canvas.height);

  const acceptedDets = record.accepted_detections || [];
  const candidateDets = record.candidate_detections || [];

  // 1. Draw accepted detections (solid bold outline)
  acceptedDets.forEach(det => {
    drawBox(ctx, det, false);
  });

  // 2. Draw candidate detections (dashed line outline)
  candidateDets.forEach(det => {
    drawBox(ctx, det, true);
  });
}

function drawBox(ctx, det, isCandidate) {
  const b = det.bounding_box;
  const cname = det.disease_class;
  const conf = det.confidence;
  const color = state.classColors[cname] || '#ffffff';

  const x1 = b.x1;
  const y1 = b.y1;
  const w = b.x2 - b.x1;
  const h = b.y2 - b.y1;

  ctx.save();

  // Bounding rectangle
  ctx.lineWidth = isCandidate ? 3 : 5;
  ctx.strokeStyle = isCandidate ? '#ff9800' : color;
  if (isCandidate) {
    ctx.setLineDash([8, 6]);
  }
  ctx.strokeRect(x1, y1, w, h);

  // Label tag
  const labelText = isCandidate 
    ? `? ${cname} ${(conf * 100).toFixed(0)}% (Candidate)` 
    : `✓ ${cname} ${(conf * 100).toFixed(0)}%`;

  ctx.font = 'bold 22px Inter, sans-serif';
  const textMetrics = ctx.measureText(labelText);
  const pad = 6;
  const boxHeight = 32;
  const boxWidth = textMetrics.width + pad * 2;

  // Background banner
  ctx.fillStyle = isCandidate ? 'rgba(255, 152, 0, 0.9)' : color;
  ctx.fillRect(x1, Math.max(0, y1 - boxHeight), boxWidth, boxHeight);

  // Text
  ctx.fillStyle = '#ffffff';
  ctx.fillText(labelText, x1 + pad, Math.max(boxHeight - 8, y1 - 8));

  ctx.restore();
}

// 6. Populate Itemized Detection Cards
function populateDetectionsList(record) {
  const container = elements.detectionsContainer;
  container.innerHTML = '';

  const acceptedDets = record.accepted_detections || [];
  const candidateDets = record.candidate_detections || [];

  if (acceptedDets.length === 0 && candidateDets.length === 0) {
    container.innerHTML = '<div class="empty-detections-text">🌿 No pathological lesions detected in this palm photograph.</div>';
    return;
  }

  // Accepted detections
  acceptedDets.forEach((det, idx) => {
    const row = createDetectionRow(det, false, idx + 1);
    container.appendChild(row);
  });

  // Candidate detections
  candidateDets.forEach((det, idx) => {
    const row = createDetectionRow(det, true, idx + 1);
    container.appendChild(row);
  });
}

function createDetectionRow(det, isCandidate, num) {
  const row = document.createElement('div');
  row.className = `detection-row ${isCandidate ? 'candidate' : 'confirmed'}`;

  const cname = det.disease_class;
  const confPct = (det.confidence * 100).toFixed(1);
  const bbox = det.bounding_box;

  row.innerHTML = `
    <div class="detection-info">
      <span class="detection-class-name">${num}. ${cname}</span>
      <span class="detection-bbox-coord">[${bbox.x1.toFixed(0)}, ${bbox.y1.toFixed(0)}, ${bbox.x2.toFixed(0)}, ${bbox.y2.toFixed(0)}]</span>
    </div>
    <div class="detection-badge-box">
      <span class="confidence-badge ${isCandidate ? 'low' : 'high'}">${confPct}%</span>
      ${isCandidate ? '<span class="candidate-tag">Candidate</span>' : ''}
    </div>
  `;
  return row;
}

// 7. Session / Scouting Walk Management
function addCurrentToSession() {
  if (!state.currentRecord) {
    showToast('No analyzed palm record to save.');
    return;
  }

  // Check if already in session
  const alreadyExists = state.scoutingSession.some(r => r.image_id === state.currentRecord.image_id);
  if (alreadyExists) {
    showToast('This palm is already included in the active scouting walk.');
    return;
  }

  state.scoutingSession.push(state.currentRecord);
  elements.sessionCountBadge.textContent = state.scoutingSession.length;
  showToast(`Palm saved to scouting walk! (${state.scoutingSession.length} total)`);
}

async function generatePlantationReport() {
  let recordsToAggregate = [];

  if (state.scoutingSession.length > 0) {
    recordsToAggregate = state.scoutingSession;
  } else if (state.currentRecord) {
    recordsToAggregate = [state.currentRecord];
  } else {
    showToast('Please analyze at least one palm before generating a plantation report.');
    return;
  }

  try {
    const res = await fetch('/api/v1/aggregate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        plantation_id: elements.inputPlantationId.value.trim() || 'COCO_BLOCK_01',
        records: recordsToAggregate
      })
    });

    if (!res.ok) throw new Error(`Aggregation failed: HTTP ${res.status}`);

    const report = await res.json();
    renderPlantationReport(report);
    showScreen('plantation');
  } catch (err) {
    showToast(`Failed to generate report: ${err.message}`);
  }
}

function renderPlantationReport(report) {
  elements.reportPlantationId.textContent = `BLOCK: ${report.meta.plantation_id}`;
  elements.reportSessionDate.textContent = report.meta.observation_session_id;

  const phi = report.plantation_health_index;
  elements.reportPhiVal.textContent = phi.phi_composite_score.toFixed(1);
  elements.reportPhiTier.textContent = phi.health_tier;

  const counts = report.observation_counts;
  const rates = report.observation_rates;
  elements.reportTotalPalms.textContent = counts.successfully_processed_images;
  elements.reportPosRate.textContent = `${rates.disease_positive_image_percentage.toFixed(1)}%`;

  let totalBoxes = 0;
  const tableBody = elements.pathologyTableBody;
  tableBody.innerHTML = '';

  const dist = report.disease_distribution;
  Object.keys(dist).forEach(cname => {
    const d = dist[cname];
    totalBoxes += d.detection_count;

    const tr = document.createElement('tr');
    tr.innerHTML = `
      <td><strong>${cname}</strong></td>
      <td>${d.detection_count}</td>
      <td>${d.unique_images_affected}</td>
      <td>${d.percentage_of_images_affected.toFixed(1)}%</td>
    `;
    tableBody.appendChild(tr);
  });

  elements.reportTotalLesions.textContent = totalBoxes;
  state.lastReport = report;
}

function exportPlantationJson() {
  if (!state.lastReport) {
    showToast('No report available to export.');
    return;
  }
  const dataStr = 'data:text/json;charset=utf-8,' + encodeURIComponent(JSON.stringify(state.lastReport, null, 2));
  const downloadAnchor = document.createElement('a');
  downloadAnchor.setAttribute('href', dataStr);
  downloadAnchor.setAttribute('download', `plantation_health_report_${Date.now()}.json`);
  document.body.appendChild(downloadAnchor);
  downloadAnchor.click();
  downloadAnchor.remove();
  showToast('Plantation report downloaded.');
}

function resetSession() {
  state.scoutingSession = [];
  elements.sessionCountBadge.textContent = '0';
  showToast('Scouting session cleared.');
  showScreen('home');
}

// Utilities
function formatBytes(bytes) {
  if (bytes === 0) return '0 Bytes';
  const k = 1024;
  const sizes = ['Bytes', 'KB', 'MB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + ' ' + sizes[i];
}

let toastTimer = null;
function showToast(message) {
  elements.toastMessage.textContent = message;
  elements.toastBanner.classList.add('show');
  if (toastTimer) clearTimeout(toastTimer);
  toastTimer = setTimeout(() => {
    elements.toastBanner.classList.remove('show');
  }, 3500);
}
