"use strict";

// When the page is opened via Live Server or another port, call the API on uvicorn.
const API_BASE =
  window.location.port === "8000" ? "" : "http://127.0.0.1:8000";

const dropzone = document.getElementById("dropzone");
const fileInput = document.getElementById("file-input");
const previewRow = document.getElementById("preview-row");
const previewImg = document.getElementById("preview-img");
const previewName = document.getElementById("preview-name");
const previewSize = document.getElementById("preview-size");
const analyzeBtn = document.getElementById("analyze-btn");
const clearBtn = document.getElementById("clear-btn");
const loader = document.getElementById("loader");
const loaderText = document.getElementById("loader-text");
const errorBox = document.getElementById("error-box");
const resultsSection = document.getElementById("results");

let selectedFile = null;
let lastRequestId = null;
let lastDetections = [];
let lastGallery = [];
let lastTimingsMs = null;
let skuCatalog = [];
let productFilter = null;
let cropFilter = "all";
let reviewTab = "crops";
let catalogQuery = "";

const PACKAGING_TYPES = ["bottle", "can", "carton"];
const NEW_SKU_VALUE = "__new__";

/* ---------------- upload handling ---------------- */

dropzone.addEventListener("click", () => fileInput.click());
dropzone.addEventListener("keydown", (e) => {
  if (e.key === "Enter" || e.key === " ") fileInput.click();
});

["dragenter", "dragover"].forEach((evt) =>
  dropzone.addEventListener(evt, (e) => {
    e.preventDefault();
    dropzone.classList.add("dragover");
  })
);
["dragleave", "drop"].forEach((evt) =>
  dropzone.addEventListener(evt, (e) => {
    e.preventDefault();
    dropzone.classList.remove("dragover");
  })
);
dropzone.addEventListener("drop", (e) => {
  const file = e.dataTransfer.files[0];
  if (file) setFile(file);
});
fileInput.addEventListener("change", () => {
  if (fileInput.files[0]) setFile(fileInput.files[0]);
});

function setFile(file) {
  if (!file.type.startsWith("image/")) {
    showError("Please choose an image file (JPG or PNG).");
    return;
  }
  selectedFile = file;
  hideError();
  previewImg.src = URL.createObjectURL(file);
  previewName.textContent = file.name;
  previewSize.textContent = formatBytes(file.size);
  previewRow.classList.remove("hidden");
  resultsSection.classList.add("hidden");
}

clearBtn.addEventListener("click", () => {
  selectedFile = null;
  fileInput.value = "";
  previewRow.classList.add("hidden");
  resultsSection.classList.add("hidden");
  hideError();
});

/* ---------------- analyze ---------------- */

analyzeBtn.addEventListener("click", async () => {
  if (!selectedFile) return;

  analyzeBtn.disabled = true;
  loader.classList.remove("hidden");
  loaderText.textContent = "Analyzing shelf\u2026";
  hideError();
  resultsSection.classList.add("hidden");

  const form = new FormData();
  form.append("file", selectedFile);

  try {
    const resp = await fetch(`${API_BASE}/api/analyze`, {
      method: "POST",
      body: form,
    });
    if (!resp.ok) {
      const detail = await resp.json().catch(() => ({}));
      throw new Error(detail.detail || `Server error (${resp.status})`);
    }
    const data = await resp.json();
    skuCatalog = await loadSkus();
    renderResults(data);
  } catch (err) {
    showError(err.message || "Something went wrong.");
  } finally {
    analyzeBtn.disabled = false;
    loader.classList.add("hidden");
  }
});

/* ---------------- rendering ---------------- */

function renderResults(data) {
  lastRequestId = data.request_id || null;
  lastDetections = (data.detections || []).map((d) => ({
    ...d,
    origSkuId: d.sku_id || null,
    origPackaging: d.packaging || null,
    origStatus: d.status,
  }));
  lastGallery = (data.gallery || []).map((g) => ({ ...g }));
  lastTimingsMs = data.timings_ms || null;
  productFilter = null;
  cropFilter = "all";
  reviewTab = "crops";
  catalogQuery = "";
  const catalogSearch = document.getElementById("catalog-search");
  if (catalogSearch) catalogSearch.value = "";
  showSaveStatus("");
  setReviewTab("crops");

  const plainUrl = apiUrl(data.image.plain_url || data.image.annotated_url) + "?t=" + Date.now();
  const annotatedUrl = apiUrl(data.image.annotated_url) + "?t=" + Date.now();
  const shelfImg = document.getElementById("annotated-img");
  shelfImg.onload = () => drawBoxes();
  shelfImg.src = plainUrl;
  document.getElementById("annotated-link").href = annotatedUrl;

  renderStats();
  renderGallery();
  applyView();

  resultsSection.classList.remove("hidden");
  resultsSection.scrollIntoView({ behavior: "smooth", block: "start" });
}

function renderStats() {
  const recognized = lastDetections.filter((d) => d.status === "recognized").length;
  const unknown = lastDetections.length - recognized;
  const changed = lastDetections.filter(isDirty).length;
  const seconds =
    lastTimingsMs && lastTimingsMs.total != null
      ? (lastTimingsMs.total / 1000).toFixed(1) + "s"
      : "\u2014";
  document.getElementById("stats-row").innerHTML = `
    ${statCard(lastDetections.length, "Products detected", "", "all")}
    ${statCard(recognized, "Recognized", "stat-green", "recognized")}
    ${statCard(unknown, "Unknown", "stat-orange", "unknown")}
    ${statCard(changed, "Changed", "stat-blue", "changed")}
    ${statCard(seconds, "Processing time", "")}
  `;

  const unknownBtn = document.getElementById("unknown-summary");
  unknownBtn.disabled = unknown === 0;
  unknownBtn.innerHTML = unknown
    ? `<strong>${unknown}</strong> unmatched product${unknown === 1 ? "" : "s"}`
    : "No unmatched products.";
}

function renderGallery() {
  const galleryList = document.getElementById("gallery-list");
  const q = catalogQuery.trim().toLowerCase();
  const items = lastGallery.filter((g) => {
    if (g.count <= 0) return false;
    if (!q) return true;
    return (
      g.sku_id.toLowerCase().includes(q) ||
      g.packaging_type.toLowerCase().includes(q)
    );
  });
  if (!items.length) {
    galleryList.innerHTML = `<p class="card-sub">No matching catalog SKUs with facings on this shelf.</p>`;
    return;
  }
  galleryList.innerHTML = items
    .map((g) => {
      return `
      <div class="gallery-item clickable"
           data-sku="${escapeAttr(g.sku_id)}"
           data-pack="${escapeAttr(g.packaging_type)}"
           role="button" tabindex="0">
        <img class="gallery-thumb" src="${apiUrl(g.image_url)}" alt="${escapeAttr(g.sku_id)}" loading="lazy" />
        <div class="gallery-info">
          <div class="gallery-sku">${escapeHtml(g.sku_id)}</div>
          <div class="gallery-pack">${escapeHtml(g.packaging_type)}</div>
        </div>
        <span class="count-badge">\u00d7 ${g.count}</span>
      </div>`;
    })
    .join("");
}

function statCard(value, label, cls, filter) {
  const extra = filter ? ` clickable" data-filter="${filter}` : "";
  return `
    <div class="stat ${cls}${extra}">
      <div class="stat-value">${value}</div>
      <div class="stat-label">${label}</div>
    </div>`;
}

function scopedDetections() {
  if (!productFilter) return lastDetections;
  return lastDetections.filter(
    (d) =>
      d.status === "recognized" &&
      d.sku_id === productFilter.sku_id &&
      d.packaging === productFilter.packaging
  );
}

function filteredDetections() {
  const scoped = scopedDetections();
  if (cropFilter === "all") return scoped;
  if (cropFilter === "changed") return scoped.filter(isDirty);
  return scoped.filter((d) => d.status === cropFilter);
}

function applyView() {
  const scoped = scopedDetections();
  const counts = {
    all: scoped.length,
    recognized: scoped.filter((d) => d.status === "recognized").length,
    unknown: scoped.filter((d) => d.status !== "recognized").length,
    changed: scoped.filter(isDirty).length,
  };
  document.querySelectorAll("#crop-filters .chip").forEach((btn) => {
    const key = btn.dataset.filter;
    const label = key.charAt(0).toUpperCase() + key.slice(1);
    btn.innerHTML = `${label} <span class="chip-count">${counts[key]}</span>`;
    btn.classList.toggle("chip-active", key === cropFilter);
  });

  document.querySelectorAll(".gallery-item").forEach((el) => {
    const on =
      productFilter &&
      el.dataset.sku === productFilter.sku_id &&
      el.dataset.pack === productFilter.packaging;
    el.classList.toggle("selected", Boolean(on));
  });
  document
    .getElementById("unknown-summary")
    .classList.toggle("selected", !productFilter && cropFilter === "unknown");

  const note = document.getElementById("shelf-filter-note");
  if (productFilter) {
    note.classList.remove("hidden");
    note.innerHTML = `Showing <strong>${escapeHtml(productFilter.sku_id)}</strong> \u00b7 ${escapeHtml(productFilter.packaging)} <button type="button" id="clear-shelf-filter">Show all</button>`;
  } else if (cropFilter === "unknown") {
    note.classList.remove("hidden");
    note.innerHTML = `Showing unknown products <button type="button" id="clear-shelf-filter">Show all</button>`;
  } else if (cropFilter === "recognized") {
    note.classList.remove("hidden");
    note.innerHTML = `Showing recognized products <button type="button" id="clear-shelf-filter">Show all</button>`;
  } else if (cropFilter === "changed") {
    note.classList.remove("hidden");
    note.innerHTML = `Showing changed crops <button type="button" id="clear-shelf-filter">Show all</button>`;
  } else {
    note.classList.add("hidden");
    note.innerHTML = "";
  }

  const shown = filteredDetections();
  document.getElementById("crop-count").textContent = shown.length;
  renderCrops(shown);
  updateSaveBar();
  drawBoxes();
}

function drawBoxes() {
  const img = document.getElementById("annotated-img");
  const canvas = document.getElementById("box-canvas");
  if (!img.naturalWidth) return;
  canvas.width = img.naturalWidth;
  canvas.height = img.naturalHeight;
  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  const thick = Math.max(2, Math.round(Math.min(canvas.width, canvas.height) / 400));
  ctx.lineWidth = thick;
  for (const det of filteredDetections()) {
    const [x1, y1, x2, y2] = det.box;
    ctx.strokeStyle = det.status === "recognized" ? "rgb(80, 175, 76)" : "rgb(255, 152, 0)";
    ctx.strokeRect(x1 + thick / 2, y1 + thick / 2, x2 - x1 - thick, y2 - y1 - thick);
  }
}

function selectProduct(sku, packaging) {
  if (
    productFilter &&
    productFilter.sku_id === sku &&
    productFilter.packaging === packaging
  ) {
    productFilter = null;
  } else {
    productFilter = { sku_id: sku, packaging: packaging };
    cropFilter = "all";
    setReviewTab("crops");
  }
  applyView();
}

document.getElementById("gallery-list").addEventListener("click", (e) => {
  const item = e.target.closest(".gallery-item.clickable");
  if (!item) return;
  selectProduct(item.dataset.sku, item.dataset.pack);
});
document.getElementById("gallery-list").addEventListener("keydown", (e) => {
  if (e.key !== "Enter" && e.key !== " ") return;
  const item = e.target.closest(".gallery-item.clickable");
  if (!item) return;
  e.preventDefault();
  selectProduct(item.dataset.sku, item.dataset.pack);
});

document.getElementById("unknown-summary").addEventListener("click", () => {
  if (document.getElementById("unknown-summary").disabled) return;
  productFilter = null;
  cropFilter = cropFilter === "unknown" ? "all" : "unknown";
  setReviewTab("crops");
  applyView();
});

document.getElementById("stats-row").addEventListener("click", (e) => {
  const stat = e.target.closest(".stat.clickable");
  if (!stat) return;
  productFilter = null;
  cropFilter = stat.dataset.filter || "all";
  setReviewTab("crops");
  applyView();
});

document.getElementById("results").addEventListener("click", (e) => {
  if (e.target.id !== "clear-shelf-filter") return;
  productFilter = null;
  cropFilter = "all";
  applyView();
});

function cropLabel(d) {
  if (isDirty(d)) {
    const { pack, sku, creating } = intendedAssignment(d);
    const name = sku || (creating ? "new SKU" : "unassigned");
    return pack ? `${name} \u00b7 ${pack}` : name;
  }
  const recognized = d.status === "recognized";
  if (recognized) {
    return d.packaging ? `${d.sku_id} \u00b7 ${d.packaging}` : d.sku_id;
  }
  return d.sku_id ? `unknown \u00b7 best ${d.sku_id}` : "unknown";
}

function editorPackaging(d) {
  return d.editPackaging != null ? d.editPackaging : d.packaging || "";
}

function editorSku(d) {
  if (d.editSku != null) return d.editSku;
  return d.status === "recognized" && d.sku_id ? d.sku_id : "";
}

function detectionById(id) {
  return lastDetections.find((d) => d.id === id);
}

function skuNamesFor(packaging) {
  const names = skuCatalog
    .filter((s) => s.packaging_type === packaging)
    .map((s) => s.sku_id);
  const unique = [...new Set(names)].sort((a, b) => a.localeCompare(b));
  return unique;
}

function skuMenuHtml(packaging, query, selected) {
  const q = (query || "").trim().toLowerCase();
  const names = skuNamesFor(packaging).filter(
    (name) => !q || name.toLowerCase().includes(q)
  );
  const items = [
    `<button type="button" class="sku-option sku-option-new${selected === NEW_SKU_VALUE ? " selected" : ""}" data-sku="${NEW_SKU_VALUE}">Create new SKU\u2026</button>`,
  ];
  for (const name of names.slice(0, 80)) {
    const sel = name === selected ? " selected" : "";
    items.push(
      `<button type="button" class="sku-option${sel}" data-sku="${escapeAttr(name)}">${escapeHtml(name)}</button>`
    );
  }
  if (!names.length) {
    items.push(`<div class="sku-empty">No SKU matches</div>`);
  }
  return items.join("");
}

function packagingOptionsHtml(selected) {
  const opts = [
    `<option value=""${selected ? "" : " selected"}>Select</option>`,
  ];
  for (const pack of PACKAGING_TYPES) {
    const sel = pack === selected ? " selected" : "";
    opts.push(`<option value="${pack}"${sel}>${pack}</option>`);
  }
  return opts.join("");
}

function renderCrops(items) {
  const grid = document.getElementById("crops-grid");
  grid.innerHTML = items.map((d) => cropCardHtml(d)).join("");
}

function intendedAssignment(d) {
  const pack = editorPackaging(d);
  let sku = editorSku(d);
  const creating = sku === NEW_SKU_VALUE;
  if (creating) sku = (d.editNewSku || "").trim();
  return { pack, sku, creating };
}

function isDirty(d) {
  const { pack, sku, creating } = intendedAssignment(d);
  if (creating) return Boolean(pack || sku || editorSku(d) === NEW_SKU_VALUE);
  const origSku = d.origStatus === "recognized" ? d.origSkuId || "" : "";
  const origPack = d.origStatus === "recognized" ? d.origPackaging || "" : "";
  if (!pack && !sku) return false;
  return sku !== origSku || pack !== origPack;
}

function isReadyToSave(d) {
  if (!isDirty(d)) return false;
  const { pack, sku } = intendedAssignment(d);
  return Boolean(pack && sku);
}

function cropCardHtml(d) {
  const recognized = d.status === "recognized";
  const label = cropLabel(d);
  const sim =
    d.similarity != null ? `sim ${d.similarity.toFixed(2)} \u00b7 ` : "";
  const why = !recognized && d.reason ? `${d.reason} \u00b7 ` : "";
  const pack = editorPackaging(d);
  const sku = editorSku(d);
  const isNew = sku === NEW_SKU_VALUE;
  const dirty = isDirty(d);
  const saved = Boolean(d.saved) && !dirty;
  const searchValue =
    d.skuMenuOpen && d.editSkuQuery != null
      ? d.editSkuQuery
      : sku && sku !== NEW_SKU_VALUE
        ? sku
        : "";
  let tagClass = "tag-unknown";
  let tagText = "unknown";
  if (saved) {
    tagClass = "tag-saved";
    tagText = "saved";
  } else if (dirty) {
    tagClass = "tag-changed";
    tagText = "changed";
  } else if (recognized) {
    tagClass = "tag-recognized";
    tagText = "match";
  }
  const cardClass = [
    "crop-card",
    dirty ? "crop-dirty" : "",
    saved ? "crop-saved" : "",
    d.saveError ? "crop-error" : "",
    d.skuMenuOpen ? "crop-picking" : "",
  ]
    .filter(Boolean)
    .join(" ");
  return `
    <div class="${cardClass}" data-det-id="${d.id}">
      <img class="crop-img" src="${apiUrl(d.crop_url)}" alt="${escapeAttr(label)}" loading="lazy" title="Click to enlarge" />
      <div class="crop-body">
        <span class="tag ${tagClass}">${tagText}</span>
        <div class="crop-label">${escapeHtml(label)}</div>
        <div class="crop-sim">${escapeHtml(why)}${escapeHtml(sim)}conf ${d.det_conf.toFixed(2)}</div>
        <div class="crop-editor">
          <label class="crop-field">
            <span>Packaging</span>
            <select class="crop-pack">${packagingOptionsHtml(pack)}</select>
          </label>
          <label class="crop-field">
            <span>SKU</span>
            <div class="sku-picker">
              <input class="sku-search" type="search" autocomplete="off" spellcheck="false"
                     placeholder="${pack ? "Search SKU\u2026" : "Pick packaging first"}"
                     ${pack ? "" : " disabled"} value="${escapeAttr(searchValue)}" />
              <div class="sku-menu${d.skuMenuOpen ? "" : " hidden"}">${skuMenuHtml(pack, d.editSkuQuery || "", sku)}</div>
            </div>
          </label>
          <input class="crop-new-sku${isNew ? "" : " hidden"}" type="text" spellcheck="false"
                 placeholder="new_sku_folder" value="${escapeAttr(d.editNewSku || "")}" />
          ${dirty ? `<button type="button" class="crop-reset">Reset</button>` : ""}
          <p class="crop-save-msg${d.saveError ? "" : " hidden"}">${escapeHtml(d.saveError || "")}</p>
        </div>
      </div>
    </div>`;
}

document.getElementById("crop-filters").addEventListener("click", (e) => {
  const btn = e.target.closest(".chip");
  if (!btn) return;
  cropFilter = btn.dataset.filter;
  if (cropFilter === "unknown") productFilter = null;
  applyView();
});

/* ---------------- crop lightbox ---------------- */

const cropLightbox = document.getElementById("crop-lightbox");
const lightboxImg = document.getElementById("lightbox-img");
const lightboxCaption = document.getElementById("lightbox-caption");

document.getElementById("crops-grid").addEventListener("click", (e) => {
  const resetBtn = e.target.closest(".crop-reset");
  if (resetBtn) {
    const card = resetBtn.closest(".crop-card");
    const det = card ? detectionById(Number(card.dataset.detId)) : null;
    if (det) {
      resetDetection(det);
      refreshAfterCorrection();
    }
    return;
  }

  const option = e.target.closest(".sku-option");
  if (option) {
    const card = option.closest(".crop-card");
    const det = card ? detectionById(Number(card.dataset.detId)) : null;
    if (det) {
      det.editSku = option.dataset.sku;
      det.skuMenuOpen = false;
      det.editSkuQuery = null;
      det.saveError = "";
      applyLocalAssignment(det);
      refreshAfterCorrection();
    }
    return;
  }

  if (!e.target.closest(".crop-img")) return;
  const card = e.target.closest(".crop-card");
  if (!card) return;
  const d = detectionById(Number(card.dataset.detId));
  if (!d) return;

  const recognized = d.status === "recognized";
  const label = cropLabel(d);
  const sim =
    d.similarity != null ? `similarity ${d.similarity.toFixed(2)} · ` : "";
  const margin =
    d.margin != null ? `margin ${d.margin.toFixed(4)} · ` : "";
  const why = !recognized && d.reason ? `${d.reason} · ` : "";

  lightboxImg.src = apiUrl(d.crop_url);
  lightboxImg.alt = label;
  lightboxCaption.innerHTML = `
    <strong>${escapeHtml(label)}</strong><br>
    ${recognized ? "Recognized match" : "Unknown"} · ${escapeHtml(why)}${escapeHtml(sim)}${escapeHtml(margin)}confidence ${d.det_conf.toFixed(2)}`;
  cropLightbox.classList.remove("hidden");
  document.body.style.overflow = "hidden";
});

document.getElementById("crops-grid").addEventListener("change", (e) => {
  const card = e.target.closest(".crop-card");
  if (!card) return;
  const det = detectionById(Number(card.dataset.detId));
  if (!det) return;

  if (e.target.classList.contains("crop-pack")) {
    det.editPackaging = e.target.value;
    det.editSku = "";
    det.editSkuQuery = "";
    det.skuMenuOpen = Boolean(det.editPackaging);
    det.saveError = "";
    applyLocalAssignment(det);
    refreshAfterCorrection();
  }
});

document.getElementById("crops-grid").addEventListener("input", (e) => {
  const card = e.target.closest(".crop-card");
  const det = card ? detectionById(Number(card.dataset.detId)) : null;
  if (!det) return;
  if (e.target.classList.contains("crop-new-sku")) {
    det.editNewSku = e.target.value;
    applyLocalAssignment(det);
    updateSaveBar();
    return;
  }
  if (e.target.classList.contains("sku-search")) {
    det.editSkuQuery = e.target.value;
    det.skuMenuOpen = true;
    const menu = card.querySelector(".sku-menu");
    menu.classList.remove("hidden");
    menu.innerHTML = skuMenuHtml(
      editorPackaging(det),
      det.editSkuQuery,
      editorSku(det)
    );
  }
});

document.getElementById("crops-grid").addEventListener("keydown", (e) => {
  if (e.key !== "Enter" || !e.target.classList.contains("sku-search")) return;
  e.preventDefault();
  const card = e.target.closest(".crop-card");
  const first =
    card.querySelector(".sku-option:not(.sku-option-new)") ||
    card.querySelector(".sku-option-new");
  if (first) first.click();
});

document.getElementById("crops-grid").addEventListener("focusin", (e) => {
  if (!e.target.classList.contains("sku-search")) return;
  const card = e.target.closest(".crop-card");
  const det = card ? detectionById(Number(card.dataset.detId)) : null;
  if (!det || !editorPackaging(det)) return;
  lastDetections.forEach((item) => {
    item.skuMenuOpen = item.id === det.id;
  });
  det.editSkuQuery = "";
  det.skuMenuOpen = true;
  e.target.select();
  document.querySelectorAll(".crop-card").forEach((el) => {
    el.classList.toggle("crop-picking", el === card);
  });
  const menu = card.querySelector(".sku-menu");
  menu.classList.remove("hidden");
  menu.innerHTML = skuMenuHtml(editorPackaging(det), "", editorSku(det));
});

document.addEventListener("click", (e) => {
  if (e.target.closest(".sku-picker")) return;
  let closed = false;
  for (const det of lastDetections) {
    if (det.skuMenuOpen) {
      det.skuMenuOpen = false;
      det.editSkuQuery = null;
      closed = true;
    }
  }
  if (closed) {
    document.querySelectorAll(".sku-menu").forEach((menu) => menu.classList.add("hidden"));
    document.querySelectorAll(".crop-card").forEach((el) => el.classList.remove("crop-picking"));
  }
});

function closeLightbox() {
  cropLightbox.classList.add("hidden");
  lightboxImg.src = "";
  document.body.style.overflow = "";
}

document.getElementById("lightbox-close").addEventListener("click", closeLightbox);
cropLightbox.addEventListener("click", (e) => {
  if (e.target === cropLightbox) closeLightbox();
});
document.addEventListener("keydown", (e) => {
  if (e.key !== "Escape") return;
  if (!cropLightbox.classList.contains("hidden")) {
    closeLightbox();
    return;
  }
  if (lastDetections.some((d) => d.skuMenuOpen)) {
    lastDetections.forEach((d) => {
      d.skuMenuOpen = false;
      d.editSkuQuery = null;
    });
    document.querySelectorAll(".sku-menu").forEach((menu) => menu.classList.add("hidden"));
  }
});

/* ---------------- crop correction ---------------- */

async function loadSkus() {
  try {
    const resp = await fetch(`${API_BASE}/api/skus`);
    if (!resp.ok) return skuCatalog;
    const data = await resp.json();
    return data.skus || [];
  } catch {
    return skuCatalog;
  }
}

function upsertSkuCatalog(skuId, packagingType) {
  const exists = skuCatalog.some(
    (s) => s.sku_id === skuId && s.packaging_type === packagingType
  );
  if (!exists) skuCatalog.push({ sku_id: skuId, packaging_type: packagingType });
}

function cropNameFromUrl(url) {
  const path = String(url || "").split("?")[0];
  return decodeURIComponent(path.split("/").pop() || "");
}

function apiErrorDetail(data, status, fallback) {
  const detail = data && data.detail;
  if (typeof detail === "string") return detail;
  if (Array.isArray(detail) && detail[0] && detail[0].msg) return detail[0].msg;
  return fallback || `Request failed (${status})`;
}

function recountGallery() {
  const counts = new Map();
  for (const d of lastDetections) {
    if (d.status !== "recognized" || !d.sku_id || !d.packaging) continue;
    const key = `${d.sku_id}\0${d.packaging}`;
    counts.set(key, (counts.get(key) || 0) + 1);
  }
  for (const g of lastGallery) {
    g.count = counts.get(`${g.sku_id}\0${g.packaging_type}`) || 0;
  }
  for (const [key, n] of counts) {
    const [skuId, packaging] = key.split("\0");
    if (
      lastGallery.some(
        (g) => g.sku_id === skuId && g.packaging_type === packaging
      )
    ) {
      continue;
    }
    const det = lastDetections.find(
      (d) => d.sku_id === skuId && d.packaging === packaging
    );
    lastGallery.push({
      sku_id: skuId,
      packaging_type: packaging,
      image_url: det ? det.crop_url : "",
      count: n,
    });
  }
  lastGallery.sort((a, b) => {
    if (b.count !== a.count) return b.count - a.count;
    const skuCmp = a.sku_id.localeCompare(b.sku_id);
    return skuCmp !== 0 ? skuCmp : a.packaging_type.localeCompare(b.packaging_type);
  });
}

function refreshAfterCorrection() {
  recountGallery();
  renderStats();
  renderGallery();
  applyView();
}

function applyLocalAssignment(d) {
  const { pack, sku } = intendedAssignment(d);
  if (pack && sku) {
    d.sku_id = sku;
    d.packaging = pack;
    d.status = "recognized";
    return;
  }
  d.sku_id = d.origSkuId;
  d.packaging = d.origPackaging;
  d.status = d.origStatus;
}

function resetDetection(d) {
  d.editPackaging = d.origPackaging || "";
  d.editSku = d.origStatus === "recognized" ? d.origSkuId || "" : "";
  d.editNewSku = "";
  d.editSkuQuery = null;
  d.skuMenuOpen = false;
  d.saveError = "";
  d.sku_id = d.origSkuId;
  d.packaging = d.origPackaging;
  d.status = d.origStatus;
}

function setReviewTab(tab) {
  reviewTab = tab;
  document.querySelectorAll(".tab-btn").forEach((btn) => {
    const on = btn.dataset.tab === tab;
    btn.classList.toggle("tab-active", on);
    btn.setAttribute("aria-selected", on ? "true" : "false");
  });
  document.getElementById("tab-crops").classList.toggle("hidden", tab !== "crops");
  document.getElementById("tab-catalog").classList.toggle("hidden", tab !== "catalog");
}

function updateSaveBar() {
  const ready = lastDetections.filter(isReadyToSave);
  const incomplete = lastDetections.filter((d) => isDirty(d) && !isReadyToSave(d));
  const btn = document.getElementById("save-corrections");
  const summary = document.getElementById("save-summary");
  if (!btn || !summary) return;
  btn.disabled = ready.length === 0;
  btn.textContent =
    ready.length === 1 ? "Save 1 correction" : `Save ${ready.length} corrections`;
  if (ready.length === 0 && incomplete.length === 0) {
    summary.textContent = "No SKU changes yet";
  } else if (ready.length === 0) {
    summary.textContent = `${incomplete.length} crop${incomplete.length === 1 ? "" : "s"} need a packaging and SKU`;
  } else if (incomplete.length) {
    summary.textContent = `${ready.length} ready to save \u00b7 ${incomplete.length} incomplete`;
  } else {
    summary.textContent = `${ready.length} crop${ready.length === 1 ? "" : "s"} will be copied to gallery folders`;
  }
}

function showSaveStatus(msg, isError) {
  const el = document.getElementById("save-status");
  if (!el) return;
  if (!msg) {
    el.classList.add("hidden");
    el.textContent = "";
    return;
  }
  el.textContent = msg;
  el.classList.toggle("error", Boolean(isError));
  el.classList.remove("hidden");
}

async function saveDirtyCorrections() {
  const ready = lastDetections.filter(isReadyToSave);
  const btn = document.getElementById("save-corrections");
  if (!ready.length) return;
  if (!lastRequestId) {
    showSaveStatus("Missing request id. Analyze again.", true);
    return;
  }

  btn.disabled = true;
  btn.textContent = "Saving\u2026";
  showSaveStatus("");
  let saved = 0;
  let failed = 0;

  for (const det of ready) {
    const { pack, sku } = intendedAssignment(det);
    try {
      const resp = await fetch(`${API_BASE}/api/save-crop`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          request_id: lastRequestId,
          crop_name: cropNameFromUrl(det.crop_url),
          sku_id: sku,
          packaging_type: pack,
        }),
      });
      const data = await resp.json().catch(() => ({}));
      if (!resp.ok) {
        throw new Error(apiErrorDetail(data, resp.status, "Save failed."));
      }
      det.status = "recognized";
      det.sku_id = data.sku_id;
      det.packaging = data.packaging_type;
      det.origSkuId = data.sku_id;
      det.origPackaging = data.packaging_type;
      det.origStatus = "recognized";
      det.editPackaging = data.packaging_type;
      det.editSku = data.sku_id;
      det.editNewSku = "";
      det.saved = true;
      det.saveError = "";
      upsertSkuCatalog(data.sku_id, data.packaging_type);
      saved += 1;
    } catch (err) {
      det.saveError = err.message || "Save failed.";
      failed += 1;
    }
  }

  refreshAfterCorrection();
  if (failed && saved) {
    showSaveStatus(`Saved ${saved}, ${failed} failed.`, true);
  } else if (failed) {
    showSaveStatus(`Could not save ${failed} crop${failed === 1 ? "" : "s"}.`, true);
  } else {
    showSaveStatus(`Saved ${saved} crop${saved === 1 ? "" : "s"} to gallery folders.`);
  }
}

document.querySelector(".review-tabs").addEventListener("click", (e) => {
  const btn = e.target.closest(".tab-btn");
  if (!btn) return;
  setReviewTab(btn.dataset.tab);
});

document.getElementById("catalog-search").addEventListener("input", (e) => {
  catalogQuery = e.target.value;
  renderGallery();
});

document.getElementById("save-corrections").addEventListener("click", () => {
  saveDirtyCorrections();
});

/* ---------------- helpers ---------------- */

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;");
}

function escapeAttr(value) {
  return escapeHtml(value).replaceAll('"', "&quot;");
}

function apiUrl(path) {
  return path.startsWith("http") ? path : `${API_BASE}${path}`;
}

function formatBytes(bytes) {
  if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(0) + " KB";
  return (bytes / (1024 * 1024)).toFixed(1) + " MB";
}

function showError(msg) {
  errorBox.textContent = msg;
  errorBox.classList.remove("hidden");
}

function hideError() {
  errorBox.classList.add("hidden");
}
