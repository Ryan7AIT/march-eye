(() => {
  "use strict";

  const $ = (id) => document.getElementById(id);
  const esc = (s) =>
    String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  const keyOf = (row) => `${row.packaging_type}/${row.sku_id}`;
  const PACKS = ["bottle", "can", "carton"];
  const HOT = 0.85;
  const WARM = 0.75;
  const FLAG_LABEL = { empty: ["bad", "empty"], single: ["bad", "1 photo"], few: ["warn", "< 3 photos"] };

  const state = {
    skus: [],
    summary: null,
    index: null,
    visualSearch: false,
    minGood: 3,
    search: "",
    pack: "all",
    coverage: "all",
    sort: "images_desc",
    selected: null,
    detail: null,
    queryFile: null,
    queryUrl: null,
    visualHits: [],
  };

  async function getJSON(url, options) {
    const res = await fetch(url, options);
    if (!res.ok) {
      let detail = res.statusText;
      try {
        const body = await res.json();
        detail = body.detail || detail;
        if (Array.isArray(detail)) detail = detail.map((d) => d.msg || d).join("; ");
      } catch (_) {}
      throw new Error(detail);
    }
    if (res.status === 204) return null;
    return res.json();
  }

  function simClass(v) {
    return v >= HOT ? "hot" : v >= WARM ? "warm" : "";
  }

  function countClass(n) {
    if (n <= 1) return "low";
    if (n < state.minGood) return "mid";
    return "good";
  }

  function flagsHtml(flags) {
    return (flags || [])
      .map((f) => {
        const info = FLAG_LABEL[f];
        if (!info) return "";
        return `<span class="flag ${info[0]}">${esc(info[1])}</span>`;
      })
      .join("");
  }

  function filteredSkus() {
    const q = state.search.trim().toLowerCase();
    const visual = new Map(state.visualHits.map((h) => [keyOf(h), h.similarity]));
    let rows = state.skus.filter((row) => {
      if (state.pack !== "all" && row.packaging_type !== state.pack) return false;
      if (state.coverage === "need" && row.n_images >= state.minGood) return false;
      if (state.coverage === "empty" && row.n_images !== 0) return false;
      if (q && !row.sku_id.toLowerCase().includes(q) && !row.packaging_type.includes(q)) return false;
      return true;
    });
    rows = rows.slice().sort((a, b) => {
      if (visual.size && state.sort === "images_desc") {
        const sa = visual.get(keyOf(a)) ?? -1;
        const sb = visual.get(keyOf(b)) ?? -1;
        if (sb !== sa) return sb - sa;
      }
      if (state.sort === "images_desc") return b.n_images - a.n_images || a.sku_id.localeCompare(b.sku_id);
      if (state.sort === "images_asc") return a.n_images - b.n_images || a.sku_id.localeCompare(b.sku_id);
      return a.sku_id.localeCompare(b.sku_id) || a.packaging_type.localeCompare(b.packaging_type);
    });
    return rows;
  }

  function renderSummary() {
    const s = state.summary;
    if (!s) {
      $("summary-row").innerHTML = "";
      return;
    }
    const idx = state.index || {};
    const needTone = s.n_need_photos ? "warn" : "ok";
    const idxTone = !idx.exists || idx.stale ? "warn" : "ok";
    const idxValue = !idx.exists ? "Missing" : idx.stale ? "Stale" : "Live";
    const idxSub = !idx.exists
      ? "run embed_gallery.py"
      : idx.stale
        ? `${idx.new_on_disk} new, ${idx.missing_on_disk} removed`
        : `${idx.num_vectors || 0} vectors`;
    $("summary-row").innerHTML = [
      [`${s.n_skus}`, "SKUs", "in the gallery", ""],
      [`${s.n_images}`, "Photos", PACKS.map((p) => `${p} ${s.by_packaging?.[p]?.images || 0}`).join(" · "), ""],
      [`${s.n_need_photos}`, "Need more photos", `below ${state.minGood} images`, needTone],
      [idxValue, "Gallery index", idxSub, idxTone],
    ]
      .map(
        ([v, l, sub, tone]) =>
          `<div class="summary ${tone}"><div class="v">${esc(v)}</div><div class="l">${esc(l)}</div><div class="l">${esc(sub)}</div></div>`
      )
      .join("");
  }

  function renderChips() {
    const chips = ["all", ...PACKS]
      .map((p) => {
        const label = p === "all" ? "All packaging" : p;
        const active = state.pack === p ? " chip-active" : "";
        return `<button type="button" class="chip${active}" data-pack="${p}">${label}</button>`;
      })
      .join("");
    $("pack-chips").innerHTML = chips;
  }

  function renderList() {
    const rows = filteredSkus();
    const visual = new Map(state.visualHits.map((h) => [keyOf(h), h.similarity]));
    $("list-status").textContent = `${rows.length} SKU${rows.length === 1 ? "" : "s"}`;
    $("sku-grid").innerHTML = rows
      .map((row) => {
        const k = keyOf(row);
        const active = state.selected === k ? " active" : "";
        const sim = visual.get(k);
        const thumb = row.thumb_url
          ? `<img src="${esc(row.thumb_url)}" alt="" />`
          : `<div class="thumb-empty">no photo</div>`;
        return `<button type="button" class="sku-card${active}" data-key="${esc(k)}">
          ${thumb}
          <div>
            <div class="name">${esc(row.sku_id)}</div>
            <div class="meta">${esc(row.packaging_type)} · <span class="count ${countClass(row.n_images)}">${row.n_images}</span> photo${row.n_images === 1 ? "" : "s"}${
              sim != null ? ` · <span class="sim ${simClass(sim)}">${(sim * 100).toFixed(1)}%</span>` : ""
            }</div>
            ${flagsHtml(row.flags)}
          </div>
        </button>`;
      })
      .join("");
  }

  function renderDetail() {
    const empty = $("detail-empty");
    const body = $("detail-body");
    if (!state.detail) {
      empty.classList.remove("hidden");
      body.classList.add("hidden");
      body.innerHTML = "";
      return;
    }
    empty.classList.add("hidden");
    body.classList.remove("hidden");
    const d = state.detail;
    const canAddQuery = Boolean(state.queryFile);
    const photos = (d.images || [])
      .map(
        (img) => `<div class="photo">
          <button type="button" class="photo-del" data-del="${esc(img.name)}" title="Delete photo">&times;</button>
          <img src="${esc(img.url)}" alt="${esc(img.name)}" data-full="${esc(img.url)}" data-caption="${esc(img.name)}" />
          <div class="photo-name">${esc(img.name)}${img.w ? ` · ${img.w}×${img.h}` : ""}</div>
        </div>`
      )
      .join("");
    const neighbors = (d.neighbors || [])
      .map(
        (n) => `<button type="button" class="neighbor" data-open="${esc(n.packaging_type)}/${esc(n.sku_id)}">
          ${n.image_url ? `<img src="${esc(n.image_url)}" alt="" />` : `<span></span>`}
          <div><div class="sku">${esc(n.sku_id)}</div><div class="pack">${esc(n.packaging_type)}</div></div>
          <strong class="${simClass(n.similarity)}">${(n.similarity * 100).toFixed(1)}%</strong>
        </button>`
      )
      .join("");
    body.innerHTML = `
      <div class="detail-head">
        <div>
          <h2 class="card-title">${esc(d.sku_id)}</h2>
          <p class="card-sub">${esc(d.packaging_type)} · ${d.n_images} photo${d.n_images === 1 ? "" : "s"} ${flagsHtml(d.flags)}</p>
        </div>
      </div>
      <div class="detail-actions">
        ${canAddQuery ? `<button type="button" class="btn btn-primary btn-sm" id="add-query-btn">Add search photo to this SKU</button>` : ""}
        <button type="button" class="btn btn-ghost btn-sm" id="rename-toggle">Rename / move</button>
        <button type="button" class="btn btn-danger btn-sm" id="delete-sku-btn">Delete SKU</button>
      </div>
      <div id="rename-row" class="detail-rename hidden">
        <input id="rename-id" value="${esc(d.sku_id)}" />
        <select id="rename-pack">${PACKS.map((p) => `<option value="${p}" ${p === d.packaging_type ? "selected" : ""}>${p}</option>`).join("")}</select>
        <button type="button" class="btn btn-primary btn-sm" id="rename-save">Save</button>
      </div>
      <div class="photo-grid">${photos || `<p class="catalog-status">No photos yet. Drop some below.</p>`}</div>
      <div id="add-photos" class="add-photos" tabindex="0">Drop photos here, or click to add
        <input id="add-files" type="file" accept="image/jpeg,image/png,image/webp,image/bmp" multiple hidden />
      </div>
      <p id="detail-status" class="catalog-status"></p>
      ${
        neighbors
          ? `<h3 class="card-title" style="margin-top:18px;font-size:14px">Looks like</h3>
             <p class="card-sub">Nearest other SKUs in the embedding gallery. High scores are the pairs the model may confuse.</p>
             <div class="neighbors">${neighbors}</div>`
          : ""
      }
    `;
  }

  function renderVisualResults() {
    const box = $("visual-results");
    if (!state.visualHits.length) {
      box.classList.add("hidden");
      box.innerHTML = "";
      return;
    }
    box.classList.remove("hidden");
    box.innerHTML = state.visualHits
      .slice(0, 12)
      .map((h) => {
        const img = h.image_url || h.thumb_url;
        return `<button type="button" class="visual-hit" data-open="${esc(h.packaging_type)}/${esc(h.sku_id)}">
          ${img ? `<img src="${esc(img)}" alt="" />` : ""}
          <div class="sku">${esc(h.sku_id)}</div>
          <div class="meta">${esc(h.packaging_type)} · ${h.n_images ?? "?"} photos</div>
          <div class="sim ${simClass(h.similarity)}">${(h.similarity * 100).toFixed(1)}%</div>
        </button>`;
      })
      .join("");
  }

  function setStatus(id, message, isError) {
    const el = $(id);
    el.textContent = message || "";
    el.classList.toggle("hidden", !message);
    el.classList.toggle("error", Boolean(isError));
  }

  async function loadList(keepSelected) {
    const data = await getJSON("/api/catalog/skus");
    state.skus = data.skus;
    state.summary = data.summary;
    state.index = data.index;
    state.visualSearch = data.visual_search;
    state.minGood = data.min_good_images || 3;
    $("visual-btn").disabled = false;
    if (!state.visualSearch) $("visual-btn").title = "Start the main app (app.py) to search with the embedding model.";
    if (keepSelected && state.selected && !state.skus.some((r) => keyOf(r) === state.selected)) {
      state.selected = null;
      state.detail = null;
    }
    renderSummary();
    renderChips();
    renderList();
    renderDetail();
  }

  async function openSku(packaging, skuId) {
    state.selected = `${packaging}/${skuId}`;
    renderList();
    try {
      state.detail = await getJSON(`/api/catalog/skus/${encodeURIComponent(packaging)}/${encodeURIComponent(skuId)}`);
    } catch (err) {
      state.detail = null;
      setStatus("list-status", err.message, true);
    }
    renderDetail();
  }

  async function uploadFiles(packaging, skuId, files) {
    if (!files.length) return;
    const body = new FormData();
    for (const file of files) body.append("files", file);
    const result = await getJSON(
      `/api/catalog/skus/${encodeURIComponent(packaging)}/${encodeURIComponent(skuId)}/images`,
      { method: "POST", body }
    );
    return result;
  }

  function bindDrop(el, onFiles) {
    el.addEventListener("dragover", (ev) => {
      ev.preventDefault();
      el.classList.add("dragover");
    });
    el.addEventListener("dragleave", () => el.classList.remove("dragover"));
    el.addEventListener("drop", (ev) => {
      ev.preventDefault();
      el.classList.remove("dragover");
      const files = [...(ev.dataTransfer?.files || [])].filter((f) => f.type.startsWith("image/"));
      if (files.length) onFiles(files);
    });
  }

  function openLightbox(src, caption) {
    $("lightbox-img").src = src;
    $("lightbox-caption").textContent = caption || "";
    $("lightbox").classList.remove("hidden");
  }

  function closeLightbox() {
    $("lightbox").classList.add("hidden");
    $("lightbox-img").src = "";
  }

  function setModal(open) {
    $("new-sku-modal").classList.toggle("hidden", !open);
    if (open) {
      $("new-sku-error").classList.add("hidden");
      $("new-sku-id").focus();
    }
  }

  async function runVisualSearch(file) {
    state.queryFile = file;
    if (state.queryUrl) URL.revokeObjectURL(state.queryUrl);
    state.queryUrl = URL.createObjectURL(file);
    $("visual-query-img").src = state.queryUrl;
    $("visual-query").classList.remove("hidden");
    setStatus("visual-status", "Searching gallery…");
    const body = new FormData();
    body.append("file", file);
    try {
      const result = await getJSON("/api/catalog/search-image", { method: "POST", body });
      state.visualHits = result.skus || [];
      setStatus("visual-status", state.visualHits.length ? `Closest ${Math.min(state.visualHits.length, 12)} SKUs` : "No gallery matches yet.");
      renderVisualResults();
      renderList();
      renderDetail();
    } catch (err) {
      setStatus("visual-status", err.message, true);
    }
  }

  function clearVisual() {
    state.queryFile = null;
    if (state.queryUrl) URL.revokeObjectURL(state.queryUrl);
    state.queryUrl = null;
    state.visualHits = [];
    $("visual-query").classList.add("hidden");
    $("visual-file").value = "";
    setStatus("visual-status", "");
    renderVisualResults();
    renderList();
    renderDetail();
  }

  $("sku-search").addEventListener("input", (ev) => {
    state.search = ev.target.value;
    renderList();
  });
  $("coverage-filter").addEventListener("change", (ev) => {
    state.coverage = ev.target.value;
    renderList();
  });
  $("sort-key").addEventListener("change", (ev) => {
    state.sort = ev.target.value;
    renderList();
  });
  $("pack-chips").addEventListener("click", (ev) => {
    const btn = ev.target.closest("[data-pack]");
    if (!btn) return;
    state.pack = btn.getAttribute("data-pack");
    renderChips();
    renderList();
  });
  $("sku-grid").addEventListener("click", (ev) => {
    const btn = ev.target.closest("[data-key]");
    if (!btn) return;
    const [packaging, skuId] = btn.getAttribute("data-key").split("/");
    openSku(packaging, skuId);
  });
  $("visual-results").addEventListener("click", (ev) => {
    const btn = ev.target.closest("[data-open]");
    if (!btn) return;
    const [packaging, skuId] = btn.getAttribute("data-open").split("/");
    openSku(packaging, skuId);
  });

  $("visual-btn").addEventListener("click", () => {
    $("visual-panel").classList.toggle("hidden");
  });
  $("visual-drop").addEventListener("click", () => $("visual-file").click());
  $("visual-file").addEventListener("change", (ev) => {
    const file = ev.target.files && ev.target.files[0];
    if (file) runVisualSearch(file);
  });
  bindDrop($("visual-drop"), (files) => runVisualSearch(files[0]));
  $("visual-clear").addEventListener("click", clearVisual);

  $("detail-body").addEventListener("click", async (ev) => {
    const openBtn = ev.target.closest("[data-open]");
    if (openBtn) {
      const [packaging, skuId] = openBtn.getAttribute("data-open").split("/");
      openSku(packaging, skuId);
      return;
    }
    const img = ev.target.closest("img[data-full]");
    if (img) {
      openLightbox(img.getAttribute("data-full"), img.getAttribute("data-caption"));
      return;
    }
    if (!state.detail) return;
    const { sku_id: skuId, packaging_type: packaging } = state.detail;
    if (ev.target.id === "rename-toggle") {
      $("rename-row").classList.toggle("hidden");
      return;
    }
    if (ev.target.id === "rename-save") {
      try {
        const result = await getJSON(
          `/api/catalog/skus/${encodeURIComponent(packaging)}/${encodeURIComponent(skuId)}`,
          {
            method: "PATCH",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ sku_id: $("rename-id").value, packaging_type: $("rename-pack").value }),
          }
        );
        await loadList(true);
        await openSku(result.packaging_type, result.sku_id);
      } catch (err) {
        setStatus("detail-status", err.message, true);
      }
      return;
    }
    if (ev.target.id === "delete-sku-btn") {
      if (!confirm(`Delete ${packaging}/${skuId} and all of its photos?`)) return;
      try {
        await getJSON(
          `/api/catalog/skus/${encodeURIComponent(packaging)}/${encodeURIComponent(skuId)}?force=true`,
          { method: "DELETE" }
        );
        state.selected = null;
        state.detail = null;
        await loadList(true);
      } catch (err) {
        setStatus("detail-status", err.message, true);
      }
      return;
    }
    if (ev.target.id === "add-query-btn" && state.queryFile) {
      try {
        await uploadFiles(packaging, skuId, [state.queryFile]);
        await loadList(true);
        await openSku(packaging, skuId);
        setStatus("detail-status", "Search photo added to this SKU.");
      } catch (err) {
        setStatus("detail-status", err.message, true);
      }
      return;
    }
    const del = ev.target.closest("[data-del]");
    if (del) {
      const name = del.getAttribute("data-del");
      if (!confirm(`Delete ${name}?`)) return;
      try {
        await getJSON(
          `/api/catalog/skus/${encodeURIComponent(packaging)}/${encodeURIComponent(skuId)}/images/${encodeURIComponent(name)}`,
          { method: "DELETE" }
        );
        await loadList(true);
        await openSku(packaging, skuId);
      } catch (err) {
        setStatus("detail-status", err.message, true);
      }
    }
  });

  $("detail-body").addEventListener("change", async (ev) => {
    if (ev.target.id !== "add-files" || !state.detail) return;
    try {
      await uploadFiles(state.detail.packaging_type, state.detail.sku_id, [...ev.target.files]);
      await loadList(true);
      await openSku(state.detail.packaging_type, state.detail.sku_id);
    } catch (err) {
      setStatus("detail-status", err.message, true);
    }
    ev.target.value = "";
  });

  $("detail-body").addEventListener("click", (ev) => {
    const zone = ev.target.closest("#add-photos");
    if (zone && !ev.target.closest("[data-del]") && ev.target.id !== "add-files") {
      $("add-files")?.click();
    }
  });
  $("detail-body").addEventListener("dragover", (ev) => {
    if (!ev.target.closest("#add-photos")) return;
    ev.preventDefault();
    ev.target.closest("#add-photos").classList.add("dragover");
  });
  $("detail-body").addEventListener("dragleave", (ev) => {
    ev.target.closest("#add-photos")?.classList.remove("dragover");
  });
  $("detail-body").addEventListener("drop", async (ev) => {
    const zone = ev.target.closest("#add-photos");
    if (!zone || !state.detail) return;
    ev.preventDefault();
    zone.classList.remove("dragover");
    const files = [...(ev.dataTransfer?.files || [])].filter((f) => f.type.startsWith("image/"));
    if (!files.length) return;
    try {
      await uploadFiles(state.detail.packaging_type, state.detail.sku_id, files);
      await loadList(true);
      await openSku(state.detail.packaging_type, state.detail.sku_id);
    } catch (err) {
      setStatus("detail-status", err.message, true);
    }
  });

  $("new-sku-btn").addEventListener("click", () => setModal(true));
  $("new-sku-cancel").addEventListener("click", () => setModal(false));
  $("new-sku-modal").addEventListener("click", (ev) => {
    if (ev.target.id === "new-sku-modal") setModal(false);
  });
  $("new-sku-save").addEventListener("click", async () => {
    const skuId = $("new-sku-id").value.trim();
    const packaging = $("new-sku-pack").value;
    const files = [...($("new-sku-files").files || [])];
    try {
      const created = await getJSON("/api/catalog/skus", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ sku_id: skuId, packaging_type: packaging }),
      });
      if (files.length) {
        await uploadFiles(created.packaging_type, created.sku_id, files);
      }
      $("new-sku-id").value = "";
      $("new-sku-files").value = "";
      setModal(false);
      await loadList(true);
      await openSku(created.packaging_type, created.sku_id);
    } catch (err) {
      const box = $("new-sku-error");
      box.textContent = err.message;
      box.classList.remove("hidden");
    }
  });

  $("lightbox-close").addEventListener("click", closeLightbox);
  $("lightbox").addEventListener("click", (ev) => {
    if (ev.target.id === "lightbox") closeLightbox();
  });
  document.addEventListener("keydown", (ev) => {
    if (ev.key !== "Escape") return;
    closeLightbox();
    setModal(false);
  });

  loadList().catch((err) => setStatus("list-status", err.message, true));
})();
