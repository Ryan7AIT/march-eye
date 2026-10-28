(() => {
  "use strict";

  const $ = (id) => document.getElementById(id);
  const esc = (s) =>
    String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  const pct = (v, d = 1) => (v == null ? "-" : (v * 100).toFixed(d) + "%");
  const num = (v) => (v == null ? "-" : Number(v).toLocaleString());
  const key = (r) => `${r.packaging_type}/${r.sku_id}`;

  const HOT = 0.85;   // nearest other SKU at least this similar = confusable
  const WARM = 0.75;

  const FLAG_INFO = {
    single: ["bad", "1 image"],
    few: ["warn", "< 3 images"],
    empty: ["bad", "empty folder"],
    no_eval: ["info", "no eval"],
    dup: ["bad", "duplicate photo"],
    leak: ["warn", "eval = gallery copy"],
    tiny: ["warn", "tiny image"],
    confusable: ["warn", "confusable"],
  };

  const state = {
    overview: null,
    sim: null,
    simByKey: new Map(),
    search: "",
    pack: "all",
    flag: "all",
    sortKey: "sku_id",
    sortDir: 1,
    expanded: new Set(),
    minGood: 3,
  };

  async function getJSON(url) {
    const res = await fetch(url);
    if (!res.ok) {
      let detail = res.statusText;
      try { detail = (await res.json()).detail || detail; } catch (_) {}
      throw new Error(detail);
    }
    return res.json();
  }

  // ------------------------------------------------------------ KPIs
  function kpi(value, label, sub, tone) {
    return `<div class="kpi tone-${tone}">
      <div class="kpi-value">${value}</div>
      <div class="kpi-label">${esc(label)}</div>
      <div class="kpi-sub">${sub}</div></div>`;
  }

  function renderKpis(o) {
    const k = o.kpis;
    const ip = k.images_per_sku;
    const bp = k.by_packaging;
    const dup = o.duplicates.counts.cross_sku + o.duplicates.counts.label_conflict;
    const idx = o.index;
    const cards = [
      kpi(num(k.n_skus), "SKUs in the gallery",
        Object.entries(bp).map(([p, v]) => `${p} ${v.skus}`).join(" &middot; "), "info"),
      kpi(num(k.n_images), "Gallery images",
        `median ${ip.median} / SKU &middot; max ${ip.max} &middot; min ${ip.min}`, "info"),
      kpi(num(k.singletons), "SKUs with one image",
        "need at least 3 views to be trusted", k.singletons ? "bad" : "ok"),
      kpi(num(k.below_min), `SKUs below ${o.min_good_images} images`,
        `${num(k.at_least_10)} SKUs have 10+`, k.below_min ? "warn" : "ok"),
      kpi(num(k.n_eval_known), "Eval crops (known)",
        `across ${num(k.skus_with_eval)} SKUs`, "info"),
      kpi(pct(k.eval_coverage, 0), "SKUs covered by eval",
        `${num(k.n_skus - k.skus_with_eval)} SKUs have no eval crop`,
        k.eval_coverage >= 0.8 ? "ok" : k.eval_coverage >= 0.5 ? "warn" : "bad"),
      kpi(num(k.n_eval_unknown), "Eval crops (unknown)",
        "products that are not in the catalog", k.n_eval_unknown ? "info" : "warn"),
      kpi(num(dup), "Duplicate photos across SKUs",
        `${num(o.duplicates.counts.eval_overlap)} eval crops copy a gallery photo`, dup ? "bad" : "ok"),
      kpi(k.imbalance_ratio == null ? "-" : `${k.imbalance_ratio}x`, "Imbalance (max / median)",
        "largest SKU vs typical SKU", k.imbalance_ratio > 5 ? "warn" : "ok"),
      kpi(idx.exists ? (idx.stale ? "Stale" : "Fresh") : "Missing", "Gallery index",
        idx.exists
          ? `${num(idx.num_vectors)} vectors &middot; ${esc(idx.model || "")}`
          : "run embed_gallery.py",
        !idx.exists || idx.stale ? "bad" : "ok"),
    ];
    $("kpis").innerHTML = cards.join("");
  }

  function renderChecks(o) {
    const label = { ok: "OK", warn: "WARN", bad: "FIX" };
    $("checks").innerHTML = o.checks.map((c) =>
      `<li class="${c.status}"><span class="check-badge">${label[c.status]}</span>
        <span><span class="check-label">${esc(c.label)}.</span>
        <span class="check-detail"> ${esc(c.detail)}</span></span></li>`).join("");
  }

  function renderCharts(o) {
    const maxSkus = Math.max(1, ...o.distribution.map((d) => d.skus));
    const tone = (label) => (label === "1" ? "bad" : label === "2" ? "warn" : label === "3-4" ? "" : "ok");
    $("dist-chart").innerHTML = o.distribution.map((d) =>
      `<div class="bar-row"><span>${esc(d.label)}</span>
        <div class="bar-track"><div class="bar-fill ${tone(d.label)}" style="width:${(d.skus / maxSkus) * 100}%"></div></div>
        <span class="bar-value">${d.skus}</span></div>`).join("");

    const bp = o.kpis.by_packaging;
    $("pack-table").innerHTML =
      `<tr><th>Packaging</th><th>SKUs</th><th>Gallery images</th><th>Eval crops</th><th>Images / SKU</th></tr>` +
      Object.entries(bp).map(([p, v]) =>
        `<tr><td>${esc(p)}</td><td>${v.skus}</td><td>${v.images}</td><td>${v.eval_crops}</td>
         <td>${v.skus ? (v.images / v.skus).toFixed(1) : "-"}</td></tr>`).join("");

    const i = o.index;
    $("index-box").innerHTML = i.exists
      ? `<b>Index</b>: ${esc(i.model || "?")} &middot; ${i.embedding_dim}-d &middot; ${num(i.num_vectors)} vectors<br>
         <b>Built</b>: ${esc(i.built_at)}
         ${i.stale ? `<br><b style="color:var(--red)">${i.new_on_disk} new and ${i.missing_on_disk} removed images since then. Re-run embed_gallery.py.</b>` : ""}`
      : "No gallery index found.";
  }

  // ------------------------------------------------------------ similarity
  function simClass(v) { return v >= HOT ? "hot" : v >= WARM ? "warm" : ""; }

  function renderHistogram(h) {
    const W = 560, H = 190, padL = 30, padB = 26, padT = 8;
    const n = h.intra.length;
    const maxV = Math.max(0.0001, ...h.intra, ...h.inter);
    const bw = (W - padL - 6) / n;
    const bar = (vals, color) => vals.map((v, i) => {
      const bh = (v / maxV) * (H - padB - padT);
      return `<rect x="${(padL + i * bw + 1).toFixed(1)}" y="${(H - padB - bh).toFixed(1)}"
        width="${(bw - 2).toFixed(1)}" height="${bh.toFixed(1)}" fill="${color}" opacity="0.6"/>`;
    }).join("");
    const ticks = [0, 0.2, 0.4, 0.6, 0.8, 1.0].map((t) => {
      const i = (t - h.edges[0]) / (h.edges[1] - h.edges[0]);
      const x = padL + i * bw;
      return `<line x1="${x}" x2="${x}" y1="${H - padB}" y2="${H - padB + 4}" stroke="#98a2b3"/>
        <text x="${x}" y="${H - 8}" font-size="10" fill="#667085" text-anchor="middle">${t.toFixed(1)}</text>`;
    }).join("");
    $("sim-hist").innerHTML = `<svg class="hist-svg" viewBox="0 0 ${W} ${H}" role="img" aria-label="Similarity histogram">
        <line x1="${padL}" x2="${W - 6}" y1="${H - padB}" y2="${H - padB}" stroke="#98a2b3"/>
        ${bar(h.inter, "#e53935")}${bar(h.intra, "#4caf50")}${ticks}
        <text x="${padL}" y="12" font-size="11" fill="#4caf50">&#9632; same SKU</text>
        <text x="${padL + 80}" y="12" font-size="11" fill="#e53935">&#9632; different SKU</text>
        <text x="${W - 6}" y="${H - 8}" font-size="10" fill="#98a2b3" text-anchor="end">cosine similarity</text>
      </svg>`;
  }

  function pairCard(p) {
    const side = (s) => `<div class="pair-side">
        ${s.image_url ? `<a href="${s.image_url}" target="_blank" rel="noopener"><img loading="lazy" src="${s.image_url}" alt=""></a>` : ""}
        <div class="sku">${esc(s.sku_id)}</div>
        <div class="meta">${esc(s.packaging_type)} &middot; ${s.n_images} img</div></div>`;
    return `<div class="pair">${side(p.a)}
      <div class="pair-score ${simClass(p.similarity)}">${pct(p.similarity)}<small>similar</small></div>
      ${side(p.b)}</div>`;
  }

  function renderSimilarity(s) {
    const top = s.pairs[0];
    $("sim-headline").innerHTML = top
      ? `<span class="big">${pct(top.similarity)}</span>
         <b>${esc(top.a.packaging_type)}/${esc(top.a.sku_id)}</b> vs
         <b>${esc(top.b.packaging_type)}/${esc(top.b.sku_id)}</b>
         <span class="muted"> &mdash; the two most alike different SKUs in the gallery</span>`
      : "No pairs match these filters.";

    renderHistogram(s.hist);
    const m = s.summary;
    const confusable = s.products.filter((p) => p.nearest_sim != null && p.nearest_sim >= HOT).length;
    const negSep = s.products.filter((p) => p.separation != null && p.separation < 0).length;
    $("sim-kpis").innerHTML = [
      ["Highest cross-SKU similarity", pct(m.inter_max)],
      [`SKUs with a neighbor &ge; ${(HOT * 100).toFixed(0)}%`, `${confusable} / ${s.n_products}`],
      ["SKUs closer to another SKU than to themselves", `${negSep}`],
      ["Same-SKU p5 vs different-SKU p95", `${m.intra_p5 ?? "-"} vs ${m.inter_p95 ?? "-"}`],
    ].map(([l, v]) => `<div class="mini-kpi"><div class="v">${v}</div><div class="l">${l}</div></div>`).join("");

    const notes = [];
    if (m.gap_p5_p95 != null) {
      notes.push(m.gap_p5_p95 >= 0.05
        ? "Same-SKU views sit clearly above almost all different-SKU pairs."
        : m.gap_p5_p95 >= 0
          ? "The gap between same-SKU and lookalike pairs is razor-thin."
          : "The distributions overlap: some different SKUs look more alike than some views of the same SKU.");
    }
    notes.push("Gallery photos are cleaner than shelf crops, so treat this as an upper bound.");
    notes.push(`Pairs shown use the saved index (${esc(s.model || "?")}); rebuild it after adding images.`);
    $("sim-notes").innerHTML = notes.map((n) => `<li>${n}</li>`).join("");

    $("sim-pairs").innerHTML = s.pairs.map(pairCard).join("");
    $("sim-result").classList.remove("hidden");
  }

  async function runSimilarity() {
    const btn = $("sim-btn");
    const status = $("sim-status");
    btn.disabled = true;
    status.className = "dash-status";
    status.textContent = "Computing similarities from the gallery index…";
    try {
      const q = new URLSearchParams({
        top: $("sim-top").value,
        same_packaging: $("sim-same-pack").checked ? "true" : "false",
      });
      const s = await getJSON(`/api/dashboard/similarity?${q}`);
      state.sim = s;
      state.simByKey = new Map(s.products.map((p) => [key(p), p]));
      status.classList.add("hidden");
      renderSimilarity(s);
      renderTable();
    } catch (err) {
      status.className = "dash-status error";
      status.textContent = `Could not compute similarity: ${err.message}`;
    } finally {
      btn.disabled = false;
    }
  }

  // ------------------------------------------------------------ SKU table
  function nearestOf(r) { return state.simByKey.get(key(r)) || null; }

  function isConfusable(r) {
    const n = nearestOf(r);
    return !!n && ((n.nearest_sim != null && n.nearest_sim >= HOT) || (n.separation != null && n.separation < 0));
  }

  function flagsOf(r) {
    const f = [...r.flags];
    if (isConfusable(r)) f.push("confusable");
    return f;
  }

  function visibleRows() {
    const q = state.search.trim().toLowerCase();
    let rows = state.overview.skus.filter((r) => {
      if (state.pack !== "all" && r.packaging_type !== state.pack) return false;
      if (q && !r.sku_id.toLowerCase().includes(q)) return false;
      if (state.flag !== "all" && !flagsOf(r).includes(state.flag)) return false;
      return true;
    });
    const k = state.sortKey;
    const val = (r) => {
      if (k === "nearest_sim") return nearestOf(r)?.nearest_sim ?? -1;
      if (k === "nearest_sku") return nearestOf(r)?.nearest_sku ?? "";
      return r[k];
    };
    rows = rows.slice().sort((a, b) => {
      const x = val(a), y = val(b);
      if (x < y) return -1 * state.sortDir;
      if (x > y) return 1 * state.sortDir;
      return a.sku_id.localeCompare(b.sku_id);
    });
    return rows;
  }

  function countCell(n) {
    const cls = n <= 1 ? "low" : n < state.minGood ? "mid" : "good";
    return `<div class="count-cell"><span class="num">${n}</span>
      <span class="count-bar ${cls}" style="width:${Math.min(n, 20) * 6}px"></span></div>`;
  }

  function thumbs(title, items) {
    if (!items.length) return `<div class="thumb-section"><div class="thumb-title">${title}</div><span class="muted">none</span></div>`;
    return `<div class="thumb-section"><div class="thumb-title">${title}</div><div class="thumbs">` +
      items.map((i) => `<a class="thumb" href="${i.url}" target="_blank" rel="noopener" title="${esc(i.name)}">
        <img loading="lazy" src="${i.url}" alt=""><span>${esc(i.name)}</span></a>`).join("") +
      `</div></div>`;
  }

  function renderTable() {
    const rows = visibleRows();
    $("sku-count").textContent = `${rows.length} / ${state.overview.skus.length}`;
    const out = [];
    for (const r of rows) {
      const n = nearestOf(r);
      const sim = n?.nearest_sim;
      out.push(`<tr class="row" data-key="${esc(key(r))}">
        <td class="sku-name">${esc(r.sku_id)}</td>
        <td>${esc(r.packaging_type)}</td>
        <td>${countCell(r.n_images)}</td>
        <td>${countCell(r.n_eval)}</td>
        <td class="sku-name">${n ? esc(n.nearest_packaging + "/" + n.nearest_sku) : '<span class="muted">run similarity</span>'}</td>
        <td>${sim == null ? "-" : `<span class="sim-val ${simClass(sim)}">${pct(sim)}</span>`}</td>
        <td>${flagsOf(r).map((f) => `<span class="flag ${FLAG_INFO[f][0]}">${FLAG_INFO[f][1]}</span>`).join("")}</td>
      </tr>`);
      if (state.expanded.has(key(r))) {
        out.push(`<tr class="detail"><td colspan="7">
          ${thumbs(`Gallery images (${r.images.length})`, r.images)}
          ${thumbs(`Eval crops (${r.eval_images.length})`, r.eval_images)}</td></tr>`);
      }
    }
    $("sku-body").innerHTML = out.join("") || `<tr><td colspan="7" class="muted" style="padding:16px">No SKUs match.</td></tr>`;

    document.querySelectorAll("#sku-table th[data-sort]").forEach((th) => {
      th.classList.toggle("sorted", th.dataset.sort === state.sortKey);
      th.classList.toggle("desc", th.dataset.sort === state.sortKey && state.sortDir < 0);
    });
  }

  function renderChips() {
    const packs = ["all", ...Object.keys(state.overview.kpis.by_packaging)];
    $("pack-chips").innerHTML = packs.map((p) =>
      `<button class="chip ${p === state.pack ? "chip-active" : ""}" data-pack="${p}" type="button">${p === "all" ? "All" : p}</button>`).join("");
  }

  // ------------------------------------------------------------ quality
  const KIND_INFO = {
    cross_sku: ["bad", "Same photo under different SKUs", "One of these is filed wrong, or the SKUs are really one product."],
    label_conflict: ["bad", "Eval label disagrees with gallery", "The eval crop is identical to a gallery photo of a different SKU."],
    eval_overlap: ["warn", "Eval crop copies a gallery photo", "It will always match itself at 100%, inflating accuracy. Replace it with a different shelf photo."],
    same_sku: ["info", "Duplicate inside one SKU", "Harmless but wasted. Delete the extra copy."],
    eval_dup: ["info", "Duplicate inside the eval set", "Counted twice in the eval numbers."],
  };

  function renderQuality(o) {
    const where = { gallery: "gallery", eval_known: "eval known", eval_unknown: "eval unknown" };
    const groups = o.duplicates.groups.map((g) => {
      const [tone, title, hint] = KIND_INFO[g.kind];
      return `<div class="q-group"><div class="q-head"><span class="flag ${tone} kind">${esc(title)}</span>
        <span class="muted">${g.files.length} identical files (md5 ${esc(g.md5)})</span></div>
        <div class="q-sub">${esc(hint)}</div>
        <div class="q-files">${g.files.map((f) => `<a class="q-file" href="${f.url}" target="_blank" rel="noopener">
          <img loading="lazy" src="${f.url}" alt="">
          <div class="where">${where[f.where]}</div>
          <div class="sku">${f.sku_id ? esc(f.packaging_type + "/" + f.sku_id) : "-"}</div>
          <div class="muted">${esc(f.name)}</div></a>`).join("")}</div></div>`;
    }).join("");

    const orphan = o.orphan_eval.length
      ? `<div class="q-section-title">Eval folders with no gallery folder (skipped by eval)</div>
         <ul class="notes">${o.orphan_eval.map((x) =>
           `<li><code>${esc(x.packaging_type)}/${esc(x.sku_id)}</code> &mdash; ${x.n_images} image(s). Rename it to match a gallery SKU or add the SKU.</li>`).join("")}</ul>`
      : "";
    const tiny = o.tiny.length
      ? `<div class="q-section-title">Tiny gallery images (short side &lt; 64 px)</div>
         <div class="q-files">${o.tiny.map((t) => `<a class="q-file" href="${t.url}" target="_blank" rel="noopener">
           <img loading="lazy" src="${t.url}" alt=""><div class="sku">${esc(t.packaging_type + "/" + t.sku_id)}</div>
           <div class="muted">${t.w}x${t.h}</div></a>`).join("")}</div>`
      : "";
    $("quality").innerHTML =
      (groups || `<p class="q-empty">No duplicate files found.</p>`) + orphan + tiny;
  }

  // ------------------------------------------------------------ load + wire
  async function load() {
    const status = $("status");
    status.className = "dash-status";
    status.textContent = "Loading data…";
    try {
      const o = await getJSON("/api/dashboard/overview");
      state.overview = o;
      state.minGood = o.min_good_images;
      status.classList.add("hidden");
      $("generated").textContent = `updated ${o.generated_at.replace("T", " ")}`;
      renderKpis(o);
      renderChecks(o);
      renderCharts(o);
      renderChips();
      renderTable();
      renderQuality(o);
    } catch (err) {
      status.className = "dash-status error";
      status.textContent = `Could not load the dashboard: ${err.message}`;
    }
  }

  $("sim-btn").addEventListener("click", runSimilarity);
  $("refresh-btn").addEventListener("click", async () => {
    state.sim = null;
    state.simByKey = new Map();
    $("sim-result").classList.add("hidden");
    await load();
  });
  $("sku-search").addEventListener("input", (e) => { state.search = e.target.value; renderTable(); });
  $("flag-filter").addEventListener("change", (e) => { state.flag = e.target.value; renderTable(); });
  $("pack-chips").addEventListener("click", (e) => {
    const b = e.target.closest("[data-pack]");
    if (!b) return;
    state.pack = b.dataset.pack;
    renderChips();
    renderTable();
  });
  document.querySelector("#sku-table thead").addEventListener("click", (e) => {
    const th = e.target.closest("th[data-sort]");
    if (!th) return;
    if (state.sortKey === th.dataset.sort) state.sortDir *= -1;
    else { state.sortKey = th.dataset.sort; state.sortDir = th.dataset.sort === "sku_id" || th.dataset.sort === "packaging_type" ? 1 : -1; }
    renderTable();
  });
  $("sku-body").addEventListener("click", (e) => {
    if (e.target.closest("a")) return;
    const tr = e.target.closest("tr.row");
    if (!tr) return;
    const k = tr.dataset.key;
    if (state.expanded.has(k)) state.expanded.delete(k); else state.expanded.add(k);
    renderTable();
  });

  load();
})();
