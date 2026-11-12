"use strict";

const progress = document.querySelector("#progress");
const back = document.querySelector("#back");
const finish = document.querySelector("#finish");
const status = document.querySelector("#status");
const choices = Array.from(document.querySelectorAll("[data-label]"));
const win = document.querySelector("#win");
const fitsBox = document.querySelector("#fits");
const KEYS = { "1": "present", "2": "absent", "3": "spike", "4": "unsure" };
const FIT_COLORS = { A: "#b4541a", B: "#1d7a52" };
let state = { n_total: 0, next_index: 0, frozen: false };
let current = null;
let index = 0;
let busy = true;

function message(text = "") {
  status.textContent = text;
  status.hidden = !text;
}

function controls() {
  choices.forEach((button) => { button.disabled = busy || state.frozen || !current; });
  back.disabled = busy || state.frozen || index <= 0;
  finish.disabled = busy || state.frozen;
  progress.textContent = `${Math.min(index + 1, state.n_total)} / ${state.n_total}`;
}

async function request(path, payload) {
  const options = payload === undefined ? {} : {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  };
  const response = await fetch(path, { ...options, cache: "no-store" });
  if (!response.ok) throw new Error(response.status === 409 ? "frozen" : "request");
  return response.json();
}

function gauss(x, c) {
  return c.amplitude * Math.exp((-4 * Math.LN2 * (x - c.center_cm1) ** 2) / c.fwhm_cm1 ** 2);
}

function plot(canvas, item) {
  const rect = canvas.getBoundingClientRect();
  const ratio = window.devicePixelRatio || 1;
  canvas.width = Math.round(rect.width * ratio);
  canvas.height = Math.round(rect.height * ratio);
  const ctx = canvas.getContext("2d");
  ctx.scale(ratio, ratio);
  const width = rect.width;
  const height = rect.height;
  ctx.clearRect(0, 0, width, height);
  if (!item) return;
  const trace = item.window;
  const left = 24, right = width - 16, top = 34, bottom = height - 44;
  const xmin = trace.x[0], xmax = trace.x[trace.x.length - 1];
  const dense = Array.from({ length: 400 }, (_, i) => xmin + ((xmax - xmin) * i) / 399);
  const curves = [];
  ["A", "B"].forEach((name) => {
    const comps = item.fits[name];
    curves.push({ name, dash: [], width: 2, y: dense.map((v) => comps.reduce((t, c) => t + gauss(v, c), 0)) });
    comps.filter((c) => c.kept).forEach((c) => {
      curves.push({ name, dash: [5, 4], width: 1, y: dense.map((v) => gauss(v, c)) });
    });
  });
  let ymin = Infinity, ymax = -Infinity;
  trace.y.concat(...curves.map((c) => c.y)).forEach((v) => { ymin = Math.min(ymin, v); ymax = Math.max(ymax, v); });
  const padding = Math.max((ymax - ymin) * 0.08, 0.001);
  ymin -= padding;
  ymax += padding;
  const x = (v) => left + ((v - xmin) / (xmax - xmin)) * (right - left);
  const y = (v) => bottom - ((v - ymin) / (ymax - ymin)) * (bottom - top);
  ctx.save();
  ctx.beginPath();
  ctx.rect(left, top, right - left, bottom - top);
  ctx.clip();
  const lo = item.band.center_cm1 - item.band.fwhm_cm1 / 2;
  const hi = item.band.center_cm1 + item.band.fwhm_cm1 / 2;
  ctx.fillStyle = "rgba(218, 155, 35, 0.18)";
  ctx.fillRect(x(lo), top, x(hi) - x(lo), bottom - top);
  ctx.strokeStyle = "#b37713";
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(x(item.band.center_cm1), top);
  ctx.lineTo(x(item.band.center_cm1), bottom);
  ctx.stroke();
  ctx.strokeStyle = "#c3ced6";
  ctx.beginPath();
  ctx.moveTo(left, y(0));
  ctx.lineTo(right, y(0));
  ctx.stroke();
  ctx.strokeStyle = "#245a7a";
  ctx.lineWidth = 1;
  ctx.beginPath();
  trace.x.forEach((v, i) => {
    if (i === 0) ctx.moveTo(x(v), y(trace.y[i]));
    else ctx.lineTo(x(v), y(trace.y[i]));
  });
  ctx.stroke();
  ctx.fillStyle = "#153e59";
  trace.x.forEach((v, i) => {
    ctx.beginPath();
    ctx.arc(x(v), y(trace.y[i]), 2, 0, Math.PI * 2);
    ctx.fill();
  });
  curves.forEach((curve) => {
    ctx.strokeStyle = FIT_COLORS[curve.name];
    ctx.lineWidth = curve.width;
    ctx.setLineDash(curve.dash);
    ctx.beginPath();
    dense.forEach((v, i) => {
      if (i === 0) ctx.moveTo(x(v), y(curve.y[i]));
      else ctx.lineTo(x(v), y(curve.y[i]));
    });
    ctx.stroke();
  });
  ctx.setLineDash([]);
  ctx.restore();
  ctx.strokeStyle = "#9babb7";
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(left, top);
  ctx.lineTo(left, bottom);
  ctx.lineTo(right, bottom);
  ctx.stroke();
  ctx.font = "12px system-ui, sans-serif";
  ctx.textAlign = "left";
  let lx = left;
  [["fit A", FIT_COLORS.A], ["fit B", FIT_COLORS.B]].forEach(([text, color]) => {
    ctx.strokeStyle = color;
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(lx, 16);
    ctx.lineTo(lx + 20, 16);
    ctx.stroke();
    ctx.fillStyle = "#192d3d";
    ctx.fillText(text, lx + 25, 20);
    lx += 80;
  });
  ctx.fillStyle = "#435c6e";
  ctx.fillText("solid: whole fit · dashed: kept components · shaded: disputed band", lx, 20);
  ctx.font = "11px system-ui, sans-serif";
  const ticks = width < 300 ? 3 : 5;
  for (let i = 0; i <= ticks; i += 1) {
    const value = xmin + ((xmax - xmin) * i) / ticks;
    ctx.textAlign = i === 0 ? "left" : i === ticks ? "right" : "center";
    ctx.fillText(Math.round(value).toString(), x(value), bottom + 16);
  }
  ctx.textAlign = "center";
  ctx.fillText("cm−1", (left + right) / 2, height - 9);
}

function cell(row, text, tag = "td") {
  const node = document.createElement(tag);
  node.textContent = text;
  row.append(node);
}

function table(item) {
  fitsBox.replaceChildren();
  if (!item) return;
  const lo = item.band.center_cm1 - item.band.fwhm_cm1 / 2;
  const hi = item.band.center_cm1 + item.band.fwhm_cm1 / 2;
  ["A", "B"].forEach((name) => {
    const heading = document.createElement("h2");
    heading.className = name.toLowerCase();
    heading.textContent = `Fit ${name}: keeps ${item.fits[name].filter((c) => c.kept).length} of ${item.fits[name].length}`;
    const grid = document.createElement("table");
    const head = document.createElement("tr");
    ["center", "FWHM", "SNR", "kept"].forEach((t) => cell(head, t, "th"));
    grid.append(head);
    item.fits[name].forEach((c) => {
      const row = document.createElement("tr");
      if (c.center_cm1 >= lo && c.center_cm1 <= hi) row.classList.add("in-band");
      if (!c.kept) row.classList.add("dropped");
      cell(row, c.center_cm1.toFixed(1));
      cell(row, c.fwhm_cm1.toFixed(1));
      cell(row, c.snr.toFixed(1));
      cell(row, c.kept ? "yes" : "no");
      grid.append(row);
    });
    fitsBox.append(heading, grid);
  });
  if (item.flags.mask_px_in_band > 0) {
    const flag = document.createElement("p");
    flag.className = "flag";
    flag.textContent = `Review flag: ${item.flags.mask_px_in_band} cosmic-ray mask pixel(s) in the band (a flag, not a veto)`;
    fitsBox.append(flag);
  }
}

function draw() {
  plot(win, current);
  table(current);
}

async function show(k) {
  index = k;
  current = k < state.n_total && !state.frozen ? await request(`/api/item/${k}`) : null;
  draw();
  controls();
  message(state.frozen ? "Finished" : !current ? "All items labeled. Finish when ready." : "");
}

async function action(work) {
  if (busy) return;
  busy = true;
  controls();
  message();
  try {
    await work();
  } catch (error) {
    if (error.message === "frozen") {
      state.frozen = true;
      current = null;
      draw();
      message("Finished");
    } else {
      // A lost response may follow a durable save. Reload resolves its status.
      current = null;
      draw();
      message("Connection or save failed. Reload to resume from saved labels.");
    }
  } finally {
    busy = false;
    controls();
  }
}

function label(value) {
  if (!current || state.frozen) return;
  action(async () => {
    await request("/api/label", { id: current.id, label: value });
    state = await request("/api/state");
    await show(index + 1);
  });
}

choices.forEach((button) => button.addEventListener("click", () => label(button.dataset.label)));
back.addEventListener("click", () => {
  if (index > 0 && !state.frozen) action(() => show(index - 1));
});
finish.addEventListener("click", () => {
  if (busy || state.frozen || !window.confirm("Freeze labels? This ends the session and prevents further changes.")) return;
  action(async () => {
    await request("/api/finish", {});
    state = await request("/api/state");
    await show(index);
  });
});
document.addEventListener("keydown", (event) => {
  if (event.altKey || event.ctrlKey || event.metaKey || event.repeat) return;
  if (Object.hasOwn(KEYS, event.key)) {
    event.preventDefault();
    label(KEYS[event.key]);
  } else if (event.key === "ArrowLeft" || event.key === "Backspace") {
    event.preventDefault();
    if (!back.disabled) back.click();
  }
});
new ResizeObserver(draw).observe(document.querySelector(".plots"));

(async () => {
  try {
    state = await request("/api/state");
    await show(state.next_index);
  } catch {
    message("Unable to load session. Reload to try again.");
  } finally {
    busy = false;
    controls();
  }
})();
