"use strict";
const $ = (id) => document.getElementById(id);
const token = document.querySelector('meta[name="review-token"]').content;
const names = { left: "左手", right: "右手" };
const statuses = {
  approved: "已通过审核",
  pending: "待审核",
  accepted: "已通过",
  rejected: "已拒绝",
  ready: "双手已通过",
  prepared: "准备完成",
  complete: "导出完成",
  failed: "失败",
  stale: "已过期",
  queued: "排队中",
  validating: "校验审核中",
  preparing: "计算偏移／写入 Parquet",
  rendering: "生成 RRD 中",
  running: "导出中",
};
let current = null,
  side = "left",
  bitmap = null,
  dirty = false,
  loading = false;
let frameRequest = 0,
  dragging = -1,
  saving = false;
let episodeRequest = 0;
let jobs = new Map(),
  episodeRows = [];
const submittedDrafts = new Map();
function currentBusy() {
  return Boolean(current && jobs.get(current.id)?.running);
}
const previews = { left: null, right: null };
const previewKeys = { left: null, right: null };
const canvas = $("frame-canvas"),
  ctx = canvas.getContext("2d");
function message(text, error = false) {
  $("message").textContent = text;
  $("message").className = error ? "error" : "";
}
async function api(url, body) {
  const response = await fetch(
    url,
    body === undefined
      ? {}
      : {
          method: "POST",
          headers: { "Content-Type": "application/json", "X-Review-Token": token },
          body: JSON.stringify(body),
        },
  );
  const result = await response.json();
  if (!response.ok) throw new Error(result.error || response.statusText);
  return result;
}
function choice() {
  return current.selected[side];
}
function markEdited() {
  dirty = true;
  choice().status = "pending";
  choice().physical_side = "unknown";
  choice().source = "manual";
  updateStatus();
  drawOverview(side);
}
function updateStatus() {
  if (!current) return;
  $("hint").textContent = currentBusy()
    ? "本条数据正在后台处理，可切换到其他数据继续审核。"
    : current.video_error ||
      (dirty ? "有未保存修改，请检查后确认；暂存不代表通过。" : current.pending_reason) ||
      "审核依据已保存，可按需导出 RRD。";
  $("side-status").textContent =
    names[side] + "：" + (statuses[choice().status] || choice().status);
  $("saved-state").textContent =
    (dirty ? "有未保存修改 · " : "已保存 · ") +
    Object.entries(current.selected)
      .map(([s, c]) => names[s] + " " + statuses[c.status])
      .join(" / ");
  for (const id of ["accept", "reject", "save", "swap"])
    $(id).disabled = loading || currentBusy() || saving;
  const hasPreviews = ["left", "right"].every(
    (s) =>
      previewKeys[s] === `${current.id}:${current.selected[s].frame_index}` &&
      previews[s] &&
      current.selected[s].quad?.length === 4,
  );
  $("approve").disabled = loading || currentBusy() || saving || !hasPreviews;
  $("approve").textContent = currentBusy()
    ? statuses[jobs.get(current.id).status] || "后台处理中"
    : "通过审核";
  $("export").disabled = loading || currentBusy() || saving || dirty || !current.approved;
  for (const id of [
    "side",
    "frame",
    "timeline",
    "candidate",
    "previous",
    "next",
    "jump",
    "notes",
    "reviewer",
    "undo",
    "clear",
    "edit-left",
    "edit-right",
  ])
    $(id).disabled = currentBusy() || saving;
  for (const s of ["left", "right"])
    $("overview-" + s + "-status").textContent = statuses[current.selected[s].status];
}
async function listEpisodes() {
  const rows = await api("/api/episodes");
  episodeRows = rows;
  $("episodes").replaceChildren();
  rows.forEach((row) => {
    const button = document.createElement("button");
    button.dataset.episodeId = row.id;
    button.textContent = episodeLabel(row);
    if (current && current.id === row.id) button.className = "active";
    button.onclick = () => guard(() => openEpisode(row.id));
    $("episodes").append(button);
  });
  updateEpisodeList();
  updateStatus();
  return rows;
}
function updateEpisodeList() {
  const readyCount = episodeRows.filter((row) => row.status === "ready").length;
  $("progress").textContent = `${readyCount}/${episodeRows.length} 已通过`;
  for (const button of $("episodes").children) {
    const row = episodeRows.find((row) => row.id === Number(button.dataset.episodeId));
    if (row) button.textContent = episodeLabel(row);
  }
}
function episodeLabel(row) {
  const job = jobs.get(row.id);
  const label = job?.running
    ? statuses[job.status]
    : job?.status === "failed"
      ? "处理失败，可重试"
      : job?.status === "complete" && job.review?.revision === row.revision
        ? "导出完成"
        : statuses[row.status];
  return `${row.id + 1}. ${row.episode.split("/").slice(-2).join("/")} · ${label}`;
}
async function openEpisode(id) {
  if (saving) return;
  if (dirty && !confirm("当前修改未保存，放弃并切换数据？")) return;
  const ticket = ++episodeRequest;
  const data = await api(`/api/episode?id=${id}`);
  if (ticket !== episodeRequest) return;
  current = submittedDrafts.has(id) ? structuredClone(submittedDrafts.get(id)) : data;
  Object.assign(current, {
    candidates: data.candidates,
    frame_count: data.frame_count,
    image_size: data.image_size,
  });
  dirty = submittedDrafts.has(id) && jobs.get(id)?.status === "failed";
  side = "left";
  $("side").value = side;
  $("episode-title").textContent = data.episode;
  $("hint").textContent =
    data.video_error || data.pending_reason || "审核依据已保存，可按需导出 RRD。";
  $("reviewer").value =
    data.reviewer || localStorage.getItem("vive-reviewer") || data.default_reviewer || "";
  $("frame-count").textContent = `共 ${data.frame_count} 帧`;
  $("timeline").max = $("frame").max = Math.max(0, data.frame_count - 1);
  $("candidate").replaceChildren(new Option("选择候选…", ""));
  data.candidates.forEach((c) =>
    $("candidate").add(
      new Option(
        `帧 ${c.frame_index} · ${c.image_side ? "图像侧提示 " + names[c.image_side] : "候选"}`,
        c.candidate_id,
      ),
    ),
  );
  $("editor").open = false;
  for (const s of ["left", "right"]) {
    previews[s]?.close();
    previews[s] = null;
    previewKeys[s] = null;
    drawOverview(s);
  }
  await Promise.all([loadPreview("left"), loadPreview("right")]);
  await showChoice();
  await listEpisodes();
}
async function loadPreview(s) {
  const id = current.id,
    frame = current.selected[s].frame_index,
    ticket = episodeRequest;
  const key = `${id}:${frame}`;
  if (previews[s] && previewKeys[s] === key) {
    drawOverview(s);
    return;
  }
  previewKeys[s] = null;
  previews[s]?.close();
  previews[s] = null;
  drawOverview(s);
  updateStatus();
  const response = await fetch(`/api/frame?id=${id}&frame=${frame}`);
  if (!response.ok) {
    $("overview-" + s + "-frame").textContent = (await response.json()).error;
    return;
  }
  const image = await createImageBitmap(await response.blob());
  if (ticket !== episodeRequest || current.id !== id || current.selected[s].frame_index !== frame) {
    image.close();
    return;
  }
  previews[s]?.close();
  previews[s] = image;
  previewKeys[s] = key;
  drawOverview(s);
  updateStatus();
}
function drawOverview(s) {
  if (!current) return;
  const image = previews[s],
    selected = current.selected[s];
  const full = $("overview-" + s),
    fc = full.getContext("2d"),
    crop = $("overview-" + s + "-crop"),
    cc = crop.getContext("2d");
  fc.clearRect(0, 0, full.width, full.height);
  cc.clearRect(0, 0, crop.width, crop.height);
  $("overview-" + s + "-frame").textContent =
    `帧 ${selected.frame_index} · ${selected.quad?.length === 4 ? "四角已标注" : "需要补标四角"}`;
  if (!image || previewKeys[s] !== `${current.id}:${selected.frame_index}`) return;
  full.width = image.width;
  full.height = image.height;
  fc.drawImage(image, 0, 0);
  paint(fc, selected.quad, full.width / Math.max(1, full.getBoundingClientRect().width));
  drawCrop(cc, crop, image, selected.quad);
}
function drawCrop(context, target, image, points, overlay = true) {
  if (!points.length) return;
  const xs = points.map((p) => p[0]),
    ys = points.map((p) => p[1]);
  const x = Math.max(0, Math.min(...xs) - 30),
    y = Math.max(0, Math.min(...ys) - 30);
  const w = Math.min(image.width - x, Math.max(...xs) - x + 30),
    h = Math.min(image.height - y, Math.max(...ys) - y + 30);
  if (w <= 0 || h <= 0) return;
  const scale = Math.min(target.width / w, target.height / h),
    ox = (target.width - w * scale) / 2,
    oy = (target.height - h * scale) / 2;
  context.drawImage(image, x, y, w, h, ox, oy, w * scale, h * scale);
  if (overlay)
    paint(
      context,
      points.map((p) => [(p[0] - x) * scale + ox, (p[1] - y) * scale + oy]),
    );
}
async function editSide(s) {
  side = s;
  $("side").value = s;
  $("editor").open = true;
  await showChoice();
  $("editor").scrollIntoView({ block: "start" });
}
async function showChoice() {
  $("frame").value = $("timeline").value = choice().frame_index;
  $("notes").value = choice().notes || "";
  $("candidate").value = choice().candidate_id || "";
  updateStatus();
  await Promise.all([loadFrame(), loadPreview(side)]);
}
async function loadFrame() {
  const ticket = ++frameRequest,
    id = current.id,
    frame = choice().frame_index;
  loading = true;
  bitmap?.close();
  bitmap = null;
  draw();
  updateStatus();
  try {
    const response = await fetch(`/api/frame?id=${id}&frame=${frame}`);
    if (!response.ok) throw new Error((await response.json()).error);
    const image = await createImageBitmap(await response.blob());
    if (ticket !== frameRequest) {
      image.close();
      return;
    }
    bitmap = image;
    canvas.width = image.width;
    canvas.height = image.height;
    resizeCanvas();
  } finally {
    if (ticket === frameRequest) {
      loading = false;
      updateStatus();
    }
  }
}
async function goFrame(frame) {
  if (saving || currentBusy()) return;
  if (!Number.isInteger(frame) || frame < 0 || frame >= current.frame_count)
    throw new Error("帧号超出视频范围");
  if (frame === choice().frame_index) return;
  Object.assign(choice(), { frame_index: frame, quad: [], candidate_id: null });
  markEdited();
  await showChoice();
}
function paint(context, points, scale = 1) {
  if (!points.length) return;
  context.strokeStyle = "#4cf2d2";
  context.fillStyle = "#4cf2d2";
  context.lineWidth = 2 * scale;
  context.beginPath();
  points.forEach(([x, y], i) => (i ? context.lineTo(x, y) : context.moveTo(x, y)));
  if (points.length === 4) context.closePath();
  context.stroke();
  points.forEach(([x, y], i) => {
    context.beginPath();
    context.arc(x, y, 5 * scale, 0, 2 * Math.PI);
    context.fill();
    context.font = `${14 * scale}px sans-serif`;
    context.fillText(`q${i}`, x + 8 * scale, y - 8 * scale);
  });
}
function resizeCanvas() {
  const width = Math.min(
    canvas.parentElement.clientWidth,
    (window.innerHeight * 0.6 * canvas.width) / canvas.height,
  );
  canvas.style.width = `${width * Number($("zoom").value)}px`;
  draw();
}
function draw() {
  if (current) drawOverview(side);
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  const crop = $("crop"),
    cc = crop.getContext("2d");
  cc.clearRect(0, 0, crop.width, crop.height);
  $("coordinates").textContent = "";
  if (!bitmap || !current) return;
  ctx.drawImage(bitmap, 0, 0);
  const points = choice().quad || [];
  if ($("overlay").checked)
    paint(ctx, points, canvas.width / Math.max(1, canvas.getBoundingClientRect().width));
  $("coordinates").textContent = points.length
    ? points.map(([x, y], i) => `q${i} (${x.toFixed(1)}, ${y.toFixed(1)})`).join(" · ")
    : "尚未标注四角";
  drawCrop(cc, crop, bitmap, points, $("overlay").checked);
}
function point(event) {
  const rect = canvas.getBoundingClientRect();
  return [
    Math.min(
      canvas.width - 1,
      Math.max(0, ((event.clientX - rect.left) * canvas.width) / rect.width),
    ),
    Math.min(
      canvas.height - 1,
      Math.max(0, ((event.clientY - rect.top) * canvas.height) / rect.height),
    ),
  ];
}
canvas.onpointerdown = (event) => {
  if (!bitmap || loading || currentBusy() || saving) return;
  const p = point(event),
    points = choice().quad,
    threshold = (14 * canvas.width) / canvas.getBoundingClientRect().width;
  dragging = points.findIndex((q) => Math.hypot(q[0] - p[0], q[1] - p[1]) < threshold);
  if (dragging < 0 && points.length < 4) {
    points.push(p);
    dragging = points.length - 1;
  }
  if (dragging >= 0) {
    canvas.setPointerCapture(event.pointerId);
    markEdited();
    draw();
  }
};
canvas.onpointermove = (event) => {
  if (dragging >= 0 && !loading && !saving && !currentBusy()) {
    choice().quad[dragging] = point(event);
    draw();
  }
};
canvas.onpointerup = canvas.onpointercancel = () => {
  dragging = -1;
};
async function save(status) {
  if (loading || saving || currentBusy()) return;
  const selected = structuredClone(current.selected);
  if (status === "accepted") {
    selected[side].physical_side = side;
  }
  if (status) selected[side].status = status;
  const reviewer = $("reviewer").value.trim();
  saving = true;
  updateStatus();
  try {
    const result = await api(`/api/review?id=${current.id}`, {
      revision: current.revision,
      binding: current.binding,
      reviewer,
      selected,
    });
    Object.assign(current, result);
    dirty = false;
    localStorage.setItem("vive-reviewer", reviewer);
    submittedDrafts.delete(current.id);
    jobs.delete(current.id);
    message(status === "accepted" ? "此侧审核已通过并保存。" : "审核记录已保存。");
    await listEpisodes();
  } finally {
    saving = false;
    updateStatus();
  }
}
async function guard(action) {
  try {
    await action();
  } catch (error) {
    message(error.message, true);
  }
}
$("side").onchange = () =>
  guard(async () => {
    if (saving) {
      $("side").value = side;
      return;
    }
    side = $("side").value;
    await showChoice();
  });
$("reviewer").oninput = () => {
  dirty = true;
  updateStatus();
};
$("notes").oninput = () => {
  choice().notes = $("notes").value;
  dirty = true;
  updateStatus();
};
$("edit-left").onclick = () => guard(() => editSide("left"));
$("edit-right").onclick = () => guard(() => editSide("right"));
$("editor").ontoggle = resizeCanvas;
$("previous").onclick = () => guard(() => goFrame(choice().frame_index - 1));
$("next").onclick = () => guard(() => goFrame(choice().frame_index + 1));
$("jump").onclick = () => guard(() => goFrame(Number($("frame").value)));
$("frame").onkeydown = (e) => {
  if (e.key === "Enter") guard(() => goFrame(Number($("frame").value)));
};
$("timeline").onchange = () => guard(() => goFrame(Number($("timeline").value)));
$("candidate").onchange = () =>
  guard(async () => {
    const c = current.candidates.find((c) => c.candidate_id === $("candidate").value);
    if (!c) return;
    Object.assign(choice(), {
      frame_index: c.frame_index,
      quad: structuredClone(c.quad),
      candidate_id: c.candidate_id,
    });
    markEdited();
    await showChoice();
  });
$("overlay").onchange = draw;
$("zoom").onchange = resizeCanvas;
$("undo").onclick = () => {
  if (!loading && !saving && !currentBusy()) {
    choice().quad.pop();
    markEdited();
    draw();
  }
};
$("clear").onclick = () => {
  if (!loading && !saving && !currentBusy()) {
    choice().quad = [];
    markEdited();
    draw();
  }
};
$("swap").onclick = () =>
  guard(async () => {
    [current.selected.left, current.selected.right] = [
      current.selected.right,
      current.selected.left,
    ];
    Object.values(current.selected).forEach((c) => {
      c.status = "pending";
      c.physical_side = "unknown";
    });
    dirty = true;
    await Promise.all([loadPreview("left"), loadPreview("right")]);
    await showChoice();
  });
$("accept").onclick = () => guard(() => save("accepted"));
$("reject").onclick = () => guard(() => save("rejected"));
$("save").onclick = () => guard(() => save());
$("approve").onclick = () =>
  guard(async () => {
    if ($("approve").disabled) return;
    const id = current.id,
      draft = structuredClone(current),
      selected = structuredClone(current.selected);
    for (const s of ["left", "right"])
      Object.assign(selected[s], { status: "accepted", physical_side: s });
    const reviewer = $("reviewer").value.trim();
    saving = true;
    updateStatus();
    try {
      await api(`/api/approve?id=${id}`, {
        revision: current.revision,
        binding: current.binding,
        reviewer,
        selected,
      });
      draft.reviewer = reviewer;
      submittedDrafts.set(id, draft);
      jobs.set(id, { index: id, kind: "approve", status: "queued", running: true });
      dirty = false;
      localStorage.setItem("vive-reviewer", reviewer);
      message("已提交审核，正在保存特征帧和重建依据；可继续审核其他数据。需要 RRD 时再点击导出。");
    } finally {
      saving = false;
      updateStatus();
    }
  });
$("export").onclick = () =>
  guard(async () => {
    if ($("export").disabled) return;
    const id = current.id;
    await api(`/api/export?id=${id}`, {});
    jobs.set(id, { index: id, kind: "export", status: "queued", running: true });
    updateStatus();
    message("本条 RRD 已提交后台导出，可继续审核其他数据。源文件不会被覆盖。");
  });
let polling = false;
async function poll() {
  if (polling) return;
  polling = true;
  try {
    const previousJobs = jobs;
    const state = await api("/api/status");
    jobs = new Map((state.jobs || []).map((j) => [j.index, j]));
    for (const job of jobs.values()) {
      if (
        previousJobs.get(job.index)?.running &&
        !job.running &&
        current?.id === job.index &&
        !dirty
      ) {
        message(
          job.status === "failed"
            ? `处理失败：${job.error || job.reason || "请查看下方详情"}`
            : job.kind === "approve"
              ? "审核依据已保存。需要查看修正结果时，点击「导出 RRD 文件」。"
              : "RRD 导出完成，源文件未覆盖。",
          job.status === "failed",
        );
      }
      if (job.review && submittedDrafts.has(job.index) && !job.running) {
        if (current?.id === job.index && !dirty) Object.assign(current, job.review);
        const row = episodeRows.find((r) => r.id === job.index);
        if (row) Object.assign(row, job.review);
        submittedDrafts.delete(job.index);
      } else if (
        job.status === "failed" &&
        submittedDrafts.has(job.index) &&
        current?.id === job.index
      ) {
        dirty = true;
      }
    }
    if (current && !dirty && !currentBusy()) {
      const id = current.id,
        revision = current.revision;
      const latest = await api(`/api/review-state?id=${id}`);
      if (current?.id === id && current.revision === revision && !dirty) {
        current.approved = latest.approved && latest.revision === revision;
        current.status = current.approved ? "ready" : "pending";
        current.pending_reason =
          latest.revision === revision
            ? latest.pending_reason
            : "审核记录已在其他位置修改，请重新选择该条数据加载。";
        const row = episodeRows.find((r) => r.id === id);
        if (row) Object.assign(row, latest);
      }
    }
    updateEpisodeList();
    updateStatus();
    const lines = [
      state.running ? "后台处理中，可继续审核其他数据。" : "审核与导出状态（RRD 需单独导出）：",
    ];
    const rows = new Map((state.summary?.entries || []).map((r) => [r.episode, r]));
    for (const job of jobs.values()) {
      const episode =
        job.episode || episodeRows.find((r) => r.id === job.index)?.episode || String(job.index);
      if (job.running || job.status === "failed" || job.status === "approved")
        rows.set(episode, { ...job, episode });
    }
    for (const row of rows.values()) {
      lines.push(
        `${row.episode}：${statuses[row.status] || row.status}${row.reused ? "（复用已有结果）" : ""}${row.error ? " · " + row.error : ""}${row.reason ? " · " + row.reason : ""}${row.recording ? "\n  " + row.recording : ""}`,
      );
    }
    $("export-status").textContent = lines.join("\n");
  } finally {
    polling = false;
  }
}
window.addEventListener("beforeunload", (event) => {
  if (dirty) {
    event.preventDefault();
    event.returnValue = "";
  }
});
window.addEventListener("resize", resizeCanvas);
guard(async () => {
  const rows = await listEpisodes();
  if (rows.length) await openEpisode(rows[0].id);
  await poll();
  message("检查左右参考图，点击「通过审核」保存依据；需要时单独导出 RRD。");
});
setInterval(() => guard(poll), 1000);
