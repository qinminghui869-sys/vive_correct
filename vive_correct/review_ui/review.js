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
  interrupted: "已中断，可重试",
  scanning: "选帧中",
  waiting: "等待选帧",
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
  bitmapKey = null,
  dirty = false,
  loading = false;
let frameRequest = 0,
  dragging = -1,
  saving = false;
let episodeRequest = 0,
  listRequest = 0,
  listVersion = null;
let jobs = new Map(),
  episodeRows = [];
const submittedDrafts = new Map();
function eyeMode() {
  return current?.coordinate_mode === "eye-reference";
}
// Changing the shared frame invalidates only annotations made on another frame.
function setReferenceFrame(frame) {
  for (const s of eyeMode() ? ["left", "right"] : [side]) {
    const c = current.selected[s];
    if (c.frame_index === frame) continue;
    Object.assign(c, {
      frame_index: frame,
      quad: [],
      candidate_id: null,
      status: "pending",
      physical_side: "unknown",
      source: "manual",
    });
    dirty = true;
    refreshAxes(s);
  }
}
function currentBusy() {
  return Boolean(current && jobs.get(current.id)?.running);
}
const previews = { left: null, right: null };
const previewKeys = { left: null, right: null };
const axes = { left: null, right: null };
const axisTimers = { left: null, right: null };
const axisTickets = { left: 0, right: 0 };
const axisControllers = { left: null, right: null };
function axisKey(s) {
  const c = current.selected[s];
  return JSON.stringify([
    current.id,
    s,
    ...(eyeMode()
      ? ["left", "right"].map((hand) => {
          const item = current.selected[hand];
          return [item.frame_index, item.quad, item.axis_angles_deg || [0, 0, 0]];
        })
      : [c.frame_index, c.quad, c.axis_angles_deg || [0, 0, 0]]),
  ]);
}
function projection(s) {
  return axes[s]?.key === axisKey(s) ? axes[s].data?.axis_projection : null;
}
function refreshAxes(s, paired = false) {
  if (eyeMode() && !paired) {
    for (const hand of ["left", "right"]) refreshAxes(hand, true);
    return;
  }
  clearTimeout(axisTimers[s]);
  axisControllers[s]?.abort();
  const ticket = ++axisTickets[s],
    key = axisKey(s),
    id = current.id;
  axes[s] = { key, error: "正在计算坐标轴…" };
  updateStatus();
  axisTimers[s] = setTimeout(async () => {
    try {
      const c = current.selected[s];
      if (axisKey(s) !== key) return;
      const controller = (axisControllers[s] = new AbortController());
      const data = await api(
        `/api/axis-preview?id=${id}`,
        {
          side: s,
          frame_index: c.frame_index,
          quad: c.quad,
          axis_angles_deg: c.axis_angles_deg || [0, 0, 0],
          axis_convention: "intrinsic_XYZ_degrees",
          ...(eyeMode() ? { selected: current.selected } : {}),
        },
        controller.signal,
      );
      if (axisTickets[s] !== ticket || axisKey(s) !== key) return;
      axes[s] = { key, data };
    } catch (error) {
      if (error.name === "AbortError") return;
      if (axisTickets[s] !== ticket || axisKey(s) !== key) return;
      axes[s] = { key, error: error.message };
    }
    drawOverview(s);
    if (s === side || eyeMode()) draw();
    updateStatus();
  }, 180);
}
function showAngles() {
  const angles = choice().axis_angles_deg || [0, 0, 0];
  angles.forEach((v, i) => {
    $("axis-angle-" + i).value = v;
    $("axis-slider-" + i).value = v;
  });
}
function setAngle(i, value) {
  if (!current || saving || currentBusy() || !Number.isFinite(value)) return;
  choice().axis_angles_deg ||= [0, 0, 0];
  choice().axis_angles_deg[i] = ((((value + 180) % 360) + 360) % 360) - 180;
  markEdited();
  showAngles();
  draw();
}
for (const [i, label] of [..."XYZ"].entries()) {
  const row = document.createElement("div");
  row.className = "toolbar";
  const name = document.createElement("label");
  name.textContent = label + "（度） ";
  const number = document.createElement("input");
  number.type = "number";
  number.step = "0.1";
  number.id = "axis-angle-" + i;
  number.oninput = () => {
    if (number.value !== "") setAngle(i, Number(number.value));
  };
  name.append(number);
  row.append(name);
  const slider = document.createElement("input");
  slider.type = "range";
  slider.min = -180;
  slider.max = 180;
  slider.step = 0.1;
  slider.id = "axis-slider-" + i;
  slider.setAttribute("aria-label", label + " 旋转角度");
  slider.oninput = () => setAngle(i, Number(slider.value));
  row.append(slider);
  for (const delta of [-90, 90]) {
    const button = document.createElement("button");
    button.textContent = `${label} ${delta > 0 ? "+" : ""}${delta}°`;
    button.onclick = () => setAngle(i, (choice().axis_angles_deg?.[i] || 0) + delta);
    row.append(button);
  }
  $("axis-inputs").append(row);
}
$("axis-reset").onclick = () => {
  if (!current || saving || currentBusy()) return;
  choice().axis_angles_deg = [0, 0, 0];
  markEdited();
  showAngles();
  draw();
};
$("axis-from-corners").onclick = () => {
  if (!current || loading || saving || currentBusy()) return;
  const proposal = axes[side]?.key === axisKey(side) && axes[side].data?.corner_correction;
  if (!proposal) return;
  choice().axis_angles_deg = [...proposal.axis_angles_deg];
  markEdited();
  showAngles();
  draw();
  message("已按四角校正方向，请核对 XYZ 轴后暂存或通过审核。");
};
const canvas = $("frame-canvas"),
  ctx = canvas.getContext("2d");
function message(text, error = false) {
  $("message").textContent = text;
  $("message").className = error ? "error" : "";
}
async function api(url, body, signal) {
  const response = await fetch(
    url,
    body === undefined
      ? { signal }
      : {
          signal,
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
  if (eyeMode()) {
    for (const c of Object.values(current.selected)) {
      c.status = "pending";
      c.physical_side = "unknown";
    }
  }
  refreshAxes(side);
  updateStatus();
  drawOverview(side);
}
function sharedFitText(data) {
  const fit = data?.shared_fit;
  if (!fit) return "坐标轴已更新 · 轴长 4 cm";
  const errors = Object.entries(fit.residuals)
    .map(
      ([s, r]) =>
        `${names[s]} ${(r.position_m * 1000).toFixed(1)} mm / ${r.rotation_deg.toFixed(1)}°`,
    )
    .join("；");
  return `${fit.provisional ? "单侧临时外参，需补齐另一侧" : fit.consistent ? "共同外参" : "共同外参不一致，无法通过审核"} · ${errors}`;
}
function updateStatus() {
  const controls = document.querySelectorAll("section button, section input, section select");
  if (!current) {
    for (const control of controls) {
      control.disabled = true;
      control.dataset.waiting = "true";
    }
    $("episode-title").textContent = "等待选帧结果";
    $("hint").textContent = "可在另一个终端运行选帧，使用同一输出目录。每条完成后会自动加入列表。";
    return;
  }
  for (const control of controls) {
    if (control.dataset.waiting) {
      control.disabled = false;
      delete control.dataset.waiting;
    }
  }
  const eyeMode = current.coordinate_mode === "eye-reference";
  $("axis-title").textContent = eyeMode
    ? "相机外参安装方向 · 输出保留原轴向 · 红 X / 绿 Y / 蓝 Z"
    : "坐标轴校准 · 本地内禀 XYZ · 红 X / 绿 Y / 蓝 Z";
  $("axis-help").textContent = eyeMode
    ? "输出仅将原点移至参考眼动相机，保留原 Standing 轴向与 Vive 姿态。画面显示原 Vive 三轴经相机投影的方向；安装角度仅约束相机外参，不旋转输出位姿。修改一侧会更新双手投影。标签中心暂近似设备原点。"
    : "原点为标签中心；初始方向来自源腕部。旋转和标签偏移将随腕部应用到整段及手部关节。";
  $("axis-reset").textContent = eyeMode ? "清零安装方向修正" : "恢复源方向";
  $("axis-controls").disabled = loading || currentBusy() || saving;
  const cornerPose = axes[side]?.key === axisKey(side) ? axes[side].data : null;
  $("axis-from-corners").disabled = !cornerPose?.corner_correction;
  $("corner-axis-status").textContent = cornerPose?.corner_correction_error
    ? "四角方向不可用：" + cornerPose.corner_correction_error
    : cornerPose?.corner_correction?.plane_ambiguous
      ? "标签平面存在多解，请核对方向，必要时换用更清晰的参考帧。"
      : "";
  $("axis-status").textContent =
    axes[side]?.key === axisKey(side)
      ? axes[side].error || sharedFitText(axes[side].data)
      : "坐标轴待计算";
  $("hint").textContent = currentBusy()
    ? "本条数据正在后台处理，可切换到其他数据继续审核。"
    : current.video_error ||
      (dirty ? "有未保存修改，请检查后确认；暂存不代表通过。" : current.pending_reason) ||
      "审核依据已保存，可按需导出 RRD。";
  if (current.coordinate_mode === "eye-reference")
    $("hint").textContent += " 眼动固定参考系：双手须同帧；请确认标签身份、50×30 mm 尺寸及轴向。";
  $("side-status").textContent =
    names[side] + "：" + (statuses[choice().status] || choice().status);
  $("shared-status").textContent =
    `帧 ${choice().frame_index} · 正在调整${names[side]} · ` +
    ["left", "right"].map((s) => names[s] + " " + statuses[current.selected[s].status]).join(" / ");
  for (const s of ["left", "right"])
    $("shared-" + s).setAttribute("aria-pressed", String(s === side));
  $("accept").textContent = eyeMode && side === "left" ? "确认左手并调整右手" : "确认此侧并保存";
  $("saved-state").textContent =
    (dirty ? "有未保存修改 · " : "已保存 · ") +
    Object.entries(current.selected)
      .map(([s, c]) => names[s] + " " + statuses[c.status])
      .join(" / ");
  for (const id of ["accept", "reject", "save", "swap"])
    $(id).disabled = loading || currentBusy() || saving;
  const hasPreviews = ["left", "right"].every(
    (s) =>
      (eyeMode
        ? bitmap && bitmapKey === `${current.id}:${current.selected[s].frame_index}`
        : previewKeys[s] === `${current.id}:${current.selected[s].frame_index}` && previews[s]) &&
      current.selected[s].quad?.length === 4,
  );
  const hasAxes = ["left", "right"].every((s) => Boolean(projection(s)));
  const sharedFit = axes[side]?.data?.shared_fit;
  $("approve").disabled =
    loading ||
    currentBusy() ||
    saving ||
    !hasPreviews ||
    !hasAxes ||
    (eyeMode && (!sharedFit || sharedFit.provisional || !sharedFit.consistent));
  $("accept").disabled ||= !projection(side);
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
    "shared-left",
    "shared-right",
  ])
    $(id).disabled = currentBusy() || saving;
  for (const s of ["left", "right"])
    $("overview-" + s + "-status").textContent =
      statuses[current.selected[s].status] + (axes[s]?.error ? " · " + axes[s].error : "");
}
async function listEpisodes() {
  const ticket = ++listRequest;
  const rows = await api("/api/episodes");
  if (ticket !== listRequest) return episodeRows;
  episodeRows = rows;
  $("episodes").replaceChildren();
  rows.forEach((row) => {
    const button = document.createElement("button");
    button.dataset.episodeId = row.id;
    button.textContent = episodeLabel(row);
    if (current && current.id === row.id) button.className = "active";
    button.disabled = Boolean(row.unavailable);
    button.title = row.error || "";
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
    reference_candidates: data.reference_candidates || [],
    frame_count: data.frame_count,
    image_size: data.image_size,
  });
  dirty = submittedDrafts.has(id) && jobs.get(id)?.status === "failed";
  side = "left";
  $("side").value = side;
  document.body.classList.toggle("eye-reference", eyeMode());
  $("shared-hands").hidden = !eyeMode();
  const viewer = canvas.parentElement;
  if (eyeMode()) $("axis-controls").before(viewer);
  else $("axis-controls").after(viewer);
  if (eyeMode() && current.selected.left.frame_index !== current.selected.right.frame_index) {
    const anchor = current.selected.left.quad?.length ? "left" : "right";
    setReferenceFrame(current.selected[anchor].frame_index);
    message("原候选不在同一帧：已统一参考帧，另一侧需在当前画面重新标注；暂存后生效。");
  }
  $("episode-title").textContent =
    data.episode +
    (data.coordinate_mode === "eye-reference" ? " · 眼动固定参考系（双手须同帧）" : "");
  $("hint").textContent =
    data.video_error || data.pending_reason || "审核依据已保存，可按需导出 RRD。";
  $("reviewer").value =
    data.reviewer || localStorage.getItem("vive-reviewer") || data.default_reviewer || "";
  $("frame-count").textContent = `共 ${data.frame_count} 帧`;
  $("timeline").max = $("frame").max = Math.max(0, data.frame_count - 1);
  $("candidate").replaceChildren(new Option(eyeMode() ? "选择双手共同帧…" : "选择候选…", ""));
  (eyeMode() ? current.reference_candidates : data.candidates).forEach((c) =>
    $("candidate").add(
      new Option(
        eyeMode()
          ? `帧 ${c.frame_index} · ${c.label_count === 2 ? "双标签共同帧" : "单标签 · 另一手待确认"}`
          : `帧 ${c.frame_index} · ${c.image_side ? "图像侧提示 " + names[c.image_side] : "候选"}`,
        c.candidate_id,
      ),
    ),
  );
  $("editor").open = eyeMode();
  for (const s of ["left", "right"]) {
    refreshAxes(s);
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
  if (eyeMode()) return;
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
function paintHand(context, s, scale = 1) {
  const data = axes[s]?.key === axisKey(s) ? axes[s].data : null;
  const points = data?.hand_projection;
  if (!points) return;
  context.save();
  context.strokeStyle = context.fillStyle = s === "left" ? "#1ed2ff" : "#f05adc";
  context.lineWidth = 2 * scale;
  for (const start of [1, 5, 9, 13, 17]) {
    const chain = [0, start, start + 1, start + 2, start + 3];
    for (let i = 1; i < chain.length; i++) {
      const a = points[chain[i - 1]],
        b = points[chain[i]];
      if (!a || !b) continue;
      context.beginPath();
      context.moveTo(...a);
      context.lineTo(...b);
      context.stroke();
    }
  }
  for (const p of points) {
    if (!p) continue;
    context.beginPath();
    context.arc(...p, 3 * scale, 0, Math.PI * 2);
    context.fill();
  }
  context.restore();
}

function drawOverview(s) {
  if (!current || eyeMode()) return;
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
  const candidate = current.candidates?.find(
    (c) => c.candidate_id === selected.candidate_id && c.frame_index === selected.frame_index,
  );
  if (candidate?.review_reasons?.includes("body_extent_unresolved"))
    $("overview-" + s + "-frame").textContent += " · 外壳范围不确定，请核对设备身份与四角";
  if (!image || previewKeys[s] !== `${current.id}:${selected.frame_index}`) return;
  full.width = image.width;
  full.height = image.height;
  fc.drawImage(image, 0, 0);
  paint(fc, selected.quad, full.width / Math.max(1, full.getBoundingClientRect().width));
  paintAxes(fc, projection(s), full.width / Math.max(1, full.getBoundingClientRect().width));
  paintHand(fc, s, full.width / Math.max(1, full.getBoundingClientRect().width));
  drawCrop(cc, crop, image, selected.quad, true, projection(s));
}
function drawCrop(context, target, image, points, overlay = true, axesProjection = null) {
  if (!points.length) return;
  const bounds = [
    ...points,
    ...(axesProjection?.endpoints || [])
      .filter(Boolean)
      .map(([x, y]) => [
        Math.max(0, Math.min(image.width, x)),
        Math.max(0, Math.min(image.height, y)),
      ]),
  ];
  const xs = bounds.map((p) => p[0]),
    ys = bounds.map((p) => p[1]);
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
  if (overlay && axesProjection) {
    const convert = (p) => p && [(p[0] - x) * scale + ox, (p[1] - y) * scale + oy];
    paintAxes(context, {
      origin: convert(axesProjection.origin),
      endpoints: axesProjection.endpoints.map(convert),
    });
  }
}
async function editSide(s) {
  if (loading || saving || currentBusy()) return;
  dragging = -1;
  side = s;
  $("side").value = s;
  $("editor").open = true;
  await showChoice();
  if (!eyeMode()) $("editor").scrollIntoView({ block: "start" });
}
async function showChoice() {
  showAngles();
  refreshAxes(side);
  $("frame").value = $("timeline").value = choice().frame_index;
  $("notes").value = choice().notes || "";
  $("candidate").value = eyeMode()
    ? current.reference_candidates.find((c) => c.frame_index === choice().frame_index)
        ?.candidate_id || ""
    : choice().candidate_id || "";
  updateStatus();
  await Promise.all([loadFrame(), loadPreview(side)]);
}
async function loadFrame() {
  const key = `${current.id}:${choice().frame_index}`;
  if (eyeMode() && bitmap && bitmapKey === key) {
    draw();
    return;
  }
  const ticket = ++frameRequest,
    id = current.id,
    frame = choice().frame_index;
  loading = true;
  bitmap?.close();
  bitmap = null;
  bitmapKey = null;
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
    bitmapKey = key;
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
  setReferenceFrame(frame);
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
function paintAxes(context, axesProjection, scale = 1) {
  if (!axesProjection?.origin) return;
  const [x, y] = axesProjection.origin;
  axesProjection.endpoints.forEach((end, i) => {
    if (!end) return;
    context.strokeStyle = context.fillStyle = ["#ff4040", "#24d35b", "#408cff"][i];
    context.lineWidth = 2.5 * scale;
    context.beginPath();
    context.moveTo(x, y);
    context.lineTo(...end);
    context.stroke();
    const angle = Math.atan2(end[1] - y, end[0] - x),
      size = 8 * scale;
    context.beginPath();
    context.moveTo(...end);
    context.lineTo(end[0] - size * Math.cos(angle - 0.4), end[1] - size * Math.sin(angle - 0.4));
    context.lineTo(end[0] - size * Math.cos(angle + 0.4), end[1] - size * Math.sin(angle + 0.4));
    context.closePath();
    context.fill();
    context.font = `bold ${14 * scale}px sans-serif`;
    context.fillText("XYZ"[i], end[0] + 5 * scale, end[1] - 5 * scale);
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
  if (eyeMode() && $("overlay").checked) {
    const other = side === "left" ? "right" : "left";
    const scale = canvas.width / Math.max(1, canvas.getBoundingClientRect().width);
    ctx.save();
    ctx.globalAlpha = 0.45;
    paint(ctx, current.selected[other].quad || [], scale);
    paintAxes(ctx, projection(other), scale);
    paintHand(ctx, other, scale);
    ctx.restore();
    for (const s of [other, side]) {
      const p = current.selected[s].quad?.[0];
      if (!p) continue;
      ctx.font = `bold ${16 * scale}px sans-serif`;
      const label = names[s] + (s === side ? " · 调整中" : "");
      ctx.lineWidth = 4 * scale;
      ctx.strokeStyle = "#101a20";
      ctx.strokeText(label, p[0], p[1] - 28 * scale);
      ctx.fillStyle = s === "left" ? "#1ed2ff" : "#f05adc";
      ctx.fillText(label, p[0], p[1] - 28 * scale);
    }
  }
  if ($("overlay").checked)
    paint(ctx, points, canvas.width / Math.max(1, canvas.getBoundingClientRect().width));
  if ($("overlay").checked)
    paintAxes(
      ctx,
      projection(side),
      canvas.width / Math.max(1, canvas.getBoundingClientRect().width),
    );
  if ($("overlay").checked)
    paintHand(ctx, side, canvas.width / Math.max(1, canvas.getBoundingClientRect().width));
  $("coordinates").textContent = points.length
    ? points.map(([x, y], i) => `q${i} (${x.toFixed(1)}, ${y.toFixed(1)})`).join(" · ")
    : "尚未标注四角";
  drawCrop(cc, crop, bitmap, points, $("overlay").checked, projection(side));
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
    refreshAxes(side);
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
$("shared-left").onclick = () => guard(() => editSide("left"));
$("shared-right").onclick = () => guard(() => editSide("right"));
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
    if (eyeMode()) {
      const frame = current.reference_candidates.find(
        (c) => c.candidate_id === $("candidate").value,
      );
      if (!frame) return;
      setReferenceFrame(frame.frame_index);
      for (const s of ["left", "right"]) {
        const item = frame.hands[s];
        Object.assign(current.selected[s], {
          frame_index: frame.frame_index,
          quad: structuredClone(item?.quad || []),
          candidate_id: item?.candidate_id || null,
          status: "pending",
          physical_side: "unknown",
          source: "manual",
        });
        refreshAxes(s);
      }
      dirty = true;
      await showChoice();
      message("已切换共同参考帧；请确认两手分处画面两半，再依次核对或补标四角。");
      return;
    }
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
    const rotations = Object.fromEntries(
      ["left", "right"].map((s) => [s, current.selected[s].axis_angles_deg || [0, 0, 0]]),
    );
    [current.selected.left, current.selected.right] = [
      current.selected.right,
      current.selected.left,
    ];
    Object.values(current.selected).forEach((c) => {
      c.status = "pending";
      c.physical_side = "unknown";
    });
    for (const s of ["left", "right"]) {
      current.selected[s].axis_angles_deg = rotations[s];
      refreshAxes(s);
    }
    dirty = true;
    await Promise.all([loadPreview("left"), loadPreview("right")]);
    await showChoice();
  });
$("accept").onclick = () =>
  guard(async () => {
    if (loading || saving || currentBusy()) return;
    await save("accepted");
    if (eyeMode() && side === "left") await editSide("right");
  });
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
    const nextVersion = JSON.stringify(state.list_version);
    if (nextVersion !== listVersion) {
      await listEpisodes();
      listVersion = nextVersion;
    }
    const scan = state.scan || { entries: [], total: 0, status: "waiting" };
    const counts = {};
    for (const row of scan.entries) counts[row.status] = (counts[row.status] || 0) + 1;
    $("scan-progress").textContent = scan.total
      ? `选帧 ${counts.complete || 0}/${scan.total} 完成 · ${counts.skipped || 0} 跳过 · ${counts.scanning || 0} 扫描中 · ${counts.queued || 0} 等待 · ${(counts.failed || 0) + (counts.interrupted || 0)} 失败或中断`
      : "等待选帧结果；也可加载已完成的输出目录。";
    $("scan-errors").textContent = [
      ...scan.entries.filter((r) => r.error).map((r) => `${r.episode}：${r.error}`),
      ...(state.discovery_errors || []).map((r) => `${r.path}：${r.error}`),
    ].join("\n");
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
      if (job.running || ["failed", "interrupted", "approved"].includes(job.status))
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
  const first = rows.find((r) => !r.unavailable);
  if (first) await openEpisode(first.id);
  else updateStatus();
  await poll();
  if (!dirty)
    message(
      eyeMode()
        ? "在同一参考帧中依次调整左手、右手，再通过审核；切换手侧会保留当前修改，换帧需重新标注双手。"
        : "检查左右参考图，点击「通过审核」保存依据；需要时单独导出 RRD。",
    );
});
setInterval(() => guard(poll), 1000);
