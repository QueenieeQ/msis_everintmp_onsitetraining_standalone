import { api, qs } from "./api.js";
import { BoxCanvas } from "./canvas.js";
import { initTrainPanel } from "./train_panel.js";
import { initProductAdmin } from "./product_admin.js";

const $ = (id) => document.getElementById(id);
let dirty = [];
let canvas;
let labelOptions = [];
let shotOptions = [];

function currentRef() {
  const [stage, camera, printer_id] = $("productSelect").value.split("|");
  return { stage, camera, printer_id };
}

function fillSelect(el, items, toOption, selectValue) {
  el.innerHTML = "";
  items.forEach((item) => {
    const opt = document.createElement("option");
    const m = toOption(item);
    opt.value = m.value;
    opt.textContent = m.label;
    el.appendChild(opt);
  });
  if (selectValue !== undefined) el.value = selectValue;
}

function renderBoxList(boxes) {
  const ul = $("boxList");
  ul.innerHTML = "";
  boxes.forEach((box, i) => {
    const li = document.createElement("li");
    const [x1, y1, x2, y2] = box.bbox.map((v) => Math.round(v));
    li.textContent = `${box.id ?? "new"}: [${x1}, ${y1}, ${x2}, ${y2}]`;
    li.className = i === canvas.selected ? "selected" : "";
    li.onclick = () => {
      canvas.selectIndex(i);
      renderBoxList(boxes);
    };
    ul.appendChild(li);
  });
}

async function refreshProducts(selectValue) {
  const ctx = await api("/api/products");
  shotOptions = ctx.shots;
  labelOptions = ctx.labels;
  fillSelect($("productSelect"), ctx.products, (p) => ({
    value: `${p.stage}|${p.camera}|${p.printer_id}`,
    label: `${p.printer_id} (${p.stage}/${p.camera})`,
  }), selectValue);
  fillSelect($("shotSelect"), shotOptions, (s) => ({ value: s, label: s }));
  fillSelect($("labelSelect"), labelOptions, (l) => ({ value: l, label: l }));
  fillSelect(
    $("trainShotSelect"),
    ["", ...shotOptions],
    (s) => (s === "" ? { value: "", label: "All shots (left + right + closer)" } : { value: s, label: s })
  );
  $("trainShotSelect").value = $("shotSelect").value;
  // Fill once so a product refresh doesn't reset the user's training choices.
  if (!$("trainModelSelect").options.length) {
    fillSelect($("trainModelSelect"), ctx.models, (m) => m);
  }
  if (!$("yoloVariantSelect").options.length) {
    fillSelect($("yoloVariantSelect"), ["", ...ctx.yolo_variants], (v) =>
      v === "" ? { value: "", label: "config default" } : { value: v, label: v }
    );
  }
  $("hint").textContent = ctx.products.length
    ? "annotate left / right / closer shots, zoom, then train"
    : "No products yet — create one below, then upload images per shot.";
  return ctx.products.length > 0;
}

async function loadShot() {
  $("saveResult").textContent = "";
  if (!$("productSelect").value) return;
  const params = { ...currentRef(), shot: $("shotSelect").value, label: $("labelSelect").value };
  const data = await api("/api/shot?" + qs(params));
  dirty = data.boxes.map((b) => ({ id: b.id, bbox: [...b.bbox] }));
  if (data.has_image) {
    canvas.loadImage("/api/image-file?" + qs({ path: data.image_path }), data.width, data.height, data.boxes);
  } else {
    canvas.clear();
    $("saveResult").textContent = "No image uploaded for this shot yet.";
  }
  renderBoxList(dirty);
}

function bindControls() {
  $("btnLoad").onclick = () => loadShot().catch((e) => alert(e.message));
  ["productSelect", "shotSelect", "labelSelect"].forEach((id) =>
    $(id).addEventListener("change", () => loadShot().catch((e) => alert(e.message)))
  );
  $("shotSelect").addEventListener("change", () => {
    $("trainShotSelect").value = $("shotSelect").value;
  });
  $("btnZoomIn").onclick = () => {
    const r = $("annoCanvas").getBoundingClientRect();
    canvas.zoomAt(1.2, r.left + r.width / 2, r.top + r.height / 2);
  };
  $("btnZoomOut").onclick = () => {
    const r = $("annoCanvas").getBoundingClientRect();
    canvas.zoomAt(1 / 1.2, r.left + r.width / 2, r.top + r.height / 2);
  };
  $("btnZoomReset").onclick = () => canvas.resetView();
  $("btnDeleteBox").onclick = () => canvas.deleteSelected();

  $("btnSave").onclick = async () => {
    try {
      const body = {
        ...currentRef(),
        shot: $("shotSelect").value,
        label: $("labelSelect").value,
        boxes: dirty.map((b) => b.bbox),
      };
      const res = await api("/api/shot", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      $("saveResult").textContent = `saved ${res.count} box(es) -> ${res.yaml_path}`;
      await loadShot();
    } catch (e) {
      $("saveResult").textContent = "ERROR: " + e.message;
    }
  };
}

async function init() {
  canvas = new BoxCanvas($("annoCanvas"), {
    onChange: (boxes) => {
      dirty = boxes;
      renderBoxList(boxes);
    },
  });

  bindControls();
  initTrainPanel(currentRef);
  initProductAdmin(
    currentRef,
    async (stage, camera, printer_id) => {
      await refreshProducts(`${stage}|${camera}|${printer_id}`);
      await loadShot();
    },
    () => loadShot()
  );

  const hasProducts = await refreshProducts();
  if (hasProducts) await loadShot();
}

init().catch((e) => alert("init failed: " + e.message));
