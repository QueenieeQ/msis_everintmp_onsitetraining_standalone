import { hitTest, applyMove, applyResize, normalizeBox, fixOrder } from "./box_edit.js";

const HANDLE_PX = 8;

/** Zoomable/pannable canvas that lets the user draw, move, resize and
 * select multiple bounding boxes over one image (image-space coords). */
export class BoxCanvas {
  constructor(canvas, { onChange } = {}) {
    this.canvas = canvas;
    this.ctx = canvas.getContext("2d");
    this.onChange = onChange || (() => {});
    this.img = null;
    this.natural = { w: 1, h: 1 };
    this.zoom = 1;
    this.panX = 0;
    this.panY = 0;
    this.boxes = []; // [{id, bbox:[x1,y1,x2,y2]}]
    this.selected = -1;
    this.drag = null; // {mode:'new'|'move'|'resize'|'pan', ...}
    this._bindEvents();
  }

  loadImage(src, width, height, boxes) {
    this.natural = { w: width, h: height };
    this.boxes = boxes.map((b) => ({ id: b.id, bbox: [...b.bbox] }));
    this.selected = -1;
    const img = new Image();
    img.onload = () => {
      this.img = img;
      const maxW = Math.max(300, this.canvas.parentElement.clientWidth - 320);
      const ratio = Math.min(1, maxW / width);
      this.canvas.width = Math.round(width * ratio);
      this.canvas.height = Math.round(height * ratio);
      this.resetView();
    };
    img.src = src;
  }

  fitScale() {
    return this.canvas.width / this.natural.w;
  }

  viewScale() {
    return this.fitScale() * this.zoom;
  }

  resetView() {
    this.zoom = 1;
    this.panX = 0;
    this.panY = 0;
    this._redraw();
  }

  zoomAt(factor, clientX, clientY) {
    if (!this.img) return;
    const rect = this.canvas.getBoundingClientRect();
    const cx = ((clientX - rect.left) * this.canvas.width) / rect.width;
    const cy = ((clientY - rect.top) * this.canvas.height) / rect.height;
    const before = this._toImageFromCanvas(cx, cy);
    this.zoom = Math.min(10, Math.max(0.2, this.zoom * factor));
    const s = this.viewScale();
    this.panX = cx - before.x * s;
    this.panY = cy - before.y * s;
    this._redraw();
  }

  clear() {
    this.img = null;
    this.boxes = [];
    this.selected = -1;
    this._redraw();
  }

  deleteSelected() {
    if (this.selected < 0) return;
    this.boxes.splice(this.selected, 1);
    this.selected = -1;
    this._redraw();
    this.onChange(this.boxes);
  }

  selectIndex(i) {
    this.selected = i;
    this._redraw();
  }

  _toImageFromCanvas(cx, cy) {
    const s = this.viewScale();
    return { x: (cx - this.panX) / s, y: (cy - this.panY) / s };
  }

  _toImage(ev) {
    const rect = this.canvas.getBoundingClientRect();
    const cx = ((ev.clientX - rect.left) * this.canvas.width) / rect.width;
    const cy = ((ev.clientY - rect.top) * this.canvas.height) / rect.height;
    return this._toImageFromCanvas(cx, cy);
  }

  _bindEvents() {
    const c = this.canvas;
    c.addEventListener("wheel", (ev) => {
      ev.preventDefault();
      this.zoomAt(ev.deltaY < 0 ? 1.15 : 1 / 1.15, ev.clientX, ev.clientY);
    }, { passive: false });

    c.addEventListener("mousedown", (ev) => {
      if (!this.img) return;
      if (ev.button === 1 || (ev.button === 0 && ev.shiftKey)) {
        this.drag = { mode: "pan", x: ev.clientX, y: ev.clientY };
        return;
      }
      if (ev.button !== 0) return;
      const pt = this._toImage(ev);
      const hit = hitTest(this.boxes, pt, HANDLE_PX / this.viewScale());
      if (hit) {
        this.selected = hit.index;
        this.drag = { ...hit, pt };
      } else {
        this.selected = -1;
        this.drag = { mode: "new", start: pt };
      }
      this._redraw();
    });

    window.addEventListener("mousemove", (ev) => {
      if (!this.drag) return;
      if (this.drag.mode === "pan") {
        this.panX += ev.clientX - this.drag.x;
        this.panY += ev.clientY - this.drag.y;
        this.drag.x = ev.clientX;
        this.drag.y = ev.clientY;
        this._redraw();
        return;
      }
      const pt = this._toImage(ev);
      if (this.drag.mode === "new") {
        this._previewBox = normalizeBox(this.drag.start, pt);
      } else if (this.drag.mode === "move") {
        applyMove(this.boxes[this.drag.index].bbox, pt, this.drag.offset);
      } else if (this.drag.mode === "resize") {
        applyResize(this.boxes[this.drag.index].bbox, pt, this.drag.corner);
      }
      this._redraw();
    });

    window.addEventListener("mouseup", () => {
      if (this.drag && this.drag.mode === "new" && this._previewBox) {
        const b = this._previewBox;
        if (Math.abs(b[2] - b[0]) > 3 && Math.abs(b[3] - b[1]) > 3) {
          this.boxes.push({ id: null, bbox: b });
          this.selected = this.boxes.length - 1;
        }
      }
      this._previewBox = null;
      if (this.drag && this.drag.mode !== "pan") {
        for (const box of this.boxes) fixOrder(box.bbox);
        this.onChange(this.boxes);
      }
      this.drag = null;
      this._redraw();
    });

    window.addEventListener("keydown", (ev) => {
      if ((ev.key === "Delete" || ev.key === "Backspace") && document.activeElement === c) {
        this.deleteSelected();
      }
    });
  }

  _redraw() {
    const { ctx, canvas } = this;
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    if (!this.img) return;
    const s = this.viewScale();
    ctx.setTransform(s, 0, 0, s, this.panX, this.panY);
    ctx.drawImage(this.img, 0, 0);
    this.boxes.forEach((box, i) => this._drawBox(box.bbox, i === this.selected, s, box.id));
    if (this._previewBox) this._drawBox(this._previewBox, true, s, null, true);
    document.getElementById("zoomLabel").textContent = Math.round(this.zoom * 100) + "%";
  }

  _drawBox([x1, y1, x2, y2], selected, s, label, dashed) {
    const ctx = this.ctx;
    ctx.strokeStyle = selected ? "#ffd166" : "#3fb950";
    ctx.lineWidth = (selected ? 3 : 2) / s;
    ctx.setLineDash(dashed ? [6 / s, 4 / s] : []);
    ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
    ctx.setLineDash([]);
    if (label) {
      ctx.fillStyle = ctx.strokeStyle;
      ctx.font = `${12 / s}px sans-serif`;
      ctx.fillText(label, x1 + 3 / s, y1 - 3 / s);
    }
  }
}
