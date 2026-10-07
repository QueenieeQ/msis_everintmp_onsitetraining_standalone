/** Pure geometry helpers for BoxCanvas: hit-testing and drag math. */

export function hitTest(boxes, pt, tol) {
  for (let i = boxes.length - 1; i >= 0; i--) {
    const [x1, y1, x2, y2] = boxes[i].bbox;
    const corner =
      near(pt.x, x2, tol) && near(pt.y, y2, tol) ? "br" :
      near(pt.x, x1, tol) && near(pt.y, y1, tol) ? "tl" :
      near(pt.x, x2, tol) && near(pt.y, y1, tol) ? "tr" :
      near(pt.x, x1, tol) && near(pt.y, y2, tol) ? "bl" : null;
    if (corner) return { index: i, mode: "resize", corner };
    if (pt.x >= x1 && pt.x <= x2 && pt.y >= y1 && pt.y <= y2) {
      return { index: i, mode: "move", offset: { x: pt.x - x1, y: pt.y - y1 } };
    }
  }
  return null;
}

function near(a, b, tol) {
  return Math.abs(a - b) < tol;
}

export function applyMove(box, pt, offset) {
  const w = box[2] - box[0];
  const h = box[3] - box[1];
  box[0] = pt.x - offset.x;
  box[1] = pt.y - offset.y;
  box[2] = box[0] + w;
  box[3] = box[1] + h;
}

export function applyResize(box, pt, corner) {
  if (corner.includes("l")) box[0] = pt.x; else box[2] = pt.x;
  if (corner.includes("t")) box[1] = pt.y; else box[3] = pt.y;
}

export function normalizeBox(p1, p2) {
  return [Math.min(p1.x, p2.x), Math.min(p1.y, p2.y), Math.max(p1.x, p2.x), Math.max(p1.y, p2.y)];
}

export function fixOrder(bbox) {
  if (bbox[0] > bbox[2]) [bbox[0], bbox[2]] = [bbox[2], bbox[0]];
  if (bbox[1] > bbox[3]) [bbox[1], bbox[3]] = [bbox[3], bbox[1]];
}
