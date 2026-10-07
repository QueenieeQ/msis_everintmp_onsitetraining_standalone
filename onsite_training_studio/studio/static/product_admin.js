import { api } from "./api.js";

function readAsDataUrl(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result);
    reader.onerror = () => reject(new Error("could not read file"));
    reader.readAsDataURL(file);
  });
}

/** Wires "New product" + "Upload image" panels.
 * `onProductCreated(stage, camera, printer_id)` refreshes the product list
 * and selects the new one; `onUploaded()` reloads the current shot. */
export function initProductAdmin(currentRef, onProductCreated, onUploaded) {
  const $ = (id) => document.getElementById(id);

  $("btnCreateProduct").onclick = async () => {
    const out = $("createResult");
    out.textContent = "creating…";
    try {
      const stage = $("newStage").value;
      const camera = $("newCamera").value;
      const printer_id = $("newPrinterId").value.trim();
      if (!printer_id) throw new Error("product name is required");
      await api("/api/products", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ stage, camera, printer_id }),
      });
      out.textContent = `created ${printer_id}`;
      $("newPrinterId").value = "";
      await onProductCreated(stage, camera, printer_id);
    } catch (e) {
      out.textContent = "ERROR: " + e.message;
    }
  };

  $("btnUpload").onclick = async () => {
    const out = $("uploadResult");
    const fileInput = $("uploadFile");
    const file = fileInput.files[0];
    if (!file) {
      out.textContent = "choose a file first";
      return;
    }
    out.textContent = "reading file…";
    try {
      const ref = currentRef();
      const image_base64 = await readAsDataUrl(file);
      out.textContent = "uploading…";
      const body = {
        ...ref,
        shot: document.getElementById("shotSelect").value,
        image_base64,
      };
      const res = await api("/api/upload-image", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      out.textContent = `uploaded ${res.width}x${res.height}`;
      fileInput.value = "";
      await onUploaded();
    } catch (e) {
      out.textContent = "ERROR: " + e.message;
    }
  };
}
