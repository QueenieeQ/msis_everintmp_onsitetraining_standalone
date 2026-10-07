import { api } from "./api.js";

const POLL_MS = 1500;

/** Wires the "Onsite training" button + polls job status until done. */
export function initTrainPanel(getProductRef) {
  const btn = document.getElementById("btnTrain");
  const out = document.getElementById("trainResult");
  const runTraining = document.getElementById("runTraining");
  const useOutput = document.getElementById("useOutputFolder");
  const outputFolder = document.getElementById("outputFolder");
  const trainShot = document.getElementById("trainShotSelect");
  const trainModel = document.getElementById("trainModelSelect");
  const yoloVariant = document.getElementById("yoloVariantSelect");

  useOutput.addEventListener("change", () => {
    outputFolder.disabled = !useOutput.checked;
  });

  btn.onclick = async () => {
    btn.disabled = true;
    out.textContent = "starting…";
    try {
      const body = {
        ...getProductRef(),
        run_training: runTraining.checked,
        output_folder: useOutput.checked ? outputFolder.value.trim() || null : null,
        shot: trainShot.value || null,
        model: trainModel.value || null,
        model_variant: yoloVariant.value || null,
      };
      const started = await api("/api/train", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      const scope = `${body.shot ? `shot=${body.shot}` : "all shots"}, ${body.model ? `model=${body.model}` : "all models"}${body.model_variant ? `, yolo=${body.model_variant}` : ""}`;
      out.textContent = `job ${started.job_id} (${scope})\ninformation_path=${started.information_path}\nrunning…`;
      poll(started.job_id, btn, out);
    } catch (e) {
      out.textContent = "ERROR: " + e.message;
      btn.disabled = false;
    }
  };
}

async function poll(jobId, btn, out) {
  try {
    const job = await api("/api/train/" + jobId);
    if (job.status !== "done") {
      setTimeout(() => poll(jobId, btn, out), POLL_MS);
      return;
    }
    out.textContent = JSON.stringify(job, null, 2);
    btn.disabled = false;
  } catch (e) {
    out.textContent = "ERROR: " + e.message;
    btn.disabled = false;
  }
}
