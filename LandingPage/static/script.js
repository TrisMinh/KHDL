const input = document.querySelector("#imageInput");
const image = document.querySelector("#previewImage");
const cropShell = document.querySelector("#cropShell");
const predictButton = document.querySelector("#predictButton");
const resetButton = document.querySelector("#resetButton");
const statusText = document.querySelector("#statusText");
const emptyState = document.querySelector("#emptyState");
const resultContent = document.querySelector("#resultContent");
const cropPreview = document.querySelector("#cropPreview");
const predictedLabel = document.querySelector("#predictedLabel");
const predictedConfidence = document.querySelector("#predictedConfidence");
const topBars = document.querySelector("#topBars");

let cropper = null;
let selectedFile = null;

function setStatus(message) {
  statusText.textContent = message;
}

function resetResult() {
  emptyState.hidden = false;
  resultContent.hidden = true;
  topBars.innerHTML = "";
}

function resetTool() {
  if (cropper) {
    cropper.destroy();
    cropper = null;
  }
  selectedFile = null;
  input.value = "";
  image.removeAttribute("src");
  cropShell.hidden = true;
  predictButton.disabled = true;
  resetButton.disabled = true;
  resetResult();
  setStatus("Select an image, then crop the sign region before prediction.");
}

function labelText(label) {
  return label.replaceAll("_", " ");
}

input.addEventListener("change", () => {
  const file = input.files && input.files[0];
  if (!file) return;

  selectedFile = file;
  resetResult();
  const url = URL.createObjectURL(file);
  image.src = url;
  cropShell.hidden = false;
  predictButton.disabled = false;
  resetButton.disabled = false;
  setStatus("Drag the crop box around the traffic sign, then press Predict.");

  image.onload = () => {
    if (cropper) cropper.destroy();
    cropper = new Cropper(image, {
      viewMode: 1,
      autoCropArea: 0.72,
      responsive: true,
      background: false,
      movable: true,
      zoomable: true,
      rotatable: false,
      scalable: false,
    });
    URL.revokeObjectURL(url);
  };
});

resetButton.addEventListener("click", resetTool);

predictButton.addEventListener("click", async () => {
  if (!selectedFile || !cropper) return;

  predictButton.disabled = true;
  predictButton.classList.add("is-loading");
  setStatus("Predicting with MobileNetV2-SE...");

  const crop = cropper.getData(true);
  const form = new FormData();
  form.append("image", selectedFile);
  form.append("crop", JSON.stringify({
    x: crop.x,
    y: crop.y,
    width: crop.width,
    height: crop.height,
  }));

  try {
    const response = await fetch("/api/predict", {
      method: "POST",
      body: form,
    });
    const data = await response.json();

    if (!response.ok) {
      throw new Error(data.error || "Prediction failed.");
    }

    renderResult(data);
    setStatus("Prediction complete. Adjust the crop and run again if needed.");
  } catch (error) {
    setStatus(error.message);
  } finally {
    predictButton.classList.remove("is-loading");
    predictButton.disabled = false;
  }
});

function renderResult(data) {
  emptyState.hidden = true;
  resultContent.hidden = false;
  cropPreview.src = data.preview;
  predictedLabel.textContent = labelText(data.prediction.label);
  predictedConfidence.textContent = `${data.prediction.confidence.toFixed(2)}% confidence`;

  topBars.innerHTML = "";
  data.top5.forEach((item, index) => {
    const row = document.createElement("div");
    row.className = "bar-row";

    const label = document.createElement("div");
    label.className = "bar-label";
    label.innerHTML = `<span>${index + 1}. ${labelText(item.label)}</span><span>${item.confidence.toFixed(2)}%</span>`;

    const track = document.createElement("div");
    track.className = "bar-track";
    const fill = document.createElement("div");
    fill.className = "bar-fill";
    if (index === 1) fill.style.background = "var(--blue)";
    if (index > 1) fill.style.background = "var(--yellow)";

    track.appendChild(fill);
    row.appendChild(label);
    row.appendChild(track);
    topBars.appendChild(row);

    requestAnimationFrame(() => {
      fill.style.width = `${Math.max(2, item.confidence)}%`;
    });
  });
}

if (window.lucide) {
  window.lucide.createIcons();
}
