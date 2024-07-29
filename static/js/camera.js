document.addEventListener("DOMContentLoaded", (event) => {
  const video = document.getElementById("video");
  const canvas = document.getElementById("canvas");
  const captureButton = document.getElementById("capture");
  const uploadForm = document.getElementById("upload-form");
  const resultDiv = document.getElementById("result");

  // Get camera access
  if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
    navigator.mediaDevices
      .getUserMedia({ video: true })
      .then(function (stream) {
        video.srcObject = stream;
        video.play();
      });
  }

  // Capture photo
  captureButton.addEventListener("click", function () {
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    canvas.getContext("2d").drawImage(video, 0, 0, canvas.width, canvas.height);

    canvas.toBlob(function (blob) {
      const formData = new FormData();
      formData.append("image", blob, "capture.jpg");

      fetch("/capture", {
        method: "POST",
        body: formData,
      })
        .then((response) => response.json())
        .then((data) => {
          displayResult(data);
        });
    }, "image/jpeg");
  });

  // Upload form submission
  uploadForm.addEventListener("submit", function (e) {
    e.preventDefault();
    const formData = new FormData(this);

    fetch("/", {
      method: "POST",
      body: formData,
    })
      .then((response) => response.json())
      .then((data) => {
        displayResult(data);
      });
  });

  function displayResult(data) {
    resultDiv.innerHTML = `
            <div class="alert alert-info">
                <h4 class="alert-heading">Analysis Result</h4>
                <p>Predicted condition: ${data.condition}</p>
                <p>Confidence: ${(data.confidence * 100).toFixed(2)}%</p>
            </div>
        `;
  }
});
