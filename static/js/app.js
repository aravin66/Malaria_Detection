document.addEventListener("DOMContentLoaded", () => {
  const profileMenu = document.querySelector("[data-profile-menu]");
  if (profileMenu) {
    const profileTrigger = profileMenu.querySelector(".profile-trigger");
    const profilePanel = profileMenu.querySelector(".profile-panel");

    const closeProfileMenu = () => {
      profileMenu.classList.remove("is-open");
      profileTrigger.setAttribute("aria-expanded", "false");
      profilePanel.hidden = true;
    };

    const openProfileMenu = () => {
      profileMenu.classList.add("is-open");
      profileTrigger.setAttribute("aria-expanded", "true");
      profilePanel.hidden = false;
    };

    profileTrigger.addEventListener("click", () => {
      if (profileMenu.classList.contains("is-open")) {
        closeProfileMenu();
        return;
      }
      openProfileMenu();
    });

    document.addEventListener("click", (event) => {
      if (!profileMenu.contains(event.target)) {
        closeProfileMenu();
      }
    });

    document.addEventListener("keydown", (event) => {
      if (event.key === "Escape") {
        closeProfileMenu();
      }
    });
  }

  const fileInput = document.querySelector("[data-file-input]");
  const dropZone = document.querySelector("[data-drop-zone]");
  const fileName = document.querySelector("[data-file-name]");

  if (fileInput && dropZone && fileName) {
    const updateFileName = (file) => {
      fileName.textContent = file ? file.name : "No file selected";
    };

    fileInput.addEventListener("change", () => {
      updateFileName(fileInput.files[0]);
    });

    ["dragenter", "dragover"].forEach((eventName) => {
      dropZone.addEventListener(eventName, (event) => {
        event.preventDefault();
        dropZone.classList.add("dragover");
      });
    });

    ["dragleave", "drop"].forEach((eventName) => {
      dropZone.addEventListener(eventName, (event) => {
        event.preventDefault();
        dropZone.classList.remove("dragover");
      });
    });

    dropZone.addEventListener("drop", (event) => {
      const files = event.dataTransfer.files;
      if (!files.length) {
        return;
      }
      const dataTransfer = new DataTransfer();
      dataTransfer.items.add(files[0]);
      fileInput.files = dataTransfer.files;
      updateFileName(files[0]);
    });
  }

  const downloadButton = document.querySelector("[data-download-result]");
  if (downloadButton) {
    downloadButton.addEventListener("click", () => {
      const content = [
        "Malaria AI Result Summary",
        "-------------------------",
        `Prediction: ${downloadButton.dataset.prediction}`,
        `Confidence: ${downloadButton.dataset.confidence}%`,
        `Model: ${downloadButton.dataset.model}`,
        `Image File: ${downloadButton.dataset.file}`,
      ].join("\n");

      const blob = new Blob([content], { type: "text/plain;charset=utf-8" });
      const url = URL.createObjectURL(blob);
      const link = document.createElement("a");
      link.href = url;
      link.download = "malaria-result.txt";
      document.body.appendChild(link);
      link.click();
      link.remove();
      URL.revokeObjectURL(url);
    });
  }

  const chartDataElement = document.getElementById("charts-data");
  if (chartDataElement && window.Chart) {
    const chartsData = JSON.parse(chartDataElement.textContent);
    const axisColor = "#9cb6d4";
    const gridColor = "rgba(24, 231, 215, 0.12)";

    const commonOptions = {
      responsive: true,
      maintainAspectRatio: false,
      animation: false,
      normalized: true,
      devicePixelRatio: 1,
      resizeDelay: 200,
      events: [],
      plugins: {
        legend: {
          labels: {
            color: axisColor,
            boxWidth: 10,
            font: {
              size: 11,
            },
          },
        },
      },
      scales: {
        x: {
          ticks: { color: axisColor, font: { size: 10 } },
          grid: { color: gridColor },
        },
        y: {
          ticks: { color: axisColor, font: { size: 10 } },
          grid: { color: gridColor },
        },
      },
    };

    const accuracyCtx = document.getElementById("accuracyChart");
    if (accuracyCtx) {
      new Chart(accuracyCtx, {
        type: "bar",
        data: {
          labels: chartsData.modelAccuracy.labels,
          datasets: [
            {
              label: "Train Accuracy (%)",
              data: chartsData.modelAccuracy.train,
              backgroundColor: "rgba(30, 182, 255, 0.72)",
              borderColor: "#1eb6ff",
              borderWidth: 1,
            },
            {
              label: "Test Accuracy (%)",
              data: chartsData.modelAccuracy.test,
              backgroundColor: "rgba(22, 231, 215, 0.72)",
              borderColor: "#16e7d7",
              borderWidth: 1,
            },
          ],
        },
        options: commonOptions,
      });
    }

    const datasetCtx = document.getElementById("datasetChart");
    if (datasetCtx) {
      new Chart(datasetCtx, {
        type: "pie",
        data: {
          labels: chartsData.datasetDistribution.labels,
          datasets: [
            {
              data: chartsData.datasetDistribution.values,
              backgroundColor: ["#1eb6ff", "#ff6780"],
              borderColor: ["#90ddff", "#ffb3c0"],
              borderWidth: 1,
            },
          ],
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          animation: false,
          devicePixelRatio: 1,
          resizeDelay: 200,
          events: [],
          layout: {
            padding: 6,
          },
          plugins: {
            legend: {
              position: "bottom",
              labels: {
                color: axisColor,
                boxWidth: 10,
                font: {
                  size: 11,
                },
              },
            },
          },
        },
      });
    }

    const splitCtx = document.getElementById("splitChart");
    if (splitCtx) {
      new Chart(splitCtx, {
        type: "bar",
        data: {
          labels: chartsData.splitDistribution.labels,
          datasets: [
            {
              label: "Training Images",
              data: chartsData.splitDistribution.train,
              backgroundColor: "rgba(30, 182, 255, 0.72)",
              borderColor: "#1eb6ff",
              borderWidth: 1,
            },
            {
              label: "Testing Images",
              data: chartsData.splitDistribution.test,
              backgroundColor: "rgba(255, 103, 128, 0.72)",
              borderColor: "#ff6780",
              borderWidth: 1,
            },
          ],
        },
        options: commonOptions,
      });
    }
  }
});
