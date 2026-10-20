/* ==========================================================================
   PlantGuard AI – Interactive UI & Upload Behavior
   ========================================================================== */

document.addEventListener("DOMContentLoaded", () => {
  initNavToggle();
  initAlertDismiss();
  initDropZone();
});

/**
 * Responsive Mobile Navigation Toggle
 */
function initNavToggle() {
  const toggle = document.getElementById("navToggle");
  const menu = document.getElementById("navMenu");
  if (!toggle || !menu) return;

  toggle.addEventListener("click", () => {
    const isOpen = menu.classList.toggle("open");
    toggle.setAttribute("aria-expanded", isOpen ? "true" : "false");
  });
}

/**
 * Dismissible Flash Alerts
 */
function initAlertDismiss() {
  const closeBtns = document.querySelectorAll(".alert-close");
  closeBtns.forEach((btn) => {
    btn.addEventListener("click", () => {
      const alert = btn.closest(".alert");
      if (alert) {
        alert.style.opacity = "0";
        alert.style.transform = "translateY(-8px)";
        alert.style.transition = "opacity 0.2s ease, transform 0.2s ease";
        setTimeout(() => alert.remove(), 200);
      }
    });
  });
}

/**
 * Drag-and-Drop Image Uploader with Preview & Validation
 */
function initDropZone() {
  const dropZone = document.getElementById("dropZone");
  const fileInput = document.querySelector('input[type="file"]');
  const dropZoneInner = document.getElementById("dropZoneInner");
  const dropPreview = document.getElementById("dropPreview");
  const previewImg = document.getElementById("previewImg");
  const previewName = document.getElementById("previewName");
  const changeImageBtn = document.getElementById("changeImageBtn");
  const uploadForm = document.getElementById("uploadForm");
  const submitBtn = document.getElementById("submitBtn");
  const submitText = document.getElementById("submitText");
  const submitSpinner = document.getElementById("submitSpinner");
  const submitLoading = document.getElementById("submitLoading");

  if (!dropZone || !fileInput) return;

  // Highlight drop zone on drag over
  ["dragenter", "dragover"].forEach((eventName) => {
    dropZone.addEventListener(eventName, (e) => {
      e.preventDefault();
      e.stopPropagation();
      dropZone.classList.add("dragover");
    });
  });

  ["dragleave", "drop"].forEach((eventName) => {
    dropZone.addEventListener(eventName, (e) => {
      e.preventDefault();
      e.stopPropagation();
      dropZone.classList.remove("dragover");
    });
  });

  // Handle dropped files
  dropZone.addEventListener("drop", (e) => {
    const dt = e.dataTransfer;
    const files = dt.files;
    if (files.length > 0) {
      fileInput.files = files;
      handleFile(files[0]);
    }
  });

  // Handle selected files via file picker
  fileInput.addEventListener("change", (e) => {
    if (fileInput.files.length > 0) {
      handleFile(fileInput.files[0]);
    }
  });

  // Change image button triggers picker
  if (changeImageBtn) {
    changeImageBtn.addEventListener("click", (e) => {
      e.stopPropagation();
      fileInput.click();
    });
  }

  function handleFile(file) {
    if (!file) return;

    // Validate type
    const validTypes = ["image/jpeg", "image/jpg", "image/png"];
    if (!validTypes.includes(file.type.toLowerCase())) {
      alert("Please upload a valid JPG, JPEG, or PNG image.");
      fileInput.value = "";
      return;
    }

    // Validate size (10 MB limit)
    const maxSize = 10 * 1024 * 1024;
    if (file.size > maxSize) {
      alert("Image file size exceeds the 10 MB limit. Please select a smaller photo.");
      fileInput.value = "";
      return;
    }

    // Display image preview
    const reader = new FileReader();
    reader.onload = (e) => {
      if (previewImg) previewImg.src = e.target.result;
      if (previewName) previewName.textContent = `${file.name} (${(file.size / 1024).toFixed(1)} KB)`;
      if (dropZoneInner) dropZoneInner.style.display = "none";
      if (dropPreview) dropPreview.style.display = "flex";
    };
    reader.readAsDataURL(file);
  }

  // Handle form submission and loading spinner
  if (uploadForm && submitBtn) {
    uploadForm.addEventListener("submit", (e) => {
      if (!fileInput.files || fileInput.files.length === 0) {
        e.preventDefault();
        alert("Please select or drop a plant leaf image first.");
        return;
      }

      // Visual loading state
      submitBtn.disabled = true;
      submitBtn.style.opacity = "0.85";
      submitBtn.style.cursor = "not-allowed";
      if (submitText) submitText.style.display = "none";
      if (submitSpinner) submitSpinner.style.display = "inline-block";
      if (submitLoading) submitLoading.style.display = "inline";
    });
  }
}
