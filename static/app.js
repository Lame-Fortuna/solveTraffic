(function () {
  const form = document.getElementById("prediction-form");
  const loading = document.getElementById("loading");
  if (form && loading) {
    form.addEventListener("submit", function () {
      loading.classList.remove("hidden");
      const button = form.querySelector("button[type='submit']");
      if (button) {
        button.disabled = true;
        button.textContent = "Predicting...";
      }
    });
  }

  const latInput = document.getElementById("latitude");
  const lonInput = document.getElementById("longitude");
  const mapEl = document.getElementById("map");
  if (!mapEl || !latInput || !lonInput || typeof L === "undefined") {
    return;
  }

  const startLat = parseFloat(latInput.value) || 12.9716;
  const startLon = parseFloat(lonInput.value) || 77.5946;
  const map = L.map("map").setView([startLat, startLon], 12);
  L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
    maxZoom: 19,
    attribution: "&copy; OpenStreetMap"
  }).addTo(map);

  const marker = L.marker([startLat, startLon], { draggable: true }).addTo(map);
  function setInputs(lat, lon) {
    latInput.value = lat.toFixed(6);
    lonInput.value = lon.toFixed(6);
  }
  function moveMarker(lat, lon) {
    marker.setLatLng([lat, lon]);
    setInputs(lat, lon);
  }

  map.on("click", function (event) {
    moveMarker(event.latlng.lat, event.latlng.lng);
  });
  marker.on("dragend", function () {
    const pos = marker.getLatLng();
    setInputs(pos.lat, pos.lng);
  });
})();
