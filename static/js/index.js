let map

async function init() {
  const color = d3.scaleOrdinal(d3.schemeCategory10)

  map = L.map('map', {
    boxZoom: false,
  }) 
    .setView([0, 0], 2)
  L.tileLayer('https://tile.openstreetmap.org/{z}/{x}/{y}.png', {
    maxZoom: 19,
    attribution: '&copy; <a href="http://www.openstreetmap.org/copyright">OpenStreetMap</a>',
  }).addTo(map);


  let lines = []

  const time = 0
  for (let ensId = 0; ensId < 50; ensId++) {
    const lineCount = await d3.json(`/api/line-count?ens-id=${ensId}&time=${time}`)

    for (let lineId = 1; lineId <= lineCount; lineId++) {
      const coords = await d3.json(`/api/coords?ens-id=${ensId}&line-id=${lineId}&time=${time}`)
      const latLons = coords.map(coord => [coord.latitude, coord.longitude])

      lines.push(L.polyline(latLons, {color: color(lineId), weight: 1}).addTo(map))
    }
  }

  alert("Loaded all lines")

  map.on("boxzoomend", function(e) {
    for (let i = 0; i < lines.length; i++) {
      const coords = lines[i].getLatLngs()
      for (let j = 0; j < coords.length; j++) {
        if (e.boxZoomBounds.contains([coords[j].lat, coords[j].lng])) {
          lines[i].setStyle({color: "red"})
          break
        }
      }
    }
  })
}


init()
