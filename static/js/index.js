let map

async function init() {
  const color = d3.scaleOrdinal(d3.schemeCategory10)

  map = L.map('map')
    .setView([0, 0], 2)
  L.tileLayer('https://tile.openstreetmap.org/{z}/{x}/{y}.png', {
    maxZoom: 19,
    attribution: '&copy; <a href="http://www.openstreetmap.org/copyright">OpenStreetMap</a>',
  }).addTo(map);

  

  for (let j = 0; j <= 0; j+=12) {
    for (let i = 1; i <= 20; i++) {
      const data = await d3.json(`/api/data?line_id=${i}&time=${j}`) 
      const latLons = data.map(coord => [coord.latitude, coord.longitude]);

      L.polyline(latLons, {color: color(i), weight: 1}).addTo(map)
    }
  }
}


init()
