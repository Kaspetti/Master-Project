let map

async function init() {
  // const width = 928;
  // const height = 500;
  // const marginTop = 20;
  // const marginRight = 30;
  // const marginBottom = 30;
  // const marginLeft = 40;
  //
  // const x = d3.scaleLinear()
  //   .domain([-200, 200])
  //   .range([marginLeft, width - marginRight])
  //
  // const y = d3.scaleLinear()
  //   .domain([-100, 100])
  //   .range([height - marginBottom, marginTop])
  //
  // const line = d3.line()
  //   .x(d => x(d.longitude))
  //   .y(d => y(d.latitude))
  //
  // const svg = d3.select("#line-chart")
  //   .append("svg")
  //   .attr("height", height)
  //   .attr("viewBox", [0, 0, width, height])
  //   .attr("style", "max-width: 100%; height: auto; height: intrinsic;")
  //
  // svg.append("g")
  //     .attr("transform", `translate(0,${height - marginBottom})`)
  //     .call(d3.axisBottom(x).ticks(width / 80).tickSizeOuter(0))
  //
  // svg.append("g")
  //     .attr("transform", `translate(${marginLeft},0)`)
  //     .call(d3.axisLeft(y).ticks(height / 40))
  //
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

      // svg.append("path")
      //   .attr("fill", "none")
      //   .attr("stroke", color(i))
      //   .attr("stroke-width", 1.5)
      //   .attr("d", line(data))
    }
  }
}


init()
