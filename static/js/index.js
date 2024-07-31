/// <reference types="leaflet" />

const dateSlider = document.getElementById("date-slider")
const dateSliderLabel = document.getElementById("date-slider-label")

const playButton = document.getElementById("play-button")

let updating = false
let updatingValue = 0
async function onSliderChange() {
  if (updating) {
    dateSlider.value = updatingValue
    return
  }
  updating = true
  updatingValue = dateSlider.value

  if (dateSlider.value > 72) {
    dateSlider.step = 6
  } else {
    dateSlider.step = 3
  }

  dateSliderLabel.innerText = dateSlider.value

  await showLines(dateSlider.value)

  updating = false
}


let isPlaying = false
let currentTime = 0
let playInterval
function playButtonPressed() {
  isPlaying = !isPlaying

  if (isPlaying) {
    playButton.innerText = "Stop"
    dateSlider.disabled = true

    playInterval = setInterval(function() {
      let currentTime = parseInt(dateSlider.value)

      if (currentTime >= 72) {
        currentTime += 6
      } else {
        currentTime += 3
      }
      if (currentTime > 240) { currentTime = 0 }

      dateSlider.value = currentTime
      dateSliderLabel.innerText = currentTime
      showLines(currentTime)
    }, 100)
  } else {
    playButton.innerText = "Play"
    dateSlider.disabled = false

    clearInterval(playInterval)
  }
}


let map
let lineLayer
let aggregateLayer

let cachedLines = {}

async function init() {

  // Handler for box select. BoxZoom but without zoom
  L.Map.BoxSelectHandler = L.Map.BoxZoom.extend({
    initialize: function(map) {
      this._map = map;
      this._container = map._container;
      this._pane = map._panes.overlayPane;
    },

    addHooks: function() {
      L.DomEvent.on(this._container, 'mousedown', this._onMouseDown, this);
    },

    removeHooks: function() {
      L.DomEvent.off(this._container, 'mousedown', this._onMouseDown);
    },

    _onMouseDown: function(e) {
      if (!e.shiftKey || ((e.which !== 1) && (e.button !== 1))) { return false; }

      L.DomUtil.disableTextSelection();

      this._startLayerPoint = this._map.mouseEventToLayerPoint(e);

      this._box = L.DomUtil.create('div', 'leaflet-zoom-box', this._pane);
      L.DomUtil.setPosition(this._box, this._startLayerPoint);

      this._container.style.cursor = 'crosshair';

      L.DomEvent
        .on(document, 'mousemove', this._onMouseMove, this)
        .on(document, 'mouseup', this._onMouseUp, this)
        .on(document, 'keydown', this._onKeyDown, this)
        .preventDefault(e);

      this._map.fire('boxselectstart');
    },

    _onMouseMove: function(e) {
      var startPoint = this._startLayerPoint,
        box = this._box,

        layerPoint = this._map.mouseEventToLayerPoint(e),
        offset = layerPoint.subtract(startPoint),

        newPos = new L.Point(
          Math.min(layerPoint.x, startPoint.x),
          Math.min(layerPoint.y, startPoint.y));

      L.DomUtil.setPosition(box, newPos);

      // TODO refactor: remove hardcoded 4 pixels
      box.style.width = (Math.max(0, Math.abs(offset.x) - 4)) + 'px';
      box.style.height = (Math.max(0, Math.abs(offset.y) - 4)) + 'px';
    },

    _finish: function() {
      this._pane.removeChild(this._box);
      this._container.style.cursor = '';

      L.DomUtil.enableTextSelection();

      L.DomEvent
        .off(document, 'mousemove', this._onMouseMove)
        .off(document, 'mouseup', this._onMouseUp)
        .off(document, 'keydown', this._onKeyDown);
    },

    _onMouseUp: function(e) {
      this._finish();

      var map = this._map,
        layerPoint = map.mouseEventToLayerPoint(e);

      if (this._startLayerPoint.equals(layerPoint)) { return; }

      var bounds = new L.LatLngBounds(
        map.layerPointToLatLng(this._startLayerPoint),
        map.layerPointToLatLng(layerPoint));

      map.fire('boxselectend', {
        boxZoomBounds: bounds
      });
    },

    _onKeyDown: function(e) {
      if (e.keyCode === 27) {
        this._finish();
      }
    }
  })

  L.Map.addInitHook("addHandler", "boxSelect", L.Map.BoxSelectHandler)

  map = L.map('map', {
    boxZoom: false,
    boxSelect: true,
  })
    .setView([0, 0], 2)

  L.tileLayer('https://tile.openstreetmap.org/{z}/{x}/{y}.png', {
    maxZoom: 19,
    attribution: '&copy; <a href="http://www.openstreetmap.org/copyright">OpenStreetMap</a>',
  }).addTo(map);

  lineLayer = L.layerGroup().addTo(map)
  aggregateLayer = L.layerGroup().addTo(map)

  // Fetches all data asynchronously and caches it
  async function fetchAllData() {
    let date = 0
    while (date <= 240) {
      if (!cachedLines[date]) {
        let ls = await d3.json(`/api/all-lines?date=${date}`)
        cachedLines[date] = ls
      }

      if (date < 72) {
        date += 3
      } else {
        date += 6
      }
    }
  }
  fetchAllData()

  await showLines(0)
}


let boxSelectHandler
async function showLines(date) {
  const color = d3.scaleOrdinal(d3.schemeCategory10)

  let lines = []
  let selection = []

  let ls = cachedLines[date]
  if (!ls) {
    ls = await d3.json(`/api/all-lines?date=${date}`)
    cachedLines[date] = ls
  }

  lineLayer.clearLayers()

  ls.forEach(function(l) {
    const min = Math.min(...l.coords.map(coord => coord.longitude))
    const max = Math.max(...l.coords.map(coord => coord.longitude))

    let latLons
    // If it crosses the anti meridian add 360 to the negative values
    if (max - min > 180) {
      latLons = l.coords.map(coord => [coord.latitude, coord.longitude < 0 ? coord.longitude + 360 : coord.longitude])
    } else {
      latLons = l.coords.map(coord => [coord.latitude, coord.longitude])
    }

    let line = L.polyline(latLons, { weight: 1 }).addTo(lineLayer)
    line.setStyle({ color: color(line._leaflet_id) })
    lines.push(line)
  })

  if (boxSelectHandler) {
    map.off("boxselectend", boxSelectHandler)
  }

  boxSelectHandler = function(e) {
    aggregateLayer.clearLayers()

    selection = []

    for (let i = 0; i < lines.length; i++) {
      const coords = lines[i].getLatLngs()
      for (let j = 0; j < coords.length; j++) {
        if (e.boxZoomBounds.contains([coords[j].lat, coords[j].lng])) {
          selection.push(lines[i])
          break
        }
      }
    }

    if (selection.length > 0) {
      lines.forEach(function(l) {
        l.setStyle({ color: "#9999" })
      })

      const min = Math.min(...selection.map(line => Math.min(...line._latlngs.map(coord => coord.lng))))
      const max = Math.max(...selection.map(line => Math.max(...line._latlngs.map(coord => coord.lng))))

      const desiredPoints = 10
      const spacing = (max - min) / (desiredPoints - 1)
      let X = []
      for (let i = min; i < max; i += spacing) {
        X.push(i)
      }

      selection.forEach(function(l) {
        // l.setStyle({ color: "red" })
        interpolateLine(l, X)
      })

      getCentroidLine(structuredClone(selection.map(l => l._latlngs)))
    } else {
      lines.forEach(function(l) {
        l.setStyle({ color: color(l._leaflet_id) })
      })
    }

  }

  map.on("boxselectend", boxSelectHandler)
}


function getCentroidLine(selection) {
  const minLength = Math.min(...selection.map(l => l.length))
  const sampledSelection = selection.map(l => sampleLine(l, minLength))

  const lineCount = sampledSelection.length

  let average = sampledSelection[0].map(coord => ([coord.lat, coord.lng]));

  sampledSelection.slice(1).forEach(function(l) {
    for (let i = 0; i < minLength; i++) {
      average[i][0] += l[i].lat
      average[i][1] += l[i].lng
    }
  })

  average = average.map(coords => [coords[0] / lineCount, coords[1] / lineCount])
  L.polyline(average, { color: "blue", weight: 3 }).addTo(aggregateLayer)
}

function sampleLine(line, samples) {
  const step = (line.length - 1) / (samples - 1);

  let newLine = [];
  for (let i = 0; i < samples; i++) {
    const index = Math.min(Math.round(i * step), line.length - 1);
    newLine.push(line[index]);
  }

  return newLine;
}

// Does not work for multivalued functions
function interpolateLine(line, X) {
  let coords = line._latlngs.sort(function(c1, c2) {
    return c1.lng > c2.lng
  })

  let interpolatedLine = []
  // TODO: Extrapolation at i=0 and i=length
  X.forEach(function(x) {
    for (let i = 0; i < coords.length-1; i++) {
      if (x == coords[i].lng) { 
        interpolatedLine.push(coords[i])
      } else if (x > coords[i].lng && x < coords[i+1].lng) {
        const c1 = coords[i]
        const c2 = coords[i+1]

        const lat = (c1.lat * (c2.lng - x) + c2.lat * (x - c1.lng)) / (c2.lng - c1.lng)
        interpolatedLine.push({lat: lat, lng: x})
      }
    }
  })

  L.polyline(interpolatedLine, { color: "green", weight: 1 }).addTo(aggregateLayer)
}


init()
