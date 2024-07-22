/// <reference types="leaflet" />

const dateSlider = document.getElementById("date-slider")
const dateSliderLabel = document.getElementById("date-slider-label")

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


let map
let lineLayer
let aggregateLayer

let cachedLines = {}

async function init() {

  // Handler for box select. BoxZoom but without zoom
  L.Map.BoxSelectHandler = L.Map.BoxZoom.extend({
    initialize: function (map) {
      this._map = map;
      this._container = map._container;
      this._pane = map._panes.overlayPane;
    },

    addHooks: function () {
      L.DomEvent.on(this._container, 'mousedown', this._onMouseDown, this);
    },

    removeHooks: function () {
      L.DomEvent.off(this._container, 'mousedown', this._onMouseDown);
    },

    _onMouseDown: function (e) {
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

    _onMouseMove: function (e) {
      var startPoint = this._startLayerPoint,
        box = this._box,

        layerPoint = this._map.mouseEventToLayerPoint(e),
        offset = layerPoint.subtract(startPoint),

        newPos = new L.Point(
          Math.min(layerPoint.x, startPoint.x),
          Math.min(layerPoint.y, startPoint.y));

      L.DomUtil.setPosition(box, newPos);

      // TODO refactor: remove hardcoded 4 pixels
      box.style.width  = (Math.max(0, Math.abs(offset.x) - 4)) + 'px';
      box.style.height = (Math.max(0, Math.abs(offset.y) - 4)) + 'px';
    },

    _finish: function () {
      this._pane.removeChild(this._box);
      this._container.style.cursor = '';

      L.DomUtil.enableTextSelection();

      L.DomEvent
        .off(document, 'mousemove', this._onMouseMove)
        .off(document, 'mouseup', this._onMouseUp)
        .off(document, 'keydown', this._onKeyDown);
    },

    _onMouseUp: function (e) {
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

    _onKeyDown: function (e) {
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

  ls.forEach(function (l) {
    const min = Math.min(...l.coords.map(coord => coord.longitude))
    const max = Math.max(...l.coords.map(coord => coord.longitude))

    let latLons
    // If it crosses the anti meridian add 360 to the negative values
    if (max - min > 180) {
      latLons = l.coords.map(coord => [coord.latitude, coord.longitude < 0 ? coord.longitude + 360 : coord.longitude])
    } else {
      latLons = l.coords.map(coord => [coord.latitude, coord.longitude])
    }

    lines.push(L.polyline(latLons, {color: color(l.id), weight: 1}).addTo(lineLayer))
  })

  map.on("boxselectend", function(e) {
    selection.forEach(function(line) {
      line.line.setStyle({color: line.color})
    })

    selection = []

    for (let i = 0; i < lines.length; i++) {
      const coords = lines[i].getLatLngs()
      for (let j = 0; j < coords.length; j++) {
        if (e.boxZoomBounds.contains([coords[j].lat, coords[j].lng])) {
          selection.push({line: lines[i], color: lines[i].options.color})
          lines[i].setStyle({color: "red"})
          break
        }
      }
    }
    
    getCentroidLine(structuredClone(selection.map(l => l.line._latlngs)))
  })
}


function getCentroidLine(selection) {
  aggregateLayer.clearLayers()

  const maxLength = Math.max(...selection.map(l => l.length))
  // const paddedSelection = selection.map(l => l.concat(Array(maxLength - l.length).fill(l[l.length-1])))
  const paddedSelection = selection.map(l => padLine(l, maxLength))

  const lineCount = paddedSelection.length

  let average = paddedSelection[0].map(point => ({lat: point.lat, lng: point.lng}));

  paddedSelection.slice(1).forEach(function(l) {
    for (let i = 0; i < maxLength; i++) {
      average[i].lat += l[i].lat
      average[i].lng += l[i].lng
    }
  })

  average = average.map(coords => [coords.lat / lineCount, coords.lng / lineCount])
  L.polyline(average, {color: "blue", weight: 3}).addTo(aggregateLayer)
}


function padLine(line, length) {
  if (length - line.length == 0) {
    return line
  }

  let padding = Array(length - line.length)

  let lastElem = line[line.length-1]

  let dLat = lastElem.lat - line[line.length - 2].lat
  let dLng = lastElem.lng - line[line.length - 2].lng
  
  for (let i = 0; i < padding.length; i++) {
    padding[i] = {
      lat: lastElem.lat + (dLat * (i + 1)),
      lng: lastElem.lng + (dLng * (i + 1)),
    }
  }

  return line.concat(padding)
}


init()
