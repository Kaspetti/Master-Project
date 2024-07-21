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

  if (dateSlider.value > 78) {
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

  await showLines(0)
}


async function showLines(date) {
  const color = d3.scaleOrdinal(d3.schemeCategory10)


  let lines = []
  let selection = []

  const ls = await d3.json(`/api/all-lines?date=${date}`)
  lineLayer.clearLayers()

  ls.forEach(function (l) {
    const latLons = l.coords.map(coord => [coord.latitude, coord.longitude])
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
  })
}


init()
