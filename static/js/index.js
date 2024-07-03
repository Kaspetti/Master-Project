/// <reference types="leaflet" />


let map

async function init() {
  const color = d3.scaleOrdinal(d3.schemeCategory10)

  // Handler for box select. BoxZoom but without zoom
  L.Map.BoxSelectHandler = L.Handler.extend({
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

  let lines = []

  const time = 0
  for (let ensId = 0; ensId < 50; ensId++) {
    const lineId = 4
    const coords = await d3.json(`/api/coords?ens-id=${ensId}&line-id=${lineId}&time=${time}`)
    const latLons = coords.map(coord => [coord.latitude, coord.longitude])

    lines.push(L.polyline(latLons, {color: color(lineId), weight: 1}).addTo(map))

    // const lineCount = await d3.json(`/api/line-count?ens-id=${ensId}&time=${time}`)
    //
    // for (let lineId = 1; lineId <= lineCount; lineId++) {
    //   const coords = await d3.json(`/api/coords?ens-id=${ensId}&line-id=${lineId}&time=${time}`)
    //   const latLons = coords.map(coord => [coord.latitude, coord.longitude])
    //
    //   lines.push(L.polyline(latLons, {color: color(lineId), weight: 1}).addTo(map))
    // }
  }

  alert("Loaded all lines")

  map.on("boxselectend", function(e) {
    console.log(e.boxZoomBounds)

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
