// package netcdf contains functionality for reading lines from netcdf files.
package netcdf

import (
	"errors"
	"fmt"

	"github.com/batchatco/go-native-netcdf/netcdf"
)

type Line struct {
	Id     int64   `json:"id"`
	Coords []Coord `json:"coords"`
}

type Coord struct {
	Latitude  float64 `json:"latitude"`
	Longitude float64 `json:"longitude"`
}

// GetAllLines gets all lines of a given date from all ensemble members.
// Dates range from 0-240 and is hours since simulation start.
func GetAllLines(date int64) ([]Line, error) {
	allLines := make([]Line, 0)
	for i := 0; i < 50; i++ {
		lines, err := getLines(int64(i), date)
		if err != nil {
			return nil, err
		}

		allLines = append(allLines, lines...)
	}

	return allLines, nil
}

func getLines(ensId int64, date int64) ([]Line, error) {
	// Opens the netCDF file of the ensamble member of id 'ensId'
	nc, err := netcdf.Open(fmt.Sprintf("./2024070112/ec.ens_%02d.2024070112.sfc.mta.nc", ensId))
	if err != nil {
		return nil, err
	}
	defer nc.Close()

	latVr, err := nc.GetVariable("latitude")
	if err != nil {
		return nil, err
	}
	lats, ok := latVr.Values.([]float64)
	if !ok {
		return nil, errors.New("Latitudes were not of type 'float64'")
	}

	lonVr, err := nc.GetVariable("longitude")
	if err != nil {
		return nil, err
	}
	lons, ok := lonVr.Values.([]float64)
	if !ok {
		return nil, errors.New("Longitudes were not of type 'float64'")
	}

	idVr, err := nc.GetVariable("line_id")
	if err != nil {
		return nil, err
	}
	ids, ok := idVr.Values.([]int64)
	if !ok {
		return nil, errors.New("Line ids were not of type 'int64'")
	}

	dateVr, err := nc.GetVariable("date")
	if err != nil {
		return nil, err
	}
	dates, ok := dateVr.Values.([]int64)
	if !ok {
		return nil, errors.New("Dates were not of type 'int64'")
	}

	lines := make([]Line, 0)
	for i := 0; i < len(ids); i++ {
		if dates[i] == date {
			id := ids[i]
			if int64(len(lines)) < id {
				lines = append(lines,
					Line{
						Id:     id,
						Coords: make([]Coord, 0),
					},
				)
			}

			lines[id-1].Coords = append(lines[id-1].Coords, Coord{lats[i], lons[i]})
		} else if dates[i] > date {
			break
		}
	}

	return lines, nil
}
