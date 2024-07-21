package main

import (
	"errors"
	"fmt"
	"net/http"
	"strconv"

	"github.com/batchatco/go-native-netcdf/netcdf"
	"github.com/gin-gonic/gin"
)


type Line struct {
    Id          int64       `json:"id"`
    Coords      []Coord     `json:"coords"`
}


type Coord struct {
    Latitude    float64     `json:"latitude"`
    Longitude   float64     `json:"longitude"`
}


func main() {
    r := gin.Default()

    r.Static("/static", "./static")

    r.LoadHTMLGlob("./templates/*")
    r.GET("/", func(ctx *gin.Context) {
        ctx.HTML(http.StatusOK, "index.html", gin.H {
            "title": "Master Project",
        })
    })

    r.GET("/api/all-lines", func(ctx *gin.Context) {
        var date int64
        if dateQuery, ok := ctx.GetQuery("date"); !ok {
            date = 0
        } else {
            var err error
            date, err = strconv.ParseInt(dateQuery, 10, 64)
            if err != nil {
                date = 0
            }
        }

        lines, err := getAllLines(date)
        if err != nil {
            ctx.JSON(http.StatusInternalServerError, gin.H{})
            return
        }

        ctx.JSON(http.StatusOK, lines)
    })

    r.Run()
}


func getAllLines(date int64) ([]Line, error) {
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
                        Id: id,
                        Coords: make([]Coord, 0),
                    },
                )
            }

            lines[id-1].Coords = append(lines[id-1].Coords, Coord {lats[i], lons[i]})
        } else if dates[i] > date {
            break
        }
    }

    return lines, nil
}
