package main

import (
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


        ctx.JSON(http.StatusOK, getAllLines(date))
    })

    r.Run()
}


func getAllLines(date int64) []Line {
    allLines := make([]Line, 0)
    for i := 0; i < 50; i++ {
        allLines = append(allLines, getLines(int64(i), date)...)
    }

    return allLines
}


func getLines(ensId int64, date int64) []Line {
    nc, err := netcdf.Open(fmt.Sprintf("./2024070112/ec.ens_%02d.2024070112.sfc.mta.nc", ensId))
    if err != nil {
        panic(err)
    }
    defer nc.Close() 

    latVr, err := nc.GetVariable("latitude")
    if err != nil {
        panic(err)
    }
    lats, ok := latVr.Values.([]float64)
    if !ok {
        fmt.Println("Lul lats wrong")
    }

    lonVr, err := nc.GetVariable("longitude")
    if err != nil {
        panic(err)
    }
    lons, ok := lonVr.Values.([]float64)
    if !ok {
        fmt.Println("Lul lons wrong")
    }

    idVr, err := nc.GetVariable("line_id")
    if err != nil {
        panic(err)
    }
    ids, ok := idVr.Values.([]int64)
    if !ok {
        fmt.Println("Lul ids wrong")
    }

    dateVr, err := nc.GetVariable("date")
    if err != nil {
        panic(err)
    }
    dates, ok := dateVr.Values.([]int64)
    if !ok {
        fmt.Println("Lul dates wrong")
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

    return lines
}
