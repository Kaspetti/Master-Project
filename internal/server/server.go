// package server contains all functions for fetching data from the server.
// This includes web pages and also the api itself.
package server

import (
	"fmt"
	"io"
	"net/http"
	"os"
	"strconv"
	"strings"

	"github.com/Kaspetti/Master-Project/internal/netcdf"
	"github.com/gin-gonic/gin"
)

// StartServer sets up the endpoints and starts the server
func StartServer() error {
	r := gin.Default()

	r.Static("/static", "./static")

	r.LoadHTMLGlob("./templates/*")
	r.GET("/", func(ctx *gin.Context) {
		ctx.HTML(http.StatusOK, "index.html", gin.H{
			"title": "Master Project",
		})
	})

	r.GET("/api/all-lines", getAllLines)

    r.GET("/api/data-exists", getDataExists)

	return r.Run(":8000")
}

// getAllLines calls the GetAllLines function from the netcdf internal package.
// It passes the date gotten from the 'date' query parameter and returns all lines
// from the 50 ensemble members.
func getAllLines(ctx *gin.Context) {
	var date string
    var time int64

    if dateQuery, ok := ctx.GetQuery("date"); !ok {
        ctx.JSON(http.StatusBadRequest, gin.H{
            "message": "date parameter missing",
        })
        return
    } else {
        date = dateQuery
    }

	if timeQuery, ok := ctx.GetQuery("time"); !ok {
		time = 0
	} else {
		var err error
		time, err = strconv.ParseInt(timeQuery, 10, 64)
		if err != nil {
			time = 0
		}
	}

    dataFolder := fmt.Sprintf("%s12", strings.ReplaceAll(date, "-", ""))
    if _, err := os.Stat(dataFolder); os.IsNotExist(err) {
        if err := fetchData(dataFolder); err != nil {
            ctx.JSON(http.StatusBadRequest, gin.H{
                "message": fmt.Sprintf("data could not be fetched from server for date %s", date),
                "error": err.Error(),
            })
            return
        }
    }

	lines, err := netcdf.GetAllLines(dataFolder, time)
	if err != nil {
		ctx.JSON(http.StatusInternalServerError, gin.H{
            "message": fmt.Sprintf("error getting lines for date %s", date),
            "error": err.Error(),
        })
		return
	}

	ctx.JSON(http.StatusOK, lines)
}


func getDataExists(ctx *gin.Context) {
	var date string

    if dateQuery, ok := ctx.GetQuery("date"); !ok {
        ctx.JSON(http.StatusBadRequest, gin.H{
            "message": "date parameter missing",
        })
        return
    } else {
        date = dateQuery
    }

    dataFolder := fmt.Sprintf("%s12", strings.ReplaceAll(date, "-", ""))
    if _, err := os.Stat(dataFolder); os.IsNotExist(err) {
        ctx.JSON(http.StatusNotFound, gin.H{})    
        return
    } else {
        ctx.JSON(http.StatusOK, gin.H{})
    }
}


func fetchData(dataFolder string) error {
    if err := os.Mkdir(dataFolder, os.ModePerm); err != nil {
        return err
    }

    for i := 0; i < 50; i++ {
        fileName := fmt.Sprintf("ec.ens_%02d.%s.sfc.mta.nc", i, dataFolder)

        path := fmt.Sprintf("./%s/%s", dataFolder, fileName)
        out, err := os.Create(path)
        if err != nil {
            return err
        }
        defer out.Close()

        resp, err := http.Get(fmt.Sprintf("https://iveret.gfi.uib.no/mtaens/%s", fileName))
        if err != nil {
            return err
        }
        defer resp.Body.Close()

        _, err = io.Copy(out, resp.Body)
        if err != nil {
            return err
        }
    }

    return nil
}
