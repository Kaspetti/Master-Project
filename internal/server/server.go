// package server contains all functions for fetching data from the server.
// This includes web pages and also the api itself.
package server

import (
	"net/http"
	"strconv"

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

	return r.Run(":8000")
}

// getAllLines calls the GetAllLines function from the netcdf internal package.
// It passes the date gotten from the 'date' query parameter and returns all lines
// from the 50 ensemble members.
func getAllLines(ctx *gin.Context) {
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

	lines, err := netcdf.GetAllLines(date)
	if err != nil {
		ctx.JSON(http.StatusInternalServerError, gin.H{})
		return
	}

	ctx.JSON(http.StatusOK, lines)
}
