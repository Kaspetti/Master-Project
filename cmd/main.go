package main

import (
	"log"

	"github.com/Kaspetti/Master-Project/internal/server"
)

func main() {
	if err := server.StartServer(); err != nil {
		log.Fatalln(err)
	}
}
