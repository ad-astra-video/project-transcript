package main

import (
	"flag"
	"fmt"
	"io"
	"log"
	"os"
	"path/filepath"

	"trickle"
)

// SaveSegments saves trickle stream segments to individual files
func main() {
	// Check command-line arguments
	baseURL := flag.String("url", "http://localhost:2939", "Base URL for the stream")
	streamName := flag.String("stream", "", "Stream name (required)")
	outputDir := flag.String("output", "segments", "Directory to save segments")
	maxSegments := flag.Int("max", 0, "Maximum number of segments to save (0 for unlimited)")
	workers := flag.Int("workers", 2, "Number of concurrent workers writing segments to disk")
	flag.Parse()

	if *streamName == "" {
		log.Fatalf("Error: stream name is required. Use -stream flag to specify the stream name.")
	}

	// Create output directory if it doesn't exist
	err := os.MkdirAll(*outputDir, 0755)
	if err != nil {
		log.Fatalf("Failed to create output directory: %v", err)
	}

	// Create trickle client
	client := trickle.NewTrickleSubscriber(*baseURL + "/" + *streamName)
	log.Printf("Starting to save segments from %s/%s to %s", *baseURL, *streamName, *outputDir)

	// Create a combined file for all segments
	combinedFilePath := filepath.Join(*outputDir, fmt.Sprintf("%s_combined.ts", *streamName))
	combinedFile, err := os.Create(combinedFilePath)
	if err != nil {
		log.Fatalf("Failed to create combined file: %v", err)
	}
	defer combinedFile.Close()

	// Channel to pass segments to writers
	segChan := make(chan struct {
		idx  int
		data []byte
	}, 20)

	// WaitGroup to wait for workers to finish
	var wg sync.WaitGroup

	// Mutex to protect combined file writes
	var combMu sync.Mutex

	// Start worker goroutines
	for w := 0; w < *workers; w++ {
		wg.Add(1)
		go func(workerID int) {
			defer wg.Done()
			for seg := range segChan {
				segmentFilePath := filepath.Join(*outputDir, fmt.Sprintf("segment_%03d.ts", seg.idx))
				segmentFile, err := os.Create(segmentFilePath)
				if err != nil {
					log.Printf("[worker %d] Failed to create segment file: %v", workerID, err)
					continue
				}

				// Write to individual file
				if _, err := segmentFile.Write(seg.data); err != nil {
					log.Printf("[worker %d] Failed to write segment %d: %v", workerID, seg.idx, err)
				}
				segmentFile.Close()

				// Append to combined file
				combMu.Lock()
				if _, err := combinedFile.Write(seg.data); err != nil {
					log.Printf("[worker %d] Failed to append segment %d to combined file: %v", workerID, seg.idx, err)
				}
				combMu.Unlock()

				log.Printf("[worker %d] Saved segment %d (%s) to %s", workerID, seg.idx, trickle.HumanBytes(int64(len(seg.data))), segmentFilePath)
			}
		}(w)
	}

	segmentCount := 0
	totalBytes := int64(0)

	for {
		// Check if we've reached the maximum number of segments
		if *maxSegments > 0 && segmentCount >= *maxSegments {
			log.Printf("Reached maximum number of segments (%d). Stopping.", *maxSegments)
			break
		}

		// Read segment
		resp, err := client.Read()
		if err != nil {
			log.Printf("Failed to read segment: %v", err)
			// Wait a bit and try again instead of exiting
			continue
		}

		idx := trickle.GetSeq(resp)
		// Read entire body quickly into memory
		data, err := io.ReadAll(resp.Body)
		resp.Body.Close()
		if err != nil {
			log.Printf("Failed to read segment body %d: %v", idx, err)
			continue
		}

		// Queue segment for async writing
		segChan <- struct {
			idx  int
			data []byte
		}{idx: idx, data: data}

		totalBytes += int64(len(data))
		segmentCount++
		
		// Every 10 segments, print a summary
		if segmentCount%10 == 0 {
			log.Printf("Progress: %d segments queued, total size: %s", segmentCount, trickle.HumanBytes(totalBytes))
		}
	}

	// Close channel and wait for workers
	close(segChan)
	wg.Wait()

	log.Printf("Completed saving %d segments, total size: %s", segmentCount, trickle.HumanBytes(totalBytes))
	log.Printf("Combined file saved to: %s", combinedFilePath)
	log.Printf("You can play the combined file with VLC: vlc \"%s\"", combinedFilePath)
}
