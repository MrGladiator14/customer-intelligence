# Customer Intelligence Platform UI

This is a simple HTML/CSS/JavaScript interface for interacting with the Meridian Customer Intelligence Platform API.

## Features

- Health check endpoint
- Single customer prediction
- Batch scoring (CSV upload)
- Complaint analysis via RAG
- Combined customer intelligence (prediction + complaint insights)

## Usage

1. Start the FastAPI server:
   ```bash
   cd /home/bryson/dev/week-13/customer-intelligence
   uvicorn src.serving.serve:app --reload
   ```

2. Open the UI:
   - Open `ui/index.html` in your web browser
   - Or serve it with a simple HTTP server:
     ```bash
     cd ui
     python -m http.server 8080
     ```
   - Then navigate to `http://localhost:8080` in your browser

## API Endpoints Used

- `GET /health` - System health check
- `POST /predict` - Single customer prediction
- `POST /batch-score` - Batch scoring with CSV upload
- `POST /ask-complaints` - Complaint analysis via RAG
- `POST /customer-intel` - Combined customer intelligence

## Requirements

- Modern web browser
- Running FastAPI server on `http://localhost:8000` (adjust API_BASE_URL in the script if different)

## Notes

- The UI uses plain HTML, CSS, and JavaScript - no frameworks required
- Error handling is included for API requests
- Results are displayed in a readable format
- Tabbed interface for easy navigation between features