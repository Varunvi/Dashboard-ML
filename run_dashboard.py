import uvicorn
import webbrowser
import threading
import time

def open_browser():
    time.sleep(1.5)
    print("Opening browser to http://localhost:8000 ...")
    webbrowser.open("http://localhost:8000")

if __name__ == "__main__":
    threading.Thread(target=open_browser, daemon=True).start()
    print("Starting Petro-Predict AI Dashboard on http://localhost:8000 ...")
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)
