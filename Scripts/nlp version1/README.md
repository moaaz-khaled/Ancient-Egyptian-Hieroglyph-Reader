### 🚀 How to Run Locally

1. Make sure the following files are in the project root directory:

   * `app.py`
   * `models.py`

2. Run the application using:

   ```bash
   python app.py
   ```

3. The backend server will start on:

   ```
   http://localhost:5000
   ```

---

### ⚙️ Configuration

* The translation model used:

  ```
  Meta 2.5G
  ```

* In your `main.js` file, make sure to set:

  ```javascript
  const API_BASE = "http://localhost:5000";
  ```

---

### ✅ Notes

* Ensure all dependencies are installed before running the app.
* The server must be running before using the frontend.
