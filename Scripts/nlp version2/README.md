### 🚀 How to Run (Kaggle Setup)

1. Open the notebook on Kaggle and run all cells.
2. Make sure to keep the session running (do not stop or disconnect).
3. The notebook will generate a public URL (via ngrok) for the backend.

---

### ⚙️ Configuration

* The translation model used: Seed-X-PPO-7B

* Copy the generated URL from the Kaggle notebook (ngrok output), for example:

  <https://irretrievably-unsimpering-darrin.ngrok-free.dev>

* In your `main.js` file, set:

  javascript
    const API_BASE = "https://irretrievably-unsimpering-darrin.ngrok-free.dev";

---

### ✅ Notes

* The Kaggle session must remain running for the API to work.
* The ngrok URL changes every time you restart the notebook, so update it accordingly in `main.js`.
* Ensure all notebook cells run successfully before using the API.
