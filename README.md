---

# 👗 AI Fashion Recommender

This is a smart fashion recommendation system that detects clothes in an image using YOLOv8 and gives fashion advice powered by ChatGPT (OpenAI).

---

## 🧰 Part 1: Install Everything You Need

### 🐍 Step 1: Install Python

1. Go to the official Python site: [https://www.python.org/downloads/](https://www.python.org/downloads/)
2. Download the latest version.
3. **Important:** ✅ Check the box that says **“Add Python to PATH”** during installation.
4. Click **“Install Now”**.
5. To verify installation:
   - Press `Win + R`, type `cmd`, hit Enter.
   - Type:

     ```bash
     python --version
     ```

   - You should see something like `Python 3.11.6`.

---

### 🧑‍💻 Step 2: Install Git

- Download Git from [https://git-scm.com/downloads](https://git-scm.com/downloads)
- Install it (default options are fine).
- You’ll now have **Git Bash** available for command-line use.

---

### 🖥️ Step 3: Install VS Code

- Download from [https://code.visualstudio.com/](https://code.visualstudio.com/)
- Install Visual Studio Code — your coding editor.

---

### 📁 Step 4: Open VS Code and Set Up Project

1. Open VS Code
2. Click on **File > Open Folder** → create a folder (e.g., `fashion_recommender`)
3. Open a terminal in VS Code (`Terminal > New Terminal`)
4. Create a virtual environment:

   ```bash
   python -m venv venv
   ```

5. Activate it (Windows):

   ```bash
   .\venv\Scripts\activate
   ```

   You should now see `(venv)` in the terminal.

---

### 📦 Step 5: Install Required Python Packages

In the terminal, install all dependencies:

```bash
pip install streamlit openai ultralytics opencv-python
```

---

## 💻 Part 2: Add the Code

Add your Python code files (e.g. `app.py`) inside your project folder.

---

## 🚀 Part 3: Run Your Project

### 🎯 Step 6: Download YOLOv8 Weights

Run this to download weights and test detection:

```bash
yolo task=detect mode=predict model=yolov8n.pt source=input.jpg
```

This will auto-download the `yolov8n.pt` weights file.

---

### 🔑 Step 7: Add OpenAI API Key

1. Go to [https://platform.openai.com/account/api-keys](https://platform.openai.com/account/api-keys)
2. Generate a new API key.
3. In the terminal (VS Code), run:

```powershell
$env:OPENAI_API_KEY="your-key-here"
```

*(Replace `"your-key-here"` with your actual OpenAI key)*

---

### 🧠 Step 8: Run the App

Start your Streamlit app with:

```bash
streamlit run app.py
```

A browser will open automatically. Now you can:

✅ Upload a picture  
👕 Detect clothing items  
✂️ Crop & display each piece  
💬 Get feedback from GPT on what to wear!

---


## 🧑‍💻 Author

- [@syedmohammedarsalanghori](https://github.com/syed Mohammed Arsalan)

---

## 📄 License

This project is licensed under the **MIT License** — feel free to use, modify, and share!

---

