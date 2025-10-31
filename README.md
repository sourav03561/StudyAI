https://github.com/user-attachments/assets/34d7b1e5-a192-44b5-b5ed-18a2d0f12150

# StudyAI — PDF to Smart Study Pack

An intelligent web app that extracts text from PDFs, summarizes key concepts, and generates flashcards, quizzes, and recommended educational videos.

## 🚀 Features
- 📄 Upload any PDF (supports scanned and digital text)
- 🧩 Auto OCR with Tesseract for scanned documents
- ✍️ Gemini AI–powered summary, key points & flashcards
- 🧠 Auto–generated quiz questions
- 🎥 Recommended YouTube videos based on key topics
- 🌐 Built with Flask (backend) + React (frontend)

## 🛠️ Tech Stack
**Frontend:** React + Vite  
**Backend:** Flask + PyMuPDF + Tesseract OCR  
**AI:** Google Gemini API  
**Video Source:** YouTube Data API (optional)  
## Usage

-Upload a PDF

-Wait for AI summary & flashcards

-View Recommended Videos in the final tab
## ⚙️ Setup

### 1. Clone & install
```bash
git clone https://github.com/yourname/studyai.git
cd studyai
```

### 2. Backend setup
```bash
cd backend
python -m venv venv
source venv/bin/activate  # (Windows: venv\Scripts\activate)
pip install -r requirements.txt
export GEMINI_API_KEY="your_gemini_key"
export YOUTUBE_API_KEY="your_youtube_key"
python app.py
```

### 3. Frontend setup
```bash
cd ../frontend
npm install
npm run dev
```

Then open 👉 [http://localhost:5173](http://localhost:5173)


