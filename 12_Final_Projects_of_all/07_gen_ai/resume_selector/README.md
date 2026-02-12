# 📄 AI Resume Screener & Candidate Matcher

An automated HR recruitment assistant built on **n8n**. This workflow utilizes **Google Gemini** to instantly analyze PDF resumes against specific Job Descriptions (JD), acting as a first-round filter to identify top technical talent without manual review.

---

## 📸 Workflow Visualization

![Resume Selector Workflow](D:\GIT_HUB\12_Final_Projects_of_all\07_gen_ai\resume_selector\workflow_image.png)

*The automation pipeline showing the resume upload, text extraction, AI evaluation, and automated email response system.*

---

## 🧠 Core Features

* **Automated PDF Parsing:** Uses the **Extract from File** node to pull raw text data from uploaded PDF resumes.
* **Expert AI Evaluation:** Leverages **Google Gemini** configured as an "Expert HR Recruiter" to compare technical skills, experience, and education against JD requirements.
* **Binary Decision Logic:** The AI is strictly prompted to provide a 'Yes' or 'No' match decision to ensure high-speed processing.
* **Automated Candidate Communication:** * **Accepted Candidates:** Receives a professional interview invitation via Gmail.
    * **Rejected Candidates:** Receives a polite status update, allowing HR to maintain a positive brand image with zero manual effort.

---

## 🛠️ Technical Workflow Architecture

### 1. Intake & Extraction
* **Form Trigger:** A user-friendly n8n form allows candidates or recruiters to upload resumes in `.pdf` format.
* **Binary-to-Text:** The workflow converts the binary PDF data into searchable text, preserving the source for downstream metadata usage.

### 2. The AI Screening Layer
* **The "HR Brain" (LLM Chain):** Powered by the **Google Gemini Chat Model**, the system evaluates the candidate specifically for a **Python Developer** role requiring **Docker** and **SQL** knowledge.
* **Zero-Shot Prompting:** The AI follows strict instructions to output only a single word, preventing parsing errors and ensuring clean logic.

### 3. Logic & Response
* **Switch Node:** Routes the workflow based on the AI's "Yes" or "No" string output.
* **Gmail Integration:**
    * **Matched (Yes):** Triggers an invitation to `kddeveloper324@gmail.com` with dynamic placeholders for candidate names.
    * **Unmatched (No):** Triggers a rejection notice to `krunalv1510@gmail.com`, citing specific technical gaps like Docker or backend development.

---

## 📋 Tech Stack

| Component | Technology |
| :--- | :--- |
| **Automation Engine** | [n8n](https://n8n.io/) |
| **AI Model** | Google Gemini (Pro/Flash via LangChain) |
| **Data Extraction** | n8n Extract from File Node |
| **Email Service** | Gmail OAuth2 Integration |
| **Input Method** | n8n Webhook-based Forms |

---

## 🔧 Setup & Installation

1.  **Import Workflow:** Import the `resume_selector.json` file into your n8n workspace.
2.  **Configure Credentials:**
    * **Google Gemini:** Connect your Google Palm/Gemini API key.
    * **Gmail:** Authenticate your Gmail account via OAuth2.
3.  **Customize the JD:** You can modify the **Basic LLM Chain** prompt to screen for different roles (e.g., Frontend Developer, Data Scientist) by simply updating the skills list in the instruction field.

---
