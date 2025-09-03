# 🎬 Movie Recommendation Engine

## 📌 Project Overview
This project presents a comprehensive data science application that constructs a **movie recommendation system** utilizing a **collaborative filtering approach**.  
The solution is implemented in **Python** and deployed as an **interactive web application using Streamlit**, thereby demonstrating the entire machine learning pipeline, from **data analysis** to a **functional end-user product**.

---

## ❓ Problem Statement
In the current digital landscape, users are often overwhelmed by the sheer volume of available content.  
The central problem this project addresses is:  

> **How can we assist a user in discovering new movies they would genuinely enjoy, without navigating through thousands of titles?**

By leveraging the **wisdom of the crowd**, this recommendation engine provides a **personalized and efficient solution** for content discovery.

---

## 🛠 Tools Used
The project utilizes the following Python libraries and frameworks:

- **Pandas & NumPy** → Data manipulation, cleaning, and matrix operations  
- **Scikit-learn** → Train-test split and calculating user-to-user similarity  
- **Streamlit** → Interactive and shareable web application  

---

## 🔎 Methodology
The recommendation system is based on **user-to-user collaborative filtering**.  
The process is as follows:

1. **Data Matrix**  
   - Rows → Users  
   - Columns → Movies  
   - Cells → Ratings  
   - Missing values (unrated movies) filled with 0  

2. **Similarity**  
   - Calculate similarity between users using **Cosine Similarity**  

3. **Prediction**  
   - Compute predicted ratings as a **weighted average** of ratings from similar users  

4. **Recommendation Generation**  
   - Recommend movies with the highest predicted ratings that the user hasn’t seen yet  

---

## 📊 Conclusion
The model was evaluated on a test set to measure prediction accuracy.  

- **RMSE (Root Mean Square Error): 0.9691**  
- Interpretation: On average, predictions are **less than 1 star away** from the actual ratings (scale of 1–5).  
- This indicates strong performance and confirms the model’s ability to provide **accurate recommendations**.  

---

## 🚀 Getting Started

### ✅ Prerequisites
- Python **3.7+**  
- `pip` (Python package installer)  

### 📥 Installation
1. Place `app.py`, `ratings.csv`, and `movies.csv` in the same directory.  
2. Open terminal and navigate to the project directory.  
3. Install dependencies:  
   ```bash
   pip install -r requirements.txt

## ▶️ Usage

Run the Streamlit app:

```bash
streamlit run app.py

## 🔮 Future Enhancements

- **Item-Based Model** → Implement and compare with the user-based approach.  
- **Hybrid Models** → Combine collaborative filtering with content-based features (e.g., genres).  
- **User Profiles** → Allow new users to rate a few movies and get personalized recommendations.  
