# Movie Recommendation System 🎬

A smart movie recommendation engine that suggests films similar to your favorites.
Built with **Python** and **Streamlit**, it uses collaborative filtering (item-based cosine similarity) on the MovieLens dataset to find movies with similar rating patterns.

---

## 🌟 Features

* **🔍 Smart Movie Search** – find any movie by full title, partial match, or even with typos (fuzzy matching using `difflib`).
* **🎯 Interactive Web Interface** – powered by Streamlit.
* **📊 Genre Display** – shows the genre of the selected movie and recommendations.
* **📈 Similarity Scores** – each recommendation includes a similarity score.
* **⚡ Fast & Cached** – similarity matrix is computed once and reused.
* **🧹 Noise Reduction** – only movies with at least 20 ratings are used to ensure reliable recommendations.

---

## 🛠️ Technologies Used

* Python 3.8+
* Streamlit – web interface
* Pandas – data manipulation
* NumPy – numerical operations
* Scikit-learn – cosine similarity computation
* Difflib – fuzzy string matching

---

## 📁 Dataset

This project uses the **MovieLens dataset**.

Required files:

* `ratings.csv` – contains user ratings for movies
* `movies.csv` – contains movie titles and genres

Place these files inside a folder named **dataset** in the project directory.

---

## 📂 Project Structure

```
movie-recommendation-system
│
├── dataset
│   ├── ratings.csv
│   └── movies.csv
│
├── app.py
├── movie_recommender.py
├── requirements.txt
└── README.md
```

---

## 🚀 Installation & Setup

### 1. Navigate to the project folder

```
cd movie-recommendation-system
```

### 2. Install required libraries

```
pip install -r requirements.txt
```

---

## ▶️ How to Run

### 🖥️ Run Web Application (Recommended)

```
python -m streamlit run app.py
```

Your browser will automatically open the interactive interface.

---

### ⌨️ Run Command-Line Version

```
python movie_recommender.py
```

Follow the instructions in the terminal to enter a movie name and view recommendations.

---

## 📖 Usage Guide (Web App)

1. Type a movie name (or part of it) in the search box.
2. Example: `toy`
3. Select the correct movie from the dropdown suggestions.
4. The system displays:

   * The genre of the selected movie
   * Top 5 similar movies
   * Similarity scores for each recommendation

Enjoy discovering new movies! 🍿

---

## 📊 How It Works

1. A **user-movie rating matrix** is created from the dataset.
2. User ratings are **mean-centered** to remove user bias.
3. **Cosine similarity** is computed between movies.
4. When a movie is selected, the system finds the **most similar movies**.
5. The top-5 most similar movies are returned as recommendations.

Similarity scores range from **-1 to 1**, where higher values indicate stronger similarity.

---

# 📜 License

This project is for **educational and research purposes**.

---

# ⭐ If You Like This Project

Consider giving it a **star ⭐ on GitHub**.

---

## 👤 Author

**Tarun Sri Ram**

This project was built as a learning project to demonstrate **collaborative filtering and interactive web applications using Streamlit**.
