# CineMatch-AI-Movie-Recommendation-System
<div align="center">

# 🎬 CineMatch AI: Intelligent Movie Recommendation Engine

### Hybrid Collaborative & Content-Based Filtering System — Implementation Specification

![Document](https://img.shields.io/badge/Document-Project%20Documentation%20(Phases%201--3)-1e3a8a?style=for-the-badge)
![Track](https://img.shields.io/badge/DEPI%202026-Microsoft%20ML%20Engineer-2563eb?style=for-the-badge)
![Scope](https://img.shields.io/badge/Scope-5--Week%20Sprint-0ea5e9?style=for-the-badge)

![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat-square&logo=scikitlearn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat-square&logo=pandas&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)
![MLflow](https://img.shields.io/badge/MLflow-0194E2?style=flat-square&logo=mlflow&logoColor=white)
![Microsoft Azure](https://img.shields.io/badge/Microsoft%20Azure-0078D4?style=flat-square&logo=microsoftazure&logoColor=white)

</div>

---

## 👥 Project Team & Resource Allocation

<table align="center">
  <tr>
    <td align="center" width="20%"><b>Youssef Hossam El Din<br>El Sayed Hadwa</b><br><sub>ID: 21007654</sub>[cite: 2]</td>
    <td align="center" width="20%"><b>Ahmed Medhat<br>Hamdy Ahmed</b><br><sub>ID: 21104763</sub>[cite: 2]</td>
    <td align="center" width="20%"><b>Ayad Mahmoud<br>Fouda Saad</b><br><sub>ID: 21165017</sub>[cite: 2]</td>
    <td align="center" width="20%"><b>Kirollos Medhat<br>Thabet Gaballah</b><br><sub>ID: 21069573</sub>[cite: 2]</td>
    <td align="center" width="20%"><b>Mai Mohamed<br>Ali Abdelhamid</b><br><sub>ID: 21131341</sub>[cite: 2]</td>
  </tr>
</table>

---

## 1️⃣ Executive Summary & Overview

Modern digital streaming platforms host tens of thousands of titles, creating severe **choice paralysis (information overload)** for viewers. Studies show that users spend an average of **18 minutes** browsing catalogues before either selecting a suboptimal title or abandoning the platform altogether.

Traditional filtering mechanisms fail by relying on rigid extremes: simple genre filters ignore community quality signals, while raw unfiltered rating averages skew heavily toward niche movies with only one or two 5-star votes.

> [!IMPORTANT]
> **CineMatch AI** delivers an end-to-end Machine Learning recommendation pipeline leveraging the **MovieLens dataset (`movies.csv` & `ratings.csv`)**. By combining **Statistical Thresholding**, **User-Item Matrix Factorization / Collaborative Filtering**, and **Genre Metadata Alignment**, the system serves real-time personalized suggestions via an interactive web application (`app.py`).

### Core System Workflow

<table align="center">
  <tr>
    <td align="center" width="25%"><h3>1</h3><b>Ingests & Merges</b><br>movies metadata & user ratings logs</td>
    <td align="center" width="25%"><h3>2</h3><b>Filters</b><br>sparse noise via minimum vote thresholds</td>
    <td align="center" width="25%"><h3>3</h3><b>Computes</b><br>Cosine / Correlation similarity across Pivot Matrix</td>
    <td align="center" width="25%"><h3>4</h3><b>Serves</b><br>Top-N personalized recommendations in &lt;1.0s</td>
  </tr>
</table>

---

## 2️⃣ Project Problem & Core Capabilities

### 2.1 Key Operational Challenges

<table>
  <tr>
    <td width="50%">
      <h4>🌀 Information Overload</h4>
      Massive movie catalogues overwhelm users and reduce platform engagement.
    </td>
    <td width="50%">
      <h4>📉 Rating Matrix Sparsity</h4>
      Most users rate less than 1% of available movies, creating noisy correlations.
    </td>
  </tr>
  <tr>
    <td width="50%">
      <h4>❄️ Cold-Start Vulnerability</h4>
      Pure rating systems fail when encountering newly added titles or new users.
    </td>
    <td width="50%">
      <h4>⚡ Real-Time Latency</h4>
      Computing pairwise similarity across large matrices on the fly causes UI lag.
    </td>
  </tr>
</table>

### 2.2 Technical Solution Delivery

<table>
  <tr>
    <td width="50%">
      <h4>🧹 Smart Data Thresholding</h4>
      Filtering inactive users and low-vote movies in <code>Movie_Recommendation_System.ipynb</code>.
    </td>
    <td width="50%">
      <h4>🧮 Vectorized Similarity Engine</h4>
      User-Item Pivot Matrix with Cosine Similarity &amp; Pearson Correlation scoring.
    </td>
  </tr>
  <tr>
    <td width="50%">
      <h4>🖥️ Interactive Web UI (<code>app.py</code>)</h4>
      Instant search-and-recommend interface with dynamic metadata display.
    </td>
    <td width="50%">
      <h4>☁️ MLOps &amp; Cloud Readiness</h4>
      Reproducible environment (<code>requirements.txt</code>) tracked via MLflow for Azure deployment.
    </td>
  </tr>
</table>

---

## 3️⃣ System Workflow Architecture

```mermaid
flowchart TD
    A["📂 Raw Datasets<br/>movies.csv + ratings.csv"] --> B["🧹 Preprocessing & EDA<br/>Merge / Clean / Vote Thresholding"]
    B --> C["🧮 Feature Engineering<br/>User-Item Pivot Matrix & Genre Vectors"]
    C --> D{"Query Type<br/>Sufficient Ratings?"}
    D -->|"Yes (Collaborative)"| E["🤝 Matrix Similarity Engine<br/>Cosine / Correlation Score"]
    D -->|"No (Cold-Start Fallback)"| F["🏷️ Content-Based Engine<br/>Genre & Metadata Matching"]
    E --> G["🖥️ Interactive Web App (app.py)<br/>Top-N Movie Recommendations"]
    F --> G

    classDef input fill:#dbeafe,stroke:#1e3a8a,color:#1e3a8a,stroke-width:2px
    classDef process fill:#e0f2fe,stroke:#0369a1,color:#0c4a6e,stroke-width:2px
    classDef decision fill:#fef3c7,stroke:#d97706,color:#78350f,stroke-width:2px
    classDef auto fill:#dcfce7,stroke:#15803d,color:#14532d,stroke-width:2px
    classDef human fill:#f3e8ff,stroke:#7e22ce,color:#581c87,stroke-width:2px

    class A input
    class B,C process
    class D decision
    class E,G auto
    class F human