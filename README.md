# COVID-19 Mobility Analysis Using Stochastic Processes

**Web Dashboard for Mobility Insights Using Markov Models, HMM, and Queuing Theory**

## 🔍 Overview

This web-based dashboard helps analyze human mobility changes during the COVID-19 pandemic using real-world data from [Google Mobility Reports](https://www.google.com/covid19/mobility/).

It combines techniques from:

* **Markov Chains** – to study mobility behavior transitions
* **Hidden Markov Models** – to infer hidden policy decisions (e.g., lockdowns)
* **Queuing Theory (M/M/1)** – to simulate public space congestion

---

## 📊 Key Features

* **Country-wise Analysis**: Select country, year, and category (e.g., retail, workplace).
* **Markov Module**:
  * Transition matrix
  * Steady-state distribution
  * Recurrence, first passage, absorption times
* **HMM Module**:
  * Viterbi & Forward algorithms
  * Custom transition/emission probabilities
  * Hidden policy visualization
* **Queuing Theory**:
  * Congestion modeling using M/M/1
  * Utilization, queue length, wait times
* **Reports & Downloads**:
  * Export results as CSV & PDF
  * Interactive visualizations

---

## 📁 Folder Structure

```
COVID_MOBILITY_SYSTEM/
│
├── app.py                    # Main Flask app
├── requirements.txt
│
├── data/
│   └── Global_Mobility_Report.csv  # Google COVID-19 mobility dataset
│
├── modules/                 # Core model logic
│   ├── hmm_model.py
│   ├── markov_model.py
│   ├── mm1_queue.py
│   ├── preprocess.py
│   └── visuals.py
│
├── static/
│   ├── styles.css
│   └── plots/               # Generated plots, PDFs, CSVs
│
├── templates/
│   ├── index.html
│   ├── markov.html
│   ├── hmm.html
│   ├── queue.html
│   └── result.html
```

---

## 📈 Sample Visualizations

* Steady-State Pie Charts
* Timeline of Mobility Behavior
* Hidden State (Viterbi) Path
* M/M/1 Queueing Summary Bar Chart

---

## 📦 Installation

```bash
git clone https://github.com/nasif1731/Covid19_MobilityAnalysisUsingStochasticProcesses.git
cd Covid19_MobilityAnalysisUsingStochasticProcesses
pip install -r requirements.txt
python app.py
```

Open your browser and navigate to: [http://localhost:5000](http://localhost:5000)

---

## 📂 Dataset

* **Source**: [Google Global Mobility Reports](https://www.gstatic.com/covid19/mobility/Global_Mobility_Report.csv)
* **Granularity**: Daily region-level records, categorized by activity type

---

## 🧠 Models Used

### ✅ Markov Chain

* Mobility states: Low, Moderate, High
* Outputs: Transition matrix, steady distribution, recurrence & absorption metrics

### 🔐 Hidden Markov Model (HMM)

* Hidden states: Strict, Moderate, Normal policies
* Observations: Mobility levels
* Algorithms: Forward (likelihood) & Viterbi (most probable policy path)

### 🏥 M/M/1 Queueing Model

* Models congestion in public places
* Metrics: Utilization, L, Lq, W, Wq, P₀

---

## 📌 Conclusion

This system transforms raw mobility data into meaningful insights using intuitive visualizations and mathematical models. It can aid policymakers, researchers, and health officials in understanding behavioral patterns and preparing for future pandemics or urban planning.

---

