# 🛡️ Insurance Pricing Pipeline — GLM vs Machine Learning

This project implements a **complete insurance pricing pipeline** focused on **claim frequency modeling** in motor insurance.
Its purpose is to **compare traditional actuarial approaches (Poisson GLM)** with **modern Machine Learning models** based on Gradient Boosting (**XGBoost**, **LightGBM**).

The project emphasizes **statistical rigor**, **proper exposure handling**, and **actuarially sound evaluation metrics**, going beyond standard Data Science practices.

---

## 🎯 Project Objective

To build a robust and reproducible framework that enables:

* Preprocessing **real-world actuarial data** (exposure, rare events, categorical features)
* Training multiple pricing models under consistent assumptions
* Performing **fair and scientifically valid comparisons**
* Quantifying the true added value of Machine Learning over industry standards

---

## 🏗️ Project Architecture

The project follows a **modular, object-oriented architecture**, designed for:

* clarity and maintainability
* code reuse and extensibility
* easy integration of new models and metrics

### 📂 Project Structure

```text
modelisation_pricing/
├── data/
│   └── data.csv                # Insurance dataset
├── logs/                       # Execution logs
├── src/
│   ├── models/
│   │   ├── base_model.py       # Abstract base pricing model
│   │   ├── glm_model.py        # Poisson GLM (statsmodels)
│   │   └── ml_models.py        # XGBoost & LightGBM
│   ├── config.py               # Global configuration
│   ├── data_preprocessing.py  # Actuarial preprocessing
│   ├── evaluation.py           # Metrics & reporting
│   ├── main.py                 # Main pipeline
│   └── __init__.py
├── README.md
└── requirements.txt
```

---

## 🛠️ Methodology & Key Steps

### 1️⃣ Actuarial Data Preprocessing (`data_preprocessing.py`)

The pipeline explicitly handles **insurance-specific data challenges**:

* **Binning & grouping**

  * Conversion of continuous variables (e.g. `VehAge`, `DrivAge`) into ordered classes
  * Ensures interpretability and GLM compatibility

* **Log transformation & clipping**

  * Treatment of skewed distributions
  * Capping of sensitive variables (e.g. `BonusMalus`)

* **Categorical encoding**

  * Regions, vehicle brands, usage types
  * Encoding strategies aligned across GLM and ML models

---

### 2️⃣ Multi-Model Pricing (`models/`)

Three model families are implemented through a unified interface:

#### 🔹 Poisson GLM

* Implemented using `statsmodels`
* Log-link function
* **Exposure integrated as an offset**
* Interpretable and fully compliant with actuarial standards

#### 🔹 Gradient Boosting Models

* XGBoost & LightGBM
* Capture nonlinearities and complex interactions
* **Exposure integrated as observation weights**
* Optimized for predictive performance

---

### 3️⃣ Actuarial Model Evaluation (`evaluation.py`)

Model assessment goes beyond generic ML metrics:

* **Weighted Poisson Deviance (WPD)**
  → Core pricing metric accounting for exposure

* **Claim Frequency Analysis**
  → Comparison of observed vs predicted frequency (train & test)

* **Improvement Index**
  → Measures performance gain relative to a benchmark (constant or GLM)

---

## 🚀 Technical Strengths

* **Object-Oriented Design**

  * Abstract base class `PricingModel`
  * Unified interface: `.train()`, `.predict()`, `.evaluate()`

* **Actuarial rigor**

  * Correct distinction between **offsets (GLM)** and **weights (ML)**
  * Respect of statistical assumptions

* **Model comparability**

  * Same dataset
  * Same exposure handling
  * Consistent evaluation metrics

* **Scalability**

  * Gradient Boosting models suitable for large insurance portfolios

---

## 📝 Conclusion

This project demonstrates a **strong ability to bridge traditional actuarial modeling with modern Data Science techniques**.
It highlights that **predictive performance alone is insufficient** in insurance pricing:
**methodological rigor**, **exposure treatment**, and **fair model comparison** are essential.

The modular architecture makes this pipeline suitable as:

* a pricing research framework
* a foundation for an industrial pricing engine
* a benchmark for GLM vs ML comparisons

---

## 🔮 Future Enhancements

* Severity modeling
* Gamma / Tweedie GLMs
* Model calibration & stability analysis
* API deployment (pricing engine)
* MLOps integration

## Author
Bertrand GAKIZA
