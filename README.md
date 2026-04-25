# 🏢 Smart Building Energy Forecasting Framework

![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)
![XGBoost](https://img.shields.io/badge/XGBoost-1.7-green.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)

> **Official Repository for the paper:** *"An Integrated Data-Driven Framework for Smart Building Energy Forecasting: From Semantic Bibliometrics to Deep Learning"*, submitted to the International Conference on Future Data and Security Engineering ().

🌐 **Live Demo & Website:** [Insert your Vercel Link here, e.g., [https://smartbuilding-ai.vercel.app](https://smartbuilding-ai.vercel.app)]

---

## 📖 Overview
Current research on smart building energy management often isolates theoretical technology adoption from technical predictive execution. This repository provides an integrated, open-source, semantically linked data-driven framework that bridges this gap. 

Our pipeline encompasses:
1. **Semantic Bibliometrics:** Analyzing 1,043 WoS papers to identify global trends.
2. **Behavioral SEM Modeling:** Validating infrastructural readiness for AI deployment.
3. **Multi-Model AI Execution:** Leveraging Lag-feature engineering and Deep Learning (ANN, XGBoost, Random Forest) on IoT data to achieve state-of-the-art accuracy.

## 🚀 Key Results
Our proposed AI framework was rigorously cross-validated on two standard UCI Machine Learning benchmark datasets.

| Dataset Context | Random Forest ($R^2$) | XGBoost ($R^2$) | Deep ANN ($R^2$) |
| :--- | :---: | :---: | :---: |
| **UCI Appliances Energy (Dynamic Time-Series)** | 0.9865 | 0.9924 | **0.9959** |
| **UCI Energy Efficiency (Static Structural)** | 0.9977 | **0.9983** | 0.9847 |

*The framework successfully handles high-noise IoT sensor data, providing highly reliable predictive capabilities suitable for Building Energy Management Systems (BEMS).*

## 📁 Repository Structure
- `/datasets/`: Contains raw bibliometric data, UCI benchmarks, and processed SEM/validation tables.
- `/src/`: Source code including the `Master_Research_Notebook.ipynb` containing the entire pipeline.
- `/models/`: Pre-trained weights for the best-performing XGBoost and Deep ANN models.
- `/figures/`: High-resolution analytical plots, SEM structural graphs, and AI convergence curves used in the paper.
- `/docs/`: The compiled LaTeX manuscript.
- `/web/`: The source code for our interactive project landing page.

## ⚙️ Quick Start

**1. Clone the repository:**
\`\`\`bash
git clone [https://github.com/username/SmartBuilding-AI-Framework.git](https://github.com/username/SmartBuilding-AI-Framework.git)
cd SmartBuilding-AI-Framework
\`\`\`

**2. Install dependencies:**
\`\`\`bash
pip install -r requirements.txt
\`\`\`

**3. Run the complete pipeline:**
You can open the Jupyter Notebook located at `src/Master_Research_Notebook.ipynb` via Google Colab or run it locally to reproduce all bibliometric charts, SEM tables, and AI forecasting metrics.

## 📝 Citation
If you use our framework, code, or datasets in your research, please cite our paper:

\`\`\`bibtex
@inproceedings{vo2026integrated,
  title={An Integrated Data-Driven Framework for Smart Building Energy Forecasting: From Semantic Bibliometrics to Deep Learning},
  author={Vo, Thi Kim Anh and [Co-author Name]},
  booktitle={},
  year={2026},
  organization={}
}
\`\`\`

## 📄 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
