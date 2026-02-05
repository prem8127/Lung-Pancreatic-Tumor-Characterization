🫁 Lung & Pancreatic Tumor Characterization Using Deep Learning
📌 Project Overview

This project presents an AI-assisted medical image analysis system for the characterization of lung and pancreatic tumors using deep learning techniques.
The system processes CT scan images, predicts tumor categories, and provides model explainability using Grad-CAM to highlight regions influencing predictions.

⚠️ Disclaimer: This project is developed strictly for academic and research purposes and is not intended for clinical diagnosis.

🎯 Key Features

✅ Lung & Pancreatic tumor classification from CT scans

✅ Deep learning using ResNet50 (Transfer Learning)

✅ Grad-CAM based visual explainability

✅ Confidence score visualization

✅ Multi-image (slice-wise) analysis

✅ Case history tracking

✅ Downloadable diagnostic report (TXT)

✅ Clean medical-grade Streamlit UI

🧠 Model & Techniques

Model Architecture: ResNet50

Frameworks: TensorFlow, Keras

Image Size: 224 × 224

Explainability: Gradient-weighted Class Activation Mapping (Grad-CAM)

Frontend: Streamlit

📂 Datasets Used

Lung Cancer Dataset:
IQ-OTH/NCCD Lung Cancer Dataset

Pancreatic Cancer Dataset:
Kaggle – Pancreatic CT Imaging Dataset

All datasets used are publicly available and anonymized.

🖥️ Application Workflow

Upload CT scan image(s)

Image preprocessing & normalization

Feature extraction using ResNet50

Tumor classification

Confidence score calculation

Grad-CAM visualization

Diagnostic summary & report generation

🧪 Explainable AI (Grad-CAM)

Grad-CAM highlights the regions of interest in CT images that contribute most to the model’s prediction, improving:

Transparency

Interpretability

Trust in medical AI systems

🛡️ Limitations

Trained on limited public datasets

Performance may vary across scanners and populations

Not evaluated on real clinical workflows

Should not replace professional medical judgment

👨‍🎓 Author Information

Name: Prem Sagar
Degree: B.Tech (3rd Year)
Specialization: Artificial Intelligence & Machine Learning (AIML)
Institution: Vignana Bharathi Institute of Technology

📧 Email: koatpremsagar10321@gmail.com

📞 Contact: 8885667196

🔗 LinkedIn: (Add your LinkedIn profile link here)
💻 GitHub: (Add your GitHub profile link here)

🚀 How to Run
pip install streamlit tensorflow numpy opencv-python pillow matplotlib
streamlit run app.py

📜 License

This project is released for academic and educational use only.