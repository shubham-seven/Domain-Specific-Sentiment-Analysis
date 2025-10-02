Domain-specific Sentiment Analysis
Overview
This project focuses on domain-specific sentiment analysis using two approaches: model-centric and data-centric. The code includes implementations for both approaches and leverages a trained model to perform sentiment classification tailored to specific domains.

Live Demo
Try out the sentiment analysis model interactively on Hugging Face Spaces:
https://huggingface.co/spaces/Shubhseven/relationship-sentiment-app

Download the Model File
The trained model weights (model.safetensors) are hosted externally due to GitHub’s file size limitations.

Download model files here: https://drive.google.com/drive/folders/1pgbdeGzdzXLO5m0i9xmeY6PrnP4G7cYn?usp=sharing

After downloading, place the model.safetensors file in the appropriate project directory before running the inference script.

Project Approaches
Model-centric Approach: • Focus: Maximizing model capabilities through representation and optimization rather than manual feature engineering.
                        • Model: Fine-tuned deep transformer architectures—primarily distilbert-base-uncased and optionally bert-base-uncased—using the Hugging Face Transformers framework in PyTorch.
                        • Preprocessing & Augmentation: Used cleaner text, emoji demojization,and aggressive data augmentation. Minority and hard classes received extra EDA and back-translation rounds.
                        • Label Encoding: Applied label encoding for compatibility with PyTorch.
                        • Custom Loss: Implemented Focal Loss to further address imbalance.
                        • Training: Used early stopping, learning rate scheduling, and class-weighted loss in a rigorous cross-validation loop (5-fold stratified).
                        • Performance: DistilBERT models showed superior generalization, especially for complex or context-dependent breakup emotions. Focal loss and augmentation further boosted minority recall.
                        • Deployment: Retrained final DistilBERT on the entire dataset and saved for API/UI deployment.

Data-centric Approach:  • Augmentation: Applied targeted EDA (Easy Data Augmentation) and back-translation (English↔French) to boost minority (“anger”, “relief”) and hard-to-predict (“confusion”, “longing_regret”, “sadness”) classes.
                                     • Feature Engineering: Combined transformer embeddings with TF-IDF (word/char n-grams) and GloVe word vectors.
                                     • Modeling: Trained Logistic Regression, Linear SVM, Random Forest,Gaussian Naive Bayes on these composite features, with aggressive custom class weighting to counter label imbalance.
Results: Acheived around 95% accuracy using Model-Centric Approach and 45% accuracy with Data-Centric Approach
