# Waste Image Classification Project

This project focuses on classifying trash images into multiple categories using deep learning models. It involves creating custom datasets, training models like MobileNet, ResNet, and YOLOv5, and evaluating their performance on various trash classification tasks.

## Dataset

1. **[Own Larger Dataset](https://www.dropbox.com/scl/fi/uwftk9pwbka66evt4sdf3/trash_classification_data.zip?rlkey=ntnhuvtmb4xyxr92zse99w7pa&st=n9tq5oh4&dl=0)**:
   - Created a comprehensive dataset containing images for trash classification.
   - Collected images from multiple sources, including the internet and existing datasets.

We used three main datasets for this project:

2. **[Organic and Inorganic Waste Dataset](https://www.kaggle.com/datasets/techsash/waste-classification-data)**:
   - Contains over 10,000 images categorized into organic and inorganic waste.
   - Used for binary classification between organic and inorganic trash.

3. **7-Class Trash Dataset**:
   - Categories: cardboard, paper, plastic, glass, e-waste, medical, and metal.
   - Consists of 600 images per class.
   - Collected from multiple sources, including:
     - [Garbage Classification](https://www.kaggle.com/datasets/asdasdasasdas/garbage-classification)
     - [E-Waste Dataset](https://www.kaggle.com/datasets/kaustubh2402/ewaste-dataset)
     - Additional images from the internet.

4. **11-Class Trash Dataset**:
   - Categories: cardboard, paper, plastic, glass, e-waste, medical, metal, clothes, shoes, hazardous, and sanitary.
   - Contains between 3000 to 5000 images per class.
   - Collected from multiple sources, including:
     - [Garbage Classification](https://www.kaggle.com/datasets/asdasdasasdas/garbage-classification)
     - [Garbage Classification (Mostafa Abla)](https://www.kaggle.com/datasets/mostafaabla/garbage-classification)
     - Roboflow datasets
     - Additional images from the internet.

## Methodology

1. **Binary Classification (Organic/Inorganic Waste)**:
   - **MobileNet**: Achieved an accuracy of 0.795.
   - **ResNet**: Achieved an accuracy of 0.877.

2. **7-Class Trash Classification**:
   - **ResNet**: Achieved an accuracy of 0.912.
   - **YOLOv5**: Achieved an accuracy of 0.87 after 100 epochs.

3. **11-Class Trash Classification**:
   - **YOLOv5**: Achieved an accuracy of 0.82 after 35 epochs.

## Results

The project demonstrates the potential of deep learning models for trash classification:
- **MobileNet** and **ResNet** performed well for binary classification between organic and inorganic waste.
- **ResNet** and **YOLOv5** showed strong performance for the 7-class classification task, with YOLOv5 also proving effective for the more complex 11-class classification.
- The results suggest that increasing the dataset size and diversity, along with further hyperparameter tuning, could improve accuracy and overall model performance.

## Future Work

Future improvements could focus on:
- Expanding the dataset with more diverse and real-world images.
- Fine-tuning the models and experimenting with different architectures.
- Implementing real-time trash classification for practical applications.

## Conclusion

This project highlights the effectiveness of deep learning models for trash classification, which can be useful in environmental applications such as waste sorting and recycling. The results show promising accuracy levels and demonstrate the potential for further improvements.

## References

- [Own Larger Dataset](https://www.dropbox.com/scl/fi/uwftk9pwbka66evt4sdf3/trash_classification_data.zip?rlkey=ntnhuvtmb4xyxr92zse99w7pa&st=n9tq5oh4&dl=0)
- [Kaggle Waste Classification Dataset](https://www.kaggle.com/datasets/techsash/waste-classification-data)
- [7-Class Garbage Classification Dataset](https://www.kaggle.com/datasets/asdasdasasdas/garbage-classification)
- [E-Waste Dataset](https://www.kaggle.com/datasets/kaustubh2402/ewaste-dataset)
- [11-Class Garbage Classification Dataset (Mostafa Abla)](https://www.kaggle.com/datasets/mostafaabla/garbage-classification)
