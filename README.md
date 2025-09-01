# Face_Recognition
Developed a Face Recognition and Detection system using FaceNet for generating facial embeddings and MTCNN for accurate face detection.




# WORKFLOW
Input Image/Video
       │
       ▼
  [Face Detection]
   (MTCNN)
       │
       ▼
 Cropped + Aligned Face
       │
       ▼
   [Feature Extraction]
       (FaceNet Embeddings)
       │
       ▼
 [Classifier / Similarity Check]
       │ (Eulidean Distance / Coisne Similtary)
 ┌───────────────┐
 │ Recognition   │ → Identify person
 │ Verification  │ → Same / Different


 
 └───────────────┘

