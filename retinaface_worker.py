# retinaface_worker.py
from detect_faces import detect_faces, print_detection_summary

if __name__ == "__main__":
    all_detections = detect_faces(
        trained_model_path="weights/Resnet50_Final.pth",
        input_folder="./data/New-Images",
        output_folder="./data/New-Results",
        confidence_threshold=0.02,
        use_cpu=False,
        save_image=True
    )
    
    print_detection_summary(all_detections)