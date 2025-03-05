
from ultralytics import YOLO
import cv2
import cvzone

cap = cv2.VideoCapture("Images/los_angeles.mp4")#imi aleg video-ul

classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]

model = YOLO('../YoloWeights/yolov8n.pt') #imi aleg fisierul care sa prelucreze informatiile

while True:
    _, img = cap.read() #imi citeste imaginile frame cu frame
    results = model(img) # am luat informatia de la YOLO
    for r in results:
        boxes = r.boxes #imi gaseste toate cadranele
        for box in boxes:
            x1 , y1, x2, y2 = box.xyxy[0] #imi stochez coordonatele in niste variabile
            x1, y1, x2, y2 =int(x1), int(y1), int(x2), int(y2)
            print(x1, y1, x2, y2)
            cv2.rectangle(img, (x1 , y1) , (x2 ,y2 ), (0 ,200 ,0) ,thickness= 2) # incadrez in dreptunghiuri in functie de coordonate
            cls = int(box.cls[0]) # scot id-ul obiectului detectat
            cvzone.putTextRect(img, f'{classNames[cls]}', (x1, y1 - 20), scale=1, thickness=1, colorR=0) #afisez deasupra ce obiect este in functie de clasele alese
    cv2.imshow("Image" , img)  #afisez noua imagine
    key = cv2.waitKey(1)  #distanta de 1ms
    if key == 27: #daca apas esc imi iese din program
        break