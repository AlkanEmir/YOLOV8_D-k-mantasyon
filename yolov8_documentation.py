""" 



YOLOV8 Dökümantasyon
23.08.2023 - 25.08.2023
Emir Alkan



***********************************************************************************************************************



'#' İle ayrılan snippetlar TEK BAŞINA çalıştırılmalı.



################################################

                CODE SNIPPET

################################################



***********************************************************************************************************************



----- KURULUM -----



Gerekenler: Python >= 3.8 PyTorch >= 1.8



- WINDOWS -

pip install ultralytics
Bu komut tek başına tüm gerekenleri (https://github.com/ultralytics/ultralytics/blob/main/requirements.txt) indirmekte.


- CONDA -

conda install -c conda-forge ultralytics
Bu komut tek başına tüm gerekenleri (https://github.com/ultralytics/ultralytics/blob/main/requirements.txt) indirmekte.


- Clone GitHub Repository -

YOLOV8'in çalışması için repoyu klonlamaya ihtiyaç yok. Klonlamak için:
git clone https://github.com/ultralytics/ultralytics
cd ultralytics
pip install -e .



***********************************************************************************************************************



Gerekenlerin yüklenmesinde problem çıkması durumunda:

git clone https://github.com/ultralytics/ultralytics
cd ultralytics
pip install -e .
pip install -r requirements.txt


Problem devam ederse:
https://github.com/ultralytics/ultralytics/blob/main/requirements.txt
Linke ulaşıp pip ile hepsini manuel indir.



"""



########################################################################################################################


import ultralytics

# Software ve Hardware bilgisi.
ultralytics.checks()    


########################################################################################################################



"""



----- DETECT -----


      
model = YOLO(path: str)           PRE-TRAINED MODELLER
             
                                    yolov8n.pt yolov8n-seg.pt yolov8n-cls.pt yolov8n-pose.pt
                                    yolov8s.pt yolov8s-seg.pt yolov8s-cls.pt yolov8s-pose.pt
                                    yolov8m.pt yolov8m-seg.pt yolov8m-cls.pt yolov8m-pose.pt
                                    yolov8l.pt yolov8l-seg.pt yolov8l-cls.pt yolov8l-pose.pt
                                    yolov8x.pt yolov8x-seg.pt yolov8x-cls.pt yolov8x-pose.pt
                                            
                                    Model Pre-trained ise model, kodun bulunduğu dir'e indirilir.

results = model.MODE(
    
                    source: str,                            # Detection yapılması istenilen kaynak.
                    
                                                            0           # Webcam (sürekli)
                                                            img.jpg     # Image
                                                            vid.mp4     # Video
                                                            screen      # Ekran (sürekli)
                                                            ...                                            
                                                                                        
                    conf:           float,                  # Confidence minimum limiti.
                    imgsz:          int,                    # Resim boyutu.
                    iou:            float,                  # IoU limiti.
                    half:           bool,                   # Half precision (FP16).
                    device:         int, str, list,         # Çalıştıralacak aygıt.
                    show:           bool,                   # Mümkünse sonuçları göster webcam display vb.
                    save:           bool,                   # Detection'ı kaydet.
                    save_txt:       bool,                   # Sonuçları .txt olarak kaydet.
                    save_conf:      bool,                   # Sonuçları confidence ile kaydet.
                    show_labels:    bool,                   # Plotta labelları göster.
                    show_conf:      bool,                   # Plotta confidence göster.
                    max_det:        int,                    # Framede max detection.
                    agnostic_nms:   bool,                   # Agnostic NMS aktifleştir.
                    classes:        int, list[int],         # Sadece seçilen class(ları) detect et.
                    
                    )                       
                        
                .MODE = train, val, predict, export, track, benchmark
                # Birinin seçilmemesi durumunda default = predict olarak seçilir.



"""



########################################################################################################################


# Basic Detection
from ultralytics import YOLO

# Modelini seç.
model = YOLO('yolov8n.pt')

# Predict işlemini tercih ettiğin parametreleri ekleyerek kullan.
results = model.predict(source = '0', conf = 0.7, imgsz = 640, save = False, show = True)


########################################################################################################################



########################################################################################################################


# OpenCV ile Detection
import cv2
from ultralytics import YOLO

# Minimum confidence, renkler ve fontu hazırla.
CONFIDENCE_THRESHOLD = 0.75
GREEN = (0, 255, 0)
WHITE = (255, 255, 255)
FONT = cv2.FONT_HERSHEY_COMPLEX

# Modelini seç.
model = YOLO('yolov8n.pt')

# Kamera aygıtını seç. '0' Webcam.
capture = cv2.VideoCapture(0)

while capture.isOpened():
    ret, frame = capture.read()
    if not ret:
        break
    
    # Aktif framede prediction işlemini yap.
    results = model.predict(frame)[0]
    
    # Aktif framede yapılan predictiondaki kutuyu tanımla.
    for r in results:
        boxes = r.boxes
        
        
        """
        
        
        
        results.PARAMS      orig_img:   ndarray     # Numpy array olarak orjinal resmi, frame'i göster.
                            path:       str         # Resmin adresini göster.
                            names:      dict        # Class isimlerini göster.
                            boxes:      Tensor      # Detect doğruysa, 2D tensor kutu koordinatlarını al.
                            masks:      Tensor      # A 3D tensor of detection masks, where each mask is a binary image.
                            probs:      Tensor      # A 1D tensor of probabilities of each class for classification task.
        
        
        
        """
        
        
        # Predictten çekilen kutunun confidence değerini çek.
        for box in boxes:
            confidence = float(box.conf[0])
            
            
            """
            
  
            
            boxes.PARAMS    xyxy:   Tensor, ndarray     # xyxy formatında koordinatları al.
                            xywh:   Tensor, ndarray     # xywh formatında koordinatları al.
                            xyxyn:  Tensor, ndarray     # xyxy formatında normalized koordinatları al.
                            xywhm:  Tensor, ndarray     # xywh formatında normalized koordinatları al.
                            conf:   Tensor, ndarray     # boxes'ın confidence değeri.
                            cls:    Tensor, ndarray     # boxes'ın class değeri.
                            id:     Tensor, ndarray     # The track IDs of the boxes (if available).
                            data:   Tensor              # Raw bboxes tensor değerleri.


            
            """
            
            
            # Kutudan çekilen confidence, tanımladığımız minimum confidencedan büyük-eşitse devam et.
            if confidence >= CONFIDENCE_THRESHOLD:
                
                # Kutudan x-y koordinatlarını al.
                x1, y1, x2, y2 = box.xyxy[0]
                
                # Aldığın kordinatları integera dönüştür.
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                
                # Aktif framede predict edilen objelerin ismini al.
                class_name = str(model.names[0])
                
                # Etiketi oluştur.
                label = f'{class_name} %{confidence:.2f}'
                frame_thickness = 2
                font_thickness = 2
                font_scale = 1
                
                # Aktif framede; çektiğimiz koordinatlara, tanımladığımız renkte, tanımladığımız kalınlıkta çerçeve oluştur.
                cv2.rectangle(frame, (x1, y1), (x2, y2), GREEN, frame_thickness)
                
                # Aktif framede; tanımladığımız etiketi, çektiğimiz koordiantın üzerine, tanımaldığımız scale, renk ve kalınlıkta etiketi koy.
                cv2.putText(frame, label, (x1, y1 - 5), FONT, font_scale, WHITE, font_thickness, lineType=cv2.LINE_AA)
            
     
    # Display oluştur, ismini koy, displayde aktif frame'i göster.       
    cv2.imshow('YOLOV8 Detection', frame)
    
    # 'Q' ya basarak displayi kapat.
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

# Kodu bitir.
capture.release()
cv2.destroyAllWindows()


########################################################################################################################



"""



----- TRAIN -----



- Google Colab YOLOV8 -


YOLOV8 Google Colab: https://colab.research.google.com/github/ultralytics/ultralytics/blob/main/examples/tutorial.ipynb
Linkteki adımları izleyerek Colab'in GPU'su kullanılarak model eğitilebilir. GPU'nuz Kuvvetli değilse bu şekilde yapılmalı.



***********************************************************************************************************************



- Şahsi PC ile Eğitim -



# Modelini seç.
model = YOLO(source: str)                   PRE-TRAINED MODELLER
             
                                            yolov8n.pt yolov8n-seg.pt yolov8n-cls.pt yolov8n-pose.pt
                                            yolov8s.pt yolov8s-seg.pt yolov8s-cls.pt yolov8s-pose.pt
                                            yolov8m.pt yolov8m-seg.pt yolov8m-cls.pt yolov8m-pose.pt
                                            yolov8l.pt yolov8l-seg.pt yolov8l-cls.pt yolov8l-pose.pt
                                            yolov8x.pt yolov8x-seg.pt yolov8x-cls.pt yolov8x-pose.pt

# Parametrelerini gir.
results = model.train(
    
                      model:            str,            # Modelin adresi.
                      data:             str,            # Datanın adresi.
                      epochs:           int,            # Epoch sayısı.
                      patience:         int,            # Gelişme olmaması durumunda erken durdurma.
                      batch:            int,            # Batch sayısı.
                      imgsz:            int,            # Resim boyutu.
                      save:             bool,           # Save train checkpoints and predict results.
                      save_period:      int,            # Seçilen epochta kaydet (eğer <1 ise inaktif).
                      cache:            bool,           # Cache aktif/inaktif.
                      device:           int, str, list  # Eğitimde kullanılacak aygıtları seç.
                      workers:          int,            # Number of worker threads for data loading (per RANK if DDP).
                      project:          str,            # Proje adı.
                      name:             str,            # Experiment adı.
                      exist_ok          bool,           # Overwrite.
                      pretrained        bool,           # Pretrained model kullanımı.
                      optimizer         str,            # Kullanılacak optimizer, seçenekler = [SGD, Adam, Adamax, AdamW, NAdam, RAdam, RMSProp, auto]
                      verbose           bool,           # Verbose outputu aktif/inaktif.
                      deterministic     bool,           # Deterministic mod aktif/inaktif.
                      single_cls        bool,           # Multi-class datayı single-class gibi eğit.
                      rect              bool,           # Rectangular training with each batch collated for minimum padding
                      cos_lr            bool,           # Cosine learning kullan.
                      close_mosaic      int,            # Disable mosaic augmentation for final epochs (0 to disable)
                      resume            bool,           # Kalınan checkpointted eğitime devam et.
                      amp               bool,           # Automatic Mixed Precision aktif/inaktif.
                      fraction          float,          # Dataset fraction to train on.
                      profile           bool,           # Profile ONNX and TensorRT speeds during training for loggers .
                      lr0               float,          # Başlangıç learning rate.
                      lrf               float,          # Bitiş learning rate.
                      momentum          float,          # SGD Momentumu.
                      weight_decay      float,          # Optimizer weight decay.
                      warmup_epochs     float,          # Warmup Epochları.
                      warmup_momentum   float,          # Başlangıç momenti için warmup.
                      warmup_bias_lr    float,          # Başlangıç bias lr'ı için warmup.
                      box               float,          # Box kaybı.
                      cls               float,          # Class kaybı.
                      dfl               float,          # DFL kaybı. 
                      pose              float,          # Pose kaybı.
                      label_smoothing   float,          # Label smoothing.
                      nbs               int,            # Nominal batch boyutu.
                      overlap_mask      bool,           # Masks should overlap during training (segment train only).
                      mask_ratio        int,            # Mask downsample ratio (segment train only).
                      dropout           float,          # Use dropout regularization (classify train only).
                      val               bool,           # Eğitim yapılırken validate et.
                      
                      )



"""



########################################################################################################################


# Train
import ultralytics
from ultralytics import YOLO

# YAML ile yeni model oluştur.
model = YOLO('yolov8n.yaml')   

# Pre-trained model ile çalış. Tercih edilen.
model = YOLO('yolov8n.pt')  

# YAML ile oluştur, weightleri transfer et.    
model = YOLO('yolov8n.yaml').load('yolov8n.pt')

# Eğitimi tercih edilen parametrelerle başlat. 
results = model.train(data = 'coco128.yaml', epochs = 100, imgsz = 640)


########################################################################################################################



"""



----- EXPORT -----



- Google Colab YOLOV8 -


YOLOV8 Google Colab: https://colab.research.google.com/github/ultralytics/ultralytics/blob/main/examples/tutorial.ipynb
Linkteki adımları izleyerek Colab'de eğitilen model tercih edilen uzantı ile export edilebilir.



***********************************************************************************************************************



- Şahsi PC ile Export -



model = YOLO(PATH_TO_MODEL)



model.export(
             
            format:     str,        # Tercih edilen format.
            
             
                                    FORMATLAR       FORMAT ARGS     MODEL                       METADATA    ARGS
                                    
                                    PyTorch         -               my_model.pt                 True        -
                                    TorchScript     torchscript     my_model.torchscript        True        imgsz, optimize
                                    ONNX            onnx            my_model.onnx               True        imgsz, half, dynamic, simplify, opset
                                    OpenVINO        openvino        my_model_openvino_model     True        imgsz, half
                                    TensorRT        engine          my_model.engine             True        imgsz, half, dynamic, simplify, workspace
                                    CoreML          coreml          my_model.mlpackage          True        imgsz, half, int8, nms
                                    TF SavedModel   saved_model     my_model_saved_model        True        imgsz, keras
                                    TF GraphDef     pb              my_model.pb                 False       imgsz
                                    TF Lite         tflite          my_model.tflite             True        imgsz, half, int8
                                    TF Edge TPU     edgetpu         my_model_edgetpu.tflite     True        imgsz
                                    TF.js           tfjs            my_model_web_model/         True        imgsz
                                    PaddlePaddle    paddle          my_model_paddle_model/      True        imgsz
                                    ncnn            ncnn            my_model_ncnn_model/        True        imgsz, half    
                                    
                                        
            imgsz:      int,        # Resim boyutu.
            keras:      bool,       # Export için TF SavedModel kullanıldıysa Keras kullan.
            optimize:   bool,       # TorchScript: mobil için optimize et.
            half:       bool,       # FP16 quantization.
            int8:       bool,       # INT8 quantization.
            dynamic:    bool,       # ONNX/TensorRT: dynamic axes.
            simplify:   bool,       # ONNX/TensorRT: simplify model
            workspace:  int,        # TensorRT: workspace size (GB)
            nms:        bool,       # CoreML: add NMS
    
            )



"""



########################################################################################################################


# Export
import ultralytics
from ultralytics import YOLO

# Pre-trained model.
model = YOLO('yolov8n.pt')      

# Custom model.
model = YOLO('path/to/best.pt') 

# Tercih edilen uzantı ile export et.
model.export(format = 'onnx')   


########################################################################################################################



"""



----- VALIDATION -----



metrics = model.val(
    
                    data:           str,                # Data dosyasının adresi.
                    imgsz:          int,                # Resim boyutu.
                    batch:          int,                # Batch boyutu.
                    save_json:      bool,               # Sonuçları JSON dosyası olarak kaydet.  
                    save_hybrid:    bool,               # Etiketlerin hibrit versiyonunu kaydet (etiket + ekstra predictionlar).
                    conf:           float,              # Minimum confidence değerini seç.
                    iou:            float,              # NMS için IoU minimum değerini seç.
                    max_det:        int,                # Resim başına maks. detection.
                    half:           bool,               # Half precision kullan (FP16).
                    device:         int, ndarray, list  # Çalışacağı aygıtı seç.
                    dnn:            bool,               # ONNX inference'ı için OpenCV DNN kullan.
                    plots:          bool,               # Eğitim sırasında plotları göster.
                    rect:           bool,               # Rectangular val with each batch collated for minimum padding.
                    split:          str,                # Dataset split to use for validation, i.e. 'val', 'test' or 'train'.
    
                    )



"""



########################################################################################################################


# Validation
import ultralytics
from ultralytics import YOLO

# Pretrained model kullan.
model = YOLO('yolov8n.pt') 

# Custom model kullan
model = YOLO('path_to_model.pt') 

# Arg gerekmiyor, dataset ve ayarları YOLOV8'in hafızasında.
metrics = model.val()
metrics.box.map    # map50-95
metrics.box.map50  # map50
metrics.box.map75  # map75
metrics.box.maps   # Her kategoriden, map50-95'ları içeren liste.


########################################################################################################################



"""



----- BENCHMARK -----


Modelin hız ve tutarlılığını test eder.
 
Export edilen formatın boyutunu, mAP50-95 değerlerini (Detection, Segmentation ve Pose için), accuracy_top5 değerlerini (Classification için)
kullanıcıya gösterir.



benchmark(
    
          model:    str,            # Modelin adresi.
          data:     str,            # .yaml dosyasının adresi
          imgsz:    int,            # Resim boyutu.
          half:     bool,           # Half Precision (FP16) quantization.
          int8:     bool,           # INT8 quantization.
          device:   int, str, list  # Kullanılacak aygıt.
          verbose:  bool,           # Do not continue on error (bool), or val floor threshold (float)
    
)



"""



########################################################################################################################


# Benchmark
import ultralytics
from ultralytics.utils.benchmarks import benchmark

benchmark(model = 'yolov8n.pt', data = 'coco8.yaml', imgsz = 640, half = False, device = 0)


########################################################################################################################



"""



----- TRACKING -----


Objenin class ve konumunu algılar, algılanan objeye özel bir ID atar, bu ID'nin sebebi her algılanan objenin değişen konumunu kaydedip izini çıkartması.




- Eğitilmiş Trackerlar -


BoT-SORT    botsort.yaml
ByteTrack   bytetrack.yaml

Default tracker BoT-SORT.



results = model.track(
    
                        source: str,                            # Detection yapılması istenilen kaynak.
                    
                                                                0           # Webcam (sürekli)
                                                                img.jpg     # Image
                                                                vid.mp4     # Video
                                                                screen      # Ekran (sürekli)
                                                                ...                                            
                                                                                        
                        conf:           float,                  # Confidence minimum limiti.
                        imgsz:          int,                    # Resim boyutu.
                        iou:            float,                  # IoU limiti.
                        half:           bool,                   # Half precision (FP16).
                        device:         int, str, list,         # Çalıştıralacak aygıt.
                        show:           bool,                   # Mümkünse sonuçları göster webcam display vb.
                        save:           bool,                   # Detection'ı kaydet.
                        save_txt:       bool,                   # Sonuçları .txt olarak kaydet.
                        save_conf:      bool,                   # Sonuçları confidence ile kaydet.
                        show_labels:    bool,                   # Plotta labelları göster.
                        show_conf:      bool,                   # Plotta confidence göster.
                        max_det:        int,                    # Framede max detection.
                        agnostic_nms:   bool,                   # Agnostic NMS aktifleştir.
                        classes:        int, list[int],         # Sadece seçilen class(ları) detect et.
                    
                      ) 



"""



########################################################################################################################


# Tracking
import ultralytics
from ultralytics import YOLO

# Herhangi bir modeli yükle.
model = YOLO('yolov8n.pt')
model = YOLO('yolov8n-seg.pt')
model = YOLO('yolov8n-pose.pt')
model = YOLO('path/to/model.pt')

# Yüklediğin model ile tracking başlat.

# BoT-SORT default olarak kullanılacak.
results = model.track(source = 'https://youtubelinki.com', show = True)

# ByteTrack ile
results = model.track(source = 'https://youtubelinki.com', show = True, tracker = 'bytetrack.yaml')


########################################################################################################################



########################################################################################################################


# OpenCV ile Tracking | Bu kod Track plotunu yapmıyor sadece ID ataması yapıyor.
import cv2
from ultralytics import YOLO

# Modeli yükle
model = YOLO('yolov8n.pt')

# Video'nun adresini tanımla.
video_path = 'path/to/video.mp4'
capture = cv2.VideoCapture(video_path)

# Her frame'i kontrol et.
while capture.isOpened():
    ret , frame = capture.read()
    
    if not ret:
        break
    
    # Aktif frame'i track et.
    results = model.track(frame, persist = True)
    
    # Sonuçları frame'e yaz.
    annotated_frame = results[0].plot()
    
    # Display
    cv2.imshow('YOLOV8 Tracking', annotated_frame)
    
    # 'Q' ya basarak loopu durdur.
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
    
# Capture'ı bırak, displayi kapat.
capture.release()
cv2.destroyAllWindows()


########################################################################################################################



########################################################################################################################


# Tracking plotları ile OpenCV
import cv2
import numpy as np
from collections import defaultdict
from ultralytics import YOLO

# Modeli Yükle
model = YOLO('yolov8n.pt')

# Video'yu kullan.
video_path = r"C:\Users\ALKAN\Downloads\Road traffic video for object recognition.mp4"
capture = cv2.VideoCapture(video_path)

# Track geçmişini kaydet.
track_history = defaultdict(lambda: [])

# Tüm frameleri kontrol et.
while capture.isOpened():
    ret, frame = capture.read()

    if not ret:
        break
    
    # Tracking işlemini framede yap.
    results = model.track(frame, persist = True)
    
    # Kutu ve ID'leri al.
    boxes = results[0].boxes.xywh.cpu()
    track_ids = results[0].boxes.id.int().cpu().tolist()

    # Frame'e sonuçları yaz.
    annotated_frame = results[0].plot()
    
    # Trackleri yazma loopu
    for box, track_id in zip(boxes, track_ids):
        x, y, w, h = box
        track = track_history[track_id]
        
        # x, y merkez noktası.
        track.append((float(x), float(y)))

        # 90 frame için 90 track
        if len(track) > 30:
            track.pop(0)
            
        # Tracking izlerini çiz.
        points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
        cv2.polylines(annotated_frame, [points], isClosed = False, color = (0, 0, 200), thickness = 5)
    
    # annotated_frame'i göster.    
    cv2.imshow('YOLOV8 Izli Tracking', annotated_frame)
    
    # 'Q' ya basılırsa loopu bitir.
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
    
# capture'ı bırak, displayleri kapat.
capture.release()
cv2.destroyAllWindows()
    
    
########################################################################################################################



########################################################################################################################


# Multithreaded Tracking, birden fazla kaynakta aynı anda tracking.
import threading
import cv2
import numpy as np
from collections import defaultdict
from ultralytics import YOLO

# Threadde tracking işlemini yapmak.
def run_tracker_in_thread(filename, model, display_name):
    
    track_history = defaultdict(lambda: [])
    
    # Kullanacağımız video.
    video = cv2.VideoCapture(filename)
    
    # Videodaki frame sayısı.
    frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    
    for _ in range(frames):
        ret, frame = video.read()
        
        if not ret:
            break
        
        # Tracking işlemini yap.
        results = model.track(source = frame, persist = True)
        
        boxes = results[0].boxes.xywh.cpu()
        track_ids = results[0].boxes.id.int().cpu().tolist()
        
        # Sonuçu plotla.
        res_plotted = results[0].plot()
        
        for box, track_id in zip(boxes, track_ids):
            x, y, w, h = box
            track = track_history[track_id]
            
            track.append((float(x), float(y)))
            
            if len(track) > 30:
                track.pop(0)
                
            points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
            cv2.polylines(res_plotted, [points], isClosed = False, color = (0, 0, 200), thickness = 5)
        
        # Displayde göster, q ya basarak bitir.
        cv2.imshow(display_name, res_plotted)
        
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
        
# Modelleri yükle.
model1 = YOLO('yolov8n.pt')
model2 = YOLO('yolov8n-seg.pt')     

# Video adreslerini tanımla.
video1 = 'path/to/video1.mp4'
video2 = 'path/to/video2.mp4'

# Tracker threadlerini oluştur.
tracker_thread1 = threading.Thread(target = run_tracker_in_thread, args = (video1, model1, 'Display 1'), daemon = True)
tracker_thread2 = threading.Thread(target = run_tracker_in_thread, args = (video2, model2, 'Display 2'), daemon = True)

# Tracker threadlerini başlat.
tracker_thread1.start()
tracker_thread2.start()

# Tracker threadlerini bitmesini bekle.
tracker_thread1.join()
tracker_thread2.join()

# Displayi kapat.
cv2.destroyAllWindows()     