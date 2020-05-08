## Face Tracking System

### Requirements

```
Python3.6
opencv-python
numpy
torch1.4.0
```

### Code Structure

```
|Projects - |data - demo.mp4  source video file
                  - test.jpg  faceboxes infrence test image
                   
            |models - |faceboxes  clone from 
            					- infrence.py detect faces using faceboxes
                   - light_cnn.py
                   - mobilenetv3.py
                           
            |output - |images  saved tracking images
                    - |videos   saved videos from tracking images
            
            |tracker- instance.py  process the tracking faces
                    - muti_object_controller.py  control the tracking instances 
                    - kalman.py   make prediction by kalman filter
                    - kcf.py  make prediction by kcf filter
                    - kcftracker.py kcf source file
                    - fhog.py  using hog features in kcf filter
                    
            |utils  - get_ID_emption.py  get face_id and face emotion
                    - make_video.py  make video from saved images
                    - util.py   tools
                    - video_helper.py  process video file
                    - visualizer.py  draw tracking info and show results
                    
            |weights   weights of models
            
            -config.py   config the parameters of project
            -face_tracking.py   main

```

### Usage

```
run face_tracking.py
```

### System Design

    input: frame of video
    output:tracking instances in the frame
    
    - 1:no tracking(detect every frame,need bbox smoothing)
    - 2：need tracking
        - A:current frame has no detection->just make prediction
        	-->object_controller.update_without_detection(frame)
        - B:current frame has detection->make prediction then correction
            - B.1:make detection (disordered)
            - B.2:make prediction (ordered)
            - B.3:make correction (match then correct prediction)
                -->object_controller.update_with_detection(detects, frame)
                    - a:current has no instances->add detections to instances
                    - b:current has instances(need match)
                    	- b.1: matched prediction with detection
                    	- b.2: correct the matched instances
                    	- b.3: non-matched instances
                    		-aa:failed to detected(has instance,no detection)
                    			-->keep prediction for a while then remove
                    		-bb:newly detected(no instance,has detection)
                    			-bb.1:add detection to instances
                    			-bb.2:add to tracking for a while then add to instances

