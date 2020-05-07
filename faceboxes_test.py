from models.faceboxes.inference import build_net, get_face_boxes


net, prior_data = build_net((500, 469), 'cpu')
print(len(get_face_boxes(net, prior_data, 'data/2007_004049.jpg', 1, 0)))

