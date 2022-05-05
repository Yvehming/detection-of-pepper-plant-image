# object_name = ['pepper', 'root', 'pepper', 'root']
# detected_boxes =[[247, 105, 567, 474], [418, 411, 445, 467], [232, 121, 514, 385], [348, 321, 368, 366]]
boxes = [['pepper',247, 105, 567, 474], ['root',418, 411, 445, 467], ['pepper',232, 121, 514, 385], ['root',348, 321, 368, 366]]
object_name = []
detected_boxes = []
for obj in boxes:
    object_name.append(obj[0])
for box in boxes:
    detected_boxes.append(box[1:5])
if object_name.count('pepper') > 1:
    height = []
    for i in range(len(object_name)):
        if object_name[i] == 'pepper':
            height.append(detected_boxes[i][3] - detected_boxes[i][1])
    print(height)