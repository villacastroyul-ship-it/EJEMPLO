import cv2

#cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)

cap = cv2.VideoCapture(
    "http://192.168.0.182:4747/video"
)

CascoBClassif = cv2.CascadeClassifier(
    'cascade.xml'
)

while True:

    ret, frame = cap.read()

    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )

    toy = CascoBClassif.detectMultiScale(
        gray,
        scaleFactor=5,
        minNeighbors=91,
        minSize=(70,78)
    )

    for (x,y,w,h) in toy:

        margen_x = int(w * 0.70)
        margen_y = int(h * 0.40)

        x1 = max(x - margen_x, 0)
        y1 = max(y - margen_y, 0)

        x2 = x + w + margen_x
        y2 = y + h + margen_y

        cv2.rectangle(
            frame,
            (x1,y1),
            (x2,y2),
            (0,255,0),
            2
        )

        cv2.putText(
            frame,
            'Casco_Operario',
            (x1,y1-10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0,255,0),
            2,
            cv2.LINE_AA
        )

    cv2.imshow(
        'frame',
        frame
    )

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
