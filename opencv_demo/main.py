import cv2
import time


#主函数
if __name__ == '__main__':

    # 开启ip摄像头
    # 设置账号密码
    video = "http://admin:admin@192.168.43.1:8081/"  # 此处@后的ipv4 地址需要修改为自己的地址
    cap = cv2.VideoCapture(video)
    # 读取 cascade 文件
    faceCascade = cv2.CascadeClassifier("./aboutxml/haarcascade_frontalface_alt2.xml")  # 改成自己存放.xml文件的路径
    num = 0
    while True:
        ret, frame = cap.read()
        frameGray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # 转换为灰度图像
        detectionResult = faceCascade.detectMultiScale(frameGray, scaleFactor=1.2, minNeighbors=3, minSize=(32, 32))
        cal = 0
        if len(detectionResult) > 0:
            for detection in detectionResult:
                cal += 1
                if cal > 5:
                    break
                x, y, w, h = detection
                cv2.rectangle(frame, (x - 10, y - 10), (x + w + 10, y + h + 10),
                              (255, 0, 0), 2)  # 加 bounding box
                boxname = "face - " + str(cal)
                cv2.putText(frame, boxname, (x, y + h), cv2.FONT_HERSHEY_PLAIN, 2,
                            (0, 0, 0), 2)  # 给 bounding box 加注解
        cv2.imshow("frame", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):  # 按“q”键退出程序
            break
    cap.release()
